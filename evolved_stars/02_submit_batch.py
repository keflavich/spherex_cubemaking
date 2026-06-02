#!/usr/bin/env python3
"""
Submit all targets in spherex_target_list.ipac to the IRSA Spectrophotometry
Tool, respecting the 2-concurrent-jobs throttle. Resumable: skips targets
whose VOTable is already on disk, and skips jobs already in flight.

Auth: requires Firefly cookies from a real browser session. Either point at a
Netscape cookies.txt (from a browser extension), or paste JSESSIONID + usrkey
into the env vars below.

Usage:
    SPHEREX_COOKIES=~/sphx_cookies.txt  python 02_submit_batch.py
    # or
    SPHEREX_JSESSIONID=...  SPHEREX_USRKEY=...  python 02_submit_batch.py

Resumes by checking spectra/<APOGEE_ID>.vot — delete to re-extract.
A small JSON ledger (jobs.json) tracks in-flight jobs across restarts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from astropy.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from spherex_spectrophot import (  # noqa: E402
    JobHandle, Target,
    download_results, is_terminal, session_from_cookies, status, submit,
)

OUT_DIR = Path(__file__).parent
TARGETS_PATH = OUT_DIR / "spherex_target_list.ipac"
SPEC_DIR = OUT_DIR / "spectra"
LEDGER = OUT_DIR / "jobs.json"

POLL_INTERVAL = 30.0       # seconds between status polls
MAX_CONCURRENT = 2         # IRSA quota
SUBMIT_GAP = 1.5           # seconds between back-to-back submissions
MIN_GAP_AFTER_DONE = 30.0  # wait after a [done] before next submit
                           # — empirically prevents the "bare response" orphan


def load_ledger() -> dict[str, dict]:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {}


def save_ledger(ledger: dict[str, dict]) -> None:
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True))


def main() -> None:
    if not TARGETS_PATH.exists():
        raise SystemExit(f"Run 01_bin_and_select.py first ({TARGETS_PATH} missing)")
    SPEC_DIR.mkdir(exist_ok=True)
    targets_tbl = Table.read(TARGETS_PATH, format="ascii.ipac")
    print(f"Loaded {len(targets_tbl)} targets from {TARGETS_PATH.name}")

    cookies_path = os.environ.get("SPHEREX_COOKIES")
    jsessionid = os.environ.get("SPHEREX_JSESSIONID")
    usrkey = os.environ.get("SPHEREX_USRKEY")
    ingresscookie = os.environ.get("SPHEREX_INGRESSCOOKIE")
    if not (cookies_path or (jsessionid and usrkey)):
        raise SystemExit(
            "Need either SPHEREX_COOKIES=path/to/cookies.txt or both "
            "SPHEREX_JSESSIONID and SPHEREX_USRKEY env vars."
        )
    sess = session_from_cookies(cookies_path=cookies_path,
                                jsessionid=jsessionid, usrkey=usrkey,
                                ingresscookie=ingresscookie)

    ledger = load_ledger()
    todo = []
    for row in targets_tbl:
        apid = str(row["APOGEE_ID"]).strip()
        if (SPEC_DIR / f"{apid}.vot").exists():
            continue
        if apid in ledger and not ledger[apid].get("done"):
            # Already submitted; will be polled in the loop.
            continue
        # Don't re-try ABANDONED jobs — they hung indefinitely once and will
        # almost certainly hang again. ERROR/ABORTED jobs are worth re-trying
        # (often transient slurm issues).
        if (apid in ledger and ledger[apid].get("done")
                and ledger[apid].get("phase") == "ABANDONED"):
            continue
        todo.append(Target(ra=float(row["RAJ2000"]),
                           dec=float(row["DEJ2000"]),
                           label=apid))

    in_flight: dict[str, JobHandle] = {}
    # Re-hydrate handles for jobs left in ledger from a prior run
    for apid, entry in ledger.items():
        if not entry.get("done") and entry.get("job_url"):
            in_flight[apid] = JobHandle(
                job_id=entry["job_id"], job_url=entry["job_url"],
                target=Target(entry["ra"], entry["dec"], apid), raw={},
            )

    print(f"To submit: {len(todo)}    in-flight resumed: {len(in_flight)}    "
          f"already-done: {len(targets_tbl) - len(todo) - len(in_flight)}")

    todo_iter = iter(todo)
    next_target = next(todo_iter, None)
    last_done_at = 0.0  # epoch seconds; gate submissions after a completion
    poll_errors: dict[str, int] = {}  # consecutive poll-error counter per job

    while in_flight or next_target is not None:
        # Fill up to MAX_CONCURRENT, but wait if we just had a completion
        while next_target is not None and len(in_flight) < MAX_CONCURRENT:
            wait = (last_done_at + MIN_GAP_AFTER_DONE) - time.time()
            if wait > 0:
                print(f"[wait] {wait:.0f}s post-done cooldown")
                time.sleep(wait)
            try:
                h = submit(next_target, sess)
            except Exception as e:
                # Catch broadly — RuntimeError (bare-response), ReadTimeout
                # (IRSA hang), ConnectionError, anything else. Log as skip and
                # move on, never crash the driver.
                err_type = type(e).__name__
                print(f"[skip] {next_target.label}: {err_type}: {e}")
                next_target = next(todo_iter, None)
                continue
            in_flight[next_target.label] = h
            ledger[next_target.label] = {
                "job_id": h.job_id, "job_url": h.job_url,
                "ra": next_target.ra, "dec": next_target.dec,
                "submitted_at": time.time(), "done": False,
            }
            save_ledger(ledger)
            print(f"[submit] {next_target.label} -> {h.job_id}  "
                  f"in_flight={len(in_flight)}/{MAX_CONCURRENT}")
            next_target = next(todo_iter, None)
            time.sleep(SUBMIT_GAP)

        # Poll all in-flight
        for apid in list(in_flight.keys()):
            h = in_flight[apid]
            try:
                st = status(h, sess)
            except Exception as e:
                # TOTAL (not consecutive) error count — counter is never reset.
                # status() can intermittently succeed (returning QUEUED) which
                # would mask a fundamentally-hung job if we used "consecutive".
                poll_errors[apid] = poll_errors.get(apid, 0) + 1
                print(f"[poll-error] {apid} (#{poll_errors[apid]}): {e}")
                if poll_errors[apid] >= 6:
                    print(f"[abandon] {apid}: {poll_errors[apid]} total poll "
                          "errors, dropping from in-flight")
                    ledger[apid].update(done=True, phase="ABANDONED",
                                        error="poll endpoint unresponsive")
                    save_ledger(ledger)
                    del in_flight[apid]
                continue
            if not is_terminal(st["phase"]):
                continue
            if st["phase"] == "COMPLETED":
                paths = download_results(
                    st, SPEC_DIR, sess, name_prefix=f"{apid}__")
                # Move first result to canonical name
                if paths:
                    canonical = SPEC_DIR / f"{apid}.vot"
                    paths[0].rename(canonical)
                    print(f"[done] {apid} -> {canonical.name}")
                ledger[apid].update(done=True, phase="COMPLETED")
                last_done_at = time.time()
            else:
                print(f"[fail] {apid}: phase={st['phase']}  "
                      f"err={st.get('error')}")
                ledger[apid].update(done=True, phase=st["phase"],
                                    error=st.get("error"))
            save_ledger(ledger)
            del in_flight[apid]

        if in_flight:
            time.sleep(POLL_INTERVAL)

    print("\nAll done.")
    n_ok = sum(1 for v in ledger.values() if v.get("phase") == "COMPLETED")
    n_fail = sum(1 for v in ledger.values()
                 if v.get("done") and v.get("phase") != "COMPLETED")
    print(f"  completed: {n_ok}    failed: {n_fail}")


if __name__ == "__main__":
    main()
