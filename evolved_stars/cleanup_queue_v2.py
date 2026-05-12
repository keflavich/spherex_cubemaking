#!/usr/bin/env python3
"""
Remove jobs from the IRSA SPHEREx queue via Firefly's removeBgJob command.

Unlike the IVOA UWS DELETE (which appears broken/slow server-side), this
calls the CmdSrv/sync endpoint with `cmd=removeBgJob` and the **Firefly
internal jobId** (from meta.jobId in the dump, NOT the UWS UUID).

For each job in the dump(s) where:
  - phase ∈ {COMPLETED, ABORTED, ERROR}, and
  - target label is on disk in spectra/ (or --force)

…POST cmd=removeBgJob to CmdSrv.

Auth via SPHEREX_JSESSIONID/SPHEREX_USRKEY/SPHEREX_INGRESSCOOKIE env vars
(+ optional SPHEREX_JOSSO).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OUT = Path(__file__).parent
SPEC_DIR = OUT / "spectra"
CMDSRV = "https://irsa.ipac.caltech.edu/applications/spherex/CmdSrv/sync"


def make_session() -> requests.Session:
    s = requests.Session()
    s.cookies.update({
        "JSESSIONID": os.environ["SPHEREX_JSESSIONID"],
        "usrkey": os.environ["SPHEREX_USRKEY"],
    })
    if "SPHEREX_INGRESSCOOKIE" in os.environ:
        s.cookies.set("INGRESSCOOKIE", os.environ["SPHEREX_INGRESSCOOKIE"])
    if "SPHEREX_JOSSO" in os.environ:
        # Note: cookie name is JOSSO_REMEMBERME_josso (no trailing s)
        s.cookies.set("JOSSO_REMEMBERME_josso", os.environ["SPHEREX_JOSSO"])
    return s


def load_skip_orphans(log_path: Path) -> list[dict]:
    """Extract Firefly jobIds from [skip] log lines.

    These are orphan jobs created by the bare-response failure mode:
    server queued them but never returned a UUID. They sit on the user's
    slurm quota indefinitely. Each [skip] line in our log records the
    Firefly internal jobId in `last response jobId=...`.
    """
    if not log_path.exists():
        return []
    import re
    pat = re.compile(
        r"\[skip\]\s+(\S+):\s+No jobUrl after \d+ attempts.*?"
        r"last response jobId=(\S+)"
    )
    out = []
    seen = set()
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        label, ff = m.group(1), m.group(2)
        if ff in seen:
            continue
        seen.add(ff)
        out.append({"uuid": None, "ff_id": ff, "label": label,
                    "phase": "ORPHAN"})
    return out


def load_jobs(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        try:
            raw = json.loads(p.read_text())
        except Exception as e:
            print(f"  WARN: can't parse {p}: {e}", file=sys.stderr)
            continue
        for j in raw.get("jobs", []):
            pt = j.get("parameters", {}).get("POINT", "")
            label = pt.rsplit(",", 1)[1] if "," in pt else ""
            ff_id = j.get("meta", {}).get("jobId")
            out.append({"uuid": j["jobId"], "ff_id": ff_id,
                        "label": label, "phase": j["phase"]})
    return out


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump_paths = ([Path(a) for a in args]
                  if args else sorted(OUT.glob("jobids_*.txt")))

    print(f"Dumps: {[p.name for p in dump_paths]}")
    jobs = load_jobs(dump_paths)
    skip_orphans = load_skip_orphans(Path("/tmp/sphx_batch.log"))
    print(f"Skip-orphan Firefly jobIds in log: {len(skip_orphans)}")
    jobs.extend(skip_orphans)
    # dedup by ff_id
    seen, uniq = set(), []
    for j in jobs:
        if not j["ff_id"] or j["ff_id"] in seen:
            continue
        seen.add(j["ff_id"])
        uniq.append(j)
    print(f"Unique jobs (with ff_id): {len(uniq)}")

    have = {p.stem for p in SPEC_DIR.glob("*.vot")}
    deletable = []
    for j in uniq:
        # Always delete ERROR / ABORTED / ORPHAN (skip-orphans) — they have no
        # recoverable data and just consume slurm budget.
        if j["phase"] in ("ERROR", "ABORTED", "ORPHAN"):
            deletable.append(j)
        # Delete COMPLETED only if we already have the spectrum on disk
        # (or --force).
        elif j["phase"] == "COMPLETED" and (force or j["label"] in have):
            deletable.append(j)
    print(f"Deletable: {len(deletable)}")
    from collections import Counter
    by_phase = Counter(j["phase"] for j in deletable)
    print(f"  by phase: {dict(by_phase)}")

    if dry_run:
        for j in deletable[:10]:
            print(f"  [{j['phase']:9s}] {j['label']:<22s} ff={j['ff_id']}")
        if len(deletable) > 10:
            print(f"  ... and {len(deletable)-10} more")
        return

    sess = make_session()

    def _remove(j):
        try:
            r = sess.post(
                f"{CMDSRV}?cmd=removeBgJob",
                data={"jobId": j["ff_id"], "cmd": "removeBgJob"},
                timeout=15,
                allow_redirects=False,
            )
            return (j, r.status_code, None)
        except Exception as e:
            return (j, None, str(e)[:80])

    n_ok = n_fail = n_err = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_remove, j) for j in deletable]
        for i, f in enumerate(as_completed(futures), 1):
            j, code, err = f.result()
            if code in (200, 204, 303):
                n_ok += 1
            elif err is not None:
                n_err += 1
            else:
                n_fail += 1
            if i <= 5 or i % 25 == 0 or i == len(deletable):
                status = (f"OK({code})" if code in (200,204,303)
                          else f"ERR" if err else f"FAIL({code})")
                print(f"  [{i:3d}/{len(deletable)}] {status:<10s} "
                      f"{j['label']:<22s} ff={j['ff_id']}", flush=True)

    print(f"\nDone: removed {n_ok}, failed {n_fail}, error {n_err}")


if __name__ == "__main__":
    main()
