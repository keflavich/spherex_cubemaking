#!/usr/bin/env python3
"""
Abort orphan jobs (jobs in the user's IRSA queue that we don't track).

Reads the latest job dump (default: latest jobids_*.txt in evolved_stars/)
and our jobs.json ledger. Any EXECUTING/QUEUED job whose UUID is NOT in
the ledger gets POST PHASE=ABORT to its UWS endpoint.

Auth via SPHEREX_JSESSIONID + SPHEREX_USRKEY + SPHEREX_INGRESSCOOKIE env vars.

Usage:
    SPHEREX_*=...  python abort_orphans.py [path/to/jobs_dump.txt]
"""

import json
import os
import sys
from pathlib import Path

import requests

OUT = Path(__file__).parent
LEDGER = OUT / "jobs.json"


def latest_dump() -> Path:
    matches = sorted(OUT.glob("jobids_*.txt"))
    if not matches:
        raise SystemExit(f"No jobids_*.txt found in {OUT}")
    return matches[-1]


def main() -> None:
    dump_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_dump()
    print(f"Reading dump: {dump_path}")
    raw = json.loads(dump_path.read_text())

    ledger = json.loads(LEDGER.read_text())
    ledger_uuids = {v["job_url"].rsplit("/", 1)[-1] for v in ledger.values()}

    js = os.environ.get("SPHEREX_JSESSIONID")
    uk = os.environ.get("SPHEREX_USRKEY")
    ic = os.environ.get("SPHEREX_INGRESSCOOKIE")
    if not (js and uk):
        raise SystemExit("SPHEREX_JSESSIONID and SPHEREX_USRKEY required")
    sess = requests.Session()
    sess.cookies.update({"JSESSIONID": js, "usrkey": uk})
    if ic:
        sess.cookies.set("INGRESSCOOKIE", ic)

    targets = []
    for j in raw["jobs"]:
        if j["phase"] not in ("EXECUTING", "QUEUED"):
            continue
        uuid = j["jobId"]
        if uuid in ledger_uuids:
            continue
        label = (j["parameters"]["POINT"].rsplit(",", 1)[1]
                 if "," in j["parameters"]["POINT"] else "?")
        targets.append((uuid, label, j["phase"]))

    print(f"Found {len(targets)} orphan in-flight jobs to abort:")
    for uuid, label, phase in targets:
        print(f"  [{phase:9s}] {label:<22s} {uuid[:8]}...")

    if not targets:
        return
    if "--dry-run" in sys.argv:
        print("(dry run — not aborting)")
        return

    for uuid, label, _ in targets:
        url = f"https://irsa.ipac.caltech.edu/api/spherex/spectrophotometry/async/{uuid}/phase"
        try:
            r = sess.post(url, data={"PHASE": "ABORT"},
                          allow_redirects=False, timeout=30)
            ok = r.status_code in (200, 303)
            print(f"  {'OK' if ok else 'FAIL':<4s} {label} ({r.status_code})")
        except Exception as e:
            print(f"  ERR  {label}: {e}")


if __name__ == "__main__":
    main()
