#!/usr/bin/env python3
"""
Delete jobs from the IRSA SPHEREx queue.

For each job in the provided dump(s) where:
  - phase is COMPLETED, ABORTED, or ERROR (terminal), AND
  - either the target label is already on disk in spectra/, OR
  - force=True

…send HTTP DELETE to the jobUrl. Per IVOA UWS spec, this removes the job
from the server.

Auth via the same SPHEREX_JSESSIONID/SPHEREX_USRKEY/SPHEREX_INGRESSCOOKIE env
vars as the rest of the pipeline.

Usage:
    SPHEREX_*=... python cleanup_queue.py [--dry-run] [--force] [dump.json ...]

Defaults to processing all jobids_*.txt files in evolved_stars/ plus the
hard-coded EXTRA list below (for new UUIDs not yet saved to a file).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

OUT = Path(__file__).parent
SPEC_DIR = OUT / "spectra"

# UUIDs from the May 11 chat-paste dump not in the older saved files.
# Format: (uuid, label, phase) — only entries where the target is on disk
# need delete; others tracked in case we want abort/cleanup separately.
EXTRA_JOBS_2026_05_11 = [
    # COMPLETED on May 11 (these targets are/will be on disk)
    ("e2b4f05c-ad8e-4a24-905d-bde812069a5c", "2M06504687-7006403", "COMPLETED"),
    ("fba50935-8d86-4f8c-99d3-abcbe2af4d4f", "2M06372680-7215095", "COMPLETED"),
    ("da8beff9-2b95-4fd1-be9d-5f8247041bf9", "2M06290911-7110551", "COMPLETED"),
    ("8f38089c-5210-4cb5-9432-54b7673ea5ab", "2M06215632-7809460", "COMPLETED"),
    ("a30b27a4-f3c7-463f-aab5-85be0434ab64", "2M06085071-7322463", "COMPLETED"),
    ("48c90314-6146-43e5-be76-e2e093cd5f8b", "2M06215632-7809460", "COMPLETED"),
    ("a678fe55-258d-4f62-85b8-8d24e27823f3", "2M06085071-7322463", "COMPLETED"),
    ("e4020033-bed0-48d6-86f0-c4056c18716d", "2M05583830-7803200", "COMPLETED"),
    ("f188efea-dc54-4e38-9a1b-342fb4e1b56e", "2M05583830-7803200", "COMPLETED"),
    ("2f8b735d-5a68-4524-b3cf-6a636d142c36", "2M05411019-7907435", "COMPLETED"),
    ("e09370b4-796b-48f6-ad18-cd54d72f0fdf", "2M05411019-7907435", "COMPLETED"),
    ("eee0bed0-f460-426f-8e9d-bde7c75daebf", "2M05225425-7205447", "COMPLETED"),
    ("fbf3fab8-2d43-4b64-8f0c-c07e6fc33e4c", "2M04562353-7638191", "COMPLETED"),
    ("db43f76f-4e5b-4e50-b048-999b748e9b6c", "2M05225425-7205447", "COMPLETED"),
    ("2ae25659-c13d-447d-8818-fa47b5ad8b24", "2M04562353-7638191", "COMPLETED"),
    ("ede5a181-b936-44e6-808e-c1f2b3aad0ad", "2M01255585-8423058", "COMPLETED"),
    ("91f4d0c0-8d7a-475a-ba5a-1c7009dcb47b", "2M23311261+8141416", "COMPLETED"),
    ("dbf67e19-477b-4f25-b34d-d30aef587ca8", "2M01255585-8423058", "COMPLETED"),
    ("dd10bc35-104d-4b51-a865-5e9b18918af1", "2M23311261+8141416", "COMPLETED"),
    ("52681d9d-ea51-4b76-bca1-04b58659ebe5", "2M21492142+7018212", "COMPLETED"),
    ("4711cc3a-240a-46b7-91d0-7f86a606a1c0", "2M18201731+7125115", "ABORTED"),
    ("2e94af32-9ddc-4a90-aaf8-319c41f1d64b", "2M18201731+7125115", "COMPLETED"),
    # ERROR and stalled in-flight (no result, but clean up)
    ("2ead7319-12b6-4afd-add8-0f7b9dd2fcfe", "2M07124937-7618392", "ERROR"),
    ("5c485a96-2524-4a54-a646-11a74f59c6d3", "2M07053659-7703286", "ERROR"),
    ("c3a66736-b8b7-4084-836f-e6e6edd783e8", "2M06504687-7006403", "ERROR"),
]


BASE = "https://irsa.ipac.caltech.edu/api/spherex/spectrophotometry/async"


def make_session() -> requests.Session:
    js = os.environ["SPHEREX_JSESSIONID"]
    uk = os.environ["SPHEREX_USRKEY"]
    ic = os.environ.get("SPHEREX_INGRESSCOOKIE")
    s = requests.Session()
    s.cookies.update({"JSESSIONID": js, "usrkey": uk})
    if ic:
        s.cookies.set("INGRESSCOOKIE", ic)
    return s


def load_jobs_from_dumps(paths: list[Path]) -> list[tuple[str, str, str]]:
    """Return (uuid, label, phase) tuples from job dump JSON files."""
    out = []
    for p in paths:
        try:
            raw = json.loads(p.read_text())
        except Exception as e:
            print(f"  WARN: can't parse {p}: {e}")
            continue
        for j in raw.get("jobs", []):
            pt = j.get("parameters", {}).get("POINT", "")
            label = pt.rsplit(",", 1)[1] if "," in pt else ""
            out.append((j["jobId"], label, j["phase"]))
    return out


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump_paths = ([Path(a) for a in args]
                  if args else sorted(OUT.glob("jobids_*.txt")))

    print(f"Dumps to process: {len(dump_paths)}")
    for p in dump_paths:
        print(f"  - {p.name}")
    print()

    # Collect all (uuid, label, phase)
    jobs = load_jobs_from_dumps(dump_paths)
    jobs.extend(EXTRA_JOBS_2026_05_11)

    # Deduplicate by UUID
    seen = set()
    uniq = []
    for u, l, ph in jobs:
        if u in seen:
            continue
        seen.add(u)
        uniq.append((u, l, ph))
    print(f"Unique jobs: {len(uniq)}")

    have = {p.stem for p in SPEC_DIR.glob("*.vot")}
    deletable = []
    for u, l, ph in uniq:
        if ph not in ("COMPLETED", "ABORTED", "ERROR"):
            continue
        if force or l in have:
            deletable.append((u, l, ph))
    print(f"Deletable (terminal + on-disk-or-forced): {len(deletable)}")
    print()

    if dry_run:
        print("=== DRY RUN — would DELETE these ===")
        for u, l, ph in deletable[:20]:
            print(f"  [{ph:9s}] {l:<22s} {u[:8]}...")
        if len(deletable) > 20:
            print(f"  ... and {len(deletable) - 20} more")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    sess = make_session()

    def _delete_one(item):
        u, l, ph = item
        try:
            r = sess.delete(f"{BASE}/{u}", allow_redirects=False, timeout=10)
            return (u, l, r.status_code, None)
        except Exception as e:
            return (u, l, None, str(e)[:80])

    n_ok = n_fail = n_timeout = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_delete_one, x) for x in deletable]
        for i, f in enumerate(as_completed(futures), 1):
            u, l, code, err = f.result()
            if code in (200, 204, 303):
                n_ok += 1
            elif err is not None:
                n_timeout += 1
            else:
                n_fail += 1
            if i <= 10 or i % 20 == 0 or i == len(deletable):
                status = ("OK" if code in (200,204,303)
                          else f"TIMEOUT" if err else f"FAIL({code})")
                print(f"  [{i:3d}/{len(deletable)}] {status:<10s} {l:<22s} {u[:8]}", flush=True)

    print(f"\nDone: deleted {n_ok}, failed {n_fail}, timeout {n_timeout}")


if __name__ == "__main__":
    main()
