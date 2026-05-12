#!/usr/bin/env python3
"""
Recover orphan results from the IRSA queue.

Reads recovery_targets.json (parsed from a UI-exported job dump), downloads
the VOTable result for each unique target label, saves to spectra/{label}.vot
(skips any already on disk).

If multiple completed jobs exist for the same target, picks the first one.
"""

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

OUT_DIR = Path(__file__).parent
SPEC_DIR = OUT_DIR / "spectra"
RECOVERY_PATH = OUT_DIR / "recovery_targets.json"


def main() -> None:
    SPEC_DIR.mkdir(exist_ok=True)
    if not RECOVERY_PATH.exists():
        raise SystemExit(f"Missing {RECOVERY_PATH}")
    records = json.loads(RECOVERY_PATH.read_text())
    print(f"Loaded {len(records)} recovery candidates")

    # Group by label, pick first per label
    by_label: dict[str, dict] = {}
    for r in records:
        by_label.setdefault(r["label"], r)
    print(f"Unique labels: {len(by_label)}")

    sess = requests.Session()  # no auth needed for completed UWS results
    sess.headers["User-Agent"] = "spherex-evolved-stars-recovery/0.1"

    n_dl = n_skip = n_err = 0
    for label, r in sorted(by_label.items()):
        dest = SPEC_DIR / f"{label}.vot"
        if dest.exists():
            n_skip += 1
            continue
        url = r["job_url"].rstrip("/") + "/results/phot-tool-result.vot"
        try:
            resp = sess.get(url, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            n_dl += 1
            print(f"  [recover] {label} ({len(resp.content):,} bytes)")
        except Exception as e:
            n_err += 1
            print(f"  [error] {label}: {e}")

    print(f"\nDone: downloaded {n_dl}, skipped {n_skip} (already on disk), errors {n_err}")


if __name__ == "__main__":
    main()
