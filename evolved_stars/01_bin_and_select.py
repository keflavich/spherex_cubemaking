#!/usr/bin/env python3
"""
Bin APOGEE evolved-star candidates and pick N per cell for SPHEREx submission.

Bin scheme (set 2026-05-08, refined after seeing the data):
  - logg:    [1.0, 1.8), [1.8, 2.4), [2.4, 3.0)   (3 RGB bins)
  - [Fe/H]:  [-0.20, -0.07), [-0.07, +0.07), [+0.07, +0.20]  (3 bins)
  - age (Gyr, astroNN proxy for mass):
      young  : 0  – 3
      mid    : 3  – 6
      old    : 6  – 14
  - RC tag: logg ∈ [2.3, 2.6) ∩ Teff ∈ [4500, 5100)
            (this overlaps the [2.4, 3.0) RGB bin — RC stars get split out)

Cells: 3 logg × 3 age × 3 [Fe/H] = 27 RGB + (3 age × 3 [Fe/H]) = 9 RC = 36
Targets: 5 / cell × 36 cells (≤ 180; some cells may be empty)
"""

from pathlib import Path

import numpy as np
from astropy.table import Table

OUT_DIR = Path(__file__).parent
JOINED = OUT_DIR / "apogee_evolved_joined.fits"
TARGETS = OUT_DIR / "spherex_target_list.ipac"

LOGG_EDGES = [(1.0, 1.8), (1.8, 2.4), (2.4, 3.0)]
FEH_EDGES = [(-0.20, -0.07), (-0.07, +0.07), (+0.07, +0.20)]
AGE_EDGES = [(0.0, 3.0), (3.0, 6.0), (6.0, 14.5)]   # Gyr

RC_LOGG = (2.3, 2.6)
RC_TEFF = (4500, 5100)

N_PER_BIN = 5


def _arr(col) -> np.ndarray:
    """Return ndarray with masked entries → NaN."""
    a = np.asarray(col, dtype=float)
    if hasattr(col, "mask"):
        a = a.copy()
        a[col.mask] = np.nan
    return a


def main() -> None:
    if not JOINED.exists():
        raise SystemExit(f"Run 00_query_apogee.py first (missing {JOINED})")
    t = Table.read(JOINED)
    print(f"Loaded {len(t):,} candidates from {JOINED.name}\n")

    teff = _arr(t["Teff"])
    logg = _arr(t["logg"])
    feh = _arr(t["[M/H]"])
    age = _arr(t["age_lowess_correct"])
    ra = _arr(t["RAJ2000"])
    is_rc = (logg >= RC_LOGG[0]) & (logg < RC_LOGG[1]) & \
            (teff >= RC_TEFF[0]) & (teff < RC_TEFF[1])

    rows = []
    rec = []   # one row per cell with counts (for SUMMARY)

    for fi, (flo, fhi) in enumerate(FEH_EDGES):
        feh_m = (feh >= flo) & (feh < fhi)
        for ai, (alo, ahi) in enumerate(AGE_EDGES):
            age_m = (age >= alo) & (age < ahi)
            # RGB bins
            for li, (llo, lhi) in enumerate(LOGG_EDGES):
                logg_m = (logg >= llo) & (logg < lhi)
                cell = feh_m & age_m & logg_m & ~is_rc
                idx = _pick(t, ra, cell, N_PER_BIN)
                rows.extend(idx)
                rec.append((f"RGB_feh{fi}_age{ai}_logg{li}",
                            int(cell.sum()), len(idx)))
            # RC bin
            cell = feh_m & age_m & is_rc
            idx = _pick(t, ra, cell, N_PER_BIN)
            rows.extend(idx)
            rec.append((f"RC_feh{fi}_age{ai}", int(cell.sum()), len(idx)))

    rows = np.unique(rows)
    sel = t[rows]
    print(f"\nSelected {len(sel)} unique targets across {len(rec)} cells\n")

    # Bin summary
    print(f"{'cell':<28s}  {'pool':>6s}  {'picked':>6s}")
    print("-" * 50)
    for name, pool, picked in rec:
        flag = "" if picked > 0 else "  EMPTY"
        print(f"  {name:<28s}  {pool:>6d}  {picked:>6d}{flag}")

    # Strip [M/H] and [a/M] FITS-incompatible names for IPAC writer
    sel = sel.copy()
    if "[M/H]" in sel.colnames: sel.rename_column("[M/H]", "MH")
    if "[a/M]" in sel.colnames: sel.rename_column("[a/M]", "aM")
    if "4.5mag" in sel.colnames: sel.rename_column("4.5mag", "W2_45mag")
    sel.write(TARGETS, format="ascii.ipac", overwrite=True)
    print(f"\nWrote {TARGETS}")


def _pick(t: Table, ra: np.ndarray, mask: np.ndarray, n: int) -> list[int]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    if len(idx) <= n:
        return list(idx)
    # Stratified across RA so we spread targets over the sky
    order = np.argsort(ra[idx])
    stride = max(1, len(idx) // n)
    return list(idx[order][::stride][:n])


if __name__ == "__main__":
    main()
