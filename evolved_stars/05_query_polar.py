#!/usr/bin/env python3
"""
Polar-cap supplement query: pull APOGEE evolved stars near the celestial poles
(|Dec| > 70°) with the same atmospheric / brightness cuts as the main sample.

Two passes:
  Pass A — same cuts as 00_query_apogee.py: K=10-13, [Fe/H]±0.2, logg 1-3,
           [α/M] < 0.1, Teff 3500-5500.
  Pass B — relaxed [α/M] (any) and looser [Fe/H] (-0.4..+0.3), since halo /
           thick disk stars are α-enhanced and metal-poor.

Reports counts both ways. Then writes a polar_target_list.ipac with whatever
we get (Pass A preferred where available).
"""

from pathlib import Path

import numpy as np
from astropy.table import Table, join, unique, vstack
from astroquery.vizier import Vizier

OUT_DIR = Path(__file__).parent
ASTRONN_PATH = OUT_DIR / "apogee_astroNN-DR17.fits"
POLAR_OUT = OUT_DIR / "polar_target_list.ipac"
MAIN_LIST = OUT_DIR / "spherex_target_list.ipac"

DEC_CUT = 70.0   # |Dec| > this is "polar"
N_PER_POLE = 30  # try to grab this many per pole; will downselect to bins later


def query_pass(label: str, dec_filter: str,
               feh_filter: str, alpha_filter: str | None) -> Table:
    print(f"=== {label} ===")
    cols = [
        "APOGEE", "RAJ2000", "DEJ2000",
        "Teff", "logg", "[M/H]", "[a/M]",
        "Jmag", "Hmag", "Ksmag", "e_Ksmag",
        "4.5mag", "Gmag", "BPmag", "RPmag",
        "AFlag", "SFlag",
    ]
    cf = {
        "Ksmag": "10.0..13.0",
        "[M/H]": feh_filter,
        "logg":  "1.0..3.0",
        "Teff":  "3500..5500",
        "DEJ2000": dec_filter,
    }
    if alpha_filter is not None:
        cf["[a/M]"] = alpha_filter
    v = Vizier(columns=cols, column_filters=cf, row_limit=-1)
    cats = v.get_catalogs("III/286")
    if not cats:
        print(f"  Vizier returned no catalogs")
        return Table()
    cat = cats[0]
    print(f"  raw rows: {len(cat):,}")

    # ASPCAP quality cut (STAR_BAD bit 23, CHI2_BAD bit 13)
    if "AFlag" in cat.colnames:
        bad = (1 << 23) | (1 << 13)
        cat = cat[(cat["AFlag"] & bad) == 0]
        print(f"  after ASPCAPflag cut: {len(cat):,}")

    cat = unique(cat, keys="APOGEE", keep="first")
    print(f"  unique APOGEE_IDs: {len(cat):,}")
    return cat


def main() -> None:
    if not ASTRONN_PATH.exists():
        raise SystemExit(f"Need {ASTRONN_PATH} (run 00_query_apogee.py first)")

    # North cap: Dec > 70
    north_a = query_pass("Pass A: NORTH cap, [α/M]<0.1, [Fe/H]±0.2",
                         dec_filter=f">{DEC_CUT}",
                         feh_filter="-0.20..0.20",
                         alpha_filter="-0.10..0.10")
    north_b = query_pass("Pass B: NORTH cap, [α/M] any, [Fe/H]-0.4..+0.3",
                         dec_filter=f">{DEC_CUT}",
                         feh_filter="-0.40..0.30",
                         alpha_filter=None)

    # South cap: Dec < -70
    south_a = query_pass("Pass A: SOUTH cap, [α/M]<0.1, [Fe/H]±0.2",
                         dec_filter=f"<{-DEC_CUT}",
                         feh_filter="-0.20..0.20",
                         alpha_filter="-0.10..0.10")
    south_b = query_pass("Pass B: SOUTH cap, [α/M] any, [Fe/H]-0.4..+0.3",
                         dec_filter=f"<{-DEC_CUT}",
                         feh_filter="-0.40..0.30",
                         alpha_filter=None)

    # Combine: prefer Pass A per pole; fall back to Pass B if A is empty
    pieces = []
    for a, b, name in [(north_a, north_b, "NORTH"),
                        (south_a, south_b, "SOUTH")]:
        if len(a) > 0:
            tag = "A"
            chosen = a
        else:
            tag = "B"
            chosen = b
        if len(chosen) == 0:
            print(f"  {name}: still empty after Pass B — APOGEE likely doesn't cover")
            continue
        chosen = chosen.copy()
        chosen["pole_tag"] = [f"{name}_{tag}"] * len(chosen)
        pieces.append(chosen)

    if not pieces:
        print("\nNo polar candidates found — APOGEE truly doesn't cover the caps "
              "with these cuts. Need a different survey (LAMOST, GALAH).")
        return

    polar = vstack(pieces)
    polar.rename_column("APOGEE", "APOGEE_ID")

    # Join astroNN ages
    astronn = Table.read(ASTRONN_PATH, hdu=1)
    astronn = unique(astronn, keys="APOGEE_ID", keep="first")
    keep = ["APOGEE_ID"]
    for c in ("age_lowess_correct", "age_total_error",
              "weighted_dist", "weighted_dist_error"):
        if c in astronn.colnames:
            keep.append(c)
    polar = join(polar, astronn[keep], keys="APOGEE_ID", join_type="left")

    # Drop targets already in the main list
    if MAIN_LIST.exists():
        main = Table.read(MAIN_LIST, format="ascii.ipac")
        main_ids = set(str(s).strip() for s in main["APOGEE_ID"])
        before = len(polar)
        polar = polar[[str(s).strip() not in main_ids for s in polar["APOGEE_ID"]]]
        print(f"\nDe-duplicated against main list: {before} → {len(polar)}")

    # FITS-incompatible names → IPAC-friendly
    if "[M/H]" in polar.colnames: polar.rename_column("[M/H]", "MH")
    if "[a/M]" in polar.colnames: polar.rename_column("[a/M]", "aM")
    if "4.5mag" in polar.colnames: polar.rename_column("4.5mag", "W2_45mag")

    # Stratify per pole, take up to N_PER_POLE each spread in RA
    out_rows = []
    for tag in sorted(set(polar["pole_tag"])):
        sub = polar[polar["pole_tag"] == tag]
        if len(sub) == 0:
            continue
        order = np.argsort(sub["RAJ2000"])
        if len(sub) <= N_PER_POLE:
            picked = sub[order]
        else:
            stride = max(1, len(sub) // N_PER_POLE)
            picked = sub[order[::stride][:N_PER_POLE]]
        print(f"  {tag}: {len(sub)} candidates -> picked {len(picked)}")
        out_rows.append(picked)

    if not out_rows:
        print("Nothing to write.")
        return
    final = vstack(out_rows)
    final.write(POLAR_OUT, format="ascii.ipac", overwrite=True)
    print(f"\nWrote {POLAR_OUT}  ({len(final)} polar targets)")
    print()
    print("Param ranges in polar sample:")
    for c in ("Teff", "logg", "MH", "aM", "Ksmag",
              "age_lowess_correct", "DEJ2000"):
        if c in final.colnames:
            v = np.asarray(final[c], dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                print(f"  {c:<22s}  {v.min():>7.2f} .. {v.max():>7.2f}  "
                      f"(median {np.median(v):.2f})")


if __name__ == "__main__":
    main()
