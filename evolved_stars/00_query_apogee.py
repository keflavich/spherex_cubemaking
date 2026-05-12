#!/usr/bin/env python3
"""
Pull APOGEE evolved-star candidates for SPHEREx 4 µm spectroscopy.

Cuts (set by user, 2026-05-08):
- K = 10..13 (Vega) — below SPHEREx saturation onset in all bands
- [M/H]   = -0.2 .. +0.2  (solar ± a bit)
- logg    =  1.0 .. 3.0   (RGB through RC)
- Teff    = 3500 .. 5500 K
- [α/M]   < 0.1            (low-α / thin-disk only, so mass-age mapping is meaningful)

Output: a single FITS table with allStar params + astroNN mass/age + RC flag.
Cross-match with APOKASC-3 (asteroseismic EVSTATE) is left to the next step,
since APOKASC-3 is Kepler-footprint only and isn't on Vizier in a clean form.
"""

from pathlib import Path

from astropy.io import fits
from astropy.table import Table, join
from astroquery.vizier import Vizier

OUT_DIR = Path(__file__).parent
ALLSTAR_OUT = OUT_DIR / "apogee_allstar_evolved.fits"
ASTRONN_OUT = OUT_DIR / "apogee_astroNN-DR17.fits"
JOINED_OUT  = OUT_DIR / "apogee_evolved_joined.fits"

ASTRONN_URL = (
    "https://data.sdss.org/sas/dr17/env/APOGEE_ASTRO_NN/"
    "apogee_astroNN-DR17.fits"
)


def fetch_allstar() -> Table:
    """Vizier III/286 = APOGEE DR17 allStar (Abdurro'uf+ 2022)."""
    print("Querying Vizier III/286 (APOGEE DR17 allStar)...")
    v = Vizier(
        columns=[
            "APOGEE_ID", "2MASS",
            "RAJ2000", "DEJ2000",
            "Teff", "logg", "[M/H]", "[a/M]",
            "Jmag", "Hmag", "Ksmag", "e_Ksmag",
            "4.5mag",  # WISE/IRAC ch2 — overlaps SPHEREx band 5
            "Gmag", "BPmag", "RPmag",
            "AFlag", "SFlag",  # Vizier-renamed ASPCAPflag, StarFlag
        ],
        column_filters={
            "Ksmag":  "10.0..13.0",
            "[M/H]": "-0.2..0.2",
            "logg":  "1.0..3.0",
            "Teff":  "3500..5500",
            "[a/M]": "-0.1..0.1",
        },
        row_limit=-1,
    )
    cats = v.get_catalogs("III/286")
    if len(cats) == 0:
        raise RuntimeError("Vizier returned no catalogs for III/286")
    cat = cats[0]
    print(f"  Vizier returned {len(cat):,} rows before quality cuts")

    # ASPCAP quality cuts: STAR_BAD (bit 23) and CHI2_BAD (bit 13) must be unset.
    aspcap_bad = (1 << 23) | (1 << 13)
    if "AFlag" in cat.colnames:
        mask = (cat["AFlag"] & aspcap_bad) == 0
        cat = cat[mask]
    else:
        print("  WARNING: AFlag column missing — skipping ASPCAP quality cut")
    print(f"  {len(cat):,} rows after ASPCAPflag quality cuts")

    cat.write(ALLSTAR_OUT, overwrite=True)
    print(f"  Wrote {ALLSTAR_OUT}")
    return cat


def fetch_astronn() -> Table:
    """Download astroNN VAC (masses + ages, row-matched to allStar DR17)."""
    if ASTRONN_OUT.exists():
        print(f"astroNN VAC already downloaded: {ASTRONN_OUT}")
        return Table.read(ASTRONN_OUT, hdu=1)
    print(f"Downloading astroNN VAC ({ASTRONN_URL})...")
    import urllib.request
    urllib.request.urlretrieve(ASTRONN_URL, ASTRONN_OUT)
    print(f"  Wrote {ASTRONN_OUT}")
    return Table.read(ASTRONN_OUT, hdu=1)


def join_catalogs(allstar: Table, astronn: Table) -> Table:
    """Inner-join allStar ↔ astroNN on APOGEE_ID. Dedupe by APOGEE_ID."""
    print("Joining allStar ↔ astroNN on APOGEE_ID...")
    # Vizier renames APOGEE_ID -> "APOGEE"; astroNN keeps "APOGEE_ID".
    if "APOGEE" in allstar.colnames and "APOGEE_ID" not in allstar.colnames:
        allstar.rename_column("APOGEE", "APOGEE_ID")
    # Dedupe: APOGEE_ID can repeat in allStar (multiple visits) and astroNN
    from astropy.table import unique
    allstar = unique(allstar, keys="APOGEE_ID", keep="first")
    astronn = unique(astronn, keys="APOGEE_ID", keep="first")
    print(f"  after dedupe: allstar={len(allstar):,}  astronn={len(astronn):,}")
    keep_astronn = [
        c for c in (
            "APOGEE_ID",
            "age_lowess_correct", "age_total_error",
            "weighted_dist", "weighted_dist_error",
            "galr", "galz",
        ) if c in astronn.colnames
    ]
    if "APOGEE_ID" not in keep_astronn:
        # Try common alternatives
        for alt in ("apogee_id", "apogeeid"):
            if alt in astronn.colnames:
                astronn.rename_column(alt, "APOGEE_ID")
                keep_astronn = ["APOGEE_ID"] + [c for c in keep_astronn if c != "APOGEE_ID"]
                break
    joined = join(allstar, astronn[keep_astronn], keys="APOGEE_ID", join_type="left")
    print(f"  Joined: {len(joined):,} rows")
    joined.write(JOINED_OUT, overwrite=True)
    print(f"  Wrote {JOINED_OUT}")
    return joined


def main() -> None:
    allstar = fetch_allstar()
    astronn = fetch_astronn()
    print(f"  astroNN columns ({len(astronn.colnames)}): "
          f"{astronn.colnames[:12]} ...")
    join_catalogs(allstar, astronn)


if __name__ == "__main__":
    main()
