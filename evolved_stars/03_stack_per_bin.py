#!/usr/bin/env python3
"""
Stack downloaded SPHEREx spectra per (logg, age, [Fe/H], RGB/RC) bin and
produce an overlay grid plot.

Reads:
  spectra/{APOGEE_ID}.vot   — per-source VOTable from the spectrophot tool
  spherex_target_list.ipac  — has APOGEE params we use to tag bin membership

Writes:
  stacks/{bin_label}.fits   — median spectrum per bin on a common λ grid
  stacks/overlay_grid.png   — RGB and RC bins side-by-side
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from spherex_votable import load  # noqa: E402
from spectrum_qc import qc  # noqa: E402

OUT_DIR = Path(__file__).parent
SPEC_DIR = OUT_DIR / "spectra"
STACK_DIR = OUT_DIR / "stacks"
TARGETS_PATH = OUT_DIR / "spherex_target_list.ipac"

# Same edges as 01_bin_and_select.py — keep in sync
LOGG_EDGES = [(1.0, 1.8), (1.8, 2.4), (2.4, 3.0)]
FEH_EDGES = [(-0.20, -0.07), (-0.07, +0.07), (+0.07, +0.20)]
AGE_EDGES = [(0.0, 3.0), (3.0, 6.0), (6.0, 14.5)]
RC_LOGG = (2.3, 2.6)
RC_TEFF = (4500, 5100)

# Common output wavelength grid: 0.7-5.2 μm, 0.02 μm step (~2x SPHEREx's R)
LAMBDA_GRID = np.arange(0.70, 5.21, 0.02)


def bin_label(row) -> str:
    logg = float(row["logg"])
    teff = float(row["Teff"])
    feh = float(row["MH"])
    age = float(row["age_lowess_correct"])

    fi = next((i for i, (lo, hi) in enumerate(FEH_EDGES) if lo <= feh < hi), None)
    ai = next((i for i, (lo, hi) in enumerate(AGE_EDGES) if lo <= age < hi), None)
    if fi is None or ai is None:
        return "OUT_OF_GRID"

    is_rc = (RC_LOGG[0] <= logg < RC_LOGG[1]) and (RC_TEFF[0] <= teff < RC_TEFF[1])
    if is_rc:
        return f"RC_feh{fi}_age{ai}"
    li = next((i for i, (lo, hi) in enumerate(LOGG_EDGES) if lo <= logg < hi), None)
    if li is None:
        return "OUT_OF_GRID"
    return f"RGB_feh{fi}_age{ai}_logg{li}"


def regrid(spec, good=None) -> tuple[np.ndarray, np.ndarray]:
    """Bin a per-source spectrum onto LAMBDA_GRID (mean of clean points/bin).

    Returns (flux_uJy, n_per_bin) on LAMBDA_GRID.
    """
    if good is None:
        good = spec.good_mask()
    wl = spec.wavelength[good]
    fl = spec.flux[good]
    if len(wl) == 0:
        return np.full_like(LAMBDA_GRID, np.nan), np.zeros_like(LAMBDA_GRID, dtype=int)
    # bin index per measurement
    half = (LAMBDA_GRID[1] - LAMBDA_GRID[0]) / 2
    edges = np.concatenate([[LAMBDA_GRID[0] - half], LAMBDA_GRID + half])
    idx = np.digitize(wl, edges) - 1
    valid = (idx >= 0) & (idx < len(LAMBDA_GRID))
    idx = idx[valid]
    fl = fl[valid]
    sums = np.bincount(idx, weights=fl, minlength=len(LAMBDA_GRID))
    counts = np.bincount(idx, minlength=len(LAMBDA_GRID))
    out = np.full_like(LAMBDA_GRID, np.nan)
    nonzero = counts > 0
    out[nonzero] = sums[nonzero] / counts[nonzero]
    return out, counts


def main() -> None:
    if not TARGETS_PATH.exists():
        raise SystemExit(f"Missing {TARGETS_PATH}")
    targets = Table.read(TARGETS_PATH, format="ascii.ipac")
    targets_by_id = {str(r["APOGEE_ID"]).strip(): r for r in targets}
    STACK_DIR.mkdir(exist_ok=True)

    # Group QC-passing VOTables by bin
    by_bin: dict[str, list[np.ndarray]] = {}
    n_rejected = 0
    for vot in sorted(SPEC_DIR.glob("*.vot")):
        apid = vot.stem
        if apid not in targets_by_id:
            continue
        spec = load(vot)[0]
        qc_res = qc(spec)
        if not qc_res.keep:
            n_rejected += 1
            continue
        label = bin_label(targets_by_id[apid])
        flux, _ = regrid(spec, good=qc_res.keep_mask)
        by_bin.setdefault(label, []).append(flux)
    if n_rejected:
        print(f"QC rejected {n_rejected} spectra before stacking")

    if not by_bin:
        print("No matching spectra yet.")
        return

    print(f"Stacking across {len(by_bin)} bins ({sum(len(v) for v in by_bin.values())} spectra)")
    stacks: dict[str, np.ndarray] = {}
    for label, fluxes in sorted(by_bin.items()):
        arr = np.vstack(fluxes)
        with np.errstate(all="ignore"):
            stacks[label] = np.nanmedian(arr, axis=0)
        print(f"  {label:<25s}  N={len(fluxes)}")

    # Plot grid: rows = (RGB logg bins + RC), cols = age bins, color = [Fe/H]
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), sharex=True, sharey=True)
    feh_colors = ["#1f77b4", "#2ca02c", "#d62728"]  # low → solar → high
    rgb_logg_labels = [f"RGB logg {lo:.1f}-{hi:.1f}" for lo, hi in LOGG_EDGES]
    rc_label = "RC (logg 2.3-2.6, Teff 4500-5100)"
    row_specs = [(False, 0), (False, 1), (False, 2), (True, None)]  # (is_rc, logg_idx)

    for r, (is_rc, li) in enumerate(row_specs):
        for c, (alo, ahi) in enumerate(AGE_EDGES):
            ax = axes[r, c]
            for fi, (flo, fhi) in enumerate(FEH_EDGES):
                key = (f"RC_feh{fi}_age{c}" if is_rc
                       else f"RGB_feh{fi}_age{c}_logg{li}")
                spec = stacks.get(key)
                if spec is None or np.all(np.isnan(spec)):
                    continue
                ax.plot(LAMBDA_GRID, spec, color=feh_colors[fi], lw=0.9,
                        label=f"[Fe/H]∈[{flo:+.2f},{fhi:+.2f}]" if (r, c) == (0, 0) else None)
            if r == 0:
                ax.set_title(f"age {alo:.0f}-{ahi:.0f} Gyr", fontsize=10)
            if c == 0:
                ax.set_ylabel(rc_label if is_rc else rgb_logg_labels[li], fontsize=9)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8, loc="lower center")
    for ax in axes[-1, :]:
        ax.set_xlabel("λ (μm)")
    fig.suptitle("SPHEREx evolved-star stacks per (logg, age, [Fe/H]) bin",
                 fontsize=12)
    fig.tight_layout()
    out = STACK_DIR / "overlay_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
