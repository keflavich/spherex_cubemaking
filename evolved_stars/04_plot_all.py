#!/usr/bin/env python3
"""
Generate per-spectrum PNG plots and a multi-panel overview for all completed
SPHEREx evolved-star spectra.

Outputs:
  spectra/{APOGEE_ID}.png       — one per source
  stacks/all_spectra_grid.png   — overview grid colored by RC vs RGB
  stacks/all_spectra_overlay.png — single-panel overlay, color = log Teff
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).parent))
from spherex_votable import load  # noqa: E402
from spectrum_qc import qc  # noqa: E402

OUT_DIR = Path(__file__).parent
SPEC_DIR = OUT_DIR / "spectra"
STACK_DIR = OUT_DIR / "stacks"
TARGETS_PATH = OUT_DIR / "spherex_target_list.ipac"


def per_spectrum(vot_path: Path, target_row=None) -> dict:
    """Plot one spectrum, return summary dict."""
    spec = load(vot_path)[0]
    qc_res = qc(spec)
    good = qc_res.keep_mask
    bad_pix_only = ((spec.flags & (1 << 32)) != 0) & ~(
        (spec.flags & ((1 << 1) | (1 << 15) | (1 << 33))) != 0)
    overflow = (spec.flags & (1 << 1)) != 0
    # outliers flagged by QC (within base good_mask, but not in keep_mask)
    outlier = spec.good_mask() & ~good

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if good.any():
        ax.errorbar(spec.wavelength[good], spec.flux[good],
                     xerr=spec.bandwidth[good]/2, yerr=spec.flux_err[good],
                     fmt=".", ms=3, color="C0", alpha=0.85,
                     ecolor="C0", elinewidth=0.4,
                     label=f"clean ({good.sum()})")
    if bad_pix_only.any():
        ax.errorbar(spec.wavelength[bad_pix_only], spec.flux[bad_pix_only],
                     xerr=spec.bandwidth[bad_pix_only]/2,
                     yerr=spec.flux_err[bad_pix_only],
                     fmt=".", ms=3, color="purple", alpha=0.5,
                     ecolor="purple", elinewidth=0.4,
                     label=f"BAD_PIXEL only ({bad_pix_only.sum()})")
    if overflow.any():
        ax.scatter(spec.wavelength[overflow], spec.flux[overflow],
                    s=8, color="red", alpha=0.5,
                    label=f"OVERFLOW ({overflow.sum()})")
    if outlier.any():
        ax.scatter(spec.wavelength[outlier], spec.flux[outlier],
                    s=30, facecolors="none", edgecolors="orange", lw=1.0,
                    label=f"outlier ({outlier.sum()})")

    title_extra = ""
    if target_row is not None:
        title_extra = (f"  Teff={target_row['Teff']:.0f}  "
                       f"logg={target_row['logg']:.2f}  "
                       f"[Fe/H]={target_row['MH']:+.2f}  "
                       f"K={target_row['Ksmag']:.2f}  "
                       f"age={target_row['age_lowess_correct']:.1f} Gyr")
    ax.set_title(f"{spec.label}{title_extra}", fontsize=10)
    ax.set_xlabel("λ (μm)")
    ax.set_ylabel("Flux (μJy)")
    ax.set_yscale("symlog", linthresh=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    for be in [0.75, 1.11, 1.64, 2.42, 3.82, 4.42, 5.00]:
        ax.axvline(be, color="gray", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout()
    out = vot_path.with_suffix(".png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return {
        "label": spec.label,
        "n_meas": spec.n_meas,
        "n_clean": int(good.sum()),
        "n_overflow": int(overflow.sum()),
        "n_bad_pix": int(bad_pix_only.sum()),
        "n_outliers": int(qc_res.n_outliers),
        "qc_keep": qc_res.keep,
        "qc_reason": qc_res.reason,
        "lambda_min": float(spec.wavelength.min()),
        "lambda_max": float(spec.wavelength.max()),
        "median_flux_clean": float(np.nanmedian(spec.flux[good])) if good.any() else float("nan"),
        "spec": spec,
    }


def main() -> None:
    targets = Table.read(TARGETS_PATH, format="ascii.ipac")
    by_id = {str(r["APOGEE_ID"]).strip(): r for r in targets}

    vots = sorted(SPEC_DIR.glob("*.vot"))
    print(f"Plotting {len(vots)} spectra...")

    summaries = []
    for vot in vots:
        apid = vot.stem
        row = by_id.get(apid)
        try:
            s = per_spectrum(vot, row)
            summaries.append((apid, row, s))
        except Exception as e:
            print(f"  ERROR on {apid}: {e}")

    print(f"  per-source PNGs written to {SPEC_DIR}")

    # All-spectra overlay (only QC-pass ones with row info, color by Teff)
    STACK_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    keep_summaries = [(a, r, s) for a, r, s in summaries
                      if r is not None and s["qc_keep"]]
    teffs = np.array([row["Teff"] for _, row, _ in keep_summaries])
    norm = Normalize(vmin=teffs.min(), vmax=teffs.max())
    cmap = plt.cm.coolwarm_r  # cool=hot star, warm=cool star
    for apid, row, s in keep_summaries:
        spec = s["spec"]
        # Use the QC keep_mask (drops the 1 outlier in "1-outlier" cases too)
        from spectrum_qc import qc as _qc
        good = _qc(spec).keep_mask
        if not good.any():
            continue
        order = np.argsort(spec.wavelength[good])
        ax.plot(spec.wavelength[good][order], spec.flux[good][order],
                color=cmap(norm(row["Teff"])), lw=0.5, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("λ (μm)")
    ax.set_ylabel("Flux (μJy)")
    n_total = sum(1 for _, r, _ in summaries if r is not None)
    n_keep = len(keep_summaries)
    ax.set_title(f"SPHEREx spectra: {n_keep} kept of {n_total} "
                 f"({n_total - n_keep} QC-rejected; color = Teff)")
    ax.grid(True, alpha=0.3)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("Teff (K)")
    for be in [0.75, 1.11, 1.64, 2.42, 3.82, 4.42, 5.00]:
        ax.axvline(be, color="gray", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout()
    out = STACK_DIR / "all_spectra_overlay.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  overlay → {out}")

    # QC summary + per-source table
    from collections import Counter
    qc_counts = Counter(s["qc_reason"] for _, r, s in summaries if r is not None)
    print(f"\nQC: {dict(qc_counts)}")
    rejected = [(a, s["qc_reason"], s["n_clean"], s["n_outliers"])
                for a, r, s in summaries
                if r is not None and not s["qc_keep"]]
    if rejected:
        print(f"REJECTED ({len(rejected)}):")
        for apid, reason, nc, no in rejected:
            print(f"  {apid:<22s}  {reason:<14s}  n_good={nc}  n_outliers={no}")

    print()
    print(f"{'APOGEE_ID':<22s} {'Teff':>6s} {'logg':>5s} {'[Fe/H]':>6s} "
          f"{'K':>5s} {'age':>5s} {'n_meas':>7s} {'clean':>6s} {'outl':>5s}  qc")
    for apid, row, s in summaries:
        if row is None:
            print(f"  {apid:<22s} (no target row — likely mira_test)")
            continue
        flag = "OK" if s["qc_keep"] else f"REJ({s['qc_reason']})"
        print(f"  {apid:<20s} {row['Teff']:>6.0f} {row['logg']:>5.2f} "
              f"{row['MH']:>+6.2f} {row['Ksmag']:>5.2f} "
              f"{row['age_lowess_correct']:>5.2f} "
              f"{s['n_meas']:>7d} {s['n_clean']:>6d} {s['n_outliers']:>5d}  {flag}")


if __name__ == "__main__":
    main()
