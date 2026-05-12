#!/usr/bin/env python3
"""Quick spectrum plot for a single SPHEREx Spectrophotometry-Tool VOTable."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from spherex_votable import load  # noqa: E402


def plot(vot_path: Path, out_png: Path) -> None:
    spec = load(vot_path)[0]
    print(f"{spec.label}: n_meas={spec.n_meas} @ ({spec.ra:.4f}, {spec.dec:.4f})")

    # Flag categories
    overflow = (spec.flags & (1 << 1)) != 0
    bad_pix = (spec.flags & (1 << 32)) != 0
    nonlin = (spec.flags & (1 << 15)) != 0
    fit_err = (spec.flags & (1 << 33)) != 0
    clean = ~(overflow | bad_pix | nonlin | fit_err)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    height_ratios=[3, 1])

    # Top: flux vs wavelength, color-coded by flag state
    for mask, label, color, alpha in [
        (clean, f"clean ({clean.sum()})", "C0", 0.85),
        (overflow & ~bad_pix, f"OVERFLOW only ({(overflow & ~bad_pix).sum()})",
         "orange", 0.5),
        (bad_pix & ~overflow, f"BAD_PIXEL only ({(bad_pix & ~overflow).sum()})",
         "purple", 0.5),
        (overflow & bad_pix, f"OVERFLOW+BAD_PIXEL ({(overflow & bad_pix).sum()})",
         "red", 0.4),
    ]:
        if mask.any():
            ax1.errorbar(spec.wavelength[mask], spec.flux[mask],
                          xerr=spec.bandwidth[mask] / 2,
                          yerr=spec.flux_err[mask],
                          fmt=".", ms=4, color=color, alpha=alpha,
                          ecolor=color, elinewidth=0.5, label=label)

    ax1.set_yscale("symlog", linthresh=1e3)
    ax1.set_ylabel("Flux (μJy)")
    ax1.set_title(f"SPHEREx spectrum — {spec.label}  "
                  f"(RA={spec.ra:.4f}°, Dec={spec.dec:.4f}°)")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="k", lw=0.5, alpha=0.3)
    # SPHEREx band edges
    band_edges = [0.75, 1.11, 1.64, 2.42, 3.82, 4.42, 5.00]
    for be in band_edges:
        ax1.axvline(be, color="gray", ls=":", lw=0.5, alpha=0.5)

    # Bottom: bandwidth and detector ID per measurement
    sc = ax2.scatter(spec.wavelength, spec.bandwidth,
                      c=spec.det_id, cmap="tab10", s=12)
    ax2.set_xlabel("Wavelength (μm)")
    ax2.set_ylabel("Bandwidth (μm)")
    ax2.grid(True, alpha=0.3)
    cb = plt.colorbar(sc, ax=ax2, pad=0.01, shrink=0.9)
    cb.set_label("det_id", rotation=270, labelpad=12)
    for be in band_edges:
        ax2.axvline(be, color="gray", ls=":", lw=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
          Path(__file__).parent / "spectra" / "mira_test.vot"
    out = src.with_suffix(".png")
    plot(src, out)
