#!/usr/bin/env python3
"""
Compute synthetic JWST NIRCam photometry from each completed SPHEREx
spectrum, then build color-magnitude (M_F444W vs F356W-F444W) and
color-color (F356W-F444W vs F200W-F356W) diagrams.

Distances come from the astroNN VAC weighted_dist (in pc), already in the
joined catalog.

Outputs:
  synth_phot.ecsv             — per-source NIRCam mags + colors + dist
  stacks/cmd_F444W.png        — color-magnitude
  stacks/ccd_color_color.png  — color-color
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table
from astroquery.svo_fps import SvoFps

sys.path.insert(0, "/Users/adam/repos/icemodels")
sys.path.insert(0, str(Path(__file__).parent))
from icemodels.core import fluxes_in_filters  # noqa: E402
from spherex_votable import load  # noqa: E402
from spectrum_qc import qc  # noqa: E402

OUT = Path(__file__).parent
SPEC_DIR = OUT / "spectra"
TARGETS = OUT / "spherex_target_list.ipac"
OUT_TBL = OUT / "synth_phot.ecsv"
STACK = OUT / "stacks"

# Stable (non-2025) NIRCam filter IDs spanning the SPHEREx range
FILTERS = [
    "JWST/NIRCam.F115W",   # ~1.15 µm
    "JWST/NIRCam.F150W",   # ~1.50 µm
    "JWST/NIRCam.F200W",   # ~2.00 µm
    "JWST/NIRCam.F277W",   # ~2.77 µm
    "JWST/NIRCam.F356W",   # ~3.56 µm
    "JWST/NIRCam.F410M",   # ~4.09 µm  (sits in SPHEREx dichroic gap)
    "JWST/NIRCam.F444W",   # ~4.44 µm
    "JWST/NIRCam.F466N",   # ~4.65 µm  (CO ice band)
    "JWST/NIRCam.F470N",   # ~4.71 µm
]


def ujy_to_abmag(flux_ujy: float) -> float:
    if not np.isfinite(flux_ujy) or flux_ujy <= 0:
        return np.nan
    # AB: m = -2.5 log10(F / 3631 Jy)
    return -2.5 * np.log10(flux_ujy * 1e-6 / 3631.0)


def main() -> None:
    targets = Table.read(TARGETS, format="ascii.ipac")
    by_id = {str(r["APOGEE_ID"]).strip(): r for r in targets}

    print("Pre-fetching NIRCam transmission curves...")
    transdata = {fid: SvoFps.get_transmission_data(fid) for fid in FILTERS}

    rows = []
    vots = sorted(SPEC_DIR.glob("*.vot"))
    print(f"Processing {len(vots)} spectra...")
    for vot in vots:
        apid = vot.stem
        if apid == "mira_test":
            continue
        row = by_id.get(apid)
        if row is None:
            continue
        spec = load(vot)[0]
        qc_res = qc(spec)
        if not qc_res.keep:
            continue
        good = qc_res.keep_mask
        wl = spec.wavelength[good] * u.um
        fl = spec.flux[good] * u.uJy
        order = np.argsort(wl)
        wl = wl[order]
        fl = fl[order]

        try:
            fluxes = fluxes_in_filters(wl, fl, filterids=FILTERS, transdata=transdata)
        except Exception as e:
            print(f"  [skip] {apid}: {e}")
            continue

        def _f(v):
            try: return float(v)
            except (ValueError, TypeError): return np.nan

        rec = {"APOGEE_ID": apid,
               "Teff": _f(row["Teff"]),
               "logg": _f(row["logg"]),
               "MH": _f(row["MH"]),
               "Ksmag": _f(row["Ksmag"]),
               "age_lowess_correct": _f(row["age_lowess_correct"]),
               "weighted_dist": _f(row["weighted_dist"]),
               "n_clean": int(good.sum())}
        # Tag RC vs RGB by the same rule used in 01_bin_and_select
        is_rc = (2.3 <= rec["logg"] < 2.6) and (4500 <= rec["Teff"] < 5100)
        rec["evol_state"] = "RC" if is_rc else "RGB"
        # Tag polar — IPAC writes null as "--"; also check for masked
        pt = row["pole_tag"] if "pole_tag" in row.colnames else None
        if pt is None or str(pt).strip() in ("", "--", "None") or \
                (hasattr(pt, "mask") and pt.mask):
            rec["pole_tag"] = ""
        else:
            rec["pole_tag"] = str(pt).strip()

        for fid in FILTERS:
            short = fid.split(".")[-1]
            f_ujy = fluxes[fid].to(u.uJy).value if hasattr(fluxes[fid], "to") else float(fluxes[fid])
            rec[f"{short}_uJy"] = f_ujy
            rec[short] = ujy_to_abmag(f_ujy)
        rows.append(rec)

    if not rows:
        raise SystemExit("No spectra processed.")
    tbl = Table(rows=rows)
    print(f"  → {len(tbl)} sources with synthetic photometry")

    # Absolute magnitudes (need distance)
    dist_pc = np.array(tbl["weighted_dist"], dtype=float)
    dist_pc[~np.isfinite(dist_pc) | (dist_pc <= 0)] = np.nan
    dm = 5 * np.log10(dist_pc / 10.0)
    for fid in FILTERS:
        short = fid.split(".")[-1]
        tbl[f"M_{short}"] = tbl[short] - dm

    tbl.write(OUT_TBL, overwrite=True)
    print(f"  wrote {OUT_TBL}")

    # ---- plots ----
    STACK.mkdir(exist_ok=True)
    is_rc = np.array([s == "RC" for s in tbl["evol_state"]])
    is_polar = np.array([t != "" for t in tbl["pole_tag"]])

    def style(ax):
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    teff_arr = np.asarray(tbl["Teff"], dtype=float)
    logg_arr = np.asarray(tbl["logg"], dtype=float)
    feh_arr = np.asarray(tbl["MH"], dtype=float)
    age_arr = np.asarray(tbl["age_lowess_correct"], dtype=float)

    # Symbol = logg bin in steps of 0.5
    LOGG_BINS = [(1.0, 1.5, "o"),    # tip RGB → circle
                 (1.5, 2.0, "s"),    # upper RGB → square
                 (2.0, 2.5, "^"),    # mid RGB / lower RC → triangle
                 (2.5, 3.0, "D")]    # base RGB / upper RC → diamond

    # Color schemes: (label, value array, cmap, vmin, vmax, cbar label)
    COLOR_SCHEMES = [
        ("teff", teff_arr, "coolwarm_r",
         np.nanmin(teff_arr), np.nanmax(teff_arr), "Teff (K)"),
        ("feh",  feh_arr,  "viridis",
         -0.20, +0.20, "[M/H] (dex)"),
        ("age",  age_arr,  "plasma",
         np.nanmin(age_arr[np.isfinite(age_arr)]),
         np.nanmax(age_arr[np.isfinite(age_arr)]), "age (Gyr, astroNN)"),
    ]

    def _scatter_logg(ax, cx, cy, c_arr, cmap, norm):
        for lo, hi, marker in LOGG_BINS:
            m = (logg_arr >= lo) & (logg_arr < hi)
            if m.any():
                ax.scatter(cx[m], cy[m], c=c_arr[m], cmap=cmap, norm=norm,
                           s=55, marker=marker, edgecolor="k", lw=0.3,
                           label=f"logg ∈ [{lo:.1f}, {hi:.1f})  N={m.sum()}")

    color_F356W_F444W = tbl["F356W"] - tbl["F444W"]
    M = tbl["M_F444W"]
    color_F200W_F356W = tbl["F200W"] - tbl["F356W"]

    for tag, c_arr, cmap_name, vmin, vmax, cb_label in COLOR_SCHEMES:
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # CMD
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_logg(ax, color_F356W_F444W, M, c_arr, cmap, norm)
        ax.invert_yaxis()
        ax.set_xlabel("F356W − F444W (AB mag)")
        ax.set_ylabel("M(F444W) (AB mag)")
        ax.set_title(f"NIRCam CMD  ({len(tbl)} stars, color = {cb_label.split(' (')[0]})")
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
        cb.set_label(cb_label)
        ax.legend(loc="lower left", fontsize=9)
        style(ax)
        fig.tight_layout()
        out = STACK / f"cmd_F444W_{tag}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  CMD ({tag}) → {out}")

        # CCD
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_logg(ax, color_F200W_F356W, color_F356W_F444W, c_arr, cmap, norm)
        ax.set_xlabel("F200W − F356W (AB mag)")
        ax.set_ylabel("F356W − F444W (AB mag)")
        ax.set_title(f"NIRCam CCD  ({len(tbl)} stars, color = {cb_label.split(' (')[0]})")
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
        cb.set_label(cb_label)
        ax.legend(loc="best", fontsize=9)
        style(ax)
        fig.tight_layout()
        out = STACK / f"ccd_color_color_{tag}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  CCD ({tag}) → {out}")

    # 4-µm CCD: F410M-F466N vs F356W-F444W, one per color scheme
    cx = tbl["F410M"] - tbl["F466N"]
    cy = tbl["F356W"] - tbl["F444W"]
    for tag, c_arr, cmap_name, vmin, vmax, cb_label in COLOR_SCHEMES:
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_logg(ax, cx, cy, c_arr, cmap, norm)
        ax.set_xlabel("F410M − F466N (AB mag)")
        ax.set_ylabel("F356W − F444W (AB mag)")
        ax.set_title(f"4-µm CCD  (N={len(tbl)}, color = {cb_label.split(' (')[0]})  "
                     "F410M is in SPHEREx dichroic gap")
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
        cb.set_label(cb_label)
        ax.legend(loc="best", fontsize=9)
        style(ax)
        fig.tight_layout()
        out = STACK / f"ccd_4um_{tag}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  4µm CCD ({tag}) → {out}")


if __name__ == "__main__":
    main()
