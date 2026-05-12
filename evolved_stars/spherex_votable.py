"""
Parse SPHEREx Spectrophotometry-Tool VOTable output.

The IRSA-emitted VOTable uses arraysize="54x*" on `obs_publisher_did` (a 2D
char field) which astropy's converter rejects. We strip that single FIELD on
the fly and then let astropy.io.votable do the rest.

Returns one Spectrum per source row (each VOTable typically has one source).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io.votable import parse


_BAD_ARRAYSIZE = re.compile(rb'arraysize="\d+x\*"')


def _patch(raw: bytes) -> bytes:
    """Replace `arraysize="<int>x*"` with `arraysize="*"` (drops 2D char dim)."""
    return _BAD_ARRAYSIZE.sub(b'arraysize="*"', raw)


@dataclass
class Spectrum:
    label: str
    ra: float
    dec: float
    wavelength: np.ndarray   # μm
    bandwidth: np.ndarray    # μm
    flux: np.ndarray         # μJy
    flux_err: np.ndarray     # μJy
    flags: np.ndarray        # uint64 bitmask per measurement
    fit_ql: np.ndarray
    det_id: np.ndarray
    lvf_id: np.ndarray
    mjd: np.ndarray
    n_meas: int

    def good_mask(self,
                  reject_bits: int = ((1 << 1) | (1 << 15) | (1 << 32) | (1 << 33))
                  ) -> np.ndarray:
        """Default reject: OVERFLOW(1), NONLINEAR(15), CONTAINS_BAD_PIXEL(32),
        FIT_ERROR(33). Returns boolean array of same length as wavelength."""
        return (self.flags & reject_bits) == 0


def load(path: str | Path) -> list[Spectrum]:
    raw = Path(path).read_bytes()
    patched = _patch(raw)
    vot = parse(io.BytesIO(patched), verify="warn")
    spectra = []
    for table in vot.iter_tables():
        a = table.array
        for i in range(len(a)):
            spectra.append(Spectrum(
                label=str(a["label"][i]).strip(),
                ra=float(a["ra"][i]),
                dec=float(a["dec"][i]),
                wavelength=np.asarray(a["lambda"][i], dtype=float),
                bandwidth=np.asarray(a["lambda_width"][i], dtype=float),
                flux=np.asarray(a["flux"][i], dtype=float),
                flux_err=np.asarray(a["flux_err"][i], dtype=float),
                flags=np.asarray(a["flags"][i], dtype=np.uint64),
                fit_ql=np.asarray(a["fit_ql"][i], dtype=float),
                det_id=np.asarray(a["det_id"][i], dtype=int),
                lvf_id=np.asarray(a["lvf_id"][i], dtype=int),
                mjd=np.asarray(a["mjd"][i], dtype=float),
                n_meas=int(a["n_meas"][i]),
            ))
    return spectra
