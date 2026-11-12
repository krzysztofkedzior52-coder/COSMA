"""Shale gouge ratio on the faces of a grid, from the cell shale-volume property VSHALE (fraction 0-1).

SGR at a point of a fault = (sum of shale thickness in the stratigraphic interval that has slipped past the point) /
throw. On the grid, for a fault face between two cells at the same layer index, k:

* throw  T = |depth of the cell centre in one column - depth of the cell centre in the other|  (same layer, so this is
  the stratigraphic offset the surfaces show at that face, taken from the grid itself);
* the slipped interval is the T of stratigraphy above the cell centre (the material that lies between the two
  juxtaposed positions), measured as cumulative thickness down the column; the last layer is included in part;
* thickness and VSHALE of a layer are the mean of the two columns' values (one stratigraphy per face);
* SGR = shale thickness in that window / T.

A face-layer is left out, and counted, when T is below the user's minimum throw or when the window would run above the
top of the model (the sequence needed is not there). Nothing is extrapolated.
"""

from __future__ import annotations

import numpy as np

OK, BELOW_MIN_THROW, ABOVE_TOP, NO_VSHALE = 0, 1, 2, 3
REASONS = {BELOW_MIN_THROW: "throw below the minimum", ABOVE_TOP: "slipped interval reaches above the model top",
           NO_VSHALE: "VSHALE undefined in the interval"}


class SGRError(ValueError):
    pass


def check_vshale(v):
    v = np.asarray(v, float)
    ok = v[np.isfinite(v)]
    if ok.size and (ok.min() < 0.0 or ok.max() > 1.0):
        raise SGRError(f"VSHALE must be a fraction between 0 and 1 (found {ok.min():g} to {ok.max():g}); "
                       "COSMA does not rescale it")
    return v


def cell_centre_depth(zcorn):
    """(nx, ny, nz) mean of the eight corner depths of each cell."""
    return np.transpose(zcorn.mean(axis=(1, 3, 5)), (2, 1, 0))


def window_sgr(zc_a, zc_b, h_a, h_b, v_a, v_b, min_throw):
    """SGR for face-layers. Inputs are (nf, nz) arrays for the two columns of each face. Returns (T, SGR, status)."""
    T = np.abs(zc_a - zc_b)
    h = 0.5 * (h_a + h_b)
    v = 0.5 * (v_a + v_b)
    nf, nz = h.shape
    ch = np.cumsum(h, axis=1)
    undefined = ~np.isfinite(v)
    cv = np.cumsum(np.where(undefined, 0.0, v) * h, axis=1)
    cu = np.cumsum(undefined * h, axis=1)                  # thickness of layers with no VSHALE, cumulative
    node_s = np.concatenate([np.zeros((nf, 1)), ch], axis=1)
    node_v = np.concatenate([np.zeros((nf, 1)), cv], axis=1)
    node_u = np.concatenate([np.zeros((nf, 1)), cu], axis=1)
    s_end = ch - 0.5 * h
    s_start = s_end - T
    sgr = np.full((nf, nz), np.nan)
    status = np.zeros((nf, nz), dtype=np.int8)
    gap = np.zeros((nf, nz), dtype=bool)
    for f in range(nf):
        v_end = np.interp(s_end[f], node_s[f], node_v[f])
        v_start = np.interp(s_start[f], node_s[f], node_v[f])
        gap[f] = (np.interp(s_end[f], node_s[f], node_u[f]) - np.interp(s_start[f], node_s[f], node_u[f])) > 1e-9
        with np.errstate(invalid="ignore", divide="ignore"):
            sgr[f] = (v_end - v_start) / T[f]
    below = T < min_throw
    above = (s_start < -1e-9) & ~below
    bad = (gap | ~np.isfinite(sgr)) & ~below & ~above
    status[below] = BELOW_MIN_THROW
    status[above] = ABOVE_TOP
    status[bad] = NO_VSHALE
    sgr = np.clip(sgr, 0.0, 1.0)
    sgr[status != OK] = np.nan
    return T, sgr, status


VSHALE_NAME = "VSHALE"


def wells_with_vshale(project) -> list:
    """Names of wells that have a curve called VSHALE (any case) in some log set. COSMA does not guess which other
    curve might be shale volume."""
    out = []
    for name, rec in project.wells.items():
        if any(c.upper() == VSHALE_NAME for ls in rec.logs.values() for c in ls.curves):
            out.append(name)
    return out


def vshale_notice(project) -> str | None:
    """A message for the user when the project has faults but no well carries VSHALE, else None."""
    if not project.faults or wells_with_vshale(project):
        return None
    return ("Faults are present, but no well has a VSHALE curve. Shale gouge ratio (SGR) fault permeability needs "
            "VSHALE (shale volume as a fraction, 0-1) as a grid property: add a log curve named VSHALE to the wells so "
            "it can be modelled at the property step. Open and fixed faults do not need it.")
