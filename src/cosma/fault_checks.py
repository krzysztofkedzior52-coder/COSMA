"""Where a well path cuts a fault.

Property modelling in COSMA is done in stratigraphic space: every sample is positioned by its depth below the
well's own top pick, wells are compared layer by layer, and distances are plan-view. So a fault displacing the layers
after they were deposited does not interrupt a property: data on both sides of it are used together. That holds for
a well only as long as its picks follow the layering. A well that crosses a fault inside an interval can have part of
the section cut out (normal fault) or repeated (reverse fault); its interval is then thinner or thicker than it was
when the layers were deposited, and the samples at a given depth below the top pick no longer belong to the same layer
as in other wells. Such a well is reported here; COSMA does not decide for you whether to drop it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .faults import Fault


@dataclass(frozen=True)
class Crossing:
    fault: str
    tvdss: float
    x: float
    y: float


def side_of_trace(trace_xy: np.ndarray, q) -> float | None:
    """Signed distance (m) from point q to the polyline, positive on its left; None when q is beside no part of
    it (nearest to an end of the polyline, outside its lateral extent) or the polyline has fewer than 2 points."""
    P = np.asarray(trace_xy, float).reshape(-1, 2)
    q = np.asarray(q, float)
    A, B = P[:-1], P[1:]
    seg = B - A
    L2 = (seg ** 2).sum(axis=1)
    ok = L2 > 0
    if not ok.any():                                    # fewer than two distinct points: no line, no side
        return None
    t = np.where(ok, ((q - A) * seg).sum(axis=1) / np.where(ok, L2, 1.0), 0.0)
    foot = A + np.clip(t, 0.0, 1.0)[:, None] * seg
    dist = np.hypot(*(q - foot).T)
    dist[~ok] = np.inf
    i = int(np.argmin(dist))
    last = len(seg) - 1
    if (i == 0 and t[0] < 0.0) or (i == last and t[last] > 1.0):
        return None
    cross = seg[i, 0] * (q[1] - A[i, 1]) - seg[i, 1] * (q[0] - A[i, 0])
    s = float(cross / math.sqrt(L2[i]))
    return s


def well_crossings(faults: dict, trajectory, md_top: float, md_base: float, *, step: float = 2.0, max_samples: int = 600) -> list:
    """All places where the well path between two measured depths passes from one side of a fault to the other.
    The fault is taken as its trace at each depth (`Fault.trace_at`), so dip and curvature are followed; the path is
    sampled every `step` metres of MD. Depths where fewer than two sticks reach, or where the well is beside the
    end of the trace (past the fault tip), give no side and no crossing."""
    if not faults:
        return []
    lo, hi = trajectory.md_range
    a, b = max(md_top, lo), min(md_base, hi)
    if not b > a:
        return []
    n = int(min(max(math.ceil((b - a) / step) + 1, 2), max_samples))
    md = np.linspace(a, b, n)
    x, y, z = trajectory.at_md(md)
    out = []
    for name, f in faults.items():
        side = []
        for k in range(n):
            tr = f.trace_at(float(z[k]))
            side.append(side_of_trace(tr.xy, (x[k], y[k])))
        for k in range(n - 1):
            s0, s1 = side[k], side[k + 1]
            if s0 is None or s1 is None or (s0 > 0) == (s1 > 0):
                continue
            r = s0 / (s0 - s1)
            out.append(Crossing(name, float(z[k] + r * (z[k + 1] - z[k])), float(x[k] + r * (x[k + 1] - x[k])),
                                float(y[k] + r * (y[k + 1] - y[k]))))
    return out


def crossing_warnings(crossings: dict, wells=None) -> list:
    """One line per well that crosses a fault inside the interval. `crossings`: well -> [Crossing]."""
    out = []
    for w, cs in crossings.items():
        if wells is not None and w not in wells:
            continue
        if cs:
            where = ", ".join(f"{c.fault} at {c.tvdss:.0f} m TVDSS" for c in cs)
            out.append(f"{w} crosses {where} inside this interval: part of the section may be missing or repeated in this "
                       "well, so its layer values may not match the same layers in other wells. Check it or leave the well out.")
    return out
