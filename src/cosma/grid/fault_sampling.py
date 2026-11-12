"""Side-aware sampling of horizons at faults (vertical faults, one trace depth per fault).

Throw is never invented: it is whatever depth step the imported surfaces show across the
fault trace. A fault whose throw is negligible against the local layer thickness stays out
of the geometry and is left to transmissibility factors.

Snapping: a cell takes the side of the trace its centre lies on; a cell face between cells
of opposite side is the fault face. That is "snap to the nearest face" with no tolerance.
A cell corner on a faulted pillar is then given the depth of ITS cell's side of the fault:
the surface is sampled one and two offsets away from the true trace on that side and
extrapolated linearly to the corner, so the surface's smeared step is never sampled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boundary import GridSpec

NEGLIGIBLE_FRACTION = 0.10


class FaultSamplingError(ValueError):
    pass


def _nearest(trace: np.ndarray, pts: np.ndarray):
    """For points (N,2): signed distance to the polyline (positive on its left), the unit left
    normal at the nearest point, and a validity mask (False beside an end of the polyline)."""
    P = np.asarray(trace, float).reshape(-1, 2)
    A, B = P[:-1], P[1:]
    seg = B - A
    L2 = (seg ** 2).sum(axis=1)
    ok = L2 > 0
    if not ok.any():
        raise FaultSamplingError("the fault trace has fewer than two distinct points")
    q = pts[:, None, :]                                   # (N,1,2)
    t = ((q - A[None]) * seg[None]).sum(axis=2) / np.where(ok, L2, 1.0)[None]
    foot = A[None] + np.clip(t, 0.0, 1.0)[..., None] * seg[None]
    dist = np.hypot(*(q - foot).transpose(2, 0, 1))
    dist[:, ~ok] = np.inf
    i = np.argmin(dist, axis=1)
    n = np.arange(len(pts))
    last = len(seg) - 1
    ti = t[n, i]
    valid = ~(((i == 0) & (ti < 0.0)) | ((i == last) & (ti > 1.0)))
    cross = seg[i, 0] * (pts[:, 1] - A[i, 1]) - seg[i, 1] * (pts[:, 0] - A[i, 0])
    length = np.sqrt(L2[i])
    signed = cross / length
    normal = np.stack([-seg[i, 1], seg[i, 0]], axis=1) / length[:, None]
    return signed, normal, valid


def local_trace(fault, depth: float, frame) -> np.ndarray:
    tr = fault.trace_at(depth)
    if len(tr.xy) < 2:
        reasons = "; ".join(f"stick {k + 1}: {r}" for k, r in tr.skipped.items()) or "fewer than two sticks reach it"
        raise FaultSamplingError(f"{fault.name}: no trace at depth {depth:g} ({reasons})")
    x, y = frame.to_local(tr.xy[:, 0], tr.xy[:, 1])
    return np.column_stack([x, y])


@dataclass
class FaultReport:
    name: str
    depth: float
    n_faces: int = 0
    snap_distance_max: float = 0.0       # farthest a fault face lies from the true trace (m)
    snap_distance_mean: float = 0.0
    median_throw: list = field(default_factory=list)       # per horizon, left side minus right side (m)
    max_abs_throw: list = field(default_factory=list)
    median_ratio: list = field(default_factory=list)       # |throw| / local layer thickness
    negligible: list = field(default_factory=list)         # per horizon
    has_throw: bool = False
    stations: int = 0
    warnings: list = field(default_factory=list)
    faces_i: object = None   # (nx-1, ny) bool: faces between (i,j) and (i+1,j) that lie on this fault
    faces_j: object = None   # (nx, ny-1) bool


def classify_cells(spec: GridSpec, trace: np.ndarray) -> np.ndarray:
    """(nx, ny) int array: +1 cell centre left of the directed trace, -1 right, 0 beside no part of it."""
    cx = 0.5 * (spec.x_edges[:-1] + spec.x_edges[1:])
    cy = 0.5 * (spec.y_edges[:-1] + spec.y_edges[1:])
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    s, _, ok = _nearest(trace, np.column_stack([X.ravel(), Y.ravel()]))
    side = np.where(ok, np.where(s > 0, 1, -1), 0)
    return side.reshape(spec.nx, spec.ny).astype(int)


def fault_faces(spec: GridSpec, side: np.ndarray):
    """Cell faces between cells of opposite side. Returns (i_faces, j_faces): boolean arrays shaped
    (nx-1, ny) [face between (i,j) and (i+1,j)] and (nx, ny-1) [between (i,j) and (i,j+1)]."""
    fi = (side[:-1, :] * side[1:, :]) == -1
    fj = (side[:, :-1] * side[:, 1:]) == -1
    return fi, fj


def _snap_distances(spec: GridSpec, trace, fi, fj) -> np.ndarray:
    pts = []
    for i, j in zip(*np.nonzero(fi)):
        pts.append((spec.x_edges[i + 1], 0.5 * (spec.y_edges[j] + spec.y_edges[j + 1])))
    for i, j in zip(*np.nonzero(fj)):
        pts.append((0.5 * (spec.x_edges[i] + spec.x_edges[i + 1]), spec.y_edges[j + 1]))
    if not pts:
        return np.zeros(0)
    s, _, _ = _nearest(trace, np.asarray(pts, float))
    return np.abs(s)


def _stations(trace: np.ndarray, step: float):
    """Points every ``step`` metres along the trace with the left normal there."""
    seg = np.diff(trace, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    total = cum[-1]
    if total <= 0:
        raise FaultSamplingError("zero-length trace")
    s = np.arange(0.5 * step, total, step) if total > step else np.array([0.5 * total])
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    f = (s - cum[idx]) / np.where(ln[idx] > 0, ln[idx], 1.0)
    pts = trace[idx] + f[:, None] * seg[idx]
    nrm = np.stack([-seg[idx, 1], seg[idx, 0]], axis=1) / np.where(ln[idx] > 0, ln[idx], 1.0)[:, None]
    return pts, nrm


def _smp(surface, frame, xy):
    mx, my = frame.to_map(xy[:, 0], xy[:, 1])
    return np.asarray(surface.sample(mx, my), float)


def surface_offset(surface) -> float:
    """Distance from the trace at which a surface is sampled: 1.5 node spacings clears the one-cell smear of a step."""
    return 1.5 * max(surface.xinc, surface.yinc)


def measure_throw(surface, frame, trace, off: float, step: float):
    """Throw (left side minus right side) at stations along the trace, from linear extrapolation of the
    surface on each side to the trace. Returns (stations, throw, z_left_at_trace, z_right_at_trace)."""
    pts, nrm = _stations(trace, step)
    z = {}
    for sgn in (1, -1):
        z[sgn] = [_smp(surface, frame, pts + sgn * m * off * nrm) for m in (1, 2)]
    zl0 = 2 * z[1][0] - z[1][1]
    zr0 = 2 * z[-1][0] - z[-1][1]
    return pts, zl0 - zr0, zl0, zr0


def assess_fault(name, depth, trace, surfaces, frame, spec, fi, fj, threshold=NEGLIGIBLE_FRACTION) -> FaultReport:
    """Throw statistics per horizon and the negligible-throw verdict (median station ratio of |throw| to the
    thinner adjacent interval's thickness, measured on the same two sides, must be below ``threshold``)."""
    rep = FaultReport(name, float(depth))
    rep.n_faces = int(fi.sum() + fj.sum())
    rep.faces_i, rep.faces_j = fi, fj
    sd = _snap_distances(spec, trace, fi, fj)
    if sd.size:
        rep.snap_distance_max, rep.snap_distance_mean = float(sd.max()), float(sd.mean())
    step = min(min(s.xinc, s.yinc) for s in surfaces)
    zl, zr, thr = [], [], []
    for s in surfaces:
        pts, th, a, b = measure_throw(s, frame, trace, surface_offset(s), step)
        zl.append(a)
        zr.append(b)
        thr.append(th)
    rep.stations = len(pts)
    nh = len(surfaces)
    for h in range(nh):
        thick = []
        for lo, hi in ((h - 1, h), (h, h + 1)):
            if lo >= 0 and hi < nh:
                # interval thickness on each side of the fault, averaged
                thick.append(0.5 * ((zl[hi] - zl[lo]) + (zr[hi] - zr[lo])))
        t = np.minimum.reduce(thick) if thick else np.full(len(pts), np.nan)
        ratio = np.abs(thr[h]) / np.where(t > 0, t, np.nan)
        ok = np.isfinite(thr[h])
        rep.median_throw.append(float(np.nanmedian(thr[h])) if ok.any() else float("nan"))
        rep.max_abs_throw.append(float(np.nanmax(np.abs(thr[h]))) if ok.any() else float("nan"))
        med = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan")
        rep.median_ratio.append(med)
        rep.negligible.append(bool(not np.isfinite(med) or med < threshold))
    rep.has_throw = not all(rep.negligible)
    if not np.isfinite(rep.median_ratio).any():
        rep.warnings.append("throw could not be measured: the surfaces do not cover the fault on both sides")
    return rep


def corner_depths(spec: GridSpec, surfaces, faults: dict, depths: dict, *, threshold=NEGLIGIBLE_FRACTION,
                  age_order=None):
    """Horizon depths at every cell corner, shape (ny, 2, nx, 2) per surface, plus one FaultReport per fault.

    ``faults``: name -> Fault; ``depths``: name -> depth at which the vertical fault trace is taken.
    Faults with negligible throw are reported but do not touch the depths.

    ``age_order``: fault names, oldest first (from the confirmed interactions). Where the reach of two faulted
    faults overlaps, the youngest fault claims the corner (it is continuous through the older one). Without it the
    nearest fault wins and a warning asks for the interaction tool."""
    if age_order is not None:
        missing = [n for n in faults if n not in age_order]
        if missing:
            raise FaultSamplingError("age order lacks " + ", ".join(missing))
        faults = {n: faults[n] for n in age_order if n in faults}
    nx, ny = spec.nx, spec.ny
    ci = np.arange(nx)[None, None, :, None] + np.arange(2)[None, None, None, :]
    cj = np.arange(ny)[:, None, None, None] + np.arange(2)[None, :, None, None]
    cx = np.broadcast_to(spec.x_edges[ci], (ny, 2, nx, 2))
    cy = np.broadcast_to(spec.y_edges[cj], (ny, 2, nx, 2))
    flat_xy = np.column_stack([cx.ravel(), cy.ravel()])
    out = [_smp(s, spec.frame, flat_xy).reshape(ny, 2, nx, 2) for s in surfaces]

    reports, best = [], np.full((ny, 2, nx, 2), np.inf)
    claimed = np.zeros((ny, 2, nx, 2), dtype=int)
    for name, fault in faults.items():
        if name not in depths:
            raise FaultSamplingError(f"{name}: no trace depth chosen (it must be explicit)")
        trace = local_trace(fault, depths[name], spec.frame)
        side = classify_cells(spec, trace)
        fi, fj = fault_faces(spec, side)
        rep = assess_fault(name, depths[name], trace, surfaces, spec.frame, spec, fi, fj, threshold)
        reports.append(rep)
        if not rep.has_throw:
            rep.warnings.append("throw is negligible against the layer thickness: the fault is left to "
                                "transmissibility factors and does not alter the geometry")
            continue
        s, nrm, valid = _nearest(trace, flat_xy)
        sig = np.broadcast_to(side.T[:, None, :, None], (ny, 2, nx, 2)).ravel()
        for h, surf in enumerate(surfaces):
            off = surface_offset(surf)
            need = valid & (sig != 0) & (sig * s < off)
            if not need.any():
                continue
            c = flat_xy[need]
            sg, nn, sv = s[need], nrm[need], sig[need]
            q1 = c + (nn * (sv * off - sg)[:, None])
            q2 = q1 + nn * (sv * off)[:, None]
            z1 = _smp(surf, spec.frame, q1)
            z2 = _smp(surf, spec.frame, q2)
            g = (z2 - z1) / off
            z = z1 - g * (off - sv * sg)
            flat = out[h].reshape(-1)
            mine = np.ones(int(need.sum()), bool) if age_order is not None else \
                np.abs(s[need]) < best.reshape(-1)[need]
            idx = np.nonzero(need)[0]
            take = idx[mine]
            flat[take] = z[mine]
        near = valid & (sig != 0) & (sig * s < surface_offset(surfaces[0]))
        claimed.reshape(-1)[near] += 1
        best_flat = best.reshape(-1)
        best_flat[near] = np.minimum(best_flat[near], np.abs(s[near]))
    crowded = int((claimed > 1).sum())
    if crowded and age_order is not None:
        for rep in reports:
            if rep.has_throw:
                rep.warnings.append(f"{crowded} cell corners lie within reach of two faulted faults; the younger fault "
                                    "claims them (confirmed fault ages). Depths there are approximate")
                break
    elif crowded:
        for rep in reports:
            if rep.has_throw:
                rep.warnings.append(f"{crowded} cell corners lie within reach of more than one faulted fault trace: "
                                    "fault intersections need the interaction tool (the nearest fault was used)")
                break
    return out, reports
