"""Sequential indicator simulation (SIS) of facies, per interval, honouring hard facies and the local proportions.

For each node on a random path (multiple-grid: coarse lattices first), every facies k is estimated by simple kriging of the
indicator residuals around the LOCAL proportion p_k(u) (from the vertical proportion curve, optionally combined with a map):
    p*_k(u) = p_k(u) + sum_a lambda_a (i_k(u_a) - p_k(u_a)).
Order-relation problems (any p* < 0 or > 1, or sum != 1) are corrected by clipping and renormalising and are COUNTED.
KNOWN LIMIT: with strongly continuous indicator variograms the sum of the kriging weights exceeds 1, the estimates leave
[0, 1] in a large share of nodes and clipping biases the simulated proportions towards the minority facies and weakens the
short-lag indicator correlation (see tests). A single realization is not forced to its target: its natural variation is part of
the uncertainty; the realization manager (M4.7) compares each realization with the target. A facies
whose local proportion is 0 stays at probability 0. If all probabilities are 0 the node is drawn from the local proportions
and counted. Intervals are simulated independently. Nothing is defaulted: neighbours, search size and multigrid levels are
arguments, and every facies of every interval needs an indicator variogram (transform 'indicator') named after the facies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .progress import report
from numba import njit

from .variogram import TYPES

MAX_TEMPLATE = 3_000_000


class SisError(ValueError):
    pass


@dataclass(frozen=True)
class SisSettings:
    max_neighbours: int        # previously simulated / hard nodes used per kriging
    search_factor: float       # search radius = factor x the largest variogram range in each direction
    levels: int                # multiple-grid levels (1 = no multigrid; level strides 2^(levels-1) ... 1)

    def __post_init__(self):
        for n, v in (("max_neighbours", self.max_neighbours), ("levels", self.levels)):
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
                raise SisError(f"{n} must be a whole number >= 1")
        if not (isinstance(self.search_factor, (int, float)) and math.isfinite(self.search_factor) and self.search_factor > 0):
            raise SisError("search_factor must be a number > 0")


# -- kernel ------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _rho(vg, vaz, vnug, vsill, f, dx, dy, dz):
    h2 = dx * dx + dy * dy + dz * dz
    if h2 == 0.0:
        return 1.0
    u = dx * vaz[f, 0] + dy * vaz[f, 1]
    v = dx * vaz[f, 1] - dy * vaz[f, 0]
    g = vnug[f]
    for s in range(3):
        if vg[f, s, 5] > 0.5:
            r = math.sqrt((u / vg[f, s, 2]) ** 2 + (v / vg[f, s, 3]) ** 2 + (dz / vg[f, s, 4]) ** 2)
            t = vg[f, s, 0]
            if t < 0.5:
                sh = 1.5 * r - 0.5 * r ** 3 if r < 1.0 else 1.0
            elif t < 1.5:
                sh = 1.0 - math.exp(-3.0 * r)
            else:
                sh = 1.0 - math.exp(-3.0 * r * r)
            g += vg[f, s, 1] * sh
    return 1.0 - g / vsill[f]


@njit(cache=True)
def _simulate(P, act, X, Y, Z, hard, vg, vaz, vnug, vsill, vgrp, tdi, tdj, tdk, nmax, strides, seed, out, stats):
    np.random.seed(seed)
    nx, ny, nk, nf = P.shape
    informed = hard >= 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nk):
                out[i, j, k] = hard[i, j, k]
    ni = np.empty(nmax, np.int64)
    nj = np.empty(nmax, np.int64)
    nkk = np.empty(nmax, np.int64)
    A = np.empty((nmax, nmax))
    b = np.empty(nmax)
    est = np.empty(nf)
    lamall = np.zeros((nf, nmax))
    for li in range(len(strides)):
        s = strides[li]
        coarse = 2 * s
        cnt = 0
        for i in range(0, nx, s):
            for j in range(0, ny, s):
                for k in range(0, nk, s):
                    if act[i, j, k] and not informed[i, j, k]:
                        if li == 0 or (i % coarse != 0 or j % coarse != 0 or k % coarse != 0):
                            cnt += 1
        nodes = np.empty((cnt, 3), np.int64)
        c = 0
        for i in range(0, nx, s):
            for j in range(0, ny, s):
                for k in range(0, nk, s):
                    if act[i, j, k] and not informed[i, j, k]:
                        if li == 0 or (i % coarse != 0 or j % coarse != 0 or k % coarse != 0):
                            nodes[c, 0] = i
                            nodes[c, 1] = j
                            nodes[c, 2] = k
                            c += 1
        order = np.random.permutation(cnt)
        for q in range(cnt):
            i = nodes[order[q], 0]
            j = nodes[order[q], 1]
            k = nodes[order[q], 2]
            n = 0
            for t in range(len(tdi)):
                ii = i + tdi[t] * s
                jj = j + tdj[t] * s
                kk = k + tdk[t] * s
                if ii < 0 or jj < 0 or kk < 0 or ii >= nx or jj >= ny or kk >= nk:
                    continue
                if informed[ii, jj, kk]:
                    ni[n] = ii
                    nj[n] = jj
                    nkk[n] = kk
                    n += 1
                    if n >= nmax:
                        break
            for f in range(nf):
                if n > 0 and vgrp[f] == f:
                    for a in range(n):
                        for a2 in range(a + 1):
                            r = _rho(vg, vaz, vnug, vsill, f, X[ni[a], nj[a], nkk[a]] - X[ni[a2], nj[a2], nkk[a2]],
                                     Y[ni[a], nj[a], nkk[a]] - Y[ni[a2], nj[a2], nkk[a2]],
                                     Z[ni[a], nj[a], nkk[a]] - Z[ni[a2], nj[a2], nkk[a2]])
                            A[a, a2] = r
                            A[a2, a] = r
                        A[a, a] = 1.0 + 1e-9
                        b[a] = _rho(vg, vaz, vnug, vsill, f, X[ni[a], nj[a], nkk[a]] - X[i, j, k],
                                    Y[ni[a], nj[a], nkk[a]] - Y[i, j, k], Z[ni[a], nj[a], nkk[a]] - Z[i, j, k])
                    lamall[f, :n] = np.linalg.solve(A[:n, :n].copy(), b[:n].copy())
            tot = 0.0
            bad = False
            for f in range(nf):
                p0 = P[i, j, k, f]
                if p0 <= 0.0:
                    est[f] = 0.0
                    continue
                e = p0
                if n > 0:
                    lam = lamall[vgrp[f], :n]
                    for a in range(n):
                        ind = 1.0 if out[ni[a], nj[a], nkk[a]] == f else 0.0
                        e += lam[a] * (ind - P[ni[a], nj[a], nkk[a], f])
                if e < -1e-9 or e > 1.0 + 1e-9:
                    bad = True
                e = min(max(e, 0.0), 1.0)
                est[f] = e
                tot += e
            stats[1] += 1
            if tot <= 1e-12:
                stats[2] += 1
                tot = 0.0
                for f in range(nf):
                    est[f] = P[i, j, k, f]
                    tot += est[f]
            else:
                if bad or abs(tot - 1.0) > 1e-9:
                    stats[0] += 1
            u = np.random.random() * tot
            acc = 0.0
            pick = nf - 1
            for f in range(nf):
                acc += est[f]
                if u < acc and est[f] > 0.0:
                    pick = f
                    break
            if est[pick] <= 0.0:                      # numerical edge: take the last facies with probability > 0
                for f in range(nf):
                    if est[f] > 0.0:
                        pick = f
            out[i, j, k] = pick
            informed[i, j, k] = True


# -- python side -------------------------------------------------------------------------------------------------------
def _pack(models):
    nf = len(models)
    vg = np.zeros((nf, 3, 6))
    vaz, vnug, vsill = np.zeros((nf, 2)), np.zeros(nf), np.zeros(nf)
    vgrp = np.arange(nf, dtype=np.int64)
    for f, m in enumerate(models):
        for s, st in enumerate(m.structures):
            vg[f, s] = (TYPES.index(st.type), st.contribution, st.range_major, st.range_minor, st.range_vert, 1.0)
        az = math.radians(m.azimuth)
        vaz[f], vnug[f], vsill[f] = (math.sin(az), math.cos(az)), m.nugget, m.sill
        for g in range(f):
            if vgrp[g] == g and replace(models[g], variable='') == replace(m, variable=''):
                vgrp[f] = g
                break
    return vg, vaz, vnug, vsill, vgrp


def make_template(rh, rv, dx, dy, dz):
    """Index offsets (di, dj, dk) inside the search ellipsoid, nearest first, excluding (0, 0, 0). rh, rv in metres;
    dx, dy, dz = typical cell sizes in metres."""
    ni, nj, nk = int(rh // dx), int(rh // dy), int(rv // dz)
    if (2 * ni + 1) * (2 * nj + 1) * (2 * nk + 1) > 40 * MAX_TEMPLATE:
        raise SisError("the search area is enormous compared with the cell size; lower the search factor")
    di, dj, dk = np.meshgrid(np.arange(-ni, ni + 1), np.arange(-nj, nj + 1), np.arange(-nk, nk + 1), indexing="ij")
    d = (di * dx / rh) ** 2 + (dj * dy / rh) ** 2 + (dk * dz / rv) ** 2
    m = (d <= 1.0) & (d > 0)
    di, dj, dk, d = di[m], dj[m], dk[m], d[m]
    if len(d) > MAX_TEMPLATE:
        raise SisError(f"the search template has {len(d)} offsets; lower the search factor")
    o = np.argsort(d, kind="stable")
    return di[o].astype(np.int64), dj[o].astype(np.int64), dk[o].astype(np.int64)


def cell_xyz(grid):
    """Cell-centre map X, Y and depth Z, each (nx, ny, nz). Pillars are taken as vertical (as for hard facies)."""
    c = grid.coord
    xl = 0.25 * (c[:-1, :-1, 0] + c[:-1, 1:, 0] + c[1:, :-1, 0] + c[1:, 1:, 0])        # (ny, nx)
    yl = 0.25 * (c[:-1, :-1, 1] + c[:-1, 1:, 1] + c[1:, :-1, 1] + c[1:, 1:, 1])
    X, Y = grid.frame.to_map(xl.T, yl.T)                                               # (nx, ny)
    Z = np.transpose(grid.zcorn.mean(axis=(1, 3, 5)), (2, 1, 0))
    nz = grid.nz
    return (np.repeat(np.asarray(X)[:, :, None], nz, 2), np.repeat(np.asarray(Y)[:, :, None], nz, 2), Z)


@dataclass
class SisResult:
    facies: np.ndarray                       # (nx, ny, nz) facies codes, 0 = inactive
    report: str
    per_interval: dict = field(default_factory=dict)


def simulate_facies(project, model, settings: SisSettings, proportions: dict, hard, seed: int, progress=None) -> SisResult:
    """proportions: {interval name: (nx, ny, nlayers, nfac)} (proportions.layer_proportions or combine_with_maps);
    hard: facies.HardFacies or None. seed: whole number >= 0; the same seed gives the same realization."""
    if not (isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**31):
        raise SisError("the seed must be a whole number between 0 and 2^31 - 1")
    if project.chart is None or project.facies_setup is None:
        raise SisError("define the chart and the facies first")
    scheme = tuple(project.facies_setup.scheme)
    codes = tuple(f.code for f in scheme)
    grid = model.grid
    li = np.asarray(model.report.layer_interval)
    X, Y, Z = cell_xyz(grid)
    from .grid.builder import _cell_thickness
    thick = _cell_thickness(grid.zcorn, grid.nx, grid.ny, grid.nz)
    active = (np.asarray(grid.actnum) > 0) & np.isfinite(thick) & (thick > 0)
    out = np.zeros(active.shape, np.int32)
    lines, per = [], {}
    by_name = {iv.name: iv for iv in project.chart.intervals}
    if hard is not None and hard.cells.shape != active.shape:
        raise SisError("the hard facies belong to a different grid; recompute them")
    done, total = 0, int(active.sum())
    for a, name in enumerate(model.interval_names):
        if name not in proportions:
            raise SisError(f"no proportions for interval {name!r}")
        ks = np.nonzero(li == a)[0]
        P = np.ascontiguousarray(proportions[name], float)
        if P.shape != active.shape[:2] + (len(ks), len(codes)):
            raise SisError(f"interval {name!r}: proportions have shape {P.shape}, expected {active.shape[:2] + (len(ks), len(codes))}")
        if not np.isfinite(P).all() or (P < 0).any() or (np.abs(P.sum(-1) - 1) > 1e-6)[active[:, :, ks]].any():
            raise SisError(f"interval {name!r}: proportions must be >= 0 and sum to 1 in every active cell")
        models = []
        for f in scheme:
            m = project.variogram_for(by_name[name].uid, f.name)
            if m is None:
                raise SisError(f"interval {name!r}: no indicator variogram for facies {f.name!r} (attach one named {f.name!r} "
                               "to this interval in the variogram tool); nothing is borrowed from another interval")
            if m.transform != "indicator":
                raise SisError(f"interval {name!r}, facies {f.name!r}: the variogram was made on '{m.transform}' data, not indicators")
            models.append(m)
        vg, vaz, vnug, vsill, vgrp = _pack(models)
        rh = settings.search_factor * max(s.range_major for m in models for s in m.structures)
        rv = settings.search_factor * max(s.range_vert for m in models for s in m.structures)
        sl = (slice(None), slice(None), ks)
        act = np.ascontiguousarray(active[sl])
        hd = np.full(act.shape, -1, np.int8)
        if hard is not None:
            h = hard.cells[sl]
            for q, c in enumerate(codes):
                hd[(h == c) & act] = q
            if ((h > 0) & ~np.isin(h, codes)).any():
                raise SisError(f"interval {name!r}: hard facies contain a code that is not in the facies scheme")
        dxm = float(np.mean(np.hypot(np.diff(X[:, :, 0], axis=0).mean() if grid.nx > 1 else 1, np.diff(Y[:, :, 0], axis=0).mean() if grid.nx > 1 else 0)))
        dym = float(np.mean(np.hypot(np.diff(X[:, :, 0], axis=1).mean() if grid.ny > 1 else 0, np.diff(Y[:, :, 0], axis=1).mean() if grid.ny > 1 else 1)))
        dzm = float(np.mean(thick[sl][act])) if act.any() else 1.0
        tdi, tdj, tdk = make_template(rh, rv, max(dxm, 1e-9), max(dym, 1e-9), max(dzm, 1e-9))
        strides = np.array([2 ** q for q in range(settings.levels - 1, -1, -1)], np.int64)
        w = thick[sl] * act
        wsum = float(w.sum())
        tgt = (P * w[..., None]).sum((0, 1, 2)) / wsum if wsum else np.zeros(len(codes))
        args = (P, act, np.ascontiguousarray(X[sl]), np.ascontiguousarray(Y[sl]), np.ascontiguousarray(Z[sl]), hd, vg, vaz, vnug,
                vsill, vgrp, tdi, tdj, tdk, settings.max_neighbours, strides)
        res = np.full(act.shape, -1, np.int8)
        stats = np.zeros(3, np.int64)
        report(progress, f"facies, interval {name}", done, total)
        _simulate(*args, seed * 1009 + a, res, stats)
        done += int(act.sum())
        if ((res < 0) & act).any():
            raise SisError(f"interval {name!r}: some active cells were not simulated")
        block = np.where(act, np.take(np.array(codes), np.clip(res, 0, None)), 0)
        real = [float((w * (block == c)).sum() / wsum) if wsum else 0.0 for c in codes]
        out[sl] = block
        nh = int((hd >= 0).sum())
        per[name] = dict(nodes=int(stats[1]), corrected=int(stats[0]), fallback=int(stats[2]), hard=nh,
                         realized=dict(zip(codes, real)), target=dict(zip(codes, map(float, tgt))))
        names = {f.code: f.name for f in scheme}
        lines.append(f"{name}: {stats[1]} cells simulated, {nh} hard; probabilities corrected in {stats[0]}, drawn from the "
                     f"local proportions in {stats[2]}; thickness-weighted facies "
                     + ", ".join(f"{names[c]} {100 * r:.1f}% (target {100 * float(t):.1f}%)" for c, r, t in zip(codes, real, tgt)))
    return SisResult(out, "\n".join(lines), per)
