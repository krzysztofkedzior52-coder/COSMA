"""Truncated Gaussian simulation (TGS) of ORDERED facies, per interval - the alternative to SIS (cosma.sis).

One Gaussian field Y ~ N(0, 1) is simulated by sequential Gaussian simulation (simple kriging, multiple-grid path) and cut at
thresholds from the LOCAL cumulative proportions along the facies order the user gives:
    facies k  where  Phi^-1(cum_{k-1}(u)) < Y(u) <= Phi^-1(cum_k(u)).
Proportions are therefore reproduced without the clipping bias of SIS, a facies with local proportion 0 can never occur, and the
indicator variogram of each facies follows from the single Gaussian variogram (it cannot be set facies by facies). Facies can only
touch their neighbours in the order. Hard facies are honoured exactly: the Gaussian values at hard cells are drawn inside their
threshold interval by Gibbs sampling (user-set sweeps) and the field is conditioned on them.
Everything is an argument: the order, the name of the Gaussian variogram attached to each interval (transform 'normal_score'),
neighbours, search, multigrid levels, Gibbs sweeps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .progress import report
from numba import njit

from .sis import SisError, SisResult, _pack, _rho, cell_xyz, make_template


class TgsError(SisError):
    pass


@dataclass(frozen=True)
class TgsSettings:
    order: tuple               # facies codes from one end of the order to the other (a permutation of the scheme codes)
    variogram: str             # name of the Gaussian variogram attached to every interval (transform 'normal_score')
    max_neighbours: int
    search_factor: float
    levels: int
    gibbs_sweeps: int          # sweeps over the hard cells that draw their Gaussian values (>= 1 when there are hard data)

    def __post_init__(self):
        if not isinstance(self.order, tuple) or len(set(self.order)) != len(self.order) or len(self.order) < 2:
            raise TgsError("order must be a tuple of at least two different facies codes")
        if not (isinstance(self.variogram, str) and self.variogram.strip()):
            raise TgsError("give the name of the Gaussian variogram")
        for n, v, lo in (("max_neighbours", self.max_neighbours, 1), ("levels", self.levels, 1), ("gibbs_sweeps", self.gibbs_sweeps, 0)):
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= lo):
                raise TgsError(f"{n} must be a whole number >= {lo}")
        if not (isinstance(self.search_factor, (int, float)) and math.isfinite(self.search_factor) and self.search_factor > 0):
            raise TgsError("search_factor must be a number > 0")


# -- normal distribution in numba --------------------------------------------------------------------------------------
@njit(cache=True)
def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@njit(cache=True)
def _ndtri(p):
    """Inverse normal CDF (Acklam's rational approximation + one Halley step; |error| < 1e-12 in (1e-300, 1 - 1e-16))."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
                  -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
                  -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
                  4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00])
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    e = _phi(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


@njit(cache=True)
def _sk(vg, vaz, vnug, vsill, X, Y, Z, informed, val, i, j, k, tdi, tdj, tdk, stride, nmax, ni, nj, nkk, A, b):
    """Simple-kriging mean and variance of the unit-variance Gaussian at (i, j, k) from informed nodes."""
    nx, ny, nk = informed.shape
    n = 0
    for t in range(len(tdi)):
        ii = i + tdi[t] * stride
        jj = j + tdj[t] * stride
        kk = k + tdk[t] * stride
        if ii < 0 or jj < 0 or kk < 0 or ii >= nx or jj >= ny or kk >= nk:
            continue
        if informed[ii, jj, kk]:
            ni[n] = ii
            nj[n] = jj
            nkk[n] = kk
            n += 1
            if n >= nmax:
                break
    if n == 0:
        return 0.0, 1.0
    for a in range(n):
        for a2 in range(a + 1):
            r = _rho(vg, vaz, vnug, vsill, 0, X[ni[a], nj[a], nkk[a]] - X[ni[a2], nj[a2], nkk[a2]],
                     Y[ni[a], nj[a], nkk[a]] - Y[ni[a2], nj[a2], nkk[a2]], Z[ni[a], nj[a], nkk[a]] - Z[ni[a2], nj[a2], nkk[a2]])
            A[a, a2] = r
            A[a2, a] = r
        A[a, a] = 1.0 + 1e-9
        b[a] = _rho(vg, vaz, vnug, vsill, 0, X[ni[a], nj[a], nkk[a]] - X[i, j, k], Y[ni[a], nj[a], nkk[a]] - Y[i, j, k],
                    Z[ni[a], nj[a], nkk[a]] - Z[i, j, k])
    lam = np.linalg.solve(A[:n, :n].copy(), b[:n].copy())
    m = 0.0
    v = 1.0
    for a in range(n):
        m += lam[a] * val[ni[a], nj[a], nkk[a]]
        v -= lam[a] * b[a]
    return m, max(v, 1e-12)


@njit(cache=True)
def _simulate(lo, hi, act, X, Y, Z, hard, vg, vaz, vnug, vsill, tdi, tdj, tdk, nmax, strides, sweeps, seed, val, stats):
    """lo, hi: (nx, ny, nk) Gaussian truncation interval of the hard facies at hard cells. hard >= 0 marks hard cells."""
    np.random.seed(seed)
    nx, ny, nk = act.shape
    informed = np.zeros(act.shape, np.bool_)
    ni = np.empty(nmax, np.int64)
    nj = np.empty(nmax, np.int64)
    nkk = np.empty(nmax, np.int64)
    A = np.empty((nmax, nmax))
    bb = np.empty(nmax)
    nh = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nk):
                if hard[i, j, k] >= 0:
                    nh += 1
    hc = np.empty((nh, 3), np.int64)
    c = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nk):
                if hard[i, j, k] >= 0:
                    hc[c, 0] = i
                    hc[c, 1] = j
                    hc[c, 2] = k
                    c += 1
    for q in range(nh):                                    # start: independent draws inside the truncation intervals
        i, j, k = hc[q, 0], hc[q, 1], hc[q, 2]
        pa, pb = _phi(lo[i, j, k]), _phi(hi[i, j, k])
        u = pa + np.random.random() * (pb - pa)
        y = _ndtri(u) if pb > pa else lo[i, j, k]
        if not np.isfinite(y):
            y = lo[i, j, k] if np.isfinite(lo[i, j, k]) else hi[i, j, k]
        val[i, j, k] = min(max(y, -8.0), 8.0)
        informed[i, j, k] = True
    for sw in range(sweeps):
        order = np.random.permutation(nh)
        for q in range(nh):
            i, j, k = hc[order[q], 0], hc[order[q], 1], hc[order[q], 2]
            informed[i, j, k] = False
            m, v = _sk(vg, vaz, vnug, vsill, X, Y, Z, informed, val, i, j, k, tdi, tdj, tdk, 1, nmax, ni, nj, nkk, A, bb)
            s = math.sqrt(v)
            pa, pb = _phi((lo[i, j, k] - m) / s), _phi((hi[i, j, k] - m) / s)
            if pb - pa > 1e-14:
                y = m + s * _ndtri(pa + np.random.random() * (pb - pa))
                if not np.isfinite(y):
                    y = min(max(m, lo[i, j, k]), hi[i, j, k])
            else:
                y = min(max(m, lo[i, j, k]), hi[i, j, k])
            val[i, j, k] = min(max(y, -8.0), 8.0)
            informed[i, j, k] = True
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
            i, j, k = nodes[order[q], 0], nodes[order[q], 1], nodes[order[q], 2]
            m, v = _sk(vg, vaz, vnug, vsill, X, Y, Z, informed, val, i, j, k, tdi, tdj, tdk, s, nmax, ni, nj, nkk, A, bb)
            val[i, j, k] = m + math.sqrt(v) * np.random.normal()
            informed[i, j, k] = True
            stats[0] += 1


def thresholds(P, order_idx):
    """(nx, ny, nk, nf + 1) Gaussian thresholds: column 0 = -inf, column nf = +inf, between: Phi^-1 of the cumulative proportions."""
    cum = np.cumsum(P[..., order_idx], axis=-1)
    cum = np.clip(cum, 0.0, 1.0)
    from scipy.special import ndtri
    t = np.empty(P.shape[:-1] + (P.shape[-1] + 1,))
    t[..., 0] = -np.inf
    t[..., -1] = np.inf
    inner = np.where(cum[..., :-1] <= 0.0, -np.inf, np.where(cum[..., :-1] >= 1.0, np.inf, ndtri(np.clip(cum[..., :-1], 1e-300, 1 - 1e-16))))
    t[..., 1:-1] = inner
    return t


def simulate_tgs(project, model, settings: TgsSettings, proportions: dict, hard, seed: int, progress=None) -> SisResult:
    """Same inputs and result as sis.simulate_facies. ``settings.order`` must be a permutation of the scheme codes."""
    if not (isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**31):
        raise TgsError("the seed must be a whole number between 0 and 2^31 - 1")
    if project.chart is None or project.facies_setup is None:
        raise TgsError("define the chart and the facies first")
    scheme = tuple(project.facies_setup.scheme)
    codes = tuple(f.code for f in scheme)
    if sorted(settings.order) != sorted(codes):
        raise TgsError(f"the order must contain exactly the facies codes {sorted(codes)}")
    oidx = np.array([codes.index(c) for c in settings.order])
    names = {f.code: f.name for f in scheme}
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
        raise TgsError("the hard facies belong to a different grid; recompute them")
    if hard is not None and settings.gibbs_sweeps < 1 and (hard.cells > 0).any():
        raise TgsError("there are hard facies: gibbs_sweeps must be >= 1")
    done, total = 0, int(active.sum())
    for a, name in enumerate(model.interval_names):
        if name not in proportions:
            raise TgsError(f"no proportions for interval {name!r}")
        ks = np.nonzero(li == a)[0]
        P = np.ascontiguousarray(proportions[name], float)
        if P.shape != active.shape[:2] + (len(ks), len(codes)):
            raise TgsError(f"interval {name!r}: proportions have shape {P.shape}, expected {active.shape[:2] + (len(ks), len(codes))}")
        if not np.isfinite(P).all() or (P < 0).any() or (np.abs(P.sum(-1) - 1) > 1e-6)[active[:, :, ks]].any():
            raise TgsError(f"interval {name!r}: proportions must be >= 0 and sum to 1 in every active cell")
        m = project.variogram_for(by_name[name].uid, settings.variogram)
        if m is None:
            raise TgsError(f"interval {name!r}: no variogram named {settings.variogram!r} is attached to it; nothing is borrowed "
                           "from another interval")
        if m.transform != "normal_score":
            raise TgsError(f"interval {name!r}: the variogram {settings.variogram!r} was made on '{m.transform}' data, not normal scores")
        vg, vaz, vnug, vsill, _ = _pack([m])
        sl = (slice(None), slice(None), ks)
        act = np.ascontiguousarray(active[sl])
        T = thresholds(P, oidx)
        hd = np.full(act.shape, -1, np.int8)
        lo = np.full(act.shape, -np.inf)
        hi = np.full(act.shape, np.inf)
        if hard is not None:
            h = hard.cells[sl]
            if ((h > 0) & ~np.isin(h, codes)).any():
                raise TgsError(f"interval {name!r}: hard facies contain a code that is not in the facies scheme")
            for q, c in enumerate(settings.order):
                sel = (h == c) & act
                hd[sel] = q
                lo[sel] = T[..., q][sel]
                hi[sel] = T[..., q + 1][sel]
        rh = settings.search_factor * max(s.range_major for s in m.structures)
        rv = settings.search_factor * max(s.range_vert for s in m.structures)
        dxm = float(np.hypot(np.diff(X[:, :, 0], axis=0).mean() if grid.nx > 1 else 1, np.diff(Y[:, :, 0], axis=0).mean() if grid.nx > 1 else 0))
        dym = float(np.hypot(np.diff(X[:, :, 0], axis=1).mean() if grid.ny > 1 else 0, np.diff(Y[:, :, 0], axis=1).mean() if grid.ny > 1 else 1))
        dzm = float(np.mean(thick[sl][act])) if act.any() else 1.0
        tdi, tdj, tdk = make_template(rh, rv, max(dxm, 1e-9), max(dym, 1e-9), max(dzm, 1e-9))
        strides = np.array([2 ** q for q in range(settings.levels - 1, -1, -1)], np.int64)
        val = np.zeros(act.shape)
        stats = np.zeros(1, np.int64)
        report(progress, f"facies, interval {name}", done, total)
        _simulate(lo, hi, act, np.ascontiguousarray(X[sl]), np.ascontiguousarray(Y[sl]), np.ascontiguousarray(Z[sl]), hd, vg, vaz,
                  vnug, vsill, tdi, tdj, tdk, settings.max_neighbours, strides, settings.gibbs_sweeps, seed * 1009 + a, val, stats)
        done += int(act.sum())
        # facies index along the order: number of thresholds strictly below the value
        pos = (val[..., None] > T[..., 1:-1]).sum(-1)
        fac = np.take(np.array(settings.order), pos)
        fac = np.where(hd >= 0, np.take(np.array(settings.order), np.clip(hd, 0, None)), fac)
        block = np.where(act, fac, 0)
        out[sl] = block
        w = thick[sl] * act
        wsum = float(w.sum())
        real = [float((w * (block == c)).sum() / wsum) if wsum else 0.0 for c in codes]
        tgt = (P * w[..., None]).sum((0, 1, 2)) / wsum if wsum else np.zeros(len(codes))
        nh = int((hd >= 0).sum())
        per[name] = dict(nodes=int(stats[0]), hard=nh, realized=dict(zip(codes, real)), target=dict(zip(codes, map(float, tgt))))
        lines.append(f"{name}: {stats[0]} cells simulated, {nh} hard; thickness-weighted facies "
                     + ", ".join(f"{names[c]} {100 * r:.1f}% (target {100 * float(t):.1f}%)" for c, r, t in zip(codes, real, tgt))
                     + f" (truncated Gaussian, order {' < '.join(names[c] for c in settings.order)})")
    return SisResult(out, "\n".join(lines), per)
