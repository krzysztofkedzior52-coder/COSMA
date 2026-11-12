"""M4.0 spike: sequential Gaussian simulation kernel on a regular grid (simple kriging, normal-score space)."""
import numpy as np
from numba import njit


@njit(cache=True)
def sph(h):
    if h >= 1.0:
        return 0.0
    return 1.0 - 1.5 * h + 0.5 * h * h * h


@njit(cache=True)
def make_template(rx, ry, rz, win):
    """Offsets (di,dj,dk) within the window, sorted by anisotropic distance; (0,0,0) excluded."""
    n = (2 * win[0] + 1) * (2 * win[1] + 1) * (2 * win[2] + 1) - 1
    off = np.empty((n, 3), np.int64)
    d = np.empty(n)
    m = 0
    for i in range(-win[0], win[0] + 1):
        for j in range(-win[1], win[1] + 1):
            for k in range(-win[2], win[2] + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                off[m, 0], off[m, 1], off[m, 2] = i, j, k
                d[m] = np.sqrt((i / rx) ** 2 + (j / ry) ** 2 + (k / rz) ** 2)
                m += 1
    order = np.argsort(d)
    return off[order], d[order]


@njit(cache=True)
def sgs(nx, ny, nz, rx, ry, rz, nmax, win, seed, hard_idx, hard_val):
    np.random.seed(seed)
    off, dist = make_template(rx, ry, rz, win)
    ntmp = off.shape[0]
    while ntmp > 0 and dist[ntmp - 1] >= 1.0:
        ntmp -= 1
    N = nx * ny * nz
    val = np.zeros(N)
    known = np.zeros(N, np.bool_)
    for t in range(hard_idx.shape[0]):
        val[hard_idx[t]] = hard_val[t]
        known[hard_idx[t]] = True
    path = np.random.permutation(N)
    px = np.empty(nmax); py = np.empty(nmax); pz = np.empty(nmax); pv = np.empty(nmax)
    C = np.empty((nmax, nmax)); L = np.empty((nmax, nmax)); c = np.empty(nmax)
    w = np.empty(nmax); y = np.empty(nmax)
    for p in range(N):
        node = path[p]
        if known[node]:
            continue
        k0 = node % nz
        j0 = (node // nz) % ny
        i0 = node // (nz * ny)
        n = 0
        for t in range(ntmp):
            i = i0 + off[t, 0]; j = j0 + off[t, 1]; k = k0 + off[t, 2]
            if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
                continue
            q = (i * ny + j) * nz + k
            if known[q]:
                px[n] = off[t, 0]; py[n] = off[t, 1]; pz[n] = off[t, 2]; pv[n] = val[q]
                c[n] = sph(dist[t])
                n += 1
                if n == nmax:
                    break
        if n == 0:
            val[node] = np.random.normal()
            known[node] = True
            continue
        for a in range(n):
            for b in range(a + 1):
                h = np.sqrt(((px[a] - px[b]) / rx) ** 2 + ((py[a] - py[b]) / ry) ** 2 + ((pz[a] - pz[b]) / rz) ** 2)
                C[a, b] = sph(h)
                C[b, a] = C[a, b]
            C[a, a] += 1e-8
        # Cholesky
        for a in range(n):
            for b in range(a + 1):
                s = C[a, b]
                for m in range(b):
                    s -= L[a, m] * L[b, m]
                if a == b:
                    L[a, a] = np.sqrt(max(s, 1e-12))
                else:
                    L[a, b] = s / L[b, b]
        for a in range(n):
            s = c[a]
            for m in range(a):
                s -= L[a, m] * y[m]
            y[a] = s / L[a, a]
        for a in range(n - 1, -1, -1):
            s = y[a]
            for m in range(a + 1, n):
                s -= L[m, a] * w[m]
            w[a] = s / L[a, a]
        mean = 0.0; var = 1.0
        for a in range(n):
            mean += w[a] * pv[a]
            var -= w[a] * c[a]
        if var < 1e-6:
            var = 1e-6
        val[node] = mean + np.sqrt(var) * np.random.normal()
        known[node] = True
    return val.reshape(nx, ny, nz)
