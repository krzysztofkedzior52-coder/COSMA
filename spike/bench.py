import sys, time, numpy as np
sys.path.insert(0, "/home/claude/cosma/spike")
from sgs_numba import sgs
hi = np.array([5, 7], np.int64); hv = np.array([1.5, -1.0])
win = np.array([12, 12, 4], np.int64)
t = time.time(); sgs(10, 10, 5, 6.0, 6.0, 2.0, 12, win, 1, hi, hv); print("compile+tiny", round(time.time() - t, 2), "s")
for n in ((100, 100, 20), (200, 200, 25)):
    N = np.prod(n)
    hi = np.array([N // 2, N // 3], np.int64)
    t = time.time(); a = sgs(*n, 15.0, 15.0, 4.0, 12, win, 7, hi, hv); dt = time.time() - t
    b = sgs(*n, 15.0, 15.0, 4.0, 12, win, 7, hi, hv)
    print(n, f"{N:,} cells", round(dt, 1), "s", f"{N/dt/1e3:.0f} k cells/s", "mean/std", round(a.mean(), 3), round(a.std(), 3),
          "repeatable", bool((a == b).all()), "hard honoured", bool(a.ravel()[hi[0]] == 1.5))
