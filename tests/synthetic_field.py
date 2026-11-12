"""Wells sampling an exactly simulated Gaussian field with a known variogram (Cholesky)."""
from __future__ import annotations

import numpy as np

from cosma.project import Project, WellRecord
from cosma.variogram import Structure, VariogramModel
from cosma.wells.model import LogSet, RawTop, Survey

TRUTH = VariogramModel(0.1, (Structure("spherical", 0.9, 450.0, 225.0, 20.0),), 30.0).check()


def synthetic(model=TRUTH, n=7, spacing=150.0, dz=2.0, seed=3, top=100.0, base=150.0):
    rng = np.random.default_rng(seed)
    pr = Project("synthetic")
    d = np.arange(0.0, 400.0, dz)
    inside = (d >= top) & (d <= base)
    wells = [(i * spacing, j * spacing) for i in range(n) for j in range(n)]
    X = np.concatenate([np.full(inside.sum(), x) for x, y in wells])
    Y = np.concatenate([np.full(inside.sum(), y) for x, y in wells])
    Z = np.tile(d[inside] - top, len(wells))
    g = model.gamma_xyz(X[:, None] - X[None], Y[:, None] - Y[None], Z[:, None] - Z[None])
    cov = model.sill - g
    cov[np.diag_indices_from(cov)] = model.sill
    f = np.linalg.cholesky(cov + 1e-9 * np.eye(len(X))) @ rng.standard_normal(len(X))
    k = 0
    for i, (x, y) in enumerate(wells):
        rec = WellRecord(f"W{i}")
        rec.header = dict(x=x, y=y, kb=0.0, td=400.0)
        rec.survey = Survey(np.array([0.0, 400.0]), np.zeros(2), np.zeros(2), "grid")
        for nm, dep in (("Seal", top - 50), ("Res", top), ("Basement", base)):
            rec.tops[nm] = RawTop(nm, dep, "md")
        v = np.full(d.size, np.nan)
        v[inside] = f[k:k + inside.sum()]
        k += inside.sum()
        rec.logs["lg"] = LogSet(d, {"PHI": v})
        pr.wells[rec.name] = rec
    pr.create_chart(["Seal", "Res"], "Basement")
    return pr


def facies_project(thresholds=(-0.4, 0.6), **kw):
    """Same simulated field, but the log is a facies code: 1 below the first threshold, 2 between, 3 above
    (truncated Gaussian)."""
    pr = synthetic(**kw)
    for rec in pr.wells.values():
        v = rec.logs["lg"].curves["PHI"]
        code = np.where(np.isnan(v), np.nan, 1.0 + (v > thresholds[0]) + (v > thresholds[1]))
        rec.logs["lg"] = type(rec.logs["lg"])(rec.logs["lg"].md, {"FACIES": code, "PHI": v})
    return pr
