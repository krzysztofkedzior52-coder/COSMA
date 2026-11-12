from types import SimpleNamespace

import numpy as np
import pytest

from cosma.facies import HardFacies  # noqa: F401
from cosma.grid.boundary import grid_spec_from_cells
from cosma.grid.builder import Layering, build_grid
from cosma.sis import SisError, SisSettings, make_template, simulate_facies
from cosma.strat import StratChart
from cosma.variogram import Structure, VariogramModel
from test_datacheck import make_project

NZ = 6


def world(nx=16, ny=12, rmaj=300.0, rvert=12.0, nug=0.0, same=True, rmin=None, azimuth=0.0):
    spec = grid_spec_from_cells(0, 0, 0, nx, ny, 100.0, 100.0)
    X, _ = np.meshgrid(spec.x_edges, spec.y_edges)
    hz = [np.full_like(X, z) for z in (1000.0, 1030.0)]
    grid, rep = build_grid(spec, hz, ["conformable"], [Layering("fixed_count", n=NZ)])
    model = SimpleNamespace(grid=grid, report=rep, interval_names=["A"])
    pr = make_project()
    pr.chart = StratChart(["A"], "Base")
    uid = pr.chart.intervals[0].uid
    for n, f in enumerate(pr.facies_setup.scheme):
        r = rmaj if (same or n == 0) else rmaj / 2
        vm = VariogramModel(nug, (Structure("spherical", 1.0, r, r if rmin is None else rmin, rvert),), azimuth, "indicator")
        pr.attach_variogram(uid, f.name, vm, "v")
    return pr, model, nx, ny


def props(nx, ny, p):
    P = np.zeros((nx, ny, NZ, 2))
    P[..., 0], P[..., 1] = p, 1 - p
    return P


SET = SisSettings(12, 1.0, 2)


def test_settings_and_template_validation():
    for bad in (dict(max_neighbours=0, search_factor=1.0, levels=1), dict(max_neighbours=4, search_factor=0, levels=1),
                dict(max_neighbours=4, search_factor=1.0, levels=0), dict(max_neighbours=2.5, search_factor=1.0, levels=1),
                dict(max_neighbours=4, search_factor=float("nan"), levels=1), dict(max_neighbours=4, search_factor=1.0, levels=True)):
        with pytest.raises(SisError):
            SisSettings(**bad)
    di, dj, dk = make_template(300.0, 12.0, 100.0, 100.0, 5.0)
    d = (di * 100 / 300) ** 2 + (dj * 100 / 300) ** 2 + (dk * 5 / 12) ** 2
    assert (d <= 1).all() and (d > 0).all() and (np.diff(d) >= 0).all()
    n = sum(1 for i in range(-3, 4) for j in range(-3, 4) for k in range(-2, 3)
            if 0 < (i / 3) ** 2 + (j / 3) ** 2 + (k * 5 / 12) ** 2 <= 1)
    assert len(di) == n


def test_same_seed_same_realization_and_seeds_differ():
    pr, model, nx, ny = world()
    P = props(nx, ny, 0.4)
    a = simulate_facies(pr, model, SET, {"A": P}, None, 7)
    b = simulate_facies(pr, model, SET, {"A": P}, None, 7)
    c = simulate_facies(pr, model, SET, {"A": P}, None, 8)
    np.testing.assert_array_equal(a.facies, b.facies)
    assert (a.facies != c.facies).mean() > 0.05
    assert set(np.unique(a.facies)) == {1, 2}
    assert a.per_interval["A"]["nodes"] == nx * ny * NZ and a.per_interval["A"]["hard"] == 0


def test_a_facies_with_zero_local_proportion_never_appears_there():
    pr, model, nx, ny = world()
    P = props(nx, ny, 0.5)
    P[:, :, :2, 0], P[:, :, :2, 1] = 0.0, 1.0           # no sand in the top two layers
    P[:, :, 4:, 0], P[:, :, 4:, 1] = 1.0, 0.0           # only sand at the bottom
    for s in range(5):
        f = simulate_facies(pr, model, SET, {"A": P}, None, s).facies
        assert (f[:, :, :2] == 2).all() and (f[:, :, 4:] == 1).all()


def hard_with(nx, ny, cells):
    h = np.zeros((nx, ny, NZ), int)
    for (i, j, k), c in cells.items():
        h[i, j, k] = c
    return SimpleNamespace(cells=h)


def test_hard_data_are_honoured_exactly_in_every_realization():
    pr, model, nx, ny = world()
    cells = {(3, 4, 0): 1, (3, 4, 1): 1, (3, 4, 2): 2, (10, 7, 3): 2, (10, 7, 4): 1, (0, 0, 5): 1, (15, 11, 0): 2}
    h = hard_with(nx, ny, cells)
    P = props(nx, ny, 0.5)
    for s in range(6):
        f = simulate_facies(pr, model, SET, {"A": P}, h, s).facies
        for (i, j, k), c in cells.items():
            assert f[i, j, k] == c
    # a hard datum contradicting a zero proportion is still honoured (the conflict is reported by the proportion step)
    P2 = props(nx, ny, 0.0)
    f = simulate_facies(pr, model, SET, {"A": P2}, hard_with(nx, ny, {(5, 5, 2): 1}), 1).facies
    assert f[5, 5, 2] == 1


def test_facies_near_a_hard_datum_resemble_it():
    pr, model, nx, ny = world(rmaj=400.0)
    P = props(nx, ny, 0.5)
    h = hard_with(nx, ny, {(8, 6, k): 1 for k in range(NZ)})
    near = far = 0.0
    for s in range(30):
        f = simulate_facies(pr, model, SisSettings(12, 1.0, 2), {"A": P}, h, s).facies
        near += (f[7:10, 5:8, :] == 1).mean()
        far += (f[0:3, 0:3, :] == 1).mean()
    assert near / 30 > 0.75 and abs(far / 30 - 0.5) < 0.2


def test_missing_wrong_or_inconsistent_inputs_are_refused():
    pr, model, nx, ny = world()
    P = props(nx, ny, 0.4)
    with pytest.raises(SisError, match="seed"):
        simulate_facies(pr, model, SET, {"A": P}, None, -1)
    with pytest.raises(SisError, match="no proportions"):
        simulate_facies(pr, model, SET, {}, None, 1)
    with pytest.raises(SisError, match="shape"):
        simulate_facies(pr, model, SET, {"A": P[:, :, :3]}, None, 1)
    bad = P.copy()
    bad[0, 0, 0] = (0.7, 0.7)
    with pytest.raises(SisError, match="sum to 1"):
        simulate_facies(pr, model, SET, {"A": bad}, None, 1)
    with pytest.raises(SisError, match="different grid"):
        simulate_facies(pr, model, SET, {"A": P}, SimpleNamespace(cells=np.zeros((2, 2, 2), int)), 1)
    uid = pr.chart.intervals[0].uid
    pr.variograms[uid].pop(pr.facies_setup.scheme[1].name)
    pr.var_primary.pop((uid, pr.facies_setup.scheme[1].name))
    with pytest.raises(SisError, match="no indicator variogram for facies"):
        simulate_facies(pr, model, SET, {"A": P}, None, 1)
    pr2, model2, nx, ny = world()
    m = pr2.variogram_for(pr2.chart.intervals[0].uid, pr2.facies_setup.scheme[0].name)
    from dataclasses import replace
    pr2.variograms[pr2.chart.intervals[0].uid][pr2.facies_setup.scheme[0].name]["v"] = replace(m, transform="none")
    with pytest.raises(SisError, match="not indicators"):
        simulate_facies(pr2, model2, SET, {"A": P}, None, 1)


def ensemble(pr, model, P, n, settings=SET, hard=None):
    return np.array([simulate_facies(pr, model, settings, {"A": P}, hard, s).facies == 1 for s in range(n)], float)


def corr(acc, axis, lag):
    m = acc.mean()
    a = np.take(acc, range(0, acc.shape[axis] - lag), axis=axis)
    b = np.take(acc, range(lag, acc.shape[axis]), axis=axis)
    return ((a * b).mean() - m * m) / (m * (1 - m))


def rho_sph(h, r):
    x = h / r
    return 1 - 1.5 * x + 0.5 * x ** 3 if x < 1 else 0.0


def test_global_proportions_are_reproduced_within_the_documented_clipping_bias():
    pr, model, nx, ny = world(nx=24, ny=24)
    for p, tol in ((0.5, 0.03), (0.3, 0.06)):
        acc = ensemble(pr, model, props(nx, ny, p), 40, SisSettings(16, 1.0, 3))
        assert abs(acc.mean() - p) < tol, (p, acc.mean())


def test_vertical_proportion_curve_is_reproduced_layer_by_layer():
    pr, model, nx, ny = world(nx=24, ny=24)
    sand = np.array([0.9, 0.7, 0.5, 0.3, 0.2, 0.1])
    P = np.zeros((nx, ny, NZ, 2))
    P[..., 0], P[..., 1] = sand, 1 - sand
    acc = ensemble(pr, model, P, 40, SisSettings(16, 1.0, 3))
    np.testing.assert_allclose(acc.mean(axis=(0, 1, 2)), sand, atol=0.07)


def test_indicator_correlation_follows_the_model_with_the_documented_short_lag_shortfall():
    pr, model, nx, ny = world(nx=30, ny=30, rmaj=800.0, rvert=20.0)
    acc = ensemble(pr, model, props(nx, ny, 0.5), 40, SisSettings(16, 1.0, 3))
    for lag in (1, 2, 4):
        sim, mod = corr(acc, 1, lag), rho_sph(lag * 100.0, 800.0)
        assert 0.75 * mod < sim < 1.05 * mod, (lag, sim, mod)
    assert abs(corr(acc, 1, 10)) < 0.1 and abs(corr(acc, 1, 14)) < 0.1             # beyond the range: no correlation
    assert abs(corr(acc, 3, 5)) < 0.12                                           # vertical lag 5 layers = 25 m > range 20 m


def test_anisotropy_and_azimuth_are_respected():
    # major axis east (azimuth 90), minor 300 m: correlation along x (axis 0) beats y (axis 1)
    pr, model, nx, ny = world(nx=30, ny=30, rmaj=800.0, rmin=200.0, rvert=20.0, azimuth=90.0)
    acc = ensemble(pr, model, props(nx, ny, 0.5), 30, SisSettings(16, 1.0, 3))
    cx, cy = corr(acc, 1, 3), corr(acc, 2, 3)
    assert cx > 0.35 and cy < 0.15 and cx > cy + 0.25, (cx, cy)
    pr, model, nx, ny = world(nx=30, ny=30, rmaj=800.0, rmin=200.0, rvert=20.0, azimuth=0.0)
    acc = ensemble(pr, model, props(nx, ny, 0.5), 30, SisSettings(16, 1.0, 3))
    assert corr(acc, 2, 3) > corr(acc, 1, 3) + 0.25


def test_multigrid_reduces_the_artefacts_of_a_single_pass_search_or_at_least_does_not_worsen_them():
    pr, model, nx, ny = world(nx=30, ny=30, rmaj=800.0, rvert=20.0)
    P = props(nx, ny, 0.5)
    err = {}
    for lv in (1, 3):
        acc = ensemble(pr, model, P, 30, SisSettings(16, 1.0, lv))
        err[lv] = sum(abs(corr(acc, 1, l) - rho_sph(l * 100.0, 800.0)) for l in (1, 2, 4, 6, 8, 10))
    assert err[3] < err[1] + 0.05, err


def test_two_intervals_are_simulated_independently():
    spec = grid_spec_from_cells(0, 0, 0, 10, 10, 100.0, 100.0)
    X, _ = np.meshgrid(spec.x_edges, spec.y_edges)
    hz = [np.full_like(X, z) for z in (1000.0, 1020.0, 1040.0)]
    grid, rep = build_grid(spec, hz, ["conformable", "conformable"], [Layering("fixed_count", n=3), Layering("fixed_count", n=3)])
    model = SimpleNamespace(grid=grid, report=rep, interval_names=["A", "B"])
    pr = make_project()
    pr.chart = StratChart(["A", "B"], "Base")
    vm = VariogramModel(0.0, (Structure("spherical", 1.0, 300.0, 300.0, 10.0),), 0.0, "indicator")
    for iv in pr.chart.intervals:
        for f in pr.facies_setup.scheme:
            pr.attach_variogram(iv.uid, f.name, vm, "v")
    PA = np.zeros((10, 10, 3, 2)); PA[..., 0] = 1.0
    PB = np.zeros((10, 10, 3, 2)); PB[..., 1] = 1.0
    r = simulate_facies(pr, model, SET, {"A": PA, "B": PB}, None, 3)
    assert (r.facies[:, :, :3] == 1).all() and (r.facies[:, :, 3:] == 2).all()
    assert set(r.per_interval) == {"A", "B"} and "A:" in r.report and "B:" in r.report
    del pr.variograms[pr.chart.intervals[1].uid]
    pr.var_primary = {k: v for k, v in pr.var_primary.items() if k[0] != pr.chart.intervals[1].uid}
    with pytest.raises(SisError, match="interval 'B': no indicator variogram"):
        simulate_facies(pr, model, SET, {"A": PA, "B": PB}, None, 3)
