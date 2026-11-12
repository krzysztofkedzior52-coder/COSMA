from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import multivariate_normal

from cosma.facies import Facies, FaciesSetup
from cosma.sis import SisError
from cosma.tgs import TgsError, TgsSettings, _ndtri, simulate_tgs, thresholds
from cosma.variogram import Structure, VariogramModel
from test_sis import NZ, hard_with, props, rho_sph, world
from test_datacheck import RULES, SAND, SILT

ST = TgsSettings((1, 2), "GAUSS", 12, 1.0, 2, 5)


def tworld(**kw):
    pr, model, nx, ny = world(**kw)
    rm, rv = kw.get("rmaj", 300.0), kw.get("rvert", 12.0)
    vm = VariogramModel(0.0, (Structure("spherical", 1.0, rm, kw.get("rmin") or rm, rv),), kw.get("azimuth", 0.0), "normal_score")
    pr.attach_variogram(pr.chart.intervals[0].uid, "GAUSS", vm, "g")
    return pr, model, nx, ny


def ens(pr, model, P, n, st=ST, hard=None):
    return np.array([simulate_tgs(pr, model, st, {"A": P}, hard, s).facies == 1 for s in range(n)], float)


def corr(acc, axis, lag):
    m = acc.mean()
    a = np.take(acc, range(0, acc.shape[axis] - lag), axis=axis)
    b = np.take(acc, range(lag, acc.shape[axis]), axis=axis)
    return ((a * b).mean() - m * m) / (m * (1 - m))


def test_inverse_normal_matches_scipy():
    ps = np.array([1e-12, 1e-6, 0.001, 0.02, 0.0243, 0.1, 0.3, 0.5, 0.77, 0.97, 0.999, 1 - 1e-9])
    np.testing.assert_allclose([_ndtri(p) for p in ps], ndtri(ps), atol=1e-8)
    assert _ndtri(0.0) == -np.inf and _ndtri(1.0) == np.inf


def test_thresholds_from_cumulative_proportions():
    P = np.array([[[[0.2, 0.3, 0.5]]]])
    t = thresholds(P, np.array([0, 1, 2]))[0, 0, 0]
    np.testing.assert_allclose(t[1:3], ndtri([0.2, 0.5]))
    assert t[0] == -np.inf and t[3] == np.inf
    t2 = thresholds(P, np.array([2, 0, 1]))[0, 0, 0]               # order: facies 2, 0, 1
    np.testing.assert_allclose(t2[1:3], ndtri([0.5, 0.7]))
    z = thresholds(np.array([[[[0.0, 1.0]]]]), np.array([0, 1]))[0, 0, 0]
    assert z[1] == -np.inf                                         # facies 0 impossible
    one = thresholds(np.array([[[[1.0, 0.0]]]]), np.array([0, 1]))[0, 0, 0]
    assert one[1] == np.inf


def test_settings_validation():
    for bad in (dict(order=(1,)), dict(order=(1, 1)), dict(order=[1, 2]), dict(variogram=""), dict(max_neighbours=0),
                dict(search_factor=0), dict(levels=0), dict(gibbs_sweeps=-1), dict(gibbs_sweeps=True)):
        kw = dict(order=(1, 2), variogram="G", max_neighbours=4, search_factor=1.0, levels=1, gibbs_sweeps=1)
        kw.update(bad)
        with pytest.raises(TgsError):
            TgsSettings(**kw)


def test_determinism_zero_proportions_and_hard_data():
    pr, model, nx, ny = tworld()
    P = props(nx, ny, 0.4)
    a = simulate_tgs(pr, model, ST, {"A": P}, None, 4)
    b = simulate_tgs(pr, model, ST, {"A": P}, None, 4)
    np.testing.assert_array_equal(a.facies, b.facies)
    assert (simulate_tgs(pr, model, ST, {"A": P}, None, 5).facies != a.facies).mean() > 0.05
    P2 = props(nx, ny, 0.5)
    P2[:, :, :2, 0], P2[:, :, :2, 1] = 0.0, 1.0
    P2[:, :, 4:, 0], P2[:, :, 4:, 1] = 1.0, 0.0
    for s in range(4):
        f = simulate_tgs(pr, model, ST, {"A": P2}, None, s).facies
        assert (f[:, :, :2] == 2).all() and (f[:, :, 4:] == 1).all()
    cells = {(3, 4, 0): 1, (3, 4, 1): 1, (3, 4, 2): 2, (10, 7, 3): 2, (10, 7, 4): 1, (0, 0, 5): 1, (15, 11, 0): 2}
    h = hard_with(nx, ny, cells)
    for s in range(5):
        f = simulate_tgs(pr, model, ST, {"A": props(nx, ny, 0.5)}, h, s).facies
        for (i, j, k), c in cells.items():
            assert f[i, j, k] == c
    f = simulate_tgs(pr, model, ST, {"A": props(nx, ny, 0.0)}, hard_with(nx, ny, {(5, 5, 2): 1}), 1).facies
    assert f[5, 5, 2] == 1                                          # hard data win even against a zero proportion


def test_facies_near_hard_data_resemble_them_through_the_gibbs_values():
    pr, model, nx, ny = tworld(rmaj=400.0)
    h = hard_with(nx, ny, {(8, 6, k): 1 for k in range(NZ)})
    near = far = 0.0
    for s in range(30):
        f = simulate_tgs(pr, model, ST, {"A": props(nx, ny, 0.5)}, h, s).facies
        near += (f[7:10, 5:8, :] == 1).mean()
        far += (f[0:3, 0:3, :] == 1).mean()
    assert near / 30 > 0.75 and abs(far / 30 - 0.5) < 0.2


def test_proportions_are_reproduced_without_the_sis_bias():
    pr, model, nx, ny = tworld(nx=24, ny=24)
    for p in (0.5, 0.3):
        acc = ens(pr, model, props(nx, ny, p), 40, TgsSettings((1, 2), "GAUSS", 16, 1.0, 3, 0))
        assert abs(acc.mean() - p) < 0.03, (p, acc.mean())


def test_vertical_proportion_curve_is_reproduced():
    pr, model, nx, ny = tworld(nx=24, ny=24)
    sand = np.array([0.9, 0.7, 0.5, 0.3, 0.2, 0.1])
    P = np.zeros((nx, ny, NZ, 2))
    P[..., 0], P[..., 1] = sand, 1 - sand
    acc = ens(pr, model, P, 40, TgsSettings((1, 2), "GAUSS", 16, 1.0, 3, 0))
    np.testing.assert_allclose(acc.mean(axis=(0, 1, 2)), sand, atol=0.05)


def test_indicator_correlation_matches_the_bivariate_gaussian_theory():
    pr, model, nx, ny = tworld(nx=30, ny=30, rmaj=800.0, rvert=20.0)
    st = TgsSettings((1, 2), "GAUSS", 16, 1.0, 3, 0)
    for p in (0.5, 0.3):
        acc = ens(pr, model, props(nx, ny, p), 40, st)
        t = ndtri(p)
        for lag in (1, 2, 4, 6):
            r = rho_sph(lag * 100.0, 800.0)
            both = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([t, t])
            theory = (both - p * p) / (p * (1 - p))
            assert abs(corr(acc, 2, lag) - theory) < 0.07, (p, lag, corr(acc, 2, lag), theory)


def test_order_controls_which_facies_can_touch():
    pr, model, nx, ny = tworld(nx=24, ny=24, rmaj=500.0, rvert=15.0)
    pr.facies_setup = FaciesSetup((SAND, SILT, Facies(3, "COAL")), RULES, min_coverage=0.5, conflict="drop")
    P = np.zeros((nx, ny, NZ, 3))
    P[..., :] = (0.3, 0.3, 0.4)

    def contacts(order):
        n = tot = 0
        for s in range(12):
            f = simulate_tgs(pr, model, TgsSettings(order, "GAUSS", 12, 1.0, 2, 0), {"A": P}, None, s).facies
            a, b = f[:-1], f[1:]
            n += (((a == 1) & (b == 2)) | ((a == 2) & (b == 1))).sum()
            tot += a.size
        return n / tot
    assert contacts((1, 3, 2)) < 0.25 * contacts((1, 2, 3))


def test_refusals():
    pr, model, nx, ny = tworld()
    P = props(nx, ny, 0.4)
    with pytest.raises(TgsError, match="seed"):
        simulate_tgs(pr, model, ST, {"A": P}, None, -2)
    with pytest.raises(TgsError, match="order must contain exactly"):
        simulate_tgs(pr, model, replace(ST, order=(1, 3)), {"A": P}, None, 1)
    with pytest.raises(TgsError, match="no proportions"):
        simulate_tgs(pr, model, ST, {}, None, 1)
    with pytest.raises(TgsError, match="sum to 1"):
        bad = P.copy()
        bad[0, 0, 0] = (0.7, 0.7)
        simulate_tgs(pr, model, ST, {"A": bad}, None, 1)
    with pytest.raises(TgsError, match="no variogram named"):
        simulate_tgs(pr, model, replace(ST, variogram="OTHER"), {"A": P}, None, 1)
    with pytest.raises(TgsError, match="gibbs_sweeps must be >= 1"):
        simulate_tgs(pr, model, replace(ST, gibbs_sweeps=0), {"A": P}, hard_with(nx, ny, {(1, 1, 1): 1}), 1)
    with pytest.raises(SisError, match="different grid"):
        simulate_tgs(pr, model, ST, {"A": P}, SimpleNamespace(cells=np.zeros((2, 2, 2), int)), 1)
    uid = pr.chart.intervals[0].uid
    pr.variograms[uid]["GAUSS"]["g"] = replace(pr.variograms[uid]["GAUSS"]["g"], transform="none")
    with pytest.raises(TgsError, match="not normal scores"):
        simulate_tgs(pr, model, ST, {"A": P}, None, 1)


def test_hard_facies_with_zero_local_proportion_is_honoured_without_poisoning_the_field():
    pr, model, nx, ny = tworld()
    pr.facies_setup = FaciesSetup((SAND, SILT, Facies(3, "COAL")), RULES, min_coverage=0.5, conflict="drop")
    P = np.zeros((nx, ny, NZ, 3))
    P[..., :] = (0.5, 0.0, 0.5)                                    # the middle facies is impossible everywhere
    h = hard_with(nx, ny, {(4, 4, 2): 2, (9, 5, 3): 2, (6, 6, 1): 1})
    st = TgsSettings((1, 2, 3), "GAUSS", 12, 1.0, 2, 3)
    for s in range(4):
        r = simulate_tgs(pr, model, st, {"A": P}, h, s)
        f = r.facies
        assert f[4, 4, 2] == 2 and f[9, 5, 3] == 2 and f[6, 6, 1] == 1
        assert (f == 2).sum() == 2 and set(np.unique(f)) == {1, 2, 3}
        assert ((f == 1) | (f == 3)).sum() == f.size - 2
