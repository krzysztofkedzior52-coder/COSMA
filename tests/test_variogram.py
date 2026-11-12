import numpy as np
import pytest

from cosma.variogram import (Structure, VariogramError, VariogramModel, experimental_variogram, fit_direction,
                             from_gslib, normal_score, shape, to_gslib)


def model(az=0.0, **kw):
    return VariogramModel(0.1, (Structure("spherical", 0.9, 1000, 500, 10),), az, **kw).check()


def test_shapes_practical_range():
    assert shape("spherical", 1.0) == 1 and shape("spherical", 2.0) == 1
    assert shape("spherical", 0.5) == pytest.approx(0.6875)
    assert shape("exponential", 1.0) == pytest.approx(1 - np.exp(-3))
    assert shape("gaussian", 1.0) == pytest.approx(1 - np.exp(-3))


def test_axes_and_azimuth():
    m = model(az=90)                      # major axis points east
    assert m.gamma_xyz(500, 0, 0) == pytest.approx(0.1 + 0.9 * 0.6875)
    assert m.gamma_xyz(0, 500, 0) == pytest.approx(1.0)          # north is the minor axis: at its range
    assert m.gamma_along([500], "major")[0] == pytest.approx(0.71875)
    assert m.gamma_along([500], "minor")[0] == pytest.approx(1.0)
    assert m.gamma_along([5], "vertical")[0] == pytest.approx(0.1 + 0.9 * 0.6875)
    assert m.gamma_xyz(0, 0, 0) == 0
    m0 = model(az=0)
    assert m0.gamma_xyz(0, 500, 0) == pytest.approx(0.71875)


def test_validation():
    with pytest.raises(VariogramError):
        VariogramModel(-1, (Structure("spherical", 1, 2, 1, 1),)).check()
    with pytest.raises(VariogramError):
        VariogramModel(0, ()).check()
    with pytest.raises(VariogramError):
        Structure("spherical", 1, 100, 200, 1).check()            # minor > major
    with pytest.raises(VariogramError):
        Structure("cubic", 1, 100, 50, 1).check()


def test_gslib_roundtrip_and_refusals():
    m = VariogramModel(0.05, (Structure("exponential", 0.5, 800, 400, 5), Structure("gaussian", 0.45, 3000, 1500, 20)), 35).check()
    assert from_gslib(to_gslib(m)) == m
    with pytest.raises(VariogramError, match="dip"):
        from_gslib("1 0.1\n1 0.9 0 10 0\n100 50 5\n")
    with pytest.raises(VariogramError, match="type 4"):
        from_gslib("1 0.1\n4 0.9 0 0 0\n100 50 5\n")
    with pytest.raises(VariogramError, match="azimuths"):
        from_gslib("2 0.1\n1 0.4 0 0 0\n100 50 5\n1 0.5 30 0 0\n200 100 5\n")
    with pytest.raises(VariogramError, match="lines"):
        from_gslib("2 0.1\n1 0.4 0 0 0\n100 50 5\n")


def test_experimental_hand_case():
    p = np.array([[0, 0, z] for z in range(4)], float)
    e = experimental_variogram(p, [0, 1, 0, 1], "vertical", lag=1, nlags=3, groups=[0] * 4)
    assert list(e.gamma) == [0.5, 0.0, 0.5] and list(e.npairs) == [3, 2, 1]


def test_vertical_never_pairs_across_wells():
    p = np.array([[0, 0, 0], [0, 0, 1], [1000, 0, 0], [1000, 0, 1]], float)
    e = experimental_variogram(p, [0, 0, 10, 10], "vertical", lag=1, nlags=1, groups=[0, 0, 1, 1])
    assert e.gamma[0] == 0 and e.npairs[0] == 2


def test_horizontal_directions():
    x, y = np.meshgrid(np.arange(10.0), np.arange(10.0))
    p = np.c_[x.ravel(), y.ravel(), np.zeros(100)]
    v = x.ravel()                                                  # varies east only
    east = experimental_variogram(p, v, "major", lag=1, nlags=3, azimuth=90, tolerance_deg=5, vertical_tolerance=1)
    north = experimental_variogram(p, v, "major", lag=1, nlags=3, azimuth=0, tolerance_deg=5, vertical_tolerance=1)
    assert east.gamma == pytest.approx([0.5, 2.0, 4.5])
    assert np.allclose(north.gamma, 0)
    minor = experimental_variogram(p, v, "minor", lag=1, nlags=3, azimuth=0, tolerance_deg=5, vertical_tolerance=1)   # minor of az 0 = east
    assert minor.gamma == pytest.approx([0.5, 2.0, 4.5])


def test_empty_bins_are_nan_not_zero():
    p = np.array([[0, 0, 0], [10, 0, 0]], float)
    e = experimental_variogram(p, [0, 1], "major", lag=5, nlags=3, azimuth=90, tolerance_deg=5, vertical_tolerance=1)
    assert np.isnan(e.gamma[0]) and e.npairs[1] == 1 and np.isnan(e.gamma[2])


def test_normal_score():
    z = normal_score([5, 1, 3, 2, 4])
    assert z[1] < z[3] < z[2] < z[4] < z[0] and abs(z.sum()) < 1e-9
    t = normal_score([1, 1, 2, 2])
    assert t[0] == t[1] and t[2] == t[3]


def test_fit_recovers_known_model():
    lag = np.arange(1, 21) * 50.0
    g = 0.1 + 0.9 * shape("spherical", lag / 600.0)
    from cosma.variogram import Experimental
    e = Experimental("major", lag, g, np.full(20, 100), 1.0)
    f = fit_direction(e, "spherical")
    assert f["nugget"] == pytest.approx(0.1, abs=1e-3) and f["contribution"] == pytest.approx(0.9, abs=1e-3)
    assert f["range"] == pytest.approx(600, rel=1e-3)
    f2 = fit_direction(e, "spherical", nugget=0.1, sill=1.0)
    assert f2["range"] == pytest.approx(600, rel=1e-3)


def test_fit_downweights_few_pair_lags():
    lag = np.arange(1, 21) * 50.0
    g = 0.1 + 0.9 * shape("spherical", lag / 600.0)
    n = np.full(20, 200)
    g2, n2 = g.copy(), n.copy()
    g2[-1] = 5.0; n2[-1] = 1                                       # one wild lag with a single pair
    from cosma.variogram import Experimental
    f = fit_direction(Experimental("major", lag, g2, n2, 1.0), "spherical")
    assert f["range"] == pytest.approx(600, rel=0.05)


def test_oblique_separation_uses_both_axes():
    m = VariogramModel(0.0, (Structure("spherical", 1.0, 1000, 500, 10),), azimuth=30).check()
    az = np.radians(30)
    dx, dy = 300 * np.sin(az) + 100 * np.cos(az), 300 * np.cos(az) - 100 * np.sin(az)   # 300 along major, 100 along minor
    r = np.hypot(300 / 1000, 100 / 500)
    assert m.gamma_xyz(dx, dy, 0) == pytest.approx(shape("spherical", r))


def test_gaussian_value_and_sill():
    assert shape("gaussian", 0.5) == pytest.approx(1 - np.exp(-0.75))
    m = VariogramModel(0.1, (Structure("spherical", 0.5, 10, 5, 1), Structure("gaussian", 0.2, 20, 10, 2))).check()
    assert m.sill == pytest.approx(0.8)
    assert m.gamma_xyz(1e6, 0, 0) == pytest.approx(0.8)


def test_normal_score_exact():
    from scipy.stats import norm
    assert normal_score([10, 20, 30, 40]) == pytest.approx(norm.ppf([0.125, 0.375, 0.625, 0.875]))


def test_fit_rmse():
    from cosma.variogram import Experimental
    lag = np.arange(1, 11) * 50.0
    g = 0.1 + 0.9 * shape("spherical", lag / 300.0)
    assert fit_direction(Experimental("major", lag, g, np.full(10, 50), 1.0), "spherical")["rmse"] == pytest.approx(0, abs=1e-6)
    noisy = g + np.where(np.arange(10) % 2, 0.05, -0.05)
    f = fit_direction(Experimental("major", lag, noisy, np.full(10, 50), 1.0), "spherical")
    assert 0.02 < f["rmse"] < 0.06


def _pts(rows):
    return np.array([r[:3] for r in rows], float), np.array([r[3] for r in rows], float), np.array([r[4] for r in rows])


def test_horizontal_pairs_are_same_layer_cross_well_only():
    # well A at x=0 (z 0 and 10), well B at x=100 (z 0 and 10); values chosen so each pair type is distinguishable
    p, v, g = _pts([(0, 0, 0, 0.0, "A"), (0, 0, 10, 10.0, "A"), (100, 0, 0, 1.0, "B"), (100, 0, 10, 13.0, "B")])
    e = experimental_variogram(p, v, "major", lag=100, nlags=1, azimuth=90, tolerance_deg=5, groups=g, vertical_tolerance=1)
    assert e.npairs[0] == 2 and e.gamma[0] == pytest.approx((0.5 * 1 ** 2 + 0.5 * 3 ** 2) / 2)    # (0,1) and (10,13) only
    e2 = experimental_variogram(p, v, "major", lag=100, nlags=1, azimuth=90, tolerance_deg=5, groups=g, vertical_tolerance=10)
    assert e2.npairs[0] == 4                                         # a thicker layer admits the cross-depth pairs
    one = experimental_variogram(p[:2], v[:2], "major", lag=1, nlags=3, azimuth=0, tolerance_deg=90, groups=["A", "A"], vertical_tolerance=100)
    assert one.npairs.sum() == 0                                      # never inside one well


def test_horizontal_needs_vertical_tolerance_and_respects_angle_and_bandwidth():
    p, v, g = _pts([(0, 0, 0, 0.0, "A"), (100, 0, 0, 2.0, "B"), (0, 100, 0, 5.0, "C"), (70.7107, 70.7107, 0, 9.0, "D")])
    with pytest.raises(VariogramError, match="vertical tolerance"):
        experimental_variogram(p, v, "major", lag=100, nlags=1, groups=g)
    az90 = experimental_variogram(p, v, "major", lag=100, nlags=1, azimuth=90, tolerance_deg=10, groups=g, vertical_tolerance=1)
    assert az90.npairs[0] == 1 and az90.gamma[0] == pytest.approx(0.5 * 2.0 ** 2)               # only A-B points east
    # 45 degrees: A-D (distance 100, exactly on the axis) and B-C (100 apart on the anti-diagonal: perpendicular, excluded)
    az45 = experimental_variogram(p, v, "major", lag=100, nlags=1, azimuth=45, tolerance_deg=10, groups=g, vertical_tolerance=1)
    assert az45.npairs[0] == 1 and az45.gamma[0] == pytest.approx(0.5 * 9.0 ** 2)
    wide = experimental_variogram(p, v, "major", lag=100, nlags=1, azimuth=90, tolerance_deg=60, groups=g, vertical_tolerance=1)
    assert wide.npairs[0] >= 2
    narrow_bw = experimental_variogram(p, v, "major", lag=100, nlags=1, azimuth=90, tolerance_deg=60, bandwidth=10, groups=g, vertical_tolerance=1)
    assert narrow_bw.npairs[0] == 1                                    # offsets beyond the bandwidth drop out


def test_horizontal_direction_is_symmetric():
    p, v, g = _pts([(0, 0, 0, 0.0, "A"), (100, 0, 0, 4.0, "B")])
    a = experimental_variogram(p, v, "major", lag=100, nlags=1, azimuth=90, tolerance_deg=5, groups=g, vertical_tolerance=1)
    b = experimental_variogram(p, v, "major", lag=100, nlags=1, azimuth=270, tolerance_deg=5, groups=g, vertical_tolerance=1)
    assert a.gamma[0] == b.gamma[0] == 8.0


def test_horizontal_never_pairs_within_one_well():
    p = np.array([[0, 0, 0], [100, 0, 0]], float)                 # a deviated well: two samples 100 m apart horizontally
    same = experimental_variogram(p, [0, 3], "major", lag=100, nlags=1, azimuth=90, tolerance_deg=5, groups=["A", "A"], vertical_tolerance=1)
    diff = experimental_variogram(p, [0, 3], "major", lag=100, nlags=1, azimuth=90, tolerance_deg=5, groups=["A", "B"], vertical_tolerance=1)
    assert same.npairs[0] == 0 and diff.npairs[0] == 1
