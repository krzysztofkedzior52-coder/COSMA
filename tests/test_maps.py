import copy

import numpy as np
import pytest

from cosma.maps import (LayerData, MapError, MapGrid, PropertyMap, _weights, grid_from_surface, grid_from_wells, krige,
                        krige_map, layer_averages, thickness_map)
from cosma.surfaces import Surface
from cosma.variogram import Structure, VariogramModel, shape
from cosma.wells.model import LogSet
from synthetic_field import facies_project, synthetic
from test_variogram_project import build, uid


def sph(nug=0.0, c=1.0, a=1000.0, b=1000.0, az=0.0):
    return VariogramModel(nug, (Structure("spherical", c, a, b, 10.0),), az).check()


# -- weights / averages -------------------------------------------------------------------------------------------------------------
def test_weights_uniform_and_gaps_not_bridged():
    z = np.arange(0.5, 50, 1.0)
    w = _weights(z, 0.0, 50.0)
    assert w == pytest.approx(np.ones(50)) and w.sum() == pytest.approx(50)
    z = np.r_[np.arange(0.0, 11.0), np.arange(40.0, 51.0)]                 # a 30 m hole in the middle
    w = _weights(z, 0.0, 50.0)
    assert w.sum() == pytest.approx(22.0)                                  # 0.5 + 9 + 1.5 | gap capped at 2 m | 1.5 + 9 + 0.5
    assert w[10] == pytest.approx(1.5) and w[11] == pytest.approx(1.5) and w[0] == w[-1] == 0.5
    assert _weights(np.array([3.0]), 0, 10)[0] == 0.0


def test_layer_average_is_thickness_weighted():
    pr = build()
    ld = layer_averages(pr, uid(pr, "Res"), "PHI")                           # PHI = md % 2 sampled each metre, 51 samples, 50 m thick
    assert ld.value == pytest.approx([0.5, 0.5, 0.5])                        # weighted 25/50; a plain mean would be 25/51
    assert ld.coverage["W-1"] == pytest.approx(1.0) and ld.thickness["W-2"] == 50
    assert list(ld.x) == [0.0, 500.0, 1000.0]
    ld = layer_averages(pr, uid(pr, "Res"), "PERM")
    assert ld.value == pytest.approx([5.0, 5.0, 5.0])


def test_coverage_skip_warn_and_errors():
    pr = build()
    u = uid(pr, "Res")
    for w, keep in (("W-1", 0.3), ("W-2", 0.7)):                             # blank out part of the interval
        lg = pr.wells[w].logs["lg"]
        md = lg.md
        v = lg.curves["PHI"].copy()
        top, base = {"W-1": (150, 200), "W-2": (170, 220)}[w]
        v[(md > top + keep * 50)] = np.nan
        pr.wells[w].logs["lg"] = LogSet(md, {"PHI": v, "PERM": lg.curves["PERM"]})
    ld = layer_averages(pr, u, "PHI")
    assert "W-1" in ld.skipped and "31%" in ld.skipped["W-1"] and ld.wells == ["W-2", "W-3"]
    assert ld.warnings == ["W-2 covers 71% of the interval"]
    ld2 = layer_averages(pr, u, "PHI", min_coverage=0.2)
    assert "W-1" in ld2.wells and not ld2.skipped
    with pytest.raises(MapError, match="between 0 and 1"):
        layer_averages(pr, u, "PHI", min_coverage=1.5)
    with pytest.raises(MapError, match="no well has enough"):
        layer_averages(pr, u, "PHI", min_coverage=1.0, wells=["W-1"])


def test_deviated_well_is_placed_at_its_weighted_centroid():
    pr = build(incl=(0, 30, 0))
    ld = layer_averages(pr, uid(pr, "Res"), "PERM")
    i = ld.wells.index("W-2")
    # the well heads east at 30 deg: x = 500 + east displacement; the centroid lies between the pick positions
    t = pr.wells["W-2"].resolved_tops()
    xs = {tp.name: tp.x for tp in t}
    assert xs["Res"] < ld.x[i] < xs["Basement"] and ld.y[i] == pytest.approx(0)
    assert ld.x[i] == pytest.approx((xs["Res"] + xs["Basement"]) / 2, abs=0.5)


def test_indicator_average_is_the_proportion():
    pr = facies_project(n=2, spacing=100.0)
    u = pr.chart.match_name("Res")
    ld = layer_averages(pr, u, "FACIES", indicator=2)
    for w, p in zip(ld.wells, ld.value):
        f = pr.wells[w].logs["lg"].curves["FACIES"]
        md = pr.wells[w].logs["lg"].md
        inside = (md >= 100) & (md <= 150)
        assert 0 <= p <= 1 and p == pytest.approx(np.mean(f[inside] == 2), abs=0.05)
    assert ld.variable == "FACIES[2]" and ld.indicator == 2


# -- kriging ------------------------------------------------------------------------------------------------------------------------------
def test_krige_single_point_and_constant_data():
    e, v = krige(sph(), [0], [0], [7.0], [[10, 10], [5000, 5000]])
    assert e == pytest.approx([7, 7])
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 1000, 6), rng.uniform(0, 1000, 6)
    e, v = krige(sph(0.2, 0.8, 500, 300, 40), x, y, np.full(6, 3.3), rng.uniform(-500, 1500, (20, 2)))
    assert e == pytest.approx(3.3) and np.all(v >= 0)


def test_krige_exact_at_data_with_zero_variance_even_with_nugget():
    rng = np.random.default_rng(1)
    x, y, z = rng.uniform(0, 1000, 5), rng.uniform(0, 1000, 5), rng.normal(size=5)
    e, v = krige(sph(0.3, 0.7, 600, 300, 20), x, y, z, np.c_[x, y])
    assert e == pytest.approx(z) and v == pytest.approx(0, abs=1e-9)


def test_krige_symmetric_midpoint_and_variance_formula():
    m = sph(0.0, 1.0, 1000, 1000)
    e, v = krige(m, [0, 400], [0, 0], [2.0, 6.0], [[200, 0]])
    assert e[0] == pytest.approx(4.0)
    g = lambda h: float(m.gamma_xyz(h, 0, 0))
    # w = (1/2, 1/2): variance = sum w g0 + mu, mu from the first row: g(400)*0.5 + mu = g(200) ... solve by hand
    mu = g(200) - 0.5 * g(400)
    assert v[0] == pytest.approx(g(200) + mu)


def test_krige_matches_independent_covariance_formulation():
    rng = np.random.default_rng(2)
    n = 7
    x, y, z = rng.uniform(0, 900, n), rng.uniform(0, 900, n), rng.normal(size=n)
    m = sph(0.15, 0.85, 700, 350, 30.0)
    m = VariogramModel(m.nugget, m.structures, 40.0)
    nodes = rng.uniform(-200, 1100, (15, 2))
    e, v = krige(m, x, y, z, nodes)
    # independent route: ordinary kriging with covariances, C(h) = sill - gamma(h) for h > 0, C(0) = sill
    az = np.radians(40.0)

    def cov(dx, dy):
        u = dx * np.sin(az) + dy * np.cos(az)
        w = dx * np.cos(az) - dy * np.sin(az)
        r = np.sqrt((u / 700) ** 2 + (w / 350) ** 2)
        return np.where((dx == 0) & (dy == 0), 1.0, 0.85 * (1 - shape("spherical", r)))
    C = cov(x[:, None] - x[None], y[:, None] - y[None])
    A = np.block([[C, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    for k, (nx, ny) in enumerate(nodes):
        rhs = np.r_[cov(nx - x, ny - y), 1.0]
        s = np.linalg.solve(A, rhs)
        assert e[k] == pytest.approx(s[:n] @ z, abs=1e-9)
        assert v[k] == pytest.approx(1.0 - s[:n] @ rhs[:n] - s[n], abs=1e-9)


def test_anisotropy_direction_matters():
    x, y, val = [0, 600], [0, 0], [0.0, 10.0]
    node = [[0, 150]]                                                     # 150 m north of the first point
    north_major = krige(sph(0, 1, 1000, 200, az=0), x, y, val, node)[0][0]
    east_major = krige(sph(0, 1, 1000, 200, az=90), x, y, val, node)[0][0]
    assert north_major < 2.0 < east_major                                  # continuity along north keeps A's value; across it, the estimate drifts
    assert east_major > north_major + 1


def test_krige_refusals():
    with pytest.raises(MapError, match="same location"):
        krige(sph(), [0, 0], [0, 0], [1, 2], [[1, 1]], names=["A", "B"])
    with pytest.raises(MapError, match="A and B"):
        krige(sph(), [0, 0], [0, 0], [1, 2], [[1, 1]], names=["A", "B"])
    with pytest.raises(MapError, match="at least one"):
        krige(sph(), [], [], [], [[0, 0]])
    with pytest.raises(MapError, match="finite"):
        krige(sph(), [0, 1], [0, 0], [1, np.nan], [[0, 0]])
    import cosma.maps as M
    old = M.MAX_WELLS
    M.MAX_WELLS = 2
    try:
        with pytest.raises(MapError, match="limited"):
            krige(sph(), [0, 1, 2], [0, 0, 0], [1, 2, 3], [[0, 0]])
    finally:
        M.MAX_WELLS = old


def test_krige_chunks_give_identical_results():
    rng = np.random.default_rng(3)
    x, y, z = rng.uniform(0, 500, 4), rng.uniform(0, 500, 4), rng.normal(size=4)
    nodes = rng.uniform(0, 500, (23, 2))
    a = krige(sph(0.1, 0.9, 300, 300), x, y, z, nodes)
    b = krige(sph(0.1, 0.9, 300, 300), x, y, z, nodes, chunk=5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# -- grids / maps ------------------------------------------------------------------------------------------------------------------------
def test_grid_helpers_and_validation():
    g = grid_from_wells([[0, 0], [100, 50]], 25.0, 50.0)
    assert (g.xori, g.yori) == (-50, -50) and g.ncol == 9 and g.nrow == 7
    X, Y = g.node_xy()
    assert X.max() >= 150 and Y.max() >= 100
    s = Surface(10, 20, 5, 6, 30.0, 1, np.zeros((4, 3)))
    g2 = grid_from_surface(s)
    assert np.allclose(g2.node_xy()[0], s.node_xy()[0]) and np.allclose(g2.node_xy()[1], s.node_xy()[1])
    for bad in (lambda: MapGrid(0, 0, 0, 1, 3, 3), lambda: MapGrid(0, 0, 1, 1, 1, 3), lambda: MapGrid(0, 0, 1, 1, 6000, 6000),
                lambda: grid_from_wells([], 1, 1), lambda: grid_from_wells([[0, 0]], 0, 1), lambda: grid_from_wells([[0, 0]], 1, -1)):
        with pytest.raises(MapError):
            bad()


def test_property_map_validates_and_samples():
    g = MapGrid(0, 0, 10, 10, 3, 3)
    v = np.arange(9.0).reshape(3, 3)
    m = PropertyMap(g, v, "PHI", "fraction", "imported")
    assert m.sample(10, 10) == pytest.approx(v[1, 1]) and np.isnan(m.sample(500, 0))
    assert m.stats() == dict(n=9, min=0.0, max=8.0, mean=4.0)
    assert PropertyMap(g, np.full((3, 3), np.nan), "x", "", "imported").stats() == dict(n=0)
    with pytest.raises(MapError, match="shape"):
        PropertyMap(g, np.zeros((2, 2)), "x", "", "imported")
    with pytest.raises(MapError, match="shape"):
        PropertyMap(g, v, "x", "", "kriged", np.zeros((2, 2)))
    with pytest.raises(MapError, match="method"):
        PropertyMap(g, v, "x", "", "guessed")


def planes(pr, top, base, mid=None):
    u = pr.chart.match_name("Res")
    def sf(d):
        return Surface(-1000, -1000, 100.0, 100.0, 0.0, 1, np.full((40, 40), float(d)))
    pr.attach_surface(u, sf(top), "t")
    pr.attach_surface("base", sf(base), "b")
    return u


def test_thickness_map_and_errors():
    pr = build()
    u = pr.chart.match_name("Res")
    g = MapGrid(0, 0, 100, 100, 5, 5)
    with pytest.raises(MapError, match="top, base"):
        thickness_map(pr, u, g)
    planes(pr, 100.0, 160.0)
    m = thickness_map(pr, u, g)
    assert m.method == "thickness" and m.unit == "m" and np.all(m.values == 60.0) and m.info["crossed_nodes"] == 0
    pr.surfaces["base"]["b"].values[:, :] = 90.0                           # base above top: crossed
    m = thickness_map(pr, u, g)
    assert np.all(np.isnan(m.values)) and m.info["crossed_nodes"] == 25


def test_krige_map_exact_at_wells_masked_clipped_and_noted():
    pr = build()
    u = planes(pr, 100.0, 160.0)
    pr.surfaces["base"]["b"].values[:15, :] = 50.0                           # in the west the base is above the top: no interval there
    m_ = VariogramModel(0.0, (Structure("spherical", 1.0, 800, 800, 10),), 0).check()
    g = MapGrid(-1000, -500, 250, 250, 9, 3)                                   # nodes at x = -1000..1000 step 250, y = -500, -250, 0 ... includes the wells at y = 0
    # make a PERM trend so wells differ
    for k, w in enumerate(pr.wells.values()):
        lg = w.logs["lg"]
        lg.curves["PERM"][:] = 10.0 * (k + 1)
    m, notes = krige_map(pr, u, "PERM", m_, g, unit="mD")
    assert m.method == "kriged" and m.unit == "mD" and m.variable == "PERM" and m.info["wells"] == ["W-1", "W-2", "W-3"]
    X, Y = g.node_xy()
    ok = ~np.isnan(m.values)
    for k, xw in enumerate((0.0, 500.0, 1000.0)):
        i = int(np.argmin(abs(X[:, 0] - xw)))
        j = int(np.argmin(abs(Y[0] - 0.0)))
        if ok[i, j]:
            assert m.values[i, j] == pytest.approx(10.0 * (k + 1)) and m.variance[i, j] == pytest.approx(0, abs=1e-7)
    assert m.info["masked"] > 0 and np.isnan(m.values[0, 0]) and np.all(np.isnan(m.variance[np.isnan(m.values)]))
    assert not any("Not masked" in n for n in notes)
    m2, n2 = krige_map(pr, u, "PERM", m_, g, bounds=(15.0, 25.0), mask="none")
    assert m2.info["clipped"] > 0 and np.nanmax(m2.values) <= 25.0 and np.nanmin(m2.values) >= 15.0 and m2.info["masked"] == 0
    assert any("clipped" in n for n in n2)
    with pytest.raises(MapError, match="mask"):
        krige_map(pr, u, "PERM", m_, g, mask="x")
    with pytest.raises(MapError, match="bounds"):
        krige_map(pr, u, "PERM", m_, g, bounds=(2, 1))


def test_krige_map_notes_for_few_wells_and_missing_surfaces():
    pr = build()
    u = pr.chart.match_name("Res")
    g = MapGrid(0, 0, 100, 100, 5, 5)
    _, notes = krige_map(pr, u, "PERM", sph(), g)
    assert any("Not masked" in n for n in notes) and any("Only 3 wells" in n for n in notes)
    m, notes = krige_map(pr, u, "PERM", sph(), g, wells=["W-1"])
    assert any("Only one well" in n for n in notes) and np.all(m.values == 5.0)
    f = facies_project(n=2, spacing=100.0)
    uu = f.chart.match_name("Res")
    mm, _ = krige_map(f, uu, "FACIES", sph(), MapGrid(0, 0, 50, 50, 5, 5), indicator=2, bounds=(0, 1), unit="fraction")
    assert mm.variable == "FACIES[2]" and 0 <= np.nanmin(mm.values) and np.nanmax(mm.values) <= 1 and mm.info["indicator"] == 2


def test_full_coverage_is_one_and_boundary_coverage_is_kept():
    pr = build()
    u = uid(pr, "Res")
    ld = layer_averages(pr, u, "PERM")
    assert all(c == pytest.approx(1.0, abs=1e-9) for c in ld.coverage.values())
    lg = pr.wells["W-2"].logs["lg"]
    v = lg.curves["PHI"].copy()
    v[lg.md > 195] = np.nan
    pr.wells["W-2"].logs["lg"] = LogSet(lg.md, {"PHI": v, "PERM": lg.curves["PERM"]})
    c = layer_averages(pr, u, "PHI").coverage["W-2"]
    assert 0.5 < c < 1.0
    assert "W-2" in layer_averages(pr, u, "PHI", min_coverage=c).wells        # exactly at the minimum is enough


def test_well_location_is_thickness_weighted_not_sample_averaged():
    pr = build(incl=(0, 30, 0))
    lg = pr.wells["W-2"].logs["lg"]
    md = np.concatenate([np.arange(0.0, 170.0, 1.0), np.arange(170.0, 195.0, 0.5), np.arange(195.0, 400.0, 1.0)])
    pr.wells["W-2"].logs["lg"] = LogSet(md, {"PHI": md * 0 + 0.2, "PERM": md * 0 + 5.0})      # dense in the upper half of Res (170-220)
    ld = layer_averages(pr, uid(pr, "Res"), "PERM")
    xs = {t.name: t.x for t in pr.wells["W-2"].resolved_tops()}
    assert ld.x[ld.wells.index("W-2")] == pytest.approx((xs["Res"] + xs["Basement"]) / 2, abs=1.0)


def test_nodes_outside_the_surfaces_are_masked():
    pr = build()
    u = planes(pr, 100.0, 160.0)                                              # surfaces cover x from -1000 to 2900
    m_ = VariogramModel(0.0, (Structure("spherical", 1.0, 800, 800, 10),), 0).check()
    g = MapGrid(2500, -250, 500, 250, 3, 3)                                     # x = 2500, 3000, 3500: beyond the surface edge
    m, _ = krige_map(pr, u, "PERM", m_, g, unit="mD")
    X, _Y = g.node_xy()
    outside = X > 2900
    assert outside.any() and np.all(np.isnan(m.values[outside])) and np.all(np.isnan(m.variance[outside]))
    assert m.info["masked"] >= int(outside.sum())
