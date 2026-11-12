import numpy as np
import pytest
from scipy.stats import kstest

from cosma.property_sim import (Distribution, PropertySetup, PropertySimError, hard_property_data, simulate_property)
from cosma.variogram import Structure, VariogramModel
from test_datacheck import add_core, make_model, make_project, phi_curve  # noqa: F401
from test_sis import NZ, rho_sph, world

SAND, SILT = "SAND", "SILT"


def pworld(rmaj=300.0, rvert=12.0, nx=16, ny=12, vname="PHI {facies}", azimuth=0.0):
    pr, model, nx, ny = world(nx=nx, ny=ny)
    uid = pr.chart.intervals[0].uid
    for f in pr.facies_setup.scheme:
        pr.attach_variogram(uid, vname.replace("{facies}", f.name),
                            VariogramModel(0.0, (Structure("spherical", 1.0, rmaj, rmaj, rvert),), azimuth, "normal_score"), "v")
    fac = np.ones((nx, ny, NZ), int)
    fac[:, :, 3:] = 2                                     # sand above, silt below
    return pr, model, fac, nx, ny


def setup(src=("user", "user"), **kw):
    k = dict(sources=((("A", 1), src[0]), (("A", 2), src[1])),
             user=tuple(((("A", c), d) for c, d, s in ((1, Distribution("normal", 0.25, 0.04), src[0]),
                                                        (2, Distribution("lognormal", 0.05, 0.02), src[1])) if s == "user")),
             max_neighbours=12, search_factor=1.0, levels=2)
    k.update(kw)
    return PropertySetup("PHI", "PHI {facies}", **k)


def rec(i, j, k, f, v):
    return dict(well="W", i=i, j=j, k=k, interval=0, facies=f, value=v, log_values=[v])


def test_distribution_roundtrips_and_validation():
    for d in (Distribution("normal", 0.2, 0.05), Distribution("lognormal", 100.0, 80.0)):
        v = np.array([0.5 * d.mean, d.mean, 1.7 * d.mean])
        np.testing.assert_allclose(d.from_z(d.to_z(v)), v)
        assert d.cdf(d.mean) > 0
    ln = Distribution("lognormal", 100.0, 80.0)
    z = np.random.default_rng(0).standard_normal(200000)
    s = ln.from_z(z)
    assert abs(s.mean() - 100) < 1.5 and abs(s.std() - 80) < 2.5
    for bad in (("x", 1, 1), ("normal", 1, 0), ("normal", float("nan"), 1), ("lognormal", -1, 1), ("lognormal", 0, 1)):
        with pytest.raises(PropertySimError):
            Distribution(*bad)


def test_setup_validation():
    d = Distribution("normal", 0.2, 0.05)
    base = dict(name="P", variogram="v", sources=((("A", 1), "user"),), user=((("A", 1), d),))
    PropertySetup(**base)
    for bad in (dict(sources=((("A", 1), "data"), (("A", 1), "user"))), dict(sources=((("A", 1), "guess"),)),
                dict(user=()), dict(sources=((("A", 1), "data"),)), dict(min_data=0), dict(conflict_probability=0.6),
                dict(search_factor=0), dict(levels=0), dict(lower=2.0, upper=1.0), dict(name=""), dict(max_neighbours=True)):
        with pytest.raises(PropertySimError):
            PropertySetup(**{**base, **bad})


def test_user_distributions_variogram_and_domains_are_reproduced():
    pr, model, fac, nx, ny = pworld(rmaj=800.0, rvert=10.0, nx=30, ny=30)
    s = setup()
    runs = [simulate_property(pr, model, fac, s, [], k) for k in range(30)]
    v = np.array([r.values for r in runs])
    assert np.isfinite(v).all()
    sand, silt = v[:, :, :, :3], v[:, :, :, 3:]
    assert abs(sand.mean() - 0.25) < 0.006 and abs(sand.std() - 0.04) < 0.006
    assert abs(silt.mean() - 0.05) < 0.004 and abs(silt.std() - 0.02) < 0.004
    assert silt.min() > 0                                              # log-normal
    # exact Gaussian variogram: correlation of the normal scores follows the model
    z = Distribution("normal", 0.25, 0.04).to_z(sand)
    for lag in (1, 2, 4, 6):
        a, b = z[:, :-lag], z[:, lag:]
        c = (a * b).mean() / z.var()
        assert abs(c - rho_sph(lag * 100.0, 800.0)) < 0.07, (lag, c)


def test_hard_values_are_exact_and_the_data_distribution_is_reproduced():
    pr, model, fac, nx, ny = pworld()
    rng = np.random.default_rng(3)
    cells = [(i, j, k) for i in range(0, 16, 2) for j in range(0, 12, 2) for k in range(3)]
    vals = rng.lognormal(np.log(0.2), 0.3, len(cells))
    data = [rec(i, j, k, 1, v) for (i, j, k), v in zip(cells, vals)]
    s = setup(src=("data", "user"), min_data=10)
    r = simulate_property(pr, model, fac, s, data, 4)
    for (i, j, k), v in zip(cells, vals):
        assert r.values[i, j, k] == pytest.approx(v, abs=1e-12)
    # simulated sand values stay within the data range and follow the data distribution
    sand = r.values[:, :, :3]
    assert sand.min() >= vals.min() - 1e-12 and sand.max() <= vals.max() + 1e-12
    vv = np.concatenate([simulate_property(pr, model, fac, s, data, k).values[:, :, :3].ravel() for k in range(10)])
    assert kstest(vv, lambda x: np.searchsorted(np.sort(vals), x) / len(vals)).statistic < 0.12
    assert r.per_domain[("A", 1)]["source"] == "data" and r.per_domain[("A", 1)]["data"] == len(cells)


def test_conditioning_pulls_neighbours_towards_the_hard_value():
    pr, model, fac, nx, ny = pworld(rmaj=500.0)
    data = [rec(8, 6, k, 1, 0.34) for k in range(3)]                    # 0.34 = about +2.25 sd for N(0.25, 0.04)
    s = setup()
    near = np.mean([simulate_property(pr, model, fac, s, data, k).values[7:10, 5:8, :3].mean() for k in range(20)])
    far = np.mean([simulate_property(pr, model, fac, s, data, k).values[0:3, 0:3, :3].mean() for k in range(20)])
    assert near > 0.31 and abs(far - 0.25) < 0.02


def test_same_seed_same_values_and_only_the_facies_cells_get_values():
    pr, model, fac, nx, ny = pworld()
    s = setup()
    a = simulate_property(pr, model, fac, s, [], 5).values
    np.testing.assert_array_equal(a, simulate_property(pr, model, fac, s, [], 5).values)
    assert (a != simulate_property(pr, model, fac, s, [], 6).values).mean() > 0.9
    fac2 = fac.copy()
    fac2[:, :, 0] = 0                                                   # layer 0 has no facies (inactive)
    b = simulate_property(pr, model, fac2, s, [], 5).values
    assert np.isnan(b[:, :, 0]).all() and np.isfinite(b[:, :, 1:]).all()


def test_bounds_clip_simulated_values_and_stop_on_hard_values_outside():
    pr, model, fac, nx, ny = pworld()
    r = simulate_property(pr, model, fac, setup(lower=0.2, upper=0.3), [], 1)
    assert np.nanmin(r.values) >= 0.2 - 1e-12 and np.nanmax(r.values) <= 0.3 + 1e-12
    with pytest.raises(PropertySimError, match="outside the bounds"):
        simulate_property(pr, model, fac, setup(lower=0.2, upper=0.3), [rec(1, 1, 1, 1, 0.31)], 1)


def test_everything_that_is_missing_or_inconsistent_stops_the_run():
    pr, model, fac, nx, ny = pworld()
    s = setup()
    with pytest.raises(PropertySimError, match="seed"):
        simulate_property(pr, model, fac, s, [], -1)
    with pytest.raises(PropertySimError, match="choose where its distribution comes from"):
        simulate_property(pr, model, fac, PropertySetup("PHI", "PHI {facies}", sources=((("A", 1), "user"),),
                                                         user=((("A", 1), Distribution("normal", 0.2, 0.05)),)), [], 1)
    with pytest.raises(PropertySimError, match="2 cell value\\(s\\), 5 needed"):
        simulate_property(pr, model, fac, setup(src=("data", "user"), min_data=5), [rec(1, 1, 1, 1, 0.2), rec(2, 2, 2, 1, 0.3)], 1)
    with pytest.raises(PropertySimError, match="very unlikely"):
        simulate_property(pr, model, fac, s, [rec(1, 1, 1, 1, 0.9)], 1)
    with pytest.raises(PropertySimError, match="<= 0"):
        simulate_property(pr, model, fac, s, [rec(1, 1, 3, 2, -0.01)], 1)
    with pytest.raises(PropertySimError, match="different grid"):
        simulate_property(pr, model, fac[:, :, :3], s, [], 1)
    with pytest.raises(PropertySimError, match="no variogram named 'OTHER Sand'"):
        simulate_property(pr, model, fac, PropertySetup("PHI", "OTHER {facies}", s.sources, s.user), [], 1)
    uid = pr.chart.intervals[0].uid
    from dataclasses import replace
    key = "PHI Sand"
    pr.variograms[uid][key]["v"] = replace(pr.variograms[uid][key]["v"], transform="none")
    with pytest.raises(PropertySimError, match="not normal scores"):
        simulate_property(pr, model, fac, s, [], 1)


def test_cell_scale_data_come_from_the_wells_through_the_data_checks():
    from cosma.datacheck import DataCheckSetup, PropertySpec
    from test_datacheck import th
    pr = make_project(extra={"PHIE": phi_curve})
    pr.set_data_checks(DataCheckSetup(th(), (PropertySpec("PHI", "PHIE", "arithmetic"),), None))
    recs = hard_property_data(pr, make_model(), "PHI")
    assert recs and all({"i", "j", "k", "interval", "facies", "value", "well"} <= set(r) for r in recs)
    assert {r["facies"] for r in recs} == {1, 2}
    with pytest.raises(PropertySimError, match="no property called"):
        hard_property_data(pr, make_model(), "OTHER")
    pr.data_checks = None
    with pytest.raises(PropertySimError, match="set up the facies and the data checks"):
        hard_property_data(pr, make_model(), "PHI")


def test_a_value_of_another_facies_is_never_used_in_this_domain():
    pr, model, fac, nx, ny = pworld()
    r = simulate_property(pr, model, fac, setup(), [rec(1, 1, 1, 2, 0.9)], 1)          # facies 2 datum lying in a sand cell
    assert 0.1 < r.values[1, 1, 1] < 0.4 and r.per_domain[("A", 1)]["data"] == 0


# -- correlated properties ---------------------------------------------------------------------------------------------
def vsetup(rho=(0.7, -0.6), **kw):
    k = dict(sources=((("A", 1), "user"), (("A", 2), "user")),
             user=((("A", 1), Distribution("normal", 0.2, 0.05)), (("A", 2), Distribution("normal", 0.6, 0.1))),
             max_neighbours=12, search_factor=1.0, levels=2, correlate_with="PHI",
             correlation=((("A", 1), rho[0]), (("A", 2), rho[1])))
    k.update(kw)
    return PropertySetup("VSH", "PHI {facies}", **k)


def scores(res, key, fac, code):
    d = res.per_domain[key]
    return d["score"](res.values[fac == code])


def test_correlation_validation():
    for bad in (dict(correlate_with="VSH"), dict(correlate_with=""), dict(correlation=((("A", 1), 1.0),)),
                dict(correlation=((("A", 1), -1.0),)), dict(correlation=((("A", 1), float("nan")),)),
                dict(correlation=((("A", 1), 0.5), (("A", 1), 0.4))), dict(correlate_with=None)):
        with pytest.raises(PropertySimError):
            vsetup(**bad)
    pr, model, fac, nx, ny = pworld()
    phi = simulate_property(pr, model, fac, setup(), [], 1)
    with pytest.raises(PropertySimError, match="simulated first"):
        simulate_property(pr, model, fac, vsetup(), [], 2)
    with pytest.raises(PropertySimError, match="no correlation set"):
        simulate_property(pr, model, fac, setup(), [], 2, primary=phi)
    with pytest.raises(PropertySimError, match="enter the correlation"):
        simulate_property(pr, model, fac, vsetup(correlation=((("A", 1), 0.5),)), [], 2, primary=phi)


def test_correlation_and_marginals_are_reproduced():
    pr, model, fac, nx, ny = pworld(nx=60, ny=60)
    r1, r2, m, sd = [], [], [], []
    for s in range(6):
        phi = simulate_property(pr, model, fac, setup(), [], s)
        vsh = simulate_property(pr, model, fac, vsetup(), [], 100 + s, primary=phi)
        r1.append(np.corrcoef(scores(phi, ("A", 1), fac, 1), scores(vsh, ("A", 1), fac, 1))[0, 1])
        r2.append(np.corrcoef(scores(phi, ("A", 2), fac, 2), scores(vsh, ("A", 2), fac, 2))[0, 1])
        m.append(vsh.values[fac == 1].mean())
        sd.append(vsh.values[fac == 1].std())
        assert vsh.per_domain[("A", 1)]["correlation"] == 0.7 and "correlation 0.7 with PHI" in vsh.report
    assert abs(np.mean(r1) - 0.7) < 0.05 and abs(np.mean(r2) + 0.6) < 0.05
    assert abs(np.mean(m) - 0.2) < 0.01 and abs(np.mean(sd) - 0.05) < 0.006        # the marginal distribution is untouched


def test_zero_correlation_is_the_independent_simulation_and_the_primary_is_not_changed():
    pr, model, fac, nx, ny = pworld()
    phi = simulate_property(pr, model, fac, setup(), [], 3)
    before = phi.values.copy()
    a = simulate_property(pr, model, fac, vsetup(rho=(0.0, 0.0)), [], 9, primary=phi)
    b = simulate_property(pr, model, fac, vsetup(correlate_with=None, correlation=()), [], 9)
    np.testing.assert_allclose(a.values, b.values, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(phi.values, before)
    # the same seeds give the same pair again
    c = simulate_property(pr, model, fac, vsetup(), [], 9, primary=phi)
    d = simulate_property(pr, model, fac, vsetup(), [], 9, primary=phi)
    np.testing.assert_array_equal(c.values, d.values)


def test_hard_values_stay_exact_and_a_contradiction_stops():
    pr, model, fac, nx, ny = pworld()
    phi_data = [rec(2, 2, 0, 1, 0.33), rec(9, 5, 1, 1, 0.21)]                     # +2 sigma and -1 sigma of N(0.25, 0.04)
    phi = simulate_property(pr, model, fac, setup(), phi_data, 1)
    ok = [rec(2, 2, 0, 1, 0.27), rec(9, 5, 1, 1, 0.16)]                           # same sign as phi: fine with rho 0.7
    v = simulate_property(pr, model, fac, vsetup(), ok, 2, primary=phi)
    assert v.values[2, 2, 0] == 0.27 and v.values[9, 5, 1] == 0.16
    assert "too few hard cells (2)" in v.report
    bad = [rec(2, 2, 0, 1, 0.05)]                                                 # -3 sigma where phi is +2 sigma, rho 0.95
    with pytest.raises(PropertySimError, match="contradict the correlation"):
        simulate_property(pr, model, fac, vsetup(rho=(0.95, 0.0)), bad, 2, primary=phi)
    three = ok + [rec(5, 5, 2, 1, 0.2)]
    v = simulate_property(pr, model, fac, vsetup(), three, 2, primary=phi)
    assert "at the 3 hard cells the scores correlate" in v.report


def test_hard_values_condition_the_residual_not_the_property():
    """With a hard secondary value z2 where the primary score is z1, the residual field must be conditioned on (z2 - rho z1)/c,
    so next to the hard cell E[residual] = rho_variogram(h) * that value."""
    pr, model, fac, nx, ny = pworld()
    rho, c = 0.9, np.sqrt(1 - 0.81)
    phi = simulate_property(pr, model, fac, setup(), [rec(8, 6, 1, 1, 0.33)], 1)          # z1 = +2 at the cell
    d1 = phi.per_domain[("A", 1)]
    hard = [rec(8, 6, 1, 1, 0.2 + 2.0 * 0.05)]                                             # z2 = +2 -> residual (2 - 1.8)/c = 0.459
    e_h = (2.0 - rho * 2.0) / c
    got = []
    for s in range(60):
        v = simulate_property(pr, model, fac, vsetup(rho=(rho, 0.0)), hard, s, primary=phi)
        for (i, j) in ((7, 6), (9, 6), (8, 5), (8, 7)):
            z2 = (v.values[i, j, 1] - 0.2) / 0.05
            z1 = d1["score"](phi.values[i, j, 1])
            got.append((z2 - rho * z1) / c)
    want = rho_sph(100.0, 300.0) * e_h
    assert abs(np.mean(got) - want) < 0.2                                                  # the unconverted value would give 1.04
    assert abs(rho_sph(100.0, 300.0) * 2.0 - want) > 0.6
