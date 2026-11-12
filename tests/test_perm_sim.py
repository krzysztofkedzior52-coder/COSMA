import numpy as np
import pytest

from cosma.datacheck import run_checks
from cosma.perm_sim import (PermSetup, PermSimError, Relation, core_relation_row, simulate_permeability)
from test_datacheck import add_core, core_setup, make_model, make_project
from test_sis import NZ, world

USER = Relation("user", intercept=1.0, slope=10.0, scatter=0.3)


def pw():
    pr, model, nx, ny = world(nx=30, ny=30)
    fac = np.ones((nx, ny, NZ), int)
    fac[:, :, 3:] = 2
    phi = np.full(fac.shape, 0.2)
    phi[:, :, 3:] = 0.1
    return pr, model, fac, phi


def setup(**kw):
    k = dict(porosity="PHI", relations=((("A", 1), USER), (("A", 2), Relation("user", intercept=-1.0, slope=8.0, scatter=0.0))),
             kv_kh=((("A", 1), 0.5), (("A", 2), 0.1)), ky_kx=0.8, core_shift=0.0, extrapolation="allow")
    k.update(kw)
    return PermSetup(**k)


def test_setup_and_relation_validation():
    for bad in (dict(source="x"), dict(source="user"), dict(source="user", intercept=1, slope=1, scatter=-1),
                dict(source="core"), dict(source="core", level="facies x interval", slope=1.0),
                dict(source="user", intercept=1, slope=1, scatter=0, level="facies x interval")):
        with pytest.raises(PermSimError):
            Relation(**bad)
    for bad in (dict(porosity=""), dict(ky_kx=0), dict(core_shift=float("nan")), dict(extrapolation="guess"), dict(lower=-1.0),
                dict(lower=5.0, upper=1.0), dict(kv_kh=((("A", 1), 0.5), (("A", 1), 0.2))), dict(kv_kh=((("A", 1), 0),))):
        with pytest.raises(PermSimError):
            setup(**bad)


def test_line_scatter_anisotropy_and_kvkh_are_exact_in_expectation():
    pr, model, fac, phi = pw()
    r = simulate_permeability(pr, model, fac, phi, setup(), [], 3)
    sand = np.log10(r.permx[:, :, :3])
    assert abs(sand.mean() - (1.0 + 10.0 * 0.2)) < 0.02 and abs(sand.std() - 0.3) < 0.01
    np.testing.assert_allclose(np.log10(r.permx[:, :, 3:]), -1.0 + 8.0 * 0.1)              # zero scatter: exactly the line
    np.testing.assert_allclose(r.permy, r.permx * 0.8)
    np.testing.assert_allclose(r.permz[:, :, :3], r.permx[:, :, :3] * 0.5)
    np.testing.assert_allclose(r.permz[:, :, 3:], r.permx[:, :, 3:] * 0.1)
    a, b = sand[:-1].ravel() - sand.mean(), sand[1:].ravel() - sand.mean()
    assert abs((a * b).mean() / sand.var()) < 0.05                                           # scatter is independent
    again = simulate_permeability(pr, model, fac, phi, setup(), [], 3)
    np.testing.assert_array_equal(again.permx, r.permx)
    assert (simulate_permeability(pr, model, fac, phi, setup(), [], 4).permx[:, :, :3] != r.permx[:, :, :3]).all()


def test_bounds_clip_and_missing_inputs_stop_the_run():
    pr, model, fac, phi = pw()
    r = simulate_permeability(pr, model, fac, phi, setup(lower=20.0, upper=60.0), [], 1)
    assert np.nanmin(r.permx) >= 20.0 - 1e-9 and np.nanmax(r.permx) <= 60.0 + 1e-9
    with pytest.raises(PermSimError, match="seed"):
        simulate_permeability(pr, model, fac, phi, setup(), [], -1)
    with pytest.raises(PermSimError, match="choose the porosity-permeability relation"):
        simulate_permeability(pr, model, fac, phi, setup(relations=((("A", 1), USER),)), [], 1)
    with pytest.raises(PermSimError, match="enter kV/kH"):
        simulate_permeability(pr, model, fac, phi, setup(kv_kh=((("A", 1), 0.5),)), [], 1)
    bad = phi.copy()
    bad[2, 2, 1] = np.nan
    with pytest.raises(PermSimError, match="porosity is missing"):
        simulate_permeability(pr, model, fac, bad, setup(), [], 1)
    with pytest.raises(PermSimError, match="shape of the grid"):
        simulate_permeability(pr, model, fac[:, :, :3], phi[:, :, :3], setup(), [], 1)


def core_world():
    pr = make_project()
    add_core(pr)
    core_setup(pr)
    rep = run_checks(pr, make_model())
    return pr, rep


def test_core_relations_use_the_chosen_level_the_shift_and_the_extrapolation_rule():
    pr, rep = core_world()
    model = make_model()
    row = core_relation_row(rep.regression, "A", 1, "facies x interval")
    assert row["ok"]
    fac = np.ones((6, 3, 10), int)
    fac[:, :, 5:] = 1
    phi = np.full(fac.shape, 0.5 * (row["phi_min"] + row["phi_max"]))
    rel = lambda lvl: Relation("core", level=lvl)                                                # noqa: E731
    st = PermSetup("PHI", ((("A", 1), rel("facies x interval")), (("B", 1), rel("facies, all intervals"))),
                   ((("A", 1), 1.0), (("B", 1), 1.0)), 1.0, 0.0, "allow")
    r = simulate_permeability(pr, model, fac, phi, st, rep.regression, 2)
    pa = r.per_domain[("A", 1)]
    assert pa["intercept"] == pytest.approx(row["intercept"]) and pa["slope"] == pytest.approx(row["slope"])
    assert pa["scatter"] == pytest.approx(row["resid_std"]) and "core, facies x interval" in pa["source"]
    row2 = core_relation_row(rep.regression, "B", 1, "facies, all intervals")
    assert r.per_domain[("B", 1)]["intercept"] == pytest.approx(row2["intercept"])
    shifted = simulate_permeability(pr, model, fac, phi, PermSetup("PHI", st.relations, st.kv_kh, 1.0, -0.3, "allow"), rep.regression, 2)
    assert shifted.per_domain[("A", 1)]["intercept"] == pytest.approx(row["intercept"] - 0.3)
    np.testing.assert_allclose(np.log10(shifted.permx[:, :, :5]), np.log10(r.permx[:, :, :5]) - 0.3)
    # porosity outside the core range
    hi = np.full(fac.shape, row["phi_max"] + 0.1)
    for rule, ok in (("allow", True), ("clamp", True), ("stop", False)):
        s2 = PermSetup("PHI", st.relations, st.kv_kh, 1.0, 0.0, rule)
        if ok:
            out = simulate_permeability(pr, model, fac, hi, s2, rep.regression, 2)
            assert out.per_domain[("A", 1)]["outside_core_range"] == 6 * 3 * 5
            lg = np.log10(out.permx[:, :, :5]).mean()
            line = row["intercept"] + row["slope"] * (row["phi_max"] + (0.1 if rule == "allow" else 0.0))
            assert abs(lg - line) < 0.5 * row["resid_std"] + 0.05
        else:
            with pytest.raises(PermSimError, match="outside the core range"):
                simulate_permeability(pr, model, fac, hi, s2, rep.regression, 2)


def test_a_core_level_that_did_not_pass_or_is_absent_stops_the_run():
    pr, rep = core_world()
    model = make_model()
    fac = np.ones((6, 3, 10), int)
    phi = np.full(fac.shape, 0.15)
    st = lambda lvl: PermSetup("PHI", ((("A", 1), Relation("core", level=lvl)),), ((("A", 1), 1.0),), 1.0, 0.0, "allow")   # noqa: E731
    # B / facies 2 has 3 pairs only (below min_core_pairs): its own level cannot be used
    fac2 = np.full((6, 3, 10), 2)
    s2 = PermSetup("PHI", ((("A", 2), Relation("core", level="facies x interval")), (("B", 2), Relation("core", level="facies x interval"))),
                   ((("A", 2), 1.0), (("B", 2), 1.0)), 1.0, 0.0, "allow")
    with pytest.raises(PermSimError, match="did not pass the data checks"):
        simulate_permeability(pr, model, fac2, phi, s2, rep.regression, 1)
    with pytest.raises(PermSimError, match="no regression at level"):
        simulate_permeability(pr, model, fac, phi, st("facies x interval"), [], 1)


def test_interval_level_uses_the_interval_row_for_every_facies():
    pr, rep = core_world()
    model = make_model()
    fac = np.ones((6, 3, 10), int)
    fac[:, :, :5] = 2
    phi = np.full(fac.shape, 0.15)
    st = PermSetup("PHI", ((("A", 1), Relation("core", level="interval, all facies")),
                           (("A", 2), Relation("core", level="interval, all facies"))), ((("A", 1), 1.0), (("A", 2), 1.0)), 1.0, 0.0, "allow")
    fac[:, :, 5:] = 1
    r = simulate_permeability(pr, model, fac, phi[:, :, :], PermSetup("PHI", st.relations + ((("B", 1), Relation("core", level="interval, all facies")),),
                                                                      st.kv_kh + ((("B", 1), 1.0),), 1.0, 0.0, "allow"), rep.regression, 1)
    row = core_relation_row(rep.regression, "A", None, "interval, all facies")
    assert ("A", 1) not in r.per_domain                       # interval A holds only facies 2 here
    assert r.per_domain[("A", 2)]["intercept"] == pytest.approx(row["intercept"])
    assert r.per_domain[("B", 1)]["intercept"] != pytest.approx(row["intercept"])


# -- spatially correlated scatter --------------------------------------------------------------------------------------
def cw(rmaj=400.0):
    from cosma.variogram import Structure, VariogramModel
    pr, model, fac, phi = pw()
    uid = pr.chart.intervals[0].uid
    for f in pr.facies_setup.scheme:
        pr.attach_variogram(uid, f"RES {f.name}", VariogramModel(0.0, (Structure("spherical", 1.0, rmaj, rmaj, 12.0),), 0.0,
                                                                 "normal_score"), "r")
    return pr, model, fac, phi


CORR = dict(scatter_mode="correlated", scatter_variogram="RES {facies}", scatter_neighbours=12, scatter_search_factor=1.0, scatter_levels=2)


def test_scatter_mode_validation():
    setup(**CORR)
    for bad in (dict(scatter_mode="x"), dict(scatter_mode="correlated"), dict(CORR, scatter_variogram=""), dict(CORR, scatter_neighbours=0),
                dict(CORR, scatter_levels=1.5), dict(CORR, scatter_search_factor=0), dict(scatter_variogram="RES {facies}"),
                dict(scatter_neighbours=4)):
        with pytest.raises(PermSimError):
            setup(**bad)


def test_correlated_scatter_keeps_the_line_and_spread_and_is_continuous():
    from test_sis import rho_sph
    pr, model, fac, phi = cw()
    two = dict(relations=((("A", 1), USER), (("A", 2), Relation("user", intercept=-1.0, slope=8.0, scatter=0.5))))
    mean, sd, lag, lag_ind = [], [], [], []
    for s in range(8):
        r = simulate_permeability(pr, model, fac, phi, setup(**two, **CORR), [], s)
        e = (np.log10(r.permx[:, :, :3]) - (1.0 + 10.0 * 0.2)) / 0.3                     # sand residuals in scatter units
        mean.append(e.mean())
        sd.append(e.std())
        lag.append(np.corrcoef(e[:-1].ravel(), e[1:].ravel())[0, 1])
        ri = simulate_permeability(pr, model, fac, phi, setup(**two), [], s)
        ei = (np.log10(ri.permx[:, :, :3]) - 3.0) / 0.3
        lag_ind.append(np.corrcoef(ei[:-1].ravel(), ei[1:].ravel())[0, 1])
        np.testing.assert_allclose(r.permy, r.permx * 0.8)
        np.testing.assert_allclose(r.permz[:, :, 3:], r.permx[:, :, 3:] * 0.1)
    assert abs(np.mean(mean)) < 0.15 and abs(np.mean(sd) - 1.0) < 0.1
    assert abs(np.mean(lag) - rho_sph(100.0, 400.0)) < 0.08                               # neighbours 100 m apart, range 400 m
    assert abs(np.mean(lag_ind)) < 0.02                                                   # independent scatter has none
    assert "scatter correlated" in r.report and r.per_domain[("A", 1)]["scatter_mode"] == "correlated"


def test_correlated_scatter_is_reproducible_reports_progress_and_needs_its_variogram():
    pr, model, fac, phi = cw()
    seen = []
    a = simulate_permeability(pr, model, fac, phi, setup(**CORR), [], 5, lambda l, d, t: seen.append((l, d, t)))
    b = simulate_permeability(pr, model, fac, phi, setup(**CORR), [], 5)
    c = simulate_permeability(pr, model, fac, phi, setup(**CORR), [], 6)
    np.testing.assert_array_equal(a.permx, b.permx)
    assert not np.array_equal(a.permx, c.permx)
    assert [x[1] for x in seen] == sorted(x[1] for x in seen) and seen[0][0].startswith("permeability, A /") and seen[0][2] == fac.size
    np.testing.assert_allclose(np.log10(a.permx[:, :, 3:]), -1.0 + 8.0 * 0.1)             # zero scatter stays exactly on the line
    with pytest.raises(PermSimError, match="no variogram named 'NOPE Sand'"):
        simulate_permeability(pr, model, fac, phi, setup(**dict(CORR, scatter_variogram="NOPE {facies}")), [], 5)
    with pytest.raises(PermSimError, match="not normal scores"):
        simulate_permeability(pr, model, fac, phi, setup(**dict(CORR, scatter_variogram="{facies}")), [], 5)
