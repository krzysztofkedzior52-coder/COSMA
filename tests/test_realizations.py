import numpy as np
import pytest

from cosma.perm_sim import PermSetup, Relation
from cosma.property_sim import Distribution, PropertySetup
from cosma.realizations import (MAX_REALIZATIONS, PropertyJob, RealizationError, RealizationManager, RealizationPlan,
                                from_state, integral_volume, seed_for, to_state, _tests)
from cosma.variogram import Structure, VariogramModel
from test_datacheck import make_model
from test_facies_sim import SIS_S, TGS_S, project


def world(method=SIS_S):
    pr, model = project(), make_model()
    pr.set_facies_sim(method)
    for iv in pr.chart.intervals:
        for f in pr.facies_setup.scheme:
            pr.attach_variogram(iv.uid, f"PHI {f.name}", VariogramModel(0.0, (Structure("spherical", 1.0, 300.0, 300.0, 10.0),), 0.0,
                                                                         "normal_score"), "p")
    return pr, model


def psetup(pr, name="PHI", mean=0.2):
    codes = [f.code for f in pr.facies_setup.scheme]
    keys = [(iv.name, c) for iv in pr.chart.intervals for c in codes]
    return PropertySetup(name, "PHI {facies}", tuple((k, "user") for k in keys),
                         tuple((k, Distribution("normal", mean, 0.03)) for k in keys), max_neighbours=8, levels=1)


def permsetup(pr):
    codes = [f.code for f in pr.facies_setup.scheme]
    keys = [(iv.name, c) for iv in pr.chart.intervals for c in codes]
    return PermSetup("PHI", tuple((k, Relation("user", intercept=1.0, slope=10.0, scatter=0.2)) for k in keys),
                     tuple((k, 0.1) for k in keys), 1.0, 0.0, "allow")


def plan(pr, **kw):
    k = dict(jobs=(PropertyJob(psetup(pr), "PORO"),), perm=permsetup(pr), count=3, base_seed=7)
    k.update(kw)
    return RealizationPlan(**k)


def test_plan_validation_and_state_roundtrip():
    pr, _ = world()
    p = plan(pr, on_fail="redraw", max_redraws=2, redraw="properties", confidence=0.9)
    assert from_state(to_state(p)) == p and from_state(None) is None
    for bad in (dict(count=0), dict(count=MAX_REALIZATIONS + 1), dict(base_seed=-1), dict(confidence=0.4), dict(confidence=1.0),
                dict(on_fail="x"), dict(on_fail="redraw"), dict(redraw="x"), dict(jobs=()),
                dict(jobs=(PropertyJob(psetup(pr), "PORO"), PropertyJob(psetup(pr, "B"), "PORO"))),
                dict(jobs=(PropertyJob(psetup(pr, "OTHER"), "PORO"),))):
        with pytest.raises(RealizationError):
            plan(pr, **bad)
    for kw in ("poro", "PERMX", "", "TOOLONGKEY", "1AB"):
        with pytest.raises(RealizationError):
            PropertyJob(psetup(pr), kw)


def test_seeds_are_deterministic_and_distinct():
    pr, _ = world()
    p = plan(pr)
    s = {seed_for(p, n, a, st) for n in range(5) for a in range(3) for st in (0, 1, 2, 100)}
    assert len(s) == 60 and seed_for(p, 2, 1, 0) == seed_for(p, 2, 1, 0)
    assert seed_for(plan(pr, base_seed=8), 2, 1, 0) != seed_for(p, 2, 1, 0)


def test_integral_volume_matches_numerical_integration():
    ax = np.linspace(-4, 4, 161)
    d = ax[1] - ax[0]
    r = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    for t, rho in (("spherical", np.where(r < 1, 1 - 1.5 * r + 0.5 * r ** 3, 0.0)), ("exponential", np.exp(-3 * r)),
                   ("gaussian", np.exp(-3 * r * r))):
        m = VariogramModel(0.0, (Structure(t, 1.0, 100.0, 50.0, 20.0),), 0.0, "normal_score")
        num = rho.sum() * d ** 3 * 100.0 * 50.0 * 20.0
        assert abs(integral_volume(m, 1.0) / num - 1) < 0.01, t
    m = VariogramModel(0.5, (Structure("spherical", 0.5, 100.0, 100.0, 10.0),), 0.0, "normal_score")      # nugget adds one cell volume
    assert abs(integral_volume(m, 8.0) - (0.5 * 8.0 + 0.5 * np.pi / 6 * 1e5) / 1.0) < 1e-6


def test_tests_accept_independent_normals_and_reject_shifted_or_squeezed():
    rng = np.random.default_rng(2)
    rej = np.zeros(3)
    for _ in range(300):
        t = _tests(rng.standard_normal(200), 200.0)
        rej += [t["p_mean"] < 0.05, t["p_variance"] < 0.05, t["p_shape"] < 0.05]
    assert (rej / 300 < 0.09).all()
    for s, key in ((rng.standard_normal(500) + 0.4, "p_mean"), (0.6 * rng.standard_normal(500), "p_variance"),
                   (rng.choice([-1.0, 1.0], 500) * 0.9 + 0.43 * rng.standard_normal(500), "p_shape")):
        assert _tests(s, 500.0)[key] < 1e-3


def test_realizations_are_reproducible_and_differ(tmp_path):
    pr, model = world()
    p = plan(pr)
    m = RealizationManager(pr, model, p)
    f0, v0, k0, c0 = m.generate(0)
    f0b, v0b, k0b, _ = m.generate(0)
    np.testing.assert_array_equal(f0, f0b)
    np.testing.assert_array_equal(v0["PORO"], v0b["PORO"])
    np.testing.assert_array_equal(k0[0], k0b[0])
    assert v0["PORO"].dtype == np.float32 and k0[0].dtype == np.float32
    f1, v1, _, _ = m.generate(1)
    assert not np.array_equal(v0["PORO"], v1["PORO"])
    # permeability follows the typed line from the stored float32 porosity
    ph = v0["PORO"].astype(float)
    ok = np.isfinite(ph)
    assert abs(np.log10(k0[0][ok]) .mean() - (1.0 + 10.0 * ph[ok]).mean()) < 0.1
    # properties-only redraw keeps the facies; both changes them
    mp = RealizationManager(pr, model, plan(pr, redraw="properties"))
    fa, va, _, _ = mp.generate(0, 0)
    fb, vb, _, _ = mp.generate(0, 1)
    np.testing.assert_array_equal(fa, fb)
    assert not np.array_equal(va["PORO"], vb["PORO"])
    import cosma.facies_sim as FS
    seen = []
    real = FS.run
    try:
        FS.run = lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1]
        for redraw, differ in (("both", True), ("properties", False)):
            seen.clear()
            mm = RealizationManager(pr, model, plan(pr, redraw=redraw))
            mm.generate(0, 0)
            mm.generate(0, 1)
            assert (seen[0] != seen[1]) == differ, redraw
    finally:
        FS.run = real
    with pytest.raises(RealizationError):
        m.generate(3)


def test_cache_roundtrip_version_and_plan_mismatch(tmp_path):
    pr, model = world()
    p = plan(pr)
    m = RealizationManager(pr, model, p, cache_dir=tmp_path)
    r = m.realize(1)
    assert not r.from_cache and (tmp_path / "realization_001.npz").exists()
    c = RealizationManager(pr, model, p, cache_dir=tmp_path).realize(1)
    assert c.from_cache and c.status == r.status
    np.testing.assert_array_equal(c.props["PORO"], r.props["PORO"])
    np.testing.assert_array_equal(c.permx, r.permx)
    import cosma.realizations as R
    old = R.GENERATOR_VERSION
    try:
        R.GENERATOR_VERSION = "other"
        c2 = RealizationManager(pr, model, p, cache_dir=tmp_path).realize(1)
        assert c2.from_cache and "would not reproduce" in c2.cache_note            # the cache wins
        np.testing.assert_array_equal(c2.props["PORO"], r.props["PORO"])
    finally:
        R.GENERATOR_VERSION = old
    with pytest.raises(RealizationError, match="different plan"):
        RealizationManager(pr, model, plan(pr, base_seed=8), cache_dir=tmp_path).realize(1)
    (tmp_path / "realization_001.npz").write_bytes(b"junk")
    with pytest.raises(RealizationError, match="unreadable"):
        RealizationManager(pr, model, p, cache_dir=tmp_path).realize(1)


def force_fail(pr):
    """A target far from what the hard data / variogram allow can't be forced on a tiny grid, so tests patch the check."""


def test_failure_handling_keep_discard_redraw(monkeypatch):
    import cosma.realizations as R
    pr, model = world()
    calls = []

    def fake(project, model, plan, facies, values, per, props, scheme, _real=R.check_realization):
        calls.append(1)
        bad = len(calls) < fake.pass_after
        return dict(rows=[], ntests=1, alpha=0.05, failed=[("PORO", "A", 1, "mean")] if bad else [], text="x")
    monkeypatch.setattr(R, "check_realization", fake)
    fake.pass_after = 10 ** 9
    assert RealizationManager(pr, model, plan(pr, on_fail="keep")).realize(0).status == "flagged"
    assert RealizationManager(pr, model, plan(pr, on_fail="discard")).realize(0).status == "discarded"
    calls.clear()
    fake.pass_after = 3                                   # the third attempt passes
    r = RealizationManager(pr, model, plan(pr, on_fail="redraw", max_redraws=3)).realize(0)
    assert r.status == "ok" and r.attempt == 2
    calls.clear()
    fake.pass_after = 10 ** 9
    r = RealizationManager(pr, model, plan(pr, on_fail="redraw", max_redraws=2)).realize(0)
    assert r.status == "flagged" and r.attempt == 2 and len(calls) == 3
    d = RealizationManager(pr, model, plan(pr, on_fail="discard")).realize(0)
    assert d.facies is None
    with pytest.raises(RealizationError, match="discarded"):
        RealizationManager(pr, model, plan(pr, on_fail="discard")).export(0, "x.grdecl")
    with pytest.raises(RealizationError, match="flagged"):
        RealizationManager(pr, model, plan(pr, on_fail="keep")).export(0, "x.grdecl")


def test_missing_inputs_are_reported():
    pr, model = world()
    pr.set_facies_sim(None)
    with pytest.raises(RealizationError, match="facies simulation"):
        RealizationManager(pr, model, plan(pr))


def full_model(model, faults=False):
    from types import SimpleNamespace
    return SimpleNamespace(grid=model.grid, report=model.report, interval_names=model.interval_names,
                           fault_reports=[1] if faults else [], fault_choices={}, notes=[], region=None, active_columns=None,
                           options=SimpleNamespace(pv=None, aquifer=None, buffer=None))


def test_export_writes_grid_properties_and_gives_vshale_to_the_faults_only(tmp_path, monkeypatch):
    import cosma.fault_transmissibility as FT
    pr, model = world()
    vs = psetup(pr, "VSH", 0.3)
    p = plan(pr, jobs=(PropertyJob(psetup(pr), "PORO"), PropertyJob(vs, "VSHALE", export=False)))
    fm = full_model(model, faults=True)
    seen = {}

    def spy(m, choices):
        seen["keys"] = sorted(m.grid.props)
        seen["vshale"] = np.array(m.grid.props["VSHALE"])
        seen["permx"] = np.array(m.grid.props["PERMX"])
        return "-- faults\n"
    monkeypatch.setattr(FT, "faults_text", spy)
    mgr = RealizationManager(pr, model, p)
    mgr.model = fm
    out = mgr.export(1, tmp_path / "r1.grdecl")
    assert [o.name for o in out] == ["r1.grdecl", "r1_faults.inc"]
    assert "VSHALE" in seen["keys"] and "PERMX" in seen["keys"] and "PORO" in seen["keys"]
    txt = (tmp_path / "r1.grdecl").read_text()
    assert "PORO" in txt and "PERMX" in txt and "PERMY" in txt and "PERMZ" in txt and "VSHALE" not in txt
    assert not {"PORO", "PERMX", "VSHALE"} & set(model.grid.props)                     # the grid is left as it was
    r = mgr.realize(1)
    np.testing.assert_array_equal(seen["vshale"], np.nan_to_num(r.props["VSHALE"], nan=0.0))
    np.testing.assert_array_equal(seen["permx"], np.nan_to_num(r.permx, nan=0.0))
    # a second export reproduces the file
    mgr.export(1, tmp_path / "r1b.grdecl")
    assert (tmp_path / "r1b.grdecl").read_text() == txt


def test_target_check_is_calibrated_and_has_power():
    from test_property_sim import pworld, setup as psetup2
    from cosma.property_sim import simulate_property
    from cosma.realizations import check_realization
    pr, model, fac, nx, ny = pworld(nx=30, ny=30)
    st = psetup2(("user", "user"))
    pl = RealizationPlan((PropertyJob(st, "PORO"),), None, 100, 1)
    fails = 0
    for s in range(20):
        r = simulate_property(pr, model, fac, st, [], s)
        c = check_realization(pr, model, pl, fac, {"PORO": r.values}, {"PORO": r.per_domain}, {}, pr.facies_setup.scheme)
        fails += bool(c["failed"])
        assert c["ntests"] == 6 and abs(c["alpha"] - 0.05 / 6) < 1e-12
    assert fails <= 2                                                                    # 5% nominal, Bonferroni makes it lower
    caught = 0
    for s in range(10):
        r = simulate_property(pr, model, fac, st, [], s)
        per = r.per_domain
        for d in per.values():
            d["score"] = (lambda f: (lambda v: f(v) + 0.3))(d["score"])                 # the target is 0.3 sigma away from the simulation
        c = check_realization(pr, model, pl, fac, {"PORO": r.values}, {"PORO": per}, {}, pr.facies_setup.scheme)
        caught += bool(c["failed"])
    assert caught >= 9


def test_facies_fractions_are_checked_against_the_target_proportions():
    from cosma.facies_sim import FaciesSimSetup
    from cosma.realizations import check_realization
    from cosma.sis import SisSettings, simulate_facies
    from test_sis import NZ, props, world as sworld, SET
    pr, model, nx, ny = sworld(nx=30, ny=30)
    pr.set_facies_sim(FaciesSimSetup("sis", SisSettings(12, 1.0, 2), None, 1))
    pl = RealizationPlan((PropertyJob(psetup_for(pr), "PORO"),), None, 100, 1)
    P = {"A": props(nx, ny, 0.5)}
    f = simulate_facies(pr, model, SET, P, None, 5).facies
    c = check_realization(pr, model, pl, f, {}, {"PORO": {}}, P, pr.facies_setup.scheme)
    assert not c["failed"] and c["ntests"] == 2
    wrong = {"A": props(nx, ny, 0.7)}
    c = check_realization(pr, model, pl, f, {}, {"PORO": {}}, wrong, pr.facies_setup.scheme)
    assert {x[3] for x in c["failed"]} == {"proportion"} and len(c["failed"]) == 2


def psetup_for(pr):
    keys = [("A", f.code) for f in pr.facies_setup.scheme]
    return PropertySetup("PHI", "PHI {facies}", tuple((k, "user") for k in keys), tuple((k, Distribution("normal", 0.2, 0.03)) for k in keys))


def test_plan_persists_in_the_project_file(tmp_path):
    from cosma.persist import PersistError, load_project, save_project
    pr, _ = world()
    p = plan(pr, on_fail="redraw", max_redraws=2)
    pr.set_realization_plan(p)
    back = load_project(save_project(pr, tmp_path / "p.cosma"))
    assert back.realization_plan == p
    pr.set_realization_plan(None)
    assert load_project(save_project(pr, tmp_path / "q.cosma")).realization_plan is None
    with pytest.raises(Exception):
        pr.set_realization_plan("x")


def test_permeability_is_computed_from_the_stored_float32_porosity():
    from cosma.perm_sim import simulate_permeability
    pr, model = world()
    p = plan(pr)
    m = RealizationManager(pr, model, p)
    f, v, k, _ = m.generate(0)
    again = simulate_permeability(pr, model, f, v["PORO"].astype(float), p.perm, None, seed_for(p, 0, 0, 100))
    np.testing.assert_array_equal(again.permx.astype(np.float32), k[0])


def test_progress_is_monotone_labelled_and_cancel_leaves_nothing_stored(tmp_path):
    from cosma.progress import Cancelled
    pr, model = world()
    p = plan(pr)
    m = RealizationManager(pr, model, p, cache_dir=tmp_path)
    got = []
    m.generate(0, 0, lambda label, f: got.append((label, f)))
    fr = [f for _, f in got]
    assert fr[0] == 0.0 and fr[-1] == 1.0 and all(a <= b + 1e-12 for a, b in zip(fr, fr[1:])) and all(0 <= f <= 1 for f in fr)
    labels = " | ".join(l for l, _ in got)
    assert "facies" in labels and "PHI" in labels and "permeability" in labels and "target" in labels
    assert all(l.startswith("realization 0:") for l, _ in got)
    assert not m.has_cache(0)
    # progress changes nothing in the result
    f1 = m.generate(0, 0)
    f2 = m.generate(0, 0, lambda *a: None)
    np.testing.assert_array_equal(f1[1]["PORO"], f2[1]["PORO"])
    # a stop part-way: nothing is cached, and the realization can be made afterwards
    n = []

    def stop_late(label, f):
        n.append(1)
        if f > 0.5:
            raise Cancelled()
    with pytest.raises(Cancelled):
        m.realize(1, stop_late)
    assert not m.has_cache(1) and not list(tmp_path.glob("*.npz"))
    m.realize(1)
    assert m.has_cache(1)


def test_engine_progress_counts_cells_and_can_cancel():
    from cosma.progress import Cancelled
    from test_property_sim import pworld, setup as psetup2
    from cosma.property_sim import simulate_property
    pr, model, fac, nx, ny = pworld(nx=12, ny=10)
    seen = []
    simulate_property(pr, model, fac, psetup2(("user", "user")), [], 1, lambda l, d, t: seen.append((l, d, t)))
    assert seen and all(t == seen[0][2] for _, _, t in seen) and seen[0][1] == 0
    d = [x[1] for x in seen]
    assert d == sorted(d) and len(set(d)) == len(d) and d[-1] < seen[0][2]       # starts at 0; each domain adds its cells
    assert "PHI" in seen[0][0] or "/" in seen[0][0]

    def stop(*a):
        raise Cancelled()
    with pytest.raises(Cancelled):
        simulate_property(pr, model, fac, psetup2(("user", "user")), [], 1, stop)


def corr_plan(pr, rho=0.8, **kw):
    import dataclasses
    keys = [(iv.name, f.code) for iv in pr.chart.intervals for f in pr.facies_setup.scheme]
    vs = dataclasses.replace(psetup(pr, "VSH", 0.3), correlate_with="PHI", correlation=tuple((k, rho) for k in keys))
    return plan(pr, jobs=(PropertyJob(psetup(pr), "PORO"), PropertyJob(vs, "VSHALE", export=False)), **kw)


def test_correlated_property_in_a_plan():
    import dataclasses
    pr, model = world()
    p = corr_plan(pr, -0.8)
    assert from_state(to_state(p)) == p
    old = to_state(plan(pr))                                           # a plan saved before correlations existed still loads
    for j in old["jobs"]:
        j["setup"].pop("correlate_with")
        j["setup"].pop("correlation")
    assert from_state(old) == plan(pr)
    vs = p.jobs[1]
    with pytest.raises(RealizationError, match="listed before"):
        plan(pr, jobs=(vs, p.jobs[0]))
    with pytest.raises(RealizationError, match="listed before"):
        plan(pr, jobs=(PropertyJob(psetup(pr), "PORO"), PropertyJob(dataclasses.replace(vs.setup, correlate_with="NOPE"), "VSHALE")))
    m = RealizationManager(pr, model, p)
    f, v, k, c = m.generate(0)
    ok = np.isfinite(v["PORO"])
    assert np.corrcoef(v["PORO"][ok], v["VSHALE"][ok])[0, 1] < -0.5
    f2, v2, _, _ = m.generate(0)
    np.testing.assert_array_equal(v["VSHALE"], v2["VSHALE"])
    # the porosity itself does not depend on whether something is correlated with it
    v0 = RealizationManager(pr, model, plan(pr)).generate(0)[1]
    np.testing.assert_array_equal(v0["PORO"], v["PORO"])


def test_correlated_permeability_scatter_in_a_plan():
    import dataclasses
    pr, model = world()
    pc = dataclasses.replace(permsetup(pr), scatter_mode="correlated", scatter_variogram="PHI {facies}", scatter_neighbours=8,
                             scatter_search_factor=1.0, scatter_levels=1)
    p = plan(pr, perm=pc)
    assert from_state(to_state(p)) == p
    old = to_state(plan(pr))                                           # saved before the choice existed: independent, as it ran
    for k in ("scatter_mode", "scatter_variogram", "scatter_neighbours", "scatter_search_factor", "scatter_levels"):
        old["perm"].pop(k)
    assert from_state(old) == plan(pr)
    m = RealizationManager(pr, model, p)
    got = []
    f, v, k, _ = m.generate(0, 0, lambda l, fr: got.append((l, fr)))
    assert any("permeability, " in l for l, _ in got) and [x for _, x in got] == sorted(x for _, x in got)
    np.testing.assert_array_equal(k[0], m.generate(0)[2][0])
    k_ind = RealizationManager(pr, model, plan(pr)).generate(0)[2]
    assert not np.array_equal(k[0], k_ind[0])
