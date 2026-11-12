"""Data checks: inventory, variogram support, cell-scale distributions, core regression levels."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cosma.datacheck import (CoreSpec, DataCheckError, DataCheckSetup, PropertySpec, Thresholds, average, describe, fit_ks,
                             from_state, regression, run_checks, to_state, usable_lag)
from cosma.facies import Condition, Facies, FaciesSetup, Rule
from cosma.grid.boundary import grid_spec_from_cells
from cosma.grid.builder import Layering, build_grid
from cosma.project import Project, WellRecord
from cosma.wells.model import LogSet, RawCore, Survey

SAND, SILT = Facies(1, "Sand"), Facies(2, "Silt")
RULES = (Rule(1, (Condition("GR", "<", 50.0),)), Rule(2, (Condition("GR", ">=", 50.0), Condition("GR", "<", 100.0))))
XS = (50.0, 250.0, 450.0)                 # three wells in cells (0,0), (2,0), (4,0)


def th(**kw):
    a = dict(min_hard_cells=1, min_wells=1, lag_v=0.7, nlags_v=4, min_pairs_v=1, min_usable_lag_v=0.7, lag_h=200.0, nlags_h=3,
             min_pairs_h=1, min_usable_lag_h=200.0, same_layer_tol=2.0, min_wells_h=2, min_core_pairs=1)
    a.update(kw)
    return Thresholds(**a)


def make_model():
    spec = grid_spec_from_cells(0, 0, 0, 6, 3, 100.0, 100.0)
    X, _ = np.meshgrid(spec.x_edges, spec.y_edges)
    hz = [np.full_like(X, z) for z in (1000.0, 1050.0, 1100.0)]
    grid, rep = build_grid(spec, hz, ["conformable", "conformable"], [Layering("fixed_count", n=5), Layering("fixed_count", n=5)])
    return SimpleNamespace(grid=grid, report=rep, interval_names=["A", "B"])


def facies_of(md):
    """Sand [1000,1020) and [1050,1080); silt [1020,1050) and [1080,1100)."""
    return np.where((md >= 1000) & (md < 1020) | (md >= 1050) & (md < 1080), 1, np.where((md >= 1000) & (md < 1100), 2, 0))


def make_project(xs=XS, step=0.5, extra=None, phi=None):
    md = np.arange(990.0, 1110.0, step)
    f = facies_of(md)
    gr = np.where(f == 1, 20.0, np.where(f == 2, 70.0, 130.0))
    curves = {"GR": gr}
    if phi is not None:
        curves["PHIE"] = phi(md)
    if extra:
        curves.update({k: v(md) for k, v in extra.items()})
    pr = Project("p")
    for n, x in enumerate(xs):
        rec = WellRecord(f"W{n}")
        rec.header = dict(x=x, y=50.0, kb=0.0, td=1200.0)
        rec.survey = Survey(np.array([0.0, 1200.0]), np.zeros(2), np.zeros(2), "grid")
        rec.logs["lg"] = LogSet(md, {k: np.array(v) for k, v in curves.items()})
        pr.wells[rec.name] = rec
    pr.set_facies_setup(FaciesSetup((SAND, SILT), RULES, min_coverage=0.5, conflict="drop"))
    return pr


def set_checks(pr, **kw):
    pr.set_data_checks(DataCheckSetup(kw.pop("thresholds", th()), kw.pop("properties", ()), kw.pop("core", None)))


# -- setup and helpers -------------------------------------------------------------------------------------------------

def test_threshold_and_spec_validation():
    th()
    for bad in (dict(min_hard_cells=0), dict(min_wells=1.5), dict(lag_v=0), dict(nlags_h=-1), dict(min_usable_lag_h=float("nan")),
                dict(same_layer_tol=0), dict(min_r2=1.5), dict(min_r2=-0.1), dict(min_pairs_v=True)):
        with pytest.raises(DataCheckError):
            th(**bad)
    th(min_r2=0.0)
    th(min_r2=0.8)
    for bad in (lambda: PropertySpec("", "C", "arithmetic"), lambda: PropertySpec("P", "C", "median"),
                lambda: CoreSpec("", "K", 0.3), lambda: CoreSpec("P", "K", 0.0),
                lambda: DataCheckSetup(th(), (PropertySpec("P", "A", "arithmetic"), PropertySpec("P", "B", "harmonic")))):
        with pytest.raises(DataCheckError):
            bad()


def test_state_roundtrip():
    s = DataCheckSetup(th(min_r2=0.4), (PropertySpec("PHI", "PHIE", "arithmetic"), PropertySpec("K", "KL", "geometric")),
                       CoreSpec("CPOR", "CKH", 0.25, "c1"))
    assert from_state(to_state(s)) == s
    s2 = DataCheckSetup(th())
    assert from_state(to_state(s2)) == s2 and from_state(None) is None and to_state(None) is None
    with pytest.raises(DataCheckError, match="bad data-check setup"):
        from_state({"thresholds": {"min_wells": 1}})


def test_describe_average_usable_lag_and_fits():
    d = describe([1.0, 2.0, 3.0, 4.0, 10.0])
    assert d["n"] == 5 and d["mean"] == 4.0 and d["p50"] == 3.0 and d["min"] == 1.0 and d["max"] == 10.0
    assert d["std"] == pytest.approx(np.std([1, 2, 3, 4, 10], ddof=1)) and describe([5.0])["std"] == 0.0 and describe([]) == {"n": 0}
    v, w = [2.0, 8.0], [1.0, 3.0]
    assert average(v, w, "arithmetic") == pytest.approx(6.5)
    assert average(v, w, "geometric") == pytest.approx(math.exp((math.log(2) + 3 * math.log(8)) / 4))
    assert average(v, w, "harmonic") == pytest.approx(4.0 / (1 / 2 + 3 / 8))
    with pytest.raises(DataCheckError, match="needs values > 0"):
        average([1.0, -1.0], [1, 1], "geometric")
    assert average([-1.0, 3.0], [1, 1], "arithmetic") == 1.0
    assert usable_lag([1, 2, 3, 4], [10, 10, 3, 10], 5) == 2.0 and usable_lag([1, 2], [0, 9], 1) == 0.0
    assert usable_lag([1, 2, 3], [9, 9, 9], 9) == 3.0
    rng = np.random.default_rng(0)
    f = fit_ks(rng.normal(5.0, 1.0, 400))
    assert abs(f["normal"]["mean"] - 5.0) < 0.2 and f["normal"]["ks"] < 0.08 and "lognormal" in f
    g = fit_ks(np.exp(rng.normal(1.0, 0.8, 400)))
    assert g["lognormal"]["ks"] < g["normal"]["ks"] and abs(g["lognormal"]["mu"] - 1.0) < 0.15
    assert "lognormal" not in fit_ks(np.array([-1.0, 0.0, 1.0, 2.0])) and fit_ks(np.array([1.0, 1.0, 1.0])) == {} and fit_ks([1.0]) == {}


def test_regression_exact_line_and_dropped_pairs():
    phi = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    r = regression(phi, 10 ** (1.0 + 8.0 * phi))
    assert r["n"] == 5 and r["slope"] == pytest.approx(8.0) and r["intercept"] == pytest.approx(1.0)
    assert r["r2"] == pytest.approx(1.0) and r["slope_ci"] == pytest.approx(0.0, abs=1e-6) and r["resid_std"] == pytest.approx(0.0, abs=1e-9)
    assert (r["phi_min"], r["phi_max"]) == (0.05, 0.25)
    r = regression(np.append(phi, [0.3, 0.35, np.nan]), np.append(10 ** (1.0 + 8.0 * phi), [0.0, -5.0, 1.0]))
    assert r["n"] == 5 and r["dropped"] == 3
    # known scatter: residual std and slope interval from the textbook formulas
    rng = np.random.default_rng(2)
    x = np.linspace(0.05, 0.3, 30)
    y = 0.5 + 6 * x + rng.normal(0, 0.2, 30)
    r = regression(x, 10 ** y)
    sxx = ((x - x.mean()) ** 2).sum()
    s = math.sqrt(((y - (r["intercept"] + r["slope"] * x)) ** 2).sum() / 28)
    assert r["resid_std"] == pytest.approx(s) and r["slope_ci"] == pytest.approx(2.048407 * s / math.sqrt(sxx), rel=1e-4)
    assert regression(phi[:2], phi[:2] + 1) == dict(n=2, dropped=0) and "slope" not in regression(np.full(5, 0.1), np.arange(1, 6.0))


# -- inventory ---------------------------------------------------------------------------------------------------------

def test_inventory_counts_cells_wells_and_path_per_interval_and_facies():
    pr = make_project()
    set_checks(pr)
    rep = run_checks(pr, make_model())
    want = {("A", 1): (6, 20.0), ("A", 2): (9, 30.0), ("B", 1): (9, 30.0), ("B", 2): (6, 20.0)}     # cells over 3 wells, path per well
    assert len(rep.inventory) == 4
    for (iv, f), (cells, path) in want.items():
        r = rep.row("inventory", iv, f)
        assert r["cells"] == cells and r["wells"] == 3 and r["path"] == pytest.approx(3 * path) and r["well_names"] == ["W0", "W1", "W2"]
        assert r["ok"] and r["why"] == ""
    set_checks(pr, thresholds=th(min_hard_cells=7, min_wells=4))
    rep = run_checks(pr, make_model())
    assert [r["ok"] for r in rep.inventory] == [False, False, False, False]
    assert "6 hard cells, 7 needed" in rep.row("inventory", "A", 1)["why"] and "3 well(s), 4 needed" in rep.row("inventory", "A", 1)["why"]
    set_checks(pr, thresholds=th(min_hard_cells=7))
    rep = run_checks(pr, make_model())
    assert {(r["interval"], r["facies"]): r["ok"] for r in rep.inventory} == {("A", 1): False, ("A", 2): True, ("B", 1): True, ("B", 2): False}


def test_a_facies_with_no_hard_data_in_an_interval_is_reported_not_skipped():
    pr = make_project()
    pr.set_facies_setup(FaciesSetup((SAND, SILT, Facies(3, "Gravel")), RULES, min_coverage=0.5, conflict="drop"))
    set_checks(pr)
    rep = run_checks(pr, make_model())
    r = rep.row("inventory", "A", 3)
    assert r["cells"] == 0 and r["wells"] == 0 and not r["ok"] and "0 hard cells" in r["why"]
    assert rep.row("vertical", "A", 3)["why"] == "no log samples of this facies here" and not rep.row("horizontal", "A", 3)["ok"]
    assert "Gravel: 0 cells" in rep.summary()


def test_missing_setups_are_refused_with_the_reason():
    pr = make_project()
    with pytest.raises(DataCheckError, match="thresholds"):
        run_checks(pr, make_model())
    set_checks(pr)
    pr.set_facies_setup(FaciesSetup((SAND, SILT), RULES))
    with pytest.raises(DataCheckError, match="minimum classified share"):
        run_checks(pr, make_model())
    pr.set_facies_setup(None)
    with pytest.raises(DataCheckError, match="facies first"):
        run_checks(pr, make_model())
    pr2 = Project("e")
    pr2.set_facies_setup(FaciesSetup((SAND,), RULES[:1], min_coverage=0.5, conflict="drop"))
    pr2.set_data_checks(DataCheckSetup(th()))
    with pytest.raises(DataCheckError, match="no well could be classified"):
        run_checks(pr2, make_model())
    from cosma.project import ProjectError
    with pytest.raises(ProjectError):
        pr.set_data_checks("nope")


# -- variogram support -------------------------------------------------------------------------------------------------

def brute_pairs(z, lag, nlags):
    d = np.abs(z[:, None] - z[None, :])[np.triu_indices(len(z), 1)]
    edges = lag * (np.arange(nlags + 1) + 0.5)
    return np.histogram(d, edges)[0]


def test_vertical_pair_counts_match_a_brute_force_histogram_and_usable_lag():
    pr = make_project()
    set_checks(pr, thresholds=th(min_pairs_v=100, min_usable_lag_v=2.8))
    rep = run_checks(pr, make_model())
    r = rep.row("vertical", "A", 1)
    z = np.arange(1000.0, 1020.0, 0.5)                                        # one well's sand samples in interval A
    want = 3 * brute_pairs(z, 0.7, 4)
    assert r["samples"] == 120 and r["wells"] == 3 and r["npairs"] == list(want)
    assert list(want) == [231, 111, 108, 207]
    assert r["usable"] == pytest.approx(2.8) and r["ok"]
    set_checks(pr, thresholds=th(min_pairs_v=120, min_usable_lag_v=2.8))
    rep = run_checks(pr, make_model())
    r = rep.row("vertical", "A", 1)
    assert r["usable"] == pytest.approx(0.7) and not r["ok"] and "supported to 0.7 m, 2.8 m needed" in r["why"]
    set_checks(pr, thresholds=th(min_pairs_v=1000))
    assert run_checks(pr, make_model()).row("vertical", "A", 1)["usable"] == 0.0


def test_pairs_never_straddle_two_wells_in_the_vertical_variogram():
    pr = make_project(xs=(50.0, 250.0))
    set_checks(pr, thresholds=th(min_pairs_v=1))
    r1 = run_checks(pr, make_model()).row("vertical", "A", 1)
    z = np.arange(1000.0, 1020.0, 0.5)
    assert r1["npairs"] == list(2 * brute_pairs(z, 0.7, 4))
    two = np.concatenate([z, z])
    assert sum(r1["npairs"]) < sum(brute_pairs(two, 0.7, 4))                  # the pooled brute force would count cross-well pairs


def test_horizontal_support_counts_cell_pairs_between_wells_in_the_same_layer():
    pr = make_project()
    set_checks(pr, thresholds=th(min_pairs_h=3, min_usable_lag_h=200.0))
    r = run_checks(pr, make_model()).row("horizontal", "A", 1)
    # sand in A = layers 0 and 1 in each of 3 wells; separations 200 (x2 well pairs) and 400 (x1), per layer
    assert r["points"] == 6 and r["wells"] == 3 and r["npairs"] == [4, 2, 0]
    assert r["usable"] == 200.0 and r["ok"] and r["why"] == ""
    set_checks(pr, thresholds=th(min_pairs_h=2, min_usable_lag_h=400.0))
    r = run_checks(pr, make_model()).row("horizontal", "A", 1)
    assert r["usable"] == 400.0 and r["ok"]
    set_checks(pr, thresholds=th(min_pairs_h=3, min_usable_lag_h=400.0, min_wells_h=4))
    r = run_checks(pr, make_model()).row("horizontal", "A", 1)
    assert not r["ok"] and "3 wells, 4 needed" in r["why"] and "supported to 200 m, 400 m needed" in r["why"]


def test_one_well_cannot_give_a_horizontal_variogram_and_the_report_says_what_to_do():
    pr = make_project(xs=(50.0,))
    set_checks(pr)
    rep = run_checks(pr, make_model())
    r = rep.row("horizontal", "A", 1)
    assert not r["ok"] and r["wells"] == 1 and "fewer than two wells" in r["why"] and "derive them from a map" in r["why"]
    assert rep.row("vertical", "A", 1)["ok"]                                  # vertical is still fine


def test_horizontal_pairs_need_the_same_layer():
    pr = make_project()
    set_checks(pr, thresholds=th(same_layer_tol=0.5))
    # layers are 10 m thick, wells share layers, so same-layer pairs survive a tight tolerance
    assert run_checks(pr, make_model()).row("horizontal", "A", 1)["npairs"] == [4, 2, 0]
    # shift one well's facies up by one layer: its sand cells no longer share layers with the others'
    pr2 = make_project()
    md = pr2.wells["W2"].logs["lg"].md
    shifted = np.where(facies_of(md - 10.0) == 1, 20.0, np.where(facies_of(md - 10.0) == 2, 70.0, 130.0))
    pr2.wells["W2"].logs["lg"] = LogSet(md, {"GR": shifted})
    set_checks(pr2, thresholds=th(same_layer_tol=0.5, min_wells_h=2))
    r = run_checks(pr2, make_model()).row("horizontal", "A", 1)
    # W2's sand is now layers 1-2: W0-W1 share layers 0 and 1 (200 m, 2 pairs), W1-W2 only layer 1 (200 m, 1 pair),
    # W0-W2 only layer 1 (400 m, 1 pair)
    assert r["npairs"] == [3, 1, 0]


# -- cell-scale distributions ------------------------------------------------------------------------------------------

def layer_of(md):
    return np.floor((md - 1000.0) / 10.0)


def sign_of(md, step=0.5):
    return np.where(np.round((md - 990.0) / step).astype(int) % 2 == 0, 1.0, -1.0)


def phi_curve(md):
    return 0.10 + 0.01 * layer_of(md) + 0.02 * sign_of(md)


def test_cell_scale_values_are_the_averages_of_their_samples_and_the_support_effect_is_shown():
    pr = make_project(extra={"PHIE": phi_curve})
    set_checks(pr, properties=(PropertySpec("PHI", "PHIE", "arithmetic"),))
    rep = run_checks(pr, make_model())
    r = next(x for x in rep.distributions if x["interval"] == "A" and x["facies"] == 1)
    c = np.array([0.10, 0.11] * 3)
    assert r["wells"] == 3 and r["cell"]["n"] == 6 and r["cell"]["mean"] == pytest.approx(0.105)
    assert r["cell"]["std"] == pytest.approx(c.std(ddof=1)) and sorted(r["values"]) == pytest.approx(sorted(c))
    lg = np.array(([0.08] * 10 + [0.12] * 10 + [0.09] * 10 + [0.13] * 10) * 3)
    assert r["log"]["n"] == 120 and r["log"]["std"] == pytest.approx(lg.std(ddof=1))
    assert r["variance_ratio"] == pytest.approx(c.var(ddof=1) / lg.var(ddof=1)) and r["variance_ratio"] < 0.1
    assert set(r["fits"]) == {"normal", "lognormal"} and 0 <= r["fits"]["normal"]["ks"] <= 1
    r2 = next(x for x in rep.distributions if x["interval"] == "B" and x["facies"] == 2)       # layers 8, 9
    assert r2["cell"]["mean"] == pytest.approx(0.185) and r2["cell"]["n"] == 6
    assert len(rep.distributions) == 4
    assert "variance ratio cell/log" in rep.summary() and "PHI: A / Sand: 6 cells (3 well(s))" in rep.summary()


def test_averaging_rule_changes_the_cell_value_as_the_closed_forms_say():
    s = 0.5
    kcurve = lambda md: 100.0 * np.exp(s * sign_of(md))                      # noqa: E731
    pr = make_project(extra={"KL": kcurve})
    set_checks(pr, properties=(PropertySpec("Ka", "KL", "arithmetic"), PropertySpec("Kg", "KL", "geometric"),
                               PropertySpec("Kh", "KL", "harmonic")))
    rep = run_checks(pr, make_model())
    got = {p: next(x for x in rep.distributions if x["property"] == p and x["interval"] == "A" and x["facies"] == 2)["cell"]["mean"]
           for p in ("Ka", "Kg", "Kh")}
    assert got["Kg"] == pytest.approx(100.0) and got["Ka"] == pytest.approx(100.0 * math.cosh(s))
    assert got["Kh"] == pytest.approx(100.0 / math.cosh(s)) and got["Kh"] < got["Kg"] < got["Ka"]


def test_cells_with_too_little_valid_log_are_left_out_and_wells_without_the_curve_are_noted():
    pr = make_project(extra={"PHIE": lambda md: np.where((md >= 1000) & (md < 1005), np.nan, phi_curve(md))})
    set_checks(pr, properties=(PropertySpec("PHI", "PHIE", "arithmetic"),))
    r = next(x for x in run_checks(pr, make_model()).distributions if x["interval"] == "A" and x["facies"] == 1)
    assert r["cell"]["n"] == 6 and r["cell"]["mean"] == pytest.approx(0.105)             # half-covered cell kept at coverage 0.5
    pr.set_facies_setup(FaciesSetup((SAND, SILT), RULES, min_coverage=0.6, conflict="drop"))
    r = next(x for x in run_checks(pr, make_model()).distributions if x["interval"] == "A" and x["facies"] == 1)
    assert r["cell"]["n"] == 3 and r["cell"]["mean"] == pytest.approx(0.11)
    del pr.wells["W2"].logs["lg"].curves["PHIE"]
    rep = run_checks(pr, make_model())
    r = next(x for x in rep.distributions if x["interval"] == "A" and x["facies"] == 2)
    assert r["wells"] == 2 and r["cell"]["n"] == 6 and any("W2 has no curve 'PHIE'" in n for n in rep.notes)
    # a property that no well has: empty rows, not a crash
    set_checks(pr, properties=(PropertySpec("X", "NOPE", "arithmetic"),))
    rep = run_checks(pr, make_model())
    assert all(x["cell"]["n"] == 0 for x in rep.distributions) and "no cell-scale values" in rep.summary()


def test_geometric_and_harmonic_averages_skip_nonpositive_samples():
    pr = make_project(extra={"KL": lambda md: np.where(sign_of(md) > 0, 100.0, -1.0)})
    set_checks(pr, properties=(PropertySpec("Kg", "KL", "geometric"),))
    r = next(x for x in run_checks(pr, make_model()).distributions if x["interval"] == "A" and x["facies"] == 2)
    assert r["cell"]["mean"] == pytest.approx(100.0) and r["log"]["min"] == 100.0 and r["cell"]["n"] == 9


# -- core regression levels --------------------------------------------------------------------------------------------

SAND_LINE = lambda phi: 10 ** (1.0 + 8.0 * np.asarray(phi))       # noqa: E731
SILT_LINE = lambda phi: 10 ** (-1.0 + 6.0 * np.asarray(phi))      # noqa: E731


def add_core(pr, sand=SAND_LINE, extra_depths=()):
    groups = [([1002.25, 1006.25, 1010.25, 1014.25], [0.12, 0.16, 0.20, 0.24], sand),      # A sand
              ([1030.25, 1040.25], [0.08, 0.12], SILT_LINE),                                # A silt
              ([1055.25, 1060.25, 1065.25, 1070.25], [0.10, 0.14, 0.18, 0.22], sand),       # B sand
              ([1090.25], [0.09], SILT_LINE)]                                               # B silt
    md = np.array([d for g in groups for d in g[0]] + list(extra_depths))
    phi = np.array([p for g in groups for p in g[1]] + [0.15] * len(extra_depths))
    perm = np.array([k for g in groups for k in g[2](g[1])] + [50.0] * len(extra_depths))
    for rec in pr.wells.values():
        rec.core["c1"] = RawCore(md, "md", {"CPOR": phi, "CKH": perm})


def core_setup(pr, **kw):
    set_checks(pr, thresholds=th(min_core_pairs=5, **kw), core=CoreSpec("CPOR", "CKH", 0.3))


def test_regression_levels_counts_slopes_and_fallbacks():
    pr = make_project()
    add_core(pr, extra_depths=(995.0, 1105.0))                    # outside the grid: not counted
    core_setup(pr)
    rep = run_checks(pr, make_model())
    lv = lambda kind, iv, f: next(r for r in rep.regression if r["level"] == kind and r["interval"] == iv and r["facies"] == f)   # noqa: E731
    assert [lv("facies x interval", iv, f)["n"] for iv, f in (("A", 1), ("A", 2), ("B", 1), ("B", 2))] == [12, 6, 12, 3]
    assert lv("facies x interval", "A", 1)["slope"] == pytest.approx(8.0) and lv("facies x interval", "A", 2)["slope"] == pytest.approx(6.0)
    b2 = lv("facies x interval", "B", 2)
    assert not b2["ok"] and b2["why"] == "3 pairs, 5 needed"
    assert lv("facies, all intervals", None, 1)["n"] == 24 and lv("facies, all intervals", None, 2)["n"] == 9
    assert lv("facies, all intervals", None, 2)["slope"] == pytest.approx(6.0) and lv("facies, all intervals", None, 2)["ok"]
    assert lv("interval, all facies", "A", None)["n"] == 18 and lv("interval, all facies", "B", None)["n"] == 15
    assert rep.fallback("A", 1)["level"] == "facies x interval" and rep.fallback("A", 1)["interval"] == "A"
    f = rep.fallback("B", 2)
    assert f["level"] == "facies, all intervals" and f["facies"] == 2                  # B silt falls back to silt over all intervals
    t = rep.summary()
    assert "B / Silt: facies, all intervals" in t and "A / Sand: facies x interval" in t and "3 pairs, 5 needed" in t


def test_when_no_level_passes_the_report_says_to_enter_the_relation_by_hand():
    pr = make_project()
    add_core(pr)
    set_checks(pr, thresholds=th(min_core_pairs=100), core=CoreSpec("CPOR", "CKH", 0.3))
    rep = run_checks(pr, make_model())
    assert rep.fallback("A", 1) is None and "none passes: enter the relation by hand" in rep.summary()


def test_the_r2_threshold_applies_to_every_level():
    pr = make_project()
    rng = np.random.default_rng(5)
    noisy = lambda phi: 10 ** (1.0 + 8.0 * np.asarray(phi) + rng.normal(0, 1.5, len(phi)))      # noqa: E731
    add_core(pr, sand=noisy)
    core_setup(pr, min_r2=0.9)
    rep = run_checks(pr, make_model())
    a_sand = next(r for r in rep.regression if r["level"] == "facies x interval" and r["interval"] == "A" and r["facies"] == 1)
    assert a_sand["r2"] < 0.9 and not a_sand["ok"] and "below 0.9" in a_sand["why"]
    a_silt = next(r for r in rep.regression if r["level"] == "facies x interval" and r["interval"] == "A" and r["facies"] == 2)
    assert a_silt["ok"] and rep.fallback("A", 1) is None


def test_core_facies_comes_from_the_nearest_classified_log_sample_within_the_tolerance():
    pr = make_project()
    add_core(pr)
    set_checks(pr, thresholds=th(min_core_pairs=1), core=CoreSpec("CPOR", "CKH", 0.01))     # core depths are 0.25 m from any sample
    rep = run_checks(pr, make_model())
    fx = [r for r in rep.regression if r["level"] == "facies x interval"]
    assert all(r["n"] == 0 for r in fx)                                                        # no facies could be assigned
    assert next(r for r in rep.regression if r["level"] == "interval, all facies" and r["interval"] == "A")["n"] == 18


def test_core_problems_are_noted_not_hidden():
    pr = make_project()
    add_core(pr)
    pr.wells["W0"].core["c2"] = pr.wells["W0"].core["c1"]                  # two sources, none named
    del pr.wells["W1"].core["c1"].columns["CKH"]                           # a missing column
    core_setup(pr)
    rep = run_checks(pr, make_model())
    assert any("W0: several sources" in n for n in rep.notes) and any("W1: lacks column" in n for n in rep.notes)
    assert next(r for r in rep.regression if r["level"] == "facies x interval" and r["interval"] == "A" and r["facies"] == 1)["n"] == 4
    set_checks(pr, thresholds=th(min_core_pairs=5), core=CoreSpec("CPOR", "CKH", 0.3, source="nope"))
    rep = run_checks(pr, make_model())
    assert rep.regression == [] and any("no source 'nope'" in n for n in rep.notes) and any("no core" in n for n in rep.notes)


def test_nonpositive_values_block_lognormal_fits_and_geometric_harmonic_averages():
    assert "lognormal" not in fit_ks([0.0, 1.0, 2.0, 3.0]) and "lognormal" in fit_ks([0.5, 1.0, 2.0, 3.0])
    for how in ("geometric", "harmonic"):
        with pytest.raises(DataCheckError):
            average([0.0, 1.0], [1.0, 1.0], how)


def test_last_resort_level_is_the_interval_with_all_its_facies_not_another_interval():
    from cosma.datacheck import DataReport
    rows = [dict(level="facies x interval", interval="A", facies=1, ok=False),
            dict(level="facies, all intervals", interval=None, facies=1, ok=False),
            dict(level="interval, all facies", interval="B", facies=None, ok=True),
            dict(level="interval, all facies", interval="A", facies=None, ok=True, tag="A")]
    rep = DataReport(["A", "B"], (SAND, SILT), th(), regression=rows)
    assert rep.fallback("A", 1)["tag"] == "A" and rep.fallback("B", 1)["interval"] == "B"
