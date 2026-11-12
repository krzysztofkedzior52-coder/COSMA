"""Vertical proportion curves: from wells (relative / depth bins, smoothing, support), typed curves, checks."""

import numpy as np
import pytest

from cosma.facies import Facies, FaciesSetup
from cosma.proportions import (Curve, ProportionError, ProportionSetup, Settings, build_curves, from_state, layer_proportions,
                               to_state)

from test_datacheck import RULES, SAND, SILT, make_model, make_project

SC = (1, 2)


def setup(settings=None, sources=("wells", "wells"), curves=()):
    st = settings or Settings("relative", nbins=5)
    return ProportionSetup(st, tuple(zip(("A", "B"), sources)), tuple(curves))


def run(pr=None, **kw):
    pr = pr or make_project()
    return build_curves(pr, make_model(), setup(**kw))


def vals(res, iv):
    return np.array(next(r for r in res.intervals if r.interval == iv).curve.values)


# -- settings and curves -----------------------------------------------------------------------------------------------

def test_settings_validation():
    Settings("relative", nbins=5)
    Settings("depth", bin_size=2.5, window=3, min_wells=2, min_cells=4)
    for bad in (dict(mode="x"), dict(mode="relative"), dict(mode="relative", nbins=0), dict(mode="relative", nbins=2.5),
                dict(mode="relative", nbins=5, bin_size=1.0), dict(mode="depth"), dict(mode="depth", bin_size=0.0),
                dict(mode="depth", bin_size=1.0, nbins=3), dict(mode="depth", bin_size=float("nan")),
                dict(mode="relative", nbins=5, window=2), dict(mode="relative", nbins=5, window=0),
                dict(mode="relative", nbins=5, min_wells=0), dict(mode="relative", nbins=5, min_cells=True)):
        with pytest.raises(ProportionError):
            Settings(**bad)


def test_curve_validation_nothing_is_rescaled():
    Curve("A", "relative", 0.5, SC, ((1.0, 0.0), (0.25, 0.75)), "user")
    for bad in (dict(values=((0.5, 0.4),)), dict(values=((1.2, -0.2),)), dict(values=((0.5, 0.5, 0.0),)),
                dict(values=()), dict(codes=(1, 1), values=((0.5, 0.5),)), dict(codes=()), dict(bin_size=0.3),
                dict(values=((float("nan"), 1.0),)), dict(mode="elsewhere"), dict(source="guess")):
        kw = dict(interval="A", mode="relative", bin_size=1.0, codes=SC, values=((0.5, 0.5),), source="user")
        kw.update(bad)
        with pytest.raises(ProportionError):
            Curve(**kw)
    with pytest.raises(ProportionError, match="sum to 0.9"):
        Curve("A", "relative", 1.0, SC, ((0.5, 0.4),), "user")


def test_setup_validation_and_state_roundtrip():
    c = Curve("A", "relative", 0.5, SC, ((1.0, 0.0), (0.25, 0.75)), "user")
    s = ProportionSetup(Settings("relative", nbins=5, window=3, min_wells=2, min_cells=3), (("A", "user"), ("B", "wells")), (c,))
    assert from_state(to_state(s)) == s and from_state(None) is None and to_state(None) is None
    for bad in (lambda: ProportionSetup(Settings("relative", nbins=5), (("A", "wells"), ("A", "user"))),
                lambda: ProportionSetup(Settings("relative", nbins=5), (("A", "guess"),)),
                lambda: ProportionSetup(Settings("relative", nbins=5), (("A", "user"),)),             # user source, no curve
                lambda: ProportionSetup(Settings("relative", nbins=5), (("A", "wells"),), (c,)),      # curve, wells source
                lambda: ProportionSetup(Settings("depth", bin_size=5.0), (("A", "user"),), (c,)),     # mode mismatch
                lambda: ProportionSetup(Settings("relative", nbins=5), (("A", "user"),), (c, c))):
        with pytest.raises(ProportionError):
            bad()
    for mut in (lambda d: d.pop("settings"), lambda d: d["settings"].update(mode="x"), lambda d: d["choices"].append(["A", "wells"]),
                lambda d: d["curves"][0].update(values=[[0.1, 0.1]]), lambda d: d["choices"].__setitem__(0, ["A"])):
        d = to_state(s)
        mut(d)
        with pytest.raises(ProportionError):
            from_state(d)


# -- curves from wells -------------------------------------------------------------------------------------------------

def test_one_bin_per_layer_reproduces_the_well_facies_sequence_exactly():
    res = run()
    assert res.ok
    # A: sand in layers 0-1, silt in 2-4 ; B: sand in layers 0-2, silt in 3-4 (three identical wells)
    assert vals(res, "A").tolist() == [[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]
    assert vals(res, "B").tolist() == [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]
    a = next(r for r in res.intervals if r.interval == "A")
    assert a.cells == 15 and a.wells == 3 and a.curve.n_cells == (3, 3, 3, 3, 3)
    assert a.global_curve == pytest.approx((0.4, 0.6)) and a.global_wells == pytest.approx((0.4, 0.6))
    assert a.conflicts == 0 and a.curve.source == "wells" and a.curve.codes == SC


def test_fewer_bins_average_the_layers_in_them():
    res = run(settings=Settings("relative", nbins=1))
    assert vals(res, "A").tolist() == [[0.4, 0.6]] and vals(res, "B").tolist() == [[0.6, 0.4]]
    res = run(settings=Settings("relative", nbins=2))                       # layer centres 0.1 .. 0.9 -> bins 0 0 1 1 1
    assert vals(res, "A").tolist() == [[1, 0], [0, 1]]
    assert vals(res, "B") == pytest.approx(np.array([[1, 0], [1 / 3, 2 / 3]]))           # layers 0,1 | 2,3,4 : 1 sand of 3


def test_moving_average_smoothing_with_truncated_ends():
    res = run(settings=Settings("relative", nbins=5, window=3))
    # window sums of the layer sequence S S s s s over 3 bins, truncated at the ends
    assert vals(res, "A")[:, 0] == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0, 0.0])
    assert vals(res, "A").sum(axis=1) == pytest.approx(1.0)
    a = next(r for r in res.intervals if r.interval == "A")
    assert a.curve.n_cells == (6, 9, 9, 9, 6)                                  # 3 wells x layers in the window


def test_depth_bins_use_true_thickness_from_the_top():
    res = run(settings=Settings("depth", bin_size=20.0))
    # layer centres at 5 15 25 35 45 m -> bins 0 0 1 1 2 (A: S S s s s)
    assert vals(res, "A").tolist() == [[1, 0], [0, 1], [0, 1]]
    assert vals(res, "B").tolist() == [[1, 0], [0.5, 0.5], [0, 1]]
    assert next(r for r in res.intervals if r.interval == "A").curve.mode == "depth"


def test_a_thin_bin_or_too_few_wells_stops_the_build_with_the_reason():
    res = run(settings=Settings("relative", nbins=5, min_cells=4))
    assert not res.ok
    r = res.intervals[0]
    assert r.curve is None and "bin(s) 1, 2, 3, 4, 5 have fewer than 4 hard cell(s)" in r.problems[0]
    with pytest.raises(ProportionError, match="have problems"):
        layer_proportions(make_model(), res)
    t = res.summary()
    assert "PROBLEM" in t and "use fewer or wider bins" in t
    res = run(settings=Settings("relative", nbins=5, window=5, min_cells=4))             # a wider window supplies the data
    assert res.ok
    res = run(settings=Settings("relative", nbins=5, min_wells=4))
    assert "3 well(s) with hard cells, 4 needed" in res.intervals[0].problems[0] and res.intervals[0].curve is None


def test_an_interval_without_hard_cells_is_a_problem_not_a_default():
    pr = make_project(xs=(50.0,))
    pr.wells["W0"].header["td"] = 1050.0                                    # the well stops at the base of A
    from cosma.wells.model import LogSet
    rec = pr.wells["W0"]
    md = rec.logs["lg"].md
    keep = md < 1050.0
    rec.logs["lg"] = LogSet(md[keep], {k: v[keep] for k, v in rec.logs["lg"].curves.items()})
    res = run(pr)
    b = res.intervals[1]
    assert "no hard cells in this interval" in b.problems[0] and b.curve is None and res.intervals[0].curve is not None
    assert not res.ok


# -- typed curves, choices and the checks against the wells -------------------------------------------------------------

def user_curve(iv, rows, bin_size=0.5):
    return Curve(iv, "relative", bin_size, SC, tuple(rows), "user")


def test_typed_curve_used_as_given_and_global_proportions_reported():
    c = user_curve("A", [(0.8, 0.2), (0.2, 0.8)])
    res = run(sources=("user", "wells"), curves=(c,))
    a, b = res.intervals
    assert a.source == "user" and a.curve is c
    assert a.global_curve == pytest.approx(((2 * 0.8 + 3 * 0.2) / 5, (2 * 0.2 + 3 * 0.8) / 5))     # layers 0-1 in bin 1, 2-4 in bin 2
    assert a.global_wells == pytest.approx((0.4, 0.6))
    assert b.source == "wells"
    t = res.summary()
    assert "A: source user" in t and "largest difference" in t


def test_conflict_when_a_curve_gives_zero_to_a_facies_a_well_saw():
    c = user_curve("A", [(0.0, 1.0), (0.0, 1.0)])                           # no sand anywhere, but wells see sand in A
    res = run(sources=("user", "wells"), curves=(c,))
    a = res.intervals[0]
    assert a.conflicts == 6 and not res.ok and "CONFLICT: 6 hard cell(s)" in res.summary()
    with pytest.raises(ProportionError):
        layer_proportions(make_model(), res)


def test_typed_curve_must_use_the_schemes_codes():
    c = Curve("A", "relative", 1.0, (1, 3), ((0.5, 0.5),), "user")
    res = run(sources=("user", "wells"), curves=(c,))
    assert res.intervals[0].curve is None and "facies codes [1, 3] but the scheme has [1, 2]" in res.intervals[0].problems[0]


def test_every_interval_needs_an_explicit_source_and_must_exist():
    pr = make_project()
    with pytest.raises(ProportionError, match="choose a proportion source for: B"):
        build_curves(pr, make_model(), ProportionSetup(Settings("relative", nbins=5), (("A", "wells"),)))
    with pytest.raises(ProportionError, match="do not exist: Z"):
        build_curves(pr, make_model(), ProportionSetup(Settings("relative", nbins=5), (("A", "wells"), ("B", "wells"), ("Z", "wells"))))
    with pytest.raises(ProportionError, match="set up the proportions"):
        build_curves(pr, make_model())
    pr.facies_setup = None
    with pytest.raises(ProportionError, match="set up the facies"):
        build_curves(pr, make_model(), setup())
    pr.facies_setup = FaciesSetup((SAND, SILT), RULES)
    with pytest.raises(ProportionError, match="minimum classified share"):
        build_curves(pr, make_model(), setup())


def test_depth_curve_shorter_than_the_interval_is_refused():
    c = Curve("A", "depth", 20.0, SC, ((1.0, 0.0), (0.0, 1.0)), "user")      # reaches 40 m; the interval is 50 m
    st = ProportionSetup(Settings("depth", bin_size=20.0), (("A", "user"), ("B", "wells")), (c,))
    res = build_curves(make_project(), make_model(), st)
    assert res.intervals[0].curve is None and "reaches 40 m but the interval is" in res.intervals[0].problems[0]


# -- on the grid --------------------------------------------------------------------------------------------------------

def test_layer_proportions_give_every_cell_its_bins_proportions():
    model = make_model()
    res = build_curves(make_project(), model, setup(settings=Settings("relative", nbins=2)))
    lp = layer_proportions(model, res)
    a = lp["A"]
    assert a.shape == (6, 3, 5, 2) and np.allclose(a.sum(axis=-1), 1.0)
    assert (a[..., :2, 0] == 1).all() and (a[..., 2:, 1] == 1).all()
    b = lp["B"]
    assert b[3, 1, 0, 0] == 1 and b[3, 1, 4, 1] == pytest.approx(2 / 3)


def test_global_proportions_are_thickness_weighted():
    c = Curve("A", "relative", 0.5, SC, ((1.0, 0.0), (0.0, 1.0)), "user")
    g = c.global_proportions([0.25, 0.75], [30.0, 10.0])
    assert g == pytest.approx([0.75, 0.25])
    assert c.bin_index([0.0, 0.49, 0.5, 1.0]).tolist() == [0, 0, 1, 1]
    d = Curve("A", "depth", 10.0, SC, ((1.0, 0.0), (0.0, 1.0)), "user")
    assert d.bin_index([0.0, 9.99, 10.0, 19.9]).tolist() == [0, 0, 1, 1]
    with pytest.raises(ProportionError, match="reaches 20 m"):
        d.bin_index([25.0])


# -- combining a curve with a proportion map ---------------------------------------------------------------------------

def _combo_setup():
    c = Curve("A", "relative", 0.5, SC, ((0.8, 0.2), (0.2, 0.8)), "user")
    return setup(Settings("relative", nbins=2), ("user", "wells"), (c,))


def _map(fr, interval="A"):
    from cosma.proportion_maps import ProportionMap
    return ProportionMap(interval, SC, np.asarray(fr, float), {}, "")


def _full(v, shape=(6, 3)):
    return np.broadcast_to(np.asarray(v, float), shape + (2,)).copy()


def test_curve_times_map_over_global_renormalised_matches_hand_calculation():
    from cosma.proportions import combine_with_maps
    model = make_model()
    res = build_curves(make_project(), model, _combo_setup())
    out, rep = combine_with_maps(model, res, {"A": _map(_full((0.7, 0.3)))}, "stop", 0.0)
    G = np.array([0.44, 0.56])                         # layers 0,1 in bin 0 (0.8,0.2); layers 2-4 in bin 1 (0.2,0.8)
    for bin_vpc, layers in (((0.8, 0.2), [0, 1]), ((0.2, 0.8), [2, 3, 4])):
        w = np.array(bin_vpc) * np.array([0.7, 0.3]) / G
        np.testing.assert_allclose(out["A"][2, 1, layers], np.tile(w / w.sum(), (len(layers), 1)))
    np.testing.assert_allclose(out["A"].sum(-1), 1.0)
    base = layer_proportions(model, res)
    np.testing.assert_array_equal(out["B"], base["B"])            # no map: the curve alone
    assert "A: map applied" in rep


def test_a_map_equal_to_the_global_proportions_changes_nothing():
    from cosma.proportions import combine_with_maps
    model = make_model()
    res = build_curves(make_project(), model, _combo_setup())
    out, _ = combine_with_maps(model, res, {"A": _map(_full((0.44, 0.56)))}, "stop", 0.0)
    np.testing.assert_allclose(out["A"], layer_proportions(model, res)["A"])


def test_columns_without_map_data_follow_the_users_choice():
    from cosma.proportions import combine_with_maps
    model = make_model()
    res = build_curves(make_project(), model, _combo_setup())
    fr = _full((0.7, 0.3))
    fr[0, 0] = np.nan
    with pytest.raises(ProportionError, match="1 active column"):
        combine_with_maps(model, res, {"A": _map(fr)}, "stop", 0.0)
    out, rep = combine_with_maps(model, res, {"A": _map(fr)}, "curve", 0.0)
    np.testing.assert_array_equal(out["A"][0, 0], layer_proportions(model, res)["A"][0, 0])
    assert not np.allclose(out["A"][1, 0], out["A"][0, 0]) and "1 active column(s) without map data" in rep


def test_incompatible_cells_are_counted_and_limited_by_the_user():
    from cosma.proportions import combine_with_maps
    model = make_model()
    # curve says only silt in bin... build a curve with a sand-only bin and a map with no sand
    c = Curve("A", "relative", 0.5, SC, ((1.0, 0.0), (0.0, 1.0)), "user")
    res = build_curves(make_project(), model, setup(Settings("relative", nbins=2), ("user", "wells"), (c,)))
    mp = _map(_full((0.0, 1.0)))
    with pytest.raises(ProportionError, match="2 of 5 cells|nothing in"):
        combine_with_maps(model, res, {"A": mp}, "stop", 0.0)
    out, rep = combine_with_maps(model, res, {"A": mp}, "stop", 0.5)
    np.testing.assert_array_equal(out["A"][..., :2, :], layer_proportions(model, res)["A"][..., :2, :])   # kept the curve
    np.testing.assert_allclose(out["A"][..., 2:, :], np.broadcast_to([0.0, 1.0], out["A"][..., 2:, :].shape))
    assert "40.0%" in rep


def test_combination_refuses_what_it_cannot_check():
    from cosma.proportions import combine_with_maps
    model = make_model()
    res = build_curves(make_project(), model, _combo_setup())
    ok = _map(_full((0.7, 0.3)))
    for kw in (dict(outside_map="guess", max_incompatible=0.0), dict(outside_map="stop", max_incompatible=1.5),
               dict(outside_map="stop", max_incompatible=None)):
        with pytest.raises(ProportionError):
            combine_with_maps(model, res, {"A": ok}, **kw)
    with pytest.raises(ProportionError, match="import the map again"):
        combine_with_maps(model, res, {"A": _map(_full((0.7, 0.3), (5, 3)))}, "stop", 0.0)
    from cosma.proportion_maps import ProportionMap
    with pytest.raises(ProportionError, match="same facies"):
        combine_with_maps(model, res, {"A": ProportionMap("A", (1, 3), _full((0.7, 0.3)), {}, "")}, "stop", 0.0)
