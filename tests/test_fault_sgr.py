"""Shale gouge ratio from VSHALE on grid faces, the relation library, and calculated multipliers by SGR."""

import numpy as np
import pytest

from cosma.fault_relations import (LIBRARY, Method, RelationError, check_new_method, compile_expression, methods_from_json,
                                   methods_from_state, methods_to_json, methods_to_state, resolve)
from cosma.fault_sgr import ABOVE_TOP, BELOW_MIN_THROW, NO_VSHALE, SGRError, check_vshale, window_sgr
from cosma.fault_transmissibility import (FaultTransmissibility as FT, PendingProperties, TransmissibilityError,
                                          calculated_detail, faults_text, from_state, slab_multiplier, to_state)

from test_fault_transmissibility import built
from test_fault_transmissibility_gui import app  # noqa: F401
from cosma.gui.main_window import MainWindow

MZ_EXPR = "-4*SGR - 0.25*log10(D)*(1 - SGR)**5"


# -- the relation ------------------------------------------------------------------------------------------------

def test_manzocchi_1999_reproduces_the_papers_own_worked_example():
    m = LIBRARY["manzocchi1999"]
    kf = 10 ** m.log10_kf(0.2, 20.0)
    assert kf == pytest.approx(0.12, abs=0.005)                    # paper: median 0.12 mD for D = 20 m, SGR = 0.2
    assert 0.012 < kf < 1.2
    # the equation by hand: log kf = -4 SGR - (1/4) log10(D) (1 - SGR)^5
    for sgr, d in ((0.0, 1.0), (0.5, 100.0), (1.0, 50.0), (0.3, 7.0)):
        assert m.log10_kf(sgr, d) == pytest.approx(-4 * sgr - 0.25 * np.log10(d) * (1 - sgr) ** 5)
    assert m.log10_kf(1.0, 50.0) == pytest.approx(-4.0)            # fully shaly: displacement no longer matters
    assert "Eq. 1" in m.citation and "SGR tends to 0" in m.note and len(LIBRARY) == 1


def test_the_evaluator_computes_arithmetic_functions_and_broadcasts():
    f = compile_expression("1 + 2*SGR - 3*log10(D) + ln(exp(2)) + sqrt(4) + abs(-1) + min(SGR, 0.5) + max(D, 5)")
    sgr, d = 0.25, 10.0
    assert f(sgr, d) == pytest.approx(1 + 0.5 - 3 + 2 + 2 + 1 + 0.25 + 10)
    out = f(np.array([0.1, 0.5]), np.array([[10.0], [100.0]]))
    assert out.shape == (2, 2) and out[1, 0] == pytest.approx(1 + 0.2 - 6 + 2 + 2 + 1 + 0.1 + 100)
    assert compile_expression("-SGR**2")(0.5, 1.0) == pytest.approx(-0.25)            # unary minus binds looser than **
    assert compile_expression("2**3**2")(0.5, 1.0) == pytest.approx(512.0)             # right-associative
    assert compile_expression("-3")(0.5, 1.0) == -3.0 and compile_expression("5")(np.array([0.1, 0.2]), 1.0).shape == (2,)


@pytest.mark.parametrize("expr,msg", [
    ("", "empty"), ("   ", "empty"), ("__import__('os').system('x')", "only these functions"), ("SGR.real", "not allowed"),
    ("a + 1", "unknown name 'a'"), ("log10", "is a function"), ("[1, 2]", "not allowed"), ("SGR if D else 1", "not allowed"),
    ("lambda: 1", "not allowed"), ("log10(SGR, D)", "takes 1 argument"), ("min(SGR)", "takes 2 argument"),
    ("log10(x=D)", "only these functions"), ("'a'", "only numbers"), ("True", "only numbers"), ("SGR +", "not a valid"),
    ("1/0", "not finite"), ("log10(SGR - 0.3)", "not finite at SGR = 0.05"), ("log10(D - 1)", "not finite at SGR = 0.05, D = 1"), ("log10(SGR - 1)", "not finite"), ("10**10**10", "not finite"), ("1" + " + 1" * 200, "longer than"),
    ("SGR @ D", "not allowed"), ("SGR % 2", "not allowed"), ("SGR // 2", "not allowed"), ("-SGR if 1 else 2", "not allowed"),
    ("not SGR", "not allowed"), ("(SGR, D)", "not allowed"), ("1e999", "finite")])
def test_the_evaluator_refuses_everything_outside_its_small_grammar(expr, msg):
    with pytest.raises(RelationError, match=msg):
        compile_expression(expr)


def test_methods_need_a_name_a_source_and_cannot_shadow_the_builtin():
    m = Method("mine", "-2 - 3*SGR", "Own calibration 2026", "n")
    check_new_method(m, {})
    for bad in (Method("manzocchi1999", "-1", "x"), Method("mine", "-1", "  ")):
        with pytest.raises(RelationError):
            check_new_method(bad, {})
    for name, expr in (("", "-1"), ("  ", "-1"), ("x", "nope")):
        with pytest.raises(RelationError):
            Method(name, expr)
    assert m.describe() == "log10 kf = -2 - 3*SGR" and m.log10_kf(0.5, 10.0) == -3.5


def test_resolve_wants_exactly_one_source():
    assert resolve("manzocchi1999", None) is LIBRARY["manzocchi1999"] and resolve(None, "-1").expression == "-1"
    for rel, ex in ((None, None), ("manzocchi1999", "-1"), ("nope", None)):
        with pytest.raises(RelationError):
            resolve(rel, ex)


def test_methods_roundtrip_through_json_and_state_and_bad_files_are_refused():
    ms = {"a": Method("a", "-3*SGR", "Paper A", "basin A"), "b": Method("b", "-1 - log10(D)", "Report B")}
    assert methods_from_json(methods_to_json(ms)) == ms and methods_from_state(methods_to_state(ms)) == ms
    for text in ("not json", "[]", '{"methods": []}', '{"cosma_sgr_methods": 1, "methods": [{"name": "x"}]}',
                 '{"cosma_sgr_methods": 1, "methods": [{"name": "x", "expression": "q", "citation": "c"}]}',
                 '{"cosma_sgr_methods": 1, "methods": [{"name": "x", "expression": "-1", "citation": ""}]}',
                 '{"cosma_sgr_methods": 1, "methods": [{"name": "manzocchi1999", "expression": "-1", "citation": "c"}]}',
                 '{"cosma_sgr_methods": 1, "methods": [{"name": "x", "expression": "-1", "citation": "c"}, '
                 '{"name": "x", "expression": "-2", "citation": "c"}]}'):
        with pytest.raises(RelationError):
            methods_from_json(text)
    with pytest.raises(RelationError):
        methods_from_state([["x", "-1", "c", "", "extra"]])
    with pytest.raises(RelationError):
        methods_from_state([["x", "-1", "c", ""], ["x", "-2", "c", ""]])


def test_the_project_stores_adds_replaces_and_removes_methods():
    from cosma.project import Project, ProjectError
    pr = Project("p")
    pr.add_sgr_method(Method("mine", "-1", "src"))
    pr.add_sgr_method(Method("mine", "-2", "src2"))
    assert pr.sgr_methods["mine"].expression == "-2"
    with pytest.raises(ProjectError):
        pr.add_sgr_method(Method("manzocchi1999", "-1", "x"))
    with pytest.raises(ProjectError):
        pr.add_sgr_method(Method("nosrc", "-1", ""))
    pr.remove_sgr_method("mine")
    with pytest.raises(ProjectError):
        pr.remove_sgr_method("mine")


# -- SGR on a face -----------------------------------------------------------------------------------------------

def cols(h, v, T):
    h, v = np.asarray(h, float)[None, :], np.asarray(v, float)[None, :]
    zc = np.cumsum(h, axis=1) - h / 2
    return zc + T, zc, h, h, v, v


def test_sgr_is_the_shale_in_the_slipped_interval_over_the_throw_with_partial_layers():
    h, v = [25, 25, 10, 40], [0.2, 0.8, 0.5, 0.1]
    T, sgr, st = window_sgr(*cols(h, v, 20.0), min_throw=1.0)
    assert T.tolist() == [[20.0] * 4]
    assert st[0].tolist() == [ABOVE_TOP, 0, 0, 0]
    assert np.isnan(sgr[0, 0])
    assert sgr[0, 1:] == pytest.approx([0.575, 0.725, 0.1])        # worked by hand from the cumulative shale thickness


def test_uniform_vshale_gives_that_sgr_for_any_throw_and_layering():
    h = [12, 7, 30, 9, 21, 15, 40]
    for T in (3.0, 20.0, 41.0):
        _, sgr, st = window_sgr(*cols(h, [0.37] * 7, T), min_throw=0.5)
        assert sgr[st == 0] == pytest.approx(0.37) and (st == 0).any()


def test_alternating_beds_half_layer_window():
    h, v = [10] * 8, [0, 1] * 4
    _, sgr, st = window_sgr(*cols(h, v, 20.0), min_throw=1.0)
    assert st[0, :2].tolist() == [ABOVE_TOP, ABOVE_TOP] and st[0, 2:].tolist() == [0] * 6
    assert sgr[0, 2:] == pytest.approx([0.5] * 6)                  # (half + whole + half) / 2


def test_the_two_walls_are_averaged_and_throw_may_have_either_sign():
    h = np.full((1, 6), 10.0)
    zc = np.cumsum(h, axis=1) - 5
    va, vb = np.full((1, 6), 0.2), np.full((1, 6), 0.6)
    _, s1, _ = window_sgr(zc + 20, zc, h, h, va, vb, 1.0)
    _, s2, _ = window_sgr(zc, zc + 20, h, h, va, vb, 1.0)
    assert np.nanmax(s1) == pytest.approx(0.4) and np.allclose(s1, s2, equal_nan=True)
    hb = np.full((1, 6), 20.0)                                      # different layer thicknesses: mean thickness used
    _, s3, st = window_sgr(zc + 20, zc, h, hb, va, vb, 1.0)
    assert st[0, 0] == ABOVE_TOP and np.isfinite(s3[0, -1])


def test_faces_below_the_minimum_throw_and_missing_vshale_are_flagged_not_calculated():
    h = [10] * 6
    _, sgr, st = window_sgr(*cols(h, [0.5] * 6, 4.0), min_throw=5.0)
    assert (st == BELOW_MIN_THROW).all() and np.isnan(sgr).all()
    v = [0.5, 0.5, np.nan, 0.5, 0.5, 0.5]
    _, sgr, st = window_sgr(*cols(h, v, 20.0), min_throw=1.0)
    assert st[0, 2:5].tolist() == [NO_VSHALE] * 3                   # windows touching the undefined layer
    assert st[0, 5] == 0 and sgr[0, 5] == pytest.approx(0.5)
    assert st[0, 0] == ABOVE_TOP


def test_vshale_must_be_a_fraction_and_is_not_rescaled():
    check_vshale(np.array([0.0, 1.0, np.nan]))
    for bad in (np.array([0.5, 1.2]), np.array([-0.1, 0.5]), np.array([30.0, 60.0])):       # e.g. percent
        with pytest.raises(SGRError, match="between 0 and 1"):
            check_vshale(bad)


# -- the choice --------------------------------------------------------------------------------------------------

def sgr_choice(**kw):
    a = dict(mode="calculated", method="sgr", relation="manzocchi1999", thickness_per_throw=1 / 66, min_throw=1.0)
    a.update(kw)
    if a.get("thickness") is not None and "thickness_per_throw" not in kw:
        a["thickness_per_throw"] = None
    return FT(**a)


def test_sgr_choice_validation_and_state_roundtrip():
    sgr_choice()
    sgr_choice(relation=None, expression=MZ_EXPR, expression_name='mz', thickness=0.3)
    for bad in (dict(relation=None), dict(expression=MZ_EXPR, expression_name="x"), dict(relation="nope"), dict(min_throw=None),
                dict(min_throw=0.0), dict(kf=1.0), dict(thickness=0.3, thickness_per_throw=0.01), dict(thickness_per_throw=None),
                dict(relation=None, expression="nope", expression_name="x"), dict(relation=None, expression=MZ_EXPR),
                dict(expression_name="x")):
        with pytest.raises((TransmissibilityError, TypeError)):
            sgr_choice(**bad)
    with pytest.raises(TransmissibilityError):
        FT("calculated", method="slab", kf=1.0, thickness=1.0, relation="manzocchi1999")
    with pytest.raises(TransmissibilityError):
        FT("calculated", method="slab", kf=1.0, thickness=1.0, min_throw=1.0)
    with pytest.raises(TransmissibilityError):
        FT("fixed", multiplier=1.0, relation="manzocchi1999")
    d = {"A": sgr_choice(), "B": sgr_choice(relation=None, expression="0.5 - 3*SGR", expression_name="mine", thickness=0.2)}
    assert from_state(to_state(d)) == d
    assert "SGR from VSHALE" in d["A"].describe() and "built-in manzocchi1999" in d["A"].describe()
    assert "own method 'mine'" in d["B"].describe() and "log10 kf = 0.5 - 3*SGR" in d["B"].describe()
    old = to_state({"C": sgr_choice()})                                        # a file from the first SGR build: four constants
    old["C"].update(relation=None, constants=[-4.0, -0.25, 5.0, 0.0], expression=None, expression_name=None)
    c = from_state(old)["C"]
    assert c.expression_name == "own constants (migrated)" and c.relation is None
    assert c.min_throw == 1.0 and resolve(None, c.expression).log10_kf(0.2, 20.0) == pytest.approx(LIBRARY["manzocchi1999"].log10_kf(0.2, 20.0))


# -- on a built grid ---------------------------------------------------------------------------------------------

def with_props(choice, vsh, perm=100.0):
    mb = built(choice, perm=perm)
    shape = (mb.grid.nx, mb.grid.ny, mb.grid.nz)
    if vsh is not None:
        mb.grid.props["VSHALE"] = np.full(shape, vsh) if np.isscalar(vsh) else vsh
    return mb


def test_uniform_vshale_and_permeability_give_the_analytic_multiplier():
    mb = with_props(sgr_choice(thickness_per_throw=0.02), 0.4)
    rep = mb.fault_reports[0]
    d = calculated_detail(mb.fault_choices["F"], rep.faces_i, rep.faces_j, mb.grid, rep)
    kf = 10 ** (-4 * 0.4 - 0.25 * np.log10(20.0) * 0.6 ** 5)         # throw is 20 m at every face; SGR = 0.4
    L = 312.0 / 4                                                      # cell size across the fault: see grid
    L = float(np.hypot(*(mb.grid.coord[0, 1, :2] - mb.grid.coord[0, 0, :2])))
    expect = float(slab_multiplier(100.0, 100.0, L, L, kf, 0.02 * 20.0))
    assert d["m"] == pytest.approx(expect, rel=1e-9)
    assert d["sgr"] == pytest.approx((0.4, 0.4, 0.4)) and d["kf"] == pytest.approx((kf, kf))
    assert sum(d["left_out"].values()) > 0 and "slipped interval reaches above the model top" in d["left_out"]
    assert d["n"] + sum(d["left_out"].values()) == int(rep.faces_i.sum()) * mb.grid.nz + int(rep.faces_j.sum()) * mb.grid.nz


def test_more_shale_means_a_stronger_barrier_and_a_constant_thickness_is_used_as_given():
    ms = []
    for v in (0.1, 0.5, 0.9):
        mb = with_props(sgr_choice(thickness=0.5), v)
        rep = mb.fault_reports[0]
        ms.append(calculated_detail(mb.fault_choices["F"], rep.faces_i, rep.faces_j, mb.grid, rep)["m"])
    assert ms[0] > ms[1] > ms[2] and all(0 < m < 1 for m in ms)
    mb = with_props(sgr_choice(relation=None, expression=MZ_EXPR, expression_name='mz', thickness=0.5), 0.5)
    rep = mb.fault_reports[0]
    m = calculated_detail(mb.fault_choices["F"], rep.faces_i, rep.faces_j, mb.grid, rep)["m"]
    assert m == pytest.approx(ms[1])                                     # own method equal to the built-in one: same


def test_the_minimum_throw_removes_faces_and_all_removed_is_an_error():
    mb = with_props(sgr_choice(min_throw=25.0), 0.4)                    # throw is 20 m everywhere
    rep = mb.fault_reports[0]
    with pytest.raises(TransmissibilityError, match="no active cell pairs.*throw below the minimum"):
        calculated_detail(mb.fault_choices["F"], rep.faces_i, rep.faces_j, mb.grid, rep)


def test_sgr_needs_vshale_and_permeability_and_says_which():
    mb = with_props(sgr_choice(), None)
    with pytest.raises(PendingProperties) as ei:
        faults_text(mb, mb.fault_choices)
    assert ei.value.missing == ("VSHALE",) and "VSHALE" in str(ei.value)
    mb = built(sgr_choice())
    with pytest.raises(PendingProperties) as ei:
        faults_text(mb, mb.fault_choices)
    assert ei.value.missing == ("PERMX", "PERMY", "VSHALE")
    mb = with_props(sgr_choice(), 30.0)                                  # percent instead of fraction
    with pytest.raises(TransmissibilityError, match="between 0 and 1"):
        faults_text(mb, mb.fault_choices)


def test_the_export_says_how_the_multiplier_was_obtained():
    mb = with_props(sgr_choice(), 0.4)
    t = faults_text(mb, mb.fault_choices)
    assert "SGR min/weighted mean/max 0.400/0.400/0.400" in t and "left out:" in t and "built-in manzocchi1999" in t
    assert "face-layers (slipped interval reaches above the model top)" in t and "MULTFLT" in t


def test_vshale_that_varies_with_depth_changes_the_multiplier_between_faces_and_layers():
    mb = with_props(sgr_choice(), 0.4)
    nz = mb.grid.nz
    v = np.full((mb.grid.nx, mb.grid.ny, nz), 0.1)
    v[:, :, nz // 2:] = 0.9
    mb2 = with_props(sgr_choice(), v)
    rep = mb2.fault_reports[0]
    d = calculated_detail(mb2.fault_choices["F"], rep.faces_i, rep.faces_j, mb2.grid, rep)
    assert d["sgr"][0] < 0.2 and d["sgr"][2] > 0.8 and 0.1 < d["sgr"][1] < 0.9


# -- independent checks ------------------------------------------------------------------------------------------

def brute_sgr(h_a, h_b, v_a, v_b, T, k, step=0.002):
    """SGR by dense sampling of the mean stratigraphy: shale fraction of the T metres above the cell centre."""
    h, v = 0.5 * (np.asarray(h_a) + np.asarray(h_b)), 0.5 * (np.asarray(v_a) + np.asarray(v_b))
    edges = np.concatenate([[0.0], np.cumsum(h)])
    centre = edges[k] + h[k] / 2
    s = np.arange(centre - T + step / 2, centre, step)
    if s.size == 0 or s[0] < 0:
        return None
    layer = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(h) - 1)
    return float(v[layer].mean())


def test_window_sgr_agrees_with_dense_sampling_for_different_walls_and_thicknesses():
    rng = np.random.default_rng(3)
    for _ in range(5):
        nz = 12
        h_a, h_b = rng.uniform(5, 30, nz), rng.uniform(5, 30, nz)
        v_a, v_b = rng.uniform(0, 1, nz), rng.uniform(0, 1, nz)
        T = float(rng.uniform(8, 60))
        zc = np.zeros((1, nz))
        out_T, sgr, st = window_sgr(zc + T, zc, h_a[None], h_b[None], v_a[None], v_b[None], 1.0)
        for k in range(nz):
            ref = brute_sgr(h_a, h_b, v_a, v_b, T, k)
            if ref is None:
                assert st[0, k] == ABOVE_TOP
            else:
                assert st[0, k] == 0 and sgr[0, k] == pytest.approx(ref, abs=2e-3), k


def test_throw_that_changes_with_depth_enters_every_face_layer_separately():
    ratio = 0.02
    mb = with_props(sgr_choice(thickness_per_throw=ratio), 0.4)
    g = mb.grid
    rep = mb.fault_reports[0]
    shift = np.linspace(0.0, 12.0, g.nz)                                  # extra throw growing with layer index
    ic = int(np.argwhere(rep.faces_i)[0][0]) + 1
    g.zcorn[:, :, :, :, ic:, :] += shift[:, None, None, None, None, None]
    d = calculated_detail(mb.fault_choices["F"], rep.faces_i, rep.faces_j, g, rep)
    from cosma.fault_sgr import cell_centre_depth
    from cosma.grid.builder import _cell_thickness
    zc, th = cell_centre_depth(g.zcorn), _cell_thickness(g.zcorn, g.nx, g.ny, g.nz)
    L = float(np.hypot(*(g.coord[0, 1, :2] - g.coord[0, 0, :2])))
    num = den = 0.0
    for ia, ja in np.argwhere(rep.faces_i):
        T = np.abs(zc[ia + 1, ja] - zc[ia, ja])
        t_layer = 0.5 * (th[ia, ja] + th[ia + 1, ja])
        edges = np.concatenate([[0.0], np.cumsum(t_layer)])
        for k in range(g.nz):
            if edges[k] + t_layer[k] / 2 - T[k] < 0:
                continue                                                   # window above the model top
            kf = 10 ** LIBRARY['manzocchi1999'].log10_kf(0.4, T[k])
            w = t_layer[k]                                                 # transmissibility without the fault ~ layer thickness
            num += w * float(slab_multiplier(100.0, 100.0, L, L, kf, ratio * T[k]))
            den += w
    assert d["m"] == pytest.approx(num / den, rel=1e-9)
    assert d["kf"][0] < d["kf"][1]                                         # permeability differs between layers


def test_the_notice_also_follows_a_las_import_without_vshale(app, tmp_path):
    from test_conflicts import D1, V, _las
    from test_faults_gui import FH, crossing_project
    from test_imports import _project
    pr = _project(tmp_path)
    pr.attach_faults(list(crossing_project().faults.values()))
    h = FH(_las(tmp_path, "a.las", D1, {"GR": V}), {})
    w = MainWindow(pr, hooks=h.dict())
    w.do_import("las")
    assert sum("no well has a VSHALE curve" in r for r in h.reports) == 1
