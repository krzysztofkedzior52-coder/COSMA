"""Fault transmissibility choices: validation, the slab multiplier against series resistance, export text."""

import numpy as np
import pytest

from cosma.fault_transmissibility import (PendingProperties, FaultTransmissibility as FT, TransmissibilityError, calculated_multiplier,
                                          deck_names, face_records, faults_text, from_state, slab_multiplier, to_state)
from cosma.grid_definition import GridDefinitionError, build_model, required_items, write_model
from cosma.project import Project, ProjectError

from test_grid_definition import gd, model, vf


# -- the choice itself -------------------------------------------------------------------------------------------------

def test_choices_are_validated():
    FT("open")
    FT("fixed", multiplier=0.0)
    FT("fixed", multiplier=2.5)
    FT("calculated", method="slab", kf=0.01, thickness=0.2)
    FT("calculated", method="slab", kf=0.01, thickness_per_throw=0.01)
    bad = [dict(mode="sealed"), dict(mode="fixed"), dict(mode="fixed", multiplier=-0.1),
           dict(mode="fixed", multiplier=float("nan")), dict(mode="fixed", multiplier=float("inf")),
           dict(mode="open", multiplier=1.0), dict(mode="open", kf=1.0),
           dict(mode="fixed", multiplier=1.0, method="slab"),
           dict(mode="calculated"), dict(mode="calculated", method="sgr", kf=1.0, thickness=1.0),
           dict(mode="calculated", method="slab", thickness=1.0),
           dict(mode="calculated", method="slab", kf=0.0, thickness=1.0),
           dict(mode="calculated", method="slab", kf=1.0),
           dict(mode="calculated", method="slab", kf=1.0, thickness=1.0, thickness_per_throw=0.1),
           dict(mode="calculated", method="slab", kf=1.0, thickness=-1.0),
           dict(mode="calculated", method="slab", kf=1.0, thickness_per_throw=0.0),
           dict(mode="calculated", method="slab", kf=float("nan"), thickness=1.0)]
    for kw in bad:
        with pytest.raises(TransmissibilityError):
            FT(**kw)


def test_state_roundtrip_and_bad_records():
    d = {"A": FT("open"), "B": FT("fixed", multiplier=0.3),
         "C": FT("calculated", method="slab", kf=0.05, thickness_per_throw=0.02)}
    assert from_state(to_state(d)) == d
    with pytest.raises(TransmissibilityError):
        from_state({"A": {}})
    with pytest.raises(TransmissibilityError):
        from_state({"A": {"mode": "fixed", "multiplier": -1}})


def test_describe_says_what_was_chosen():
    assert "open" in FT("open").describe()
    assert "sealing" in FT("fixed", multiplier=0.0).describe()
    assert "0.25" in FT("fixed", multiplier=0.25).describe()
    c = FT("calculated", method="slab", kf=0.5, thickness_per_throw=0.02).describe()
    assert "0.5 mD" in c and "0.02 x throw" in c


# -- the slab multiplier against series resistance ----------------------------------------------------------------------

def series_truth(ki, kj, Li, Lj, kf, t):
    """Half of each cell is the path from its centre to the face. The fault slab replaces t/2 of each side."""
    r0 = Li / 2 / ki + Lj / 2 / kj
    r1 = (Li / 2 - t / 2) / ki + (Lj / 2 - t / 2) / kj + t / kf
    return r0 / r1


@pytest.mark.parametrize("ki,kj,Li,Lj,kf,t", [
    (100.0, 100.0, 100.0, 100.0, 1.0, 0.5), (100.0, 20.0, 100.0, 50.0, 0.01, 0.2),
    (5.0, 800.0, 30.0, 300.0, 3.0, 2.0), (100.0, 100.0, 100.0, 100.0, 1000.0, 0.5)])
def test_slab_multiplier_equals_series_resistance(ki, kj, Li, Lj, kf, t):
    assert float(slab_multiplier(ki, kj, Li, Lj, kf, t)) == pytest.approx(series_truth(ki, kj, Li, Lj, kf, t), rel=1e-12)


def test_slab_multiplier_limits():
    assert float(slab_multiplier(100, 100, 100, 100, 100.0, 1.0)) == pytest.approx(1.0)      # same rock as the matrix
    assert float(slab_multiplier(100, 100, 100, 100, 1e-9, 1.0)) < 1e-6                      # sealing as kf -> 0
    assert float(slab_multiplier(100, 100, 100, 100, 1.0, 1e-9)) == pytest.approx(1.0)       # no thickness, no effect
    assert float(slab_multiplier(100, 100, 100, 100, 1000.0, 5.0)) > 1.0                     # leakier than matrix
    # published worked example scale: 20 cm of 1 mD fault between 200 cm blocks of 1000 mD -> large reduction
    assert float(slab_multiplier(1000.0, 1000.0, 2.0, 2.0, 1.0, 0.2)) == pytest.approx(1 / (1 + 0.2 * (1000 - 1) / 2.0))


# -- names and face records ----------------------------------------------------------------------------------------------

def test_deck_names_are_short_unique_and_mapped():
    n = deck_names(["Main", "Main fault, N", "toolongname", "main", "A"])
    assert n["Main"] == "MAIN" and n["A"] == "A"
    assert n["Main fault, N"].startswith("FLT") and n["toolongname"].startswith("FLT")
    assert n["main"].startswith("FLT")                  # clashes with MAIN once upper-cased
    assert len(set(n.values())) == 5 and all(len(v) <= 8 for v in n.values())


def test_face_records_merge_runs_and_skip_inactive():
    nx, ny = 5, 4
    act = np.ones((nx, ny), bool)
    fi = np.zeros((nx - 1, ny), bool)
    fj = np.zeros((nx, ny - 1), bool)
    fi[1, :] = True                                    # a straight N-S fault between i=1 and 2
    assert face_records(fi, fj, act) == [(2, 2, 1, 4, "X")]
    fi[1, 2] = False                                   # a hole splits the run
    assert face_records(fi, fj, act) == [(2, 2, 1, 2, "X"), (2, 2, 4, 4, "X")]
    fi[:] = False
    fj[:, 1] = True                                    # E-W fault between j=1 and 2
    assert face_records(fi, fj, act) == [(1, 5, 2, 2, "Y")]
    act[:, :] = False
    act[2, :] = True                                   # only column i=2 active: faces elsewhere skipped
    assert face_records(fi, fj, act) == [(3, 3, 2, 2, "Y")]
    fi[1, :] = True
    act[:, :] = False
    assert face_records(fi, fj, act) == []


# -- the project and the build gate -------------------------------------------------------------------------------------

def faulted(step=20.0):
    pr = model(step=(312.0, step))
    pr.attach_faults([vf("F", (312.0, -100.0), (312.0, 500.0))])
    pr.set_fault_depth("F", 1500.0)
    return pr


def test_project_stores_and_forgets_choices():
    pr = faulted()
    with pytest.raises(ProjectError):
        pr.set_fault_transmissibility("Z", FT("open"))
    with pytest.raises(ProjectError):
        pr.set_fault_transmissibility("F", "open")
    pr.set_fault_transmissibility("F", FT("fixed", multiplier=0.1))
    assert pr.fault_transmissibility["F"].multiplier == 0.1
    pr.remove_fault("F")
    assert pr.fault_transmissibility == {}


def test_every_fault_needs_a_choice_before_a_build():
    pr = faulted()
    g = gd(pr)
    assert any("transmissibility of fault 'F'" in i for i in required_items(pr, g))
    with pytest.raises(GridDefinitionError, match="transmissibility of fault 'F'"):
        build_model(pr, g)
    pr.set_fault_transmissibility("F", FT("open"))
    assert required_items(pr, g) == []


def test_calculated_is_refused_for_a_fault_without_throw():
    pr = faulted(step=2.0)                             # 2 m against 50 m layers: negligible
    pr.set_fault_transmissibility("F", FT("calculated", method="slab", kf=0.1, thickness=0.1))
    with pytest.raises(GridDefinitionError, match="negligible throw"):
        build_model(pr, gd(pr))
    pr.set_fault_transmissibility("F", FT("fixed", multiplier=0.0))      # a negligible fault may be open or fixed
    assert build_model(pr, gd(pr)).fault_choices["F"].multiplier == 0.0


# -- the exported text -------------------------------------------------------------------------------------------------

def built(choice, step=20.0, perm=None):
    pr = faulted(step)
    pr.set_fault_transmissibility("F", choice)
    mb = build_model(pr, gd(pr))
    if perm is not None:
        shape = (mb.grid.nx, mb.grid.ny, mb.grid.nz)
        mb.grid.props["PERMX"] = np.full(shape, perm) if np.isscalar(perm) else perm[0]
        mb.grid.props["PERMY"] = np.full(shape, perm) if np.isscalar(perm) else perm[1]
    return mb


def multflt(text):
    lines = text.splitlines()
    k = lines.index("MULTFLT")
    return float(lines[k + 1].split()[1])


def test_open_and_fixed_write_their_multiplier_and_every_face():
    for choice, value in ((FT("open"), 1.0), (FT("fixed", multiplier=0.0), 0.0), (FT("fixed", multiplier=0.35), 0.35)):
        mb = built(choice)
        text = faults_text(mb, mb.fault_choices)
        assert multflt(text) == value
        assert "FAULTS\n" in text and "'F'" in text
        rep = mb.fault_reports[0]
        recs = face_records(rep.faces_i, rep.faces_j, mb.active_columns)
        assert sum((r[1] - r[0] + 1) * (r[3] - r[2] + 1) for r in recs) == rep.n_faces      # every face written once
        assert all(r[4] in "XY" for r in recs)
        assert f"1 {mb.grid.nz} " in text                                                    # all layers


def test_a_fault_without_cell_faces_writes_no_dangling_multiplier():
    pr = faulted()
    pr.attach_faults([vf("Far", (5000.0, -100.0), (5000.0, 500.0))])
    pr.set_fault_depth("Far", 1500.0)
    pr.refresh_fault_interactions(5.0)
    pr.set_fault_transmissibility("F", FT("fixed", multiplier=0.5))
    pr.set_fault_transmissibility("Far", FT("fixed", multiplier=0.5))
    mb = build_model(pr, gd(pr))
    text = faults_text(mb, mb.fault_choices)
    assert "'F' 0.5 /" in text and "'Far'" not in text.split("MULTFLT")[1] and "nothing written" in text


def test_calculated_waits_for_the_property_step_and_the_geometry_still_exports(tmp_path):
    mb = built(FT("calculated", method="slab", kf=0.01, thickness=0.2))
    with pytest.raises(PendingProperties, match="property step") as ei:
        faults_text(mb, mb.fault_choices)
    assert ei.value.missing == ("PERMX", "PERMY") and ei.value.fault == "F"
    files = write_model(mb, tmp_path / "x.grdecl")
    assert [f.name for f in files] == ["x.grdecl"]                      # no fault include, but the grid is there
    assert any("fault transmissibility include NOT written" in n and "PERMX, PERMY" in n for n in mb.notes)
    assert any("NOT written" in line for line in mb.boundary_lines())
    write_model(mb, tmp_path / "y.grdecl")                              # a second export does not duplicate the note
    assert sum("fault transmissibility include" in n for n in mb.notes) == 1
    mb.grid.props["PERMX"] = mb.grid.props["PERMY"] = np.full((mb.grid.nx, mb.grid.ny, mb.grid.nz), 100.0)
    files = write_model(mb, tmp_path / "z.grdecl")                      # properties exist: the include is written
    assert sorted(f.name for f in files) == ["z.grdecl", "z_faults.inc"]


def test_calculated_uniform_permeability_equals_the_analytic_multiplier():
    kf, t, k = 0.02, 0.3, 150.0
    mb = built(FT("calculated", method="slab", kf=kf, thickness=t), perm=k)
    m, n = calculated_multiplier(mb.fault_choices["F"], mb.fault_reports[0].faces_i, mb.fault_reports[0].faces_j,
                                 mb.grid, mb.fault_reports[0])
    assert n > 0
    assert m == pytest.approx(float(slab_multiplier(k, k, 100.0, 100.0, kf, t)), rel=1e-12)
    assert multflt(faults_text(mb, mb.fault_choices)) == pytest.approx(m, rel=1e-5)


def test_calculated_thickness_from_throw_uses_the_median_throw():
    kf, ratio, k = 0.02, 0.01, 150.0
    mb = built(FT("calculated", method="slab", kf=kf, thickness_per_throw=ratio), perm=k)
    rep = mb.fault_reports[0]
    throw = float(np.median([abs(x) for x in rep.median_throw]))
    assert throw == pytest.approx(20.0, abs=0.5)
    m, _ = calculated_multiplier(mb.fault_choices["F"], rep.faces_i, rep.faces_j, mb.grid, rep)
    assert m == pytest.approx(float(slab_multiplier(k, k, 100.0, 100.0, kf, ratio * throw)), rel=1e-12)


def test_calculated_mean_is_weighted_by_the_transmissibility_without_the_fault():
    mb = built(FT("calculated", method="slab", kf=0.05, thickness=0.4))
    g = mb.grid
    rng = np.random.default_rng(3)
    px = rng.uniform(5.0, 500.0, (g.nx, g.ny, g.nz))
    py = rng.uniform(5.0, 500.0, (g.nx, g.ny, g.nz))
    px[2, 1, :] = 0.0                                  # a dead cell next to the fault: no flow pair there
    px[3, 2, 0] = -1.0                                 # and a nonsense value
    delta = rng.uniform(-4.0, 4.0, g.zcorn[0, 1].shape)
    g.zcorn[0, 1] += delta                             # layer 0 thicker/thinner column by column ...
    g.zcorn[1, 0] += delta                             # ... layer 1 takes the difference, so the cells still touch
    g.props["PERMX"], g.props["PERMY"] = px, py
    rep = mb.fault_reports[0]
    m, n = calculated_multiplier(mb.fault_choices["F"], rep.faces_i, rep.faces_j, g, rep)
    # independent loop over faces (this fault is N-S, so only I faces); cells are 100 m wide, thickness per cell
    from cosma.grid.builder import _cell_thickness
    th = _cell_thickness(g.zcorn, g.nx, g.ny, g.nz)
    num = den = 0.0
    cnt = 0
    for (i, j) in zip(*np.nonzero(rep.faces_i)):
        for k in range(g.nz):
            if g.actnum[i, j, k] and g.actnum[i + 1, j, k] and px[i, j, k] > 0 and px[i + 1, j, k] > 0:
                a, b = px[i, j, k], px[i + 1, j, k]
                area = 100.0 * 0.5 * (th[i, j, k] + th[i + 1, j, k])
                t0 = area / (100.0 / 2 / a + 100.0 / 2 / b)
                num += t0 * series_truth(a, b, 100.0, 100.0, 0.05, 0.4)
                den += t0
                cnt += 1
    assert n == cnt and m == pytest.approx(num / den, rel=1e-12)
    assert 0 < m < 1


def test_calculated_needs_cell_pairs_and_a_throw():
    mb = built(FT("calculated", method="slab", kf=0.01, thickness=0.2), perm=100.0)
    rep = mb.fault_reports[0]
    mb.grid.actnum[:] = 0
    with pytest.raises(TransmissibilityError, match="no active cell pairs"):
        calculated_multiplier(mb.fault_choices["F"], rep.faces_i, rep.faces_j, mb.grid, rep)
    mb2 = built(FT("calculated", method="slab", kf=0.01, thickness_per_throw=0.1), perm=100.0)
    rep2 = mb2.fault_reports[0]
    rep2.median_throw = [float("nan")] * len(rep2.median_throw)
    with pytest.raises(TransmissibilityError, match="no measurable throw"):
        calculated_multiplier(mb2.fault_choices["F"], rep2.faces_i, rep2.faces_j, mb2.grid, rep2)


def test_write_model_writes_the_faults_include(tmp_path):
    mb = built(FT("fixed", multiplier=0.2))
    out = write_model(mb, tmp_path / "g.grdecl")
    names = [p.name for p in out]
    assert "g_faults.inc" in names
    assert multflt((tmp_path / "g_faults.inc").read_text()) == 0.2


def test_the_simulator_parser_accepts_the_include(tmp_path):
    parser = pytest.importorskip("opm.io.parser")
    mb = built(FT("fixed", multiplier=0.2))
    write_model(mb, tmp_path / "g.grdecl")
    deck = ("RUNSPEC\nDIMENS\n 6 4 %d /\nGRID\nINCLUDE\n 'g.grdecl' /\nINCLUDE\n 'g_faults.inc' /\nEND\n" % mb.grid.nz)
    (tmp_path / "t.DATA").write_text(deck)
    d = parser.Parser().parse(str(tmp_path / "t.DATA"))
    rec = d["FAULTS"][0]
    assert rec[0].get_str(0) == "F" and [rec[k].get_int(0) for k in range(1, 7)] == [3, 3, 1, 4, 1, mb.grid.nz]
    assert rec[7].get_str(0) == "X"
    m = d["MULTFLT"][0]
    assert m[0].get_str(0) == "F" and m[1].get_raw(0) == 0.2


def test_thickness_from_throw_takes_the_median_over_horizons():
    mb = built(FT("calculated", method="slab", kf=0.02, thickness_per_throw=0.01), perm=150.0)
    rep = mb.fault_reports[0]
    rep.median_throw = [0.0, -10.0, 30.0]              # median of |throw| is 10, the maximum 30
    m, _ = calculated_multiplier(mb.fault_choices["F"], rep.faces_i, rep.faces_j, mb.grid, rep)
    assert m == pytest.approx(float(slab_multiplier(150.0, 150.0, 100.0, 100.0, 0.02, 0.01 * 10.0)), rel=1e-12)


def test_the_text_itself_refuses_calculation_for_a_fault_without_throw():
    mb = built(FT("fixed", multiplier=0.5), step=2.0, perm=100.0)
    assert not mb.fault_reports[0].has_throw
    with pytest.raises(TransmissibilityError, match="negligible throw"):
        faults_text(mb, {"F": FT("calculated", method="slab", kf=0.01, thickness=0.1)})
    with pytest.raises(TransmissibilityError, match="no transmissibility choice"):
        faults_text(mb, {})


def test_a_build_records_which_faults_have_visible_throw():
    pr = faulted(step=20.0)
    pr.set_fault_transmissibility("F", FT("open"))
    assert pr.fault_throw_seen == {}
    build_model(pr, gd(pr))
    assert pr.fault_throw_seen == {"F": True}
    pr2 = faulted(step=2.0)
    pr2.set_fault_transmissibility("F", FT("open"))
    build_model(pr2, gd(pr2))
    assert pr2.fault_throw_seen == {"F": False}
    pr2.set_fault_transmissibility("F", FT("calculated", method="slab", kf=0.1, thickness=0.1))
    with pytest.raises(GridDefinitionError):
        build_model(pr2, gd(pr2))
    assert pr2.fault_throw_seen == {"F": False}             # still recorded when the build is refused
