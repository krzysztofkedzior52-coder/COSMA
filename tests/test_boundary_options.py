import numpy as np
import pytest
from shapely.geometry import Polygon, box

from cosma.grid.boundary import grid_spec_from_cells
from cosma.grid.boundary_options import (AquiferSpec, BoundaryOptionError, aqudims_text, BoundaryOptions, PVMultiplier,
                                         aquifer_text, boundary_layout, combination_warnings, connection_runs,
                                         edge_columns, edge_distance, face_cells, multpv_cube, multpv_text)

SPEC = grid_spec_from_cells(0, 0, 0, 10, 10, 100.0, 100.0)


def sq(x0, y0, x1, y1):
    return box(x0, y0, x1, y1)


def fet(**kw):
    a = dict(kind="fetkovich", faces=("I-", "I+"), datum_depth=1500.0, initial_pressure=160.0, compressibility=1e-4,
             influx_multiplier=1.0, volume=1e9, productivity_index=50.0)
    a.update(kw)
    return AquiferSpec(**a)


def ct(**kw):
    a = dict(kind="carter_tracy", faces=("J+",), datum_depth=1500.0, initial_pressure=None, compressibility=1e-4,
             influx_multiplier=1.0, permeability=200.0, porosity=0.2, external_radius=5000.0, thickness=50.0,
             influx_angle=180.0, table=1)
    a.update(kw)
    return AquiferSpec(**a)


# -- layout -----------------------------------------------------------------------------------------------------------------

def test_centre_rule_and_closed_default():
    act, reg, w = boundary_layout(SPEC, sq(200, 200, 700, 700), BoundaryOptions())
    assert act.sum() == 25 and act[2:7, 2:7].all() and (reg == act).all() and not w


def test_fraction_rule_uses_the_minimum_fraction():
    poly = sq(250, 250, 650, 650)                     # edge columns are covered exactly 50 %
    a0, *_ = boundary_layout(SPEC, poly, BoundaryOptions(rule="fraction", min_fraction=0.25))
    a1, *_ = boundary_layout(SPEC, poly, BoundaryOptions(rule="fraction", min_fraction=0.5))
    a2, *_ = boundary_layout(SPEC, poly, BoundaryOptions(rule="fraction", min_fraction=0.51))
    assert a0.sum() == 25 and a1.sum() == 21 and not a1[2, 2] and a1[3, 2] and a2.sum() == 9   # corners 25 %, edges 50 %


def test_buffer_adds_a_second_region_within_the_distance():
    act, reg, w = boundary_layout(SPEC, sq(200, 200, 700, 700), BoundaryOptions(buffer=100.0))
    assert act.sum() == 49 and (reg == 1).sum() == 25 and (reg == 2).sum() == 24 and not w
    assert reg[1, 1] == 2 and reg[0, 0] == 0 and reg[3, 3] == 1


def test_buffer_larger_than_the_grid_is_reported():
    act, reg, w = boundary_layout(SPEC, sq(0, 0, 300, 300), BoundaryOptions(buffer=100.0))
    assert w and "too small for the buffer" in w[0]


def test_polygon_in_a_rotated_frame_matches_the_local_square():
    sp = grid_spec_from_cells(5000, 7000, 30.0, 10, 10, 100.0, 100.0)
    pts = [sp.frame.to_map(x, y) for x, y in ((200, 200), (700, 200), (700, 700), (200, 700))]
    pts = [(float(a), float(b)) for a, b in pts]
    act, *_ = boundary_layout(sp, Polygon(pts), BoundaryOptions())
    assert act.sum() == 25 and act[2:7, 2:7].all()


def test_polygon_missing_the_grid_is_an_error():
    with pytest.raises(BoundaryOptionError, match="no grid columns"):
        boundary_layout(SPEC, sq(5000, 5000, 6000, 6000), BoundaryOptions())


# -- rings and faces ----------------------------------------------------------------------------------------------------

def test_edge_rings():
    a = np.ones((5, 5), bool)
    assert edge_columns(a, 1).sum() == 16 and edge_columns(a, 2).sum() == 24 and edge_columns(a, 3).sum() == 25
    assert edge_distance(a)[2, 2] == 3 and edge_distance(a)[0, 2] == 1
    a[2, 2] = False                                    # a hole makes its neighbours edge columns too
    assert edge_columns(a, 1).sum() == 16 + 4


def test_multpv_cube_and_text_round_trip(tmp_path):
    from cosma.io.grdecl import scan_keywords
    act = np.zeros((6, 4), bool)
    act[1:5, 1:3] = True
    pv = PVMultiplier(7.5, 1)
    cube = multpv_cube(act, pv, 3)
    assert cube.shape == (6, 4, 3) and (cube[act] == 7.5).all() and (cube[~act] == 1.0).all()
    text = multpv_text(cube, pv)
    f = tmp_path / "m.inc"
    f.write_text(text)
    assert scan_keywords(f) == {"MULTPV": 6 * 4 * 3}
    toks = " ".join(l for l in text.splitlines() if not l.startswith("--") and l.strip() not in ("MULTPV",)).replace("/", "").split()
    vals = []
    for t in toks:
        n, _, v = t.partition("*") if "*" in t else ("1", "", t)
        vals += [float(v)] * int(n)
    vals = np.array(vals).reshape(3, 4, 6)             # K, J, I: I fastest
    assert np.array_equal(np.transpose(vals, (2, 1, 0)), cube)


def test_faces_and_connection_runs_conserve_every_face():
    act = np.ones((5, 5), bool)
    act[0, 2] = False                                  # notch in the low-I edge splits the I- face into two runs
    fc = face_cells(act, ("I-", "I+", "J-", "J+"))
    assert fc["I-"].sum() == 4 + 1                     # 4 edge cells on the left, plus the cell right of the notch
    runs = connection_runs(act, ("I-",))
    assert sorted(runs) == [(1, 1, 1, 2, "I-"), (1, 1, 4, 5, "I-"), (2, 2, 3, 3, "I-")]
    allruns = connection_runs(act, ("I-", "I+", "J-", "J+"))
    total = sum((i2 - i1 + 1) * (j2 - j1 + 1) for i1, i2, j1, j2, _ in allruns)
    assert total == sum(int(m.sum()) for m in face_cells(act, ("I-", "I+", "J-", "J+")).values())
    assert sorted(connection_runs(act, ("J+",))) == [(1, 1, 2, 2, "J+"), (1, 5, 5, 5, "J+")]   # the notch also faces J+
    assert connection_runs(np.pad(np.ones((3, 3), bool), 1), ("J+",)) == [(2, 4, 4, 4, "J+")]


# -- validation -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: fet(faces=()), lambda: fet(faces=("X+",)), lambda: fet(faces=("I-", "I-")), lambda: fet(volume=None),
    lambda: fet(volume=-1.0), lambda: fet(productivity_index=0.0), lambda: fet(kind="darcy"),
    lambda: fet(compressibility=0.0), lambda: fet(initial_pressure=-3.0), lambda: fet(influx_multiplier=0.0),
    lambda: fet(datum_depth=float("nan")), lambda: fet(permeability=100.0),
    lambda: ct(porosity=1.5), lambda: ct(influx_angle=400.0), lambda: ct(table=1.5), lambda: ct(thickness=None),
    lambda: ct(volume=1.0),
    lambda: PVMultiplier(0.0, 1), lambda: PVMultiplier(2.0, 0), lambda: PVMultiplier(2.0, 1.5),
    lambda: BoundaryOptions(rule="other"), lambda: BoundaryOptions(min_fraction=0.0), lambda: BoundaryOptions(buffer=-5.0),
])
def test_every_bad_input_is_refused(make):
    with pytest.raises(BoundaryOptionError):
        make()


def test_state_round_trip_and_combination_warnings():
    o = BoundaryOptions("fraction", 0.4, 250.0, PVMultiplier(3.0, 2), fet())
    assert BoundaryOptions.from_state(o.to_state()) == o
    o2 = BoundaryOptions(aquifer=ct())
    assert BoundaryOptions.from_state(o2.to_state()) == o2 and BoundaryOptions().closed and not o.closed
    with pytest.raises(BoundaryOptionError):
        BoundaryOptions.from_state({"rule": "center"})
    w = " ".join(combination_warnings(o))
    assert "double-count" in w and "outer ring of the buffer" in w
    assert combination_warnings(BoundaryOptions(buffer=10.0)) == []


# -- deck text against the OPM parser (the one OPM Flow uses) --------------------------------------------------------------

def _parse(text, tmp_path, runspec=True):
    P = pytest.importorskip("opm.io.parser")
    f = tmp_path / "x.inc"
    if runspec:                      # OPM insists that AQUDIMS is declared in RUNSPEC before any analytic aquifer
        text = "RUNSPEC\n" + aquifer_runspec + "\nGRID\nSOLUTION\n" + text
    f.write_text(text)
    return P.Parser().parse(str(f), P.ParseContext([("PARSE_RANDOM_SLASH", P.action.ignore)]))


def test_fetkovich_text_parses_with_the_expected_items(tmp_path):
    act = np.pad(np.ones((3, 3), bool), 1)
    global aquifer_runspec
    aquifer_runspec = aqudims_text(fet(), act)
    deck = _parse(aquifer_text(fet(), act, 4), tmp_path)
    rec = deck["AQUFETP"][0]
    assert rec[0].get_int(0) == 1 and rec[1].get_raw(0) == 1500.0 and rec[2].get_raw(0) == 160.0
    assert rec[3].get_raw(0) == 1e9 and rec[4].get_raw(0) == 1e-4 and rec[5].get_raw(0) == 50.0
    con = deck["AQUANCON"]
    assert len(con) == 2                                # one I- run and one I+ run
    assert [con[0][k].get_int(0) for k in (1, 2, 3, 4, 5, 6)] == [2, 2, 2, 4, 1, 4]


def test_carter_tracy_text_parses_and_equilibrated_pressure_is_defaulted(tmp_path):
    act = np.pad(np.ones((3, 3), bool), 1)
    global aquifer_runspec
    aquifer_runspec = aqudims_text(ct(), act)
    deck = _parse(aquifer_text(ct(), act, 2), tmp_path)
    rec = deck["AQUCT"][0]
    assert rec[3].get_raw(0) == 200.0 and rec[8].get_raw(0) == 180.0
    assert len(deck["AQUANCON"]) == 1
    assert " 1500 1* 200 " in aquifer_text(ct(), act, 2)             # pressure left to equilibration, not guessed


def test_aquifer_without_faces_to_connect_is_refused():
    full = np.ones((3, 3), bool)
    with pytest.raises(BoundaryOptionError):
        aquifer_text(fet(faces=("I-",)), np.zeros((3, 3), bool), 2)
    assert aquifer_text(fet(faces=("I-",)), full, 2)


def test_influx_multiplier_and_face_names_reach_the_aquifer_records(tmp_path):
    act = np.pad(np.ones((3, 3), bool), 1)
    global aquifer_runspec
    a = fet(influx_multiplier=2.5, faces=("J-",))
    aquifer_runspec = aqudims_text(a, act)
    con = _parse(aquifer_text(a, act, 4), tmp_path)["AQUANCON"]
    assert len(con) == 1 and con[0][9].get_raw(0) == 2.5 and con[0][7].get_str(0) == "J-"
    assert [con[0][k].get_int(0) for k in (1, 2, 3, 4, 5, 6)] == [2, 4, 2, 2, 1, 4]
