"""Fault sticks as an import kind: confirmation, conflicts, advice, storage."""

import math

import numpy as np
import pytest

from cosma.faults import Fault
from cosma.imports import ImportCancelled, ImportRejected, conflicts_for, advice_for, import_file, inspect, scripted
from cosma.project import Action, Project, ProjectError
from cosma.surfaces import Surface
from test_faults import plane_sticks, write
from test_variogram_project import build

OK = {"xy_unit": "m", "z_unit": "m", "z_is": "depth"}


def sticks_csv(tmp_path, name="f.csv", faults=(("A", 60, 90), ("B", 85, 0)), ft=1.0):
    rows = ["x,y,depth,fault,stick"]
    for fn, dip, dd in faults:
        for k, s in enumerate(plane_sticks(dip, dd, [0, 300, 700], origin=(5000.0, 6000.0)), 1):
            rows += [f"{float(x) / ft!r},{float(y) / ft!r},{float(z) / ft!r},{fn},{k}" for x, y, z in s]
    return write(tmp_path, name, "\n".join(rows) + "\n")


def test_preview_states_what_is_in_the_file_and_asks_for_everything(tmp_path):
    prev = inspect("fault_sticks", sticks_csv(tmp_path))
    f = dict(prev.found)
    assert f["Layout"].startswith("table") and f["Sticks are identified by"] == "a stick id column" and f["Faults / points"] == "2 / 36"
    assert "3 stick(s), 18 points" in f["  A"] and "fault <- 'fault'" in f["Columns recognised"]
    keys = {p.key: p for p in prev.params}
    assert set(keys) == {"xy_unit", "z_unit", "z_is"}
    assert all(p.value is None and p.source == "required" for p in keys.values())          # nothing is pre-filled
    for missing in ({}, {"xy_unit": "m", "z_unit": "m"}, {"xy_unit": "ft", "z_unit": "m", "z_is": "depth"}, {**OK, "z_is": "time"}):
        with pytest.raises(ImportRejected):
            import_file("fault_sticks", sticks_csv(tmp_path), scripted(missing))
    with pytest.raises(ImportCancelled):
        import_file("fault_sticks", sticks_csv(tmp_path), lambda p: None)


def test_headerless_file_asks_for_a_fault_name_with_the_file_stem_as_default(tmp_path):
    p = write(tmp_path, "Main_fault.txt", "0 0 10\n0 0 90\n999 999 999\n100 0 10\n100 0 90\n")
    prev = inspect("fault_sticks", p)
    nm = next(x for x in prev.params if x.key == "fault_name")
    assert nm.value == "Main_fault" and nm.source == "default"
    (f,) = import_file("fault_sticks", p, scripted(OK))
    assert f.name == "Main_fault" and f.n_sticks == 2
    (g,) = import_file("fault_sticks", p, scripted({**OK, "fault_name": "Renamed"}))
    assert g.name == "Renamed"
    with pytest.raises(ImportRejected, match="must be chosen"):
        import_file("fault_sticks", p, scripted({**OK, "fault_name": " "}))


def test_notes_about_stick_grouping_and_time_are_shown(tmp_path):
    prev = inspect("fault_sticks", write(tmp_path, "t.csv", "x,y,twt,fault\n0,0,1,A\n1,0,2,A\n"))
    assert any("ONE polyline" in w for w in prev.warnings) and any("time-domain" in w for w in prev.warnings)
    assert dict(prev.found)["Sticks are identified by"].startswith("nothing")


def test_import_into_a_project_converts_units_and_stores_faults(tmp_path):
    pr = build()
    p = sticks_csv(tmp_path, ft=0.3048)                               # file in feet
    res = import_file("fault_sticks", p, scripted({"xy_unit": "ft_intl", "z_unit": "ft_intl", "z_is": "depth"}), project=pr)
    assert res.kind == "fault_sticks" and list(pr.faults) == ["A", "B"] and len(res.applied) == 2
    a = pr.faults["A"]
    assert a.sticks[0][0] == pytest.approx([5000.0, 6000.0, 0.0], abs=1e-6)
    assert a.fit().dip_deg == pytest.approx(60, abs=1e-6) and pr.faults["B"].fit().dip_deg == pytest.approx(85, abs=1e-6)
    assert a.info["xy_unit"] == "ft_intl" and a.source == "f"


def test_elevation_is_negated(tmp_path):
    pr = Project()
    p = write(tmp_path, "e.csv", "x,y,z,fault\n0,0,-100,A\n0,0,-300,A\n10,0,-100,B\n10,0,-300,B\n")
    import_file("fault_sticks", p, scripted({**OK, "z_is": "elevation"}), project=pr)
    assert pr.faults["A"].sticks[0][:, 2].tolist() == [100.0, 300.0]


def test_conflicts_replace_skip_suffix_and_identical(tmp_path):
    pr = Project()
    p = sticks_csv(tmp_path)
    import_file("fault_sticks", p, scripted(OK), project=pr)
    base = dict(pr.faults)
    prev = inspect("fault_sticks", p, pr)
    cs = conflicts_for(prev, OK)
    assert [c.key for c in cs] == ["fault|A", "fault|B"] and all(c.identical for c in cs)
    assert "3 stick(s)" in cs[0].existing
    with pytest.raises(ImportRejected, match="choose replace"):
        import_file("fault_sticks", p, scripted(OK), project=pr)
    # a different geometry is not "identical"
    p2 = sticks_csv(tmp_path, "g.csv", faults=(("A", 50, 90), ("C", 70, 20)))
    cs2 = conflicts_for(inspect("fault_sticks", p2, pr), OK)
    assert [c.key for c in cs2] == ["fault|A"] and not cs2[0].identical
    r = import_file("fault_sticks", p2, scripted({**OK, "conflict_actions": {"fault|A": Action("skip")}}), project=pr)
    assert pr.faults["A"] is base["A"] and "C" in pr.faults and "not imported" in r.applied[0]
    import_file("fault_sticks", p2, scripted({**OK, "conflict_actions": {"fault|A": Action("suffix", "_v2"), "fault|C": Action("skip")}}), project=pr)
    assert pr.faults["A_v2"].fit().dip_deg == pytest.approx(50, abs=1e-6) and pr.faults["A"] is base["A"]
    import_file("fault_sticks", p2, scripted({**OK, "conflict_actions": {"fault|A": Action("replace"), "fault|C": Action("skip")}}), project=pr)
    assert pr.faults["A"].fit().dip_deg == pytest.approx(50, abs=1e-6)
    with pytest.raises(ImportRejected, match="already exists"):
        import_file("fault_sticks", p2, scripted({**OK, "conflict_actions": {"fault|A": Action("suffix", "_v2"), "fault|C": Action("skip")}}), project=pr)
    with pytest.raises(ImportRejected, match="not in conflict"):
        import_file("fault_sticks", p2, scripted({**OK, "conflict_actions": {"fault|A": Action("skip"), "fault|Z": Action("skip")}}), project=pr)


def test_suffix_may_not_collide_with_another_fault_in_the_same_file(tmp_path):
    pr = Project()
    pr.faults["A"] = Fault("A", plane_sticks(60, 90, [0, 300]))
    p = write(tmp_path, "x.csv", "x,y,z,fault\n0,0,1,A\n0,0,5,A\n9,0,1,A\n9,0,5,A\n0,0,1,A2\n0,0,5,A2\n9,0,1,A2\n9,0,5,A2\n"
                                 )
    with pytest.raises(ImportRejected, match="already exists"):
        import_file("fault_sticks", p, scripted({**OK, "conflict_actions": {"fault|A": Action("suffix", "2")}}), project=pr)


def test_attach_refuses_duplicates_and_unanswered_conflicts():
    pr = Project()
    f = Fault("A", plane_sticks(60, 90, [0, 300]))
    pr.attach_faults([f])
    with pytest.raises(ProjectError, match="choose replace"):
        pr.attach_faults([f])
    with pytest.raises(ProjectError, match="same name"):
        pr.attach_faults([f, f], {})
    pr.remove_fault("A")
    assert pr.faults == {}
    with pytest.raises(ProjectError, match="no fault"):
        pr.remove_fault("A")


def test_without_a_project_the_faults_are_returned(tmp_path):
    fs = import_file("fault_sticks", sticks_csv(tmp_path), scripted(OK))
    assert [f.name for f in fs] == ["A", "B"]


def test_advice_dip_single_stick_and_position(tmp_path):
    pr = build()
    pr.attach_surface("base", Surface(0, 0, 100.0, 100.0, 0.0, 1, np.full((20, 20), 1000.0)), "b")     # x, y in 0..1900, depth 1000
    u = pr.chart.match_name("Res")
    pr.attach_surface(u, Surface(0, 0, 100.0, 100.0, 0.0, 1, np.full((20, 20), 800.0)), "t")
    sticks = {"Steep": plane_sticks(88, 90, [0, 300], length=30.0, origin=(500.0, 500.0)),
              "Gentle": plane_sticks(40, 90, [0, 300], origin=(500.0, 500.0)),
              "Lone": plane_sticks(70, 90, [0], origin=(500.0, 500.0)),
              "Faraway": plane_sticks(88, 90, [0, 300], length=30.0, origin=(90000.0, 90000.0))}
    rows = ["x,y,depth,fault,stick"]
    for n, ss in sticks.items():
        for k, s in enumerate(ss):
            rows += [f"{float(x)!r},{float(y)!r},{float(z) + 700!r},{n},{k}" for x, y, z in s]
    p = write(tmp_path, "adv.csv", "\n".join(rows) + "\n")
    prev = inspect("fault_sticks", p, pr)
    adv = advice_for(prev, OK)
    text = "\n".join(adv)
    assert "Gentle: dip is 40 deg but v1 treats faults as vertical" in text and "moves sideways by" in text
    assert "Lone: dip not determined" in text and "seismic section" in text
    assert "Faraway: lies outside every surface" in text
    assert "Steep" not in text                                              # a near-vertical fault with good data raises nothing
    # depths nowhere near the surfaces
    deep = write(tmp_path, "deep.csv", "x,y,z,fault\n500,500,5000,D\n500,500,6000,D\n600,500,5000,D\n600,500,6000,D\n")
    assert "do not overlap the project's surfaces" in "\n".join(advice_for(inspect("fault_sticks", deep, pr), OK))


def test_shift_number_in_the_warning_is_right(tmp_path):
    p = sticks_csv(tmp_path, faults=(("Gentle", 45, 0),))
    adv = advice_for(inspect("fault_sticks", p), OK)
    z = 1000.0 * math.tan(math.radians(45))
    assert any(f"({z:.0f} m) the fault moves sideways by {z:.0f} m" in a for a in adv)


def test_unreadable_files_are_reported_at_inspect(tmp_path):
    from cosma.faults import FaultError
    with pytest.raises(FaultError, match="no column for z"):
        inspect("fault_sticks", write(tmp_path, "b.csv", "x,y,fault\n1,2,A\n"))
