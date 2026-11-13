"""Confirmation layer: previews show what the file says, nothing is guessed, nothing runs unconfirmed."""

from __future__ import annotations

import json
import os

import numpy as np

import pytest

from cosma import imports as imp
from cosma.imports import ImportCancelled, ImportRejected, import_file, inspect, scripted
from cosma.wells.io import read_deviation, read_well_headers
from test_surfaces import write_irap

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def _param(prev, key):
    return next(p for p in prev.params if p.key == key)


def _found(prev):
    return dict(prev.found)


# -- surface ---------------------------------------------------------------------------------------
@pytest.fixture
def surf(tmp_path):
    p = tmp_path / "top.irap"
    write_irap(p, 5, 4, 1000.0, 2000.0, 10.0, 20.0, 0.0, lambda x, y: 1500.0 + 0.05 * (x - 1000))
    return p


def test_surface_preview_reports_file_facts_without_units(surf):
    prev = inspect("surface", surf)
    f = _found(prev)
    assert f["Nodes (columns x rows)"] == "5 x 4"
    assert f["Origin (x, y) as in file"] == "1000, 2000"
    assert f["Increment (x, y) as in file"] == "10, 20"
    assert f["Value range as in file"] == "1500 to 1502"
    assert f["Defined nodes"] == "20 of 20"
    for k in ("xy_unit", "z_unit"):
        p = _param(prev, k)
        assert p.value is None and p.source == "required"       # never pre-filled
    assert _param(prev, "positive").source == "default"           # shown as a default, to be checked


def test_unit_missing_is_rejected_not_defaulted(surf):
    with pytest.raises(ImportRejected, match="Horizontal unit.*Vertical unit"):
        import_file("surface", surf, scripted({"positive": "down"}))
    with pytest.raises(ImportRejected, match="not one of"):
        import_file("surface", surf, scripted({"xy_unit": "ft", "z_unit": "m"}))
    with pytest.raises(ImportRejected, match="unknown parameter"):
        import_file("surface", surf, scripted({"xy_unit": "m", "z_unit": "m", "bogus": 1}))


def test_cancel_never_runs_the_reader(surf, monkeypatch):
    calls = []
    monkeypatch.setattr(imp, "read_surface", lambda *a, **k: calls.append(1))
    with pytest.raises(ImportCancelled):
        import_file("surface", surf, lambda prev: None)
    assert calls == []
    import_file("surface", surf, scripted({"xy_unit": "m", "z_unit": "m"}))
    assert calls == [1]


def test_confirmed_values_reach_the_reader(surf):
    s = import_file("surface", surf, scripted({"xy_unit": "ft_intl", "z_unit": "ft_intl", "positive": "down"}))
    assert s.xinc == pytest.approx(3.048) and s.values[0, 0] == pytest.approx(1500 * 0.3048)


# -- polygon -----------------------------------------------------------------------------------------
SQ = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
LL = [[-87.0, 0.0], [-86.99, 0.0], [-86.99, 0.01], [-87.0, 0.01], [-87.0, 0.0]]


def _gj(tmp_path, rings):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": f"P{i}"}, "geometry": {"type": "Polygon", "coordinates": [r]}}
        for i, r in enumerate(rings)]}
    return _w(tmp_path, "a.geojson", json.dumps(fc))


def test_polygon_text_needs_no_crs(tmp_path):
    p = _w(tmp_path, "a.csv", "x,y\n0,0\n10,0\n10,10\n0,10\n")
    prev = inspect("polygon", p)
    assert _found(prev)["Polygons"] == "1"
    assert not _param(prev, "target_crs").required
    poly = import_file("polygon", p, scripted({}))
    assert poly.area == pytest.approx(100.0)


def test_polygon_geojson_requires_target_crs_and_shows_file_crs(tmp_path):
    p = _gj(tmp_path, [LL])
    prev = inspect("polygon", p)
    assert "4326" in _found(prev)["Coordinate system in file"]
    assert _param(prev, "target_crs").required and not _param(prev, "source_crs").required
    with pytest.raises(ImportRejected, match="Model CRS"):
        import_file("polygon", p, scripted({}))
    poly = import_file("polygon", p, scripted({"target_crs": "EPSG:32616"}))
    assert 1.0e6 < poly.area < 1.5e6


def test_polygon_several_must_be_chosen(tmp_path):
    p = _gj(tmp_path, [SQ, [[20, 20], [50, 20], [50, 50], [20, 50], [20, 20]]])
    prev = inspect("polygon", p)
    assert _found(prev)["Polygons"] == "2"
    assert "P1" in " ".join(k for k, _ in prev.found)
    with pytest.raises(ImportRejected, match="Polygon to use"):
        import_file("polygon", p, scripted({"target_crs": "EPSG:32616", "source_crs": "EPSG:32616"}))
    poly = import_file("polygon", p, scripted({"which": "largest", "source_crs": "EPSG:32616", "target_crs": "EPSG:32616"}))
    assert poly.area == pytest.approx(900.0)
    poly0 = import_file("polygon", p, scripted({"which": "0", "source_crs": "EPSG:32616", "target_crs": "EPSG:32616"}))
    assert poly0.area == pytest.approx(100.0)


# -- wells -------------------------------------------------------------------------------------------------
from cosma.project import Action, Project, WellLink  # noqa: E402

REPL, SKIP = Action("replace"), Action("skip")
from cosma.wells.io import WellDataError  # noqa: E402

HEADERS = "Well,X,Y,KB,TD\nW-1,1000,2000,25,1500\nW-2,3000,2000,30,\n"
DEV = "well,md,incl,azim\nW-1,0,0,0\nW-1,500,30,90\nW-1,1500,30,90\n"
UM = {"xy_unit": "m", "z_unit": "m"}


def _project(tmp_path, dev=True):
    pr = Project()
    import_file("well_headers", _w(tmp_path, "h.csv", HEADERS), scripted(UM), project=pr)
    if dev:
        import_file("deviation", _w(tmp_path, "d.csv", DEV), scripted({"z_unit": "m", "azimuth_reference": "grid"}), project=pr)
    return pr


def test_headers_preview_and_import(tmp_path):
    p = _w(tmp_path, "h.csv", HEADERS)
    prev = inspect("well_headers", p, Project())
    f = _found(prev)
    assert f["Wells"] == "W-1, W-2" and f["KB range as in file"] == "25 to 30"
    assert "kb <- 'kb'" in f["Columns recognised"]
    assert any("ABOVE mean sea level" in w for w in prev.warnings)
    assert all(l.mode == "new" for l, _ in prev.links.values())          # empty project: everything is new
    pr = Project()
    res = import_file("well_headers", p, scripted({"xy_unit": "m", "z_unit": "ft_intl"}), project=pr)
    assert pr.names() == ["W-1", "W-2"] and pr.wells["W-1"].header["kb"] == pytest.approx(25 * 0.3048)
    assert len(res.applied) == 2


def test_headers_for_existing_well_replace_it_and_say_so(tmp_path):
    pr = _project(tmp_path)
    p2 = _w(tmp_path, "h2.csv", "well,x,y,kb\nW-1,1100,2000,40\n")
    prev = inspect("well_headers", p2, pr)
    assert prev.links["W-1"][0] == WellLink("existing", "W-1")
    with pytest.raises(ImportRejected, match="W-1 header.*choose replace, skip"):
        import_file("well_headers", p2, scripted(UM), project=pr)                  # a choice is forced
    assert pr.wells["W-1"].header["x"] == 1000.0
    res = import_file("well_headers", p2, scripted({**UM, "conflict_actions": {"W-1|header|header": REPL}}), project=pr)
    assert any("header replaced" in n for n in res.notes)
    # the trajectory follows the new header: derived, not stored
    assert pr.wells["W-1"].trajectory.x[0] == 1100.0 and pr.wells["W-1"].trajectory.kb == 40.0


def test_deviation_convergence_only_when_true(tmp_path):
    pr = _project(tmp_path, dev=False)
    dp = _w(tmp_path, "d.csv", DEV)
    prev = inspect("deviation", dp, pr)
    assert _found(prev)["Stations per well"] == "W-1: 3"
    assert _param(prev, "azimuth_reference").value is None
    with pytest.raises(ImportRejected, match="Azimuth is relative to"):
        import_file("deviation", dp, scripted({"z_unit": "m"}), project=pr)
    with pytest.raises(ImportRejected, match="Grid convergence"):
        import_file("deviation", dp, scripted({"z_unit": "m", "azimuth_reference": "true"}), project=pr)
    import_file("deviation", dp, scripted({"z_unit": "m", "azimuth_reference": "grid"}), project=pr)
    x_grid = pr.wells["W-1"].trajectory.x[-1]
    import_file("deviation", dp, scripted({"z_unit": "m", "azimuth_reference": "true", "convergence_deg": "2",
                                           "conflict_actions": {"W-1|survey|survey": REPL}}), project=pr)
    assert pr.wells["W-1"].trajectory.x[-1] != x_grid
    with pytest.raises(ImportRejected, match="needs project"):
        import_file("deviation", dp, scripted({}))


def test_data_order_does_not_matter(tmp_path):
    """Survey first, header later: the trajectory appears when both exist."""
    pr = Project()
    res = import_file("deviation", _w(tmp_path, "d.csv", DEV), scripted({"z_unit": "m", "azimuth_reference": "grid"}), project=pr)
    assert any("no header yet" in n for n in res.notes) and pr.wells["W-1"].trajectory is None
    h = inspect("well_headers", _w(tmp_path, "h.csv", HEADERS), pr)
    assert h.links["W-1"][0] == WellLink("existing", "W-1")                 # matched by name
    import_file("well_headers", _w(tmp_path, "h.csv", HEADERS), scripted(UM), project=pr)
    tr = pr.wells["W-1"].trajectory
    R, th = 500.0 / (np.pi / 6), np.pi / 6               # 0 -> 30 deg build over 500 m, then a 1000 m tangent
    assert tr.x[-1] == pytest.approx(1000.0 + R * (1 - np.cos(th)) + 1000.0 * np.sin(th), abs=1e-6)
    assert tr.x[0] == 1000.0 and tr.y[0] == 2000.0


def test_tops_core_and_unplaced_storage(tmp_path):
    pr = _project(tmp_path)
    tp = _w(tmp_path, "t.csv", "well,top,depth\nW-1,Seal,1000\nW-1,Res,1200\n")
    prev = inspect("tops", tp, pr)
    assert _found(prev)["Depth range as in file"] == "1000 to 1200" and _param(prev, "depth_type").value is None
    res = import_file("tops", tp, scripted({"z_unit": "m", "depth_type": "md"}), project=pr)
    assert [t.name for t in pr.wells["W-1"].resolved_tops()] == ["Seal", "Res"]
    res2 = import_file("tops", _w(tmp_path, "t2.csv", "well,top,depth\nW-1,Seal,1010\n"),
                       scripted({"z_unit": "m", "depth_type": "md", "conflict_actions": {"W-1|top|Seal": REPL}}), project=pr)
    assert any("top Seal replaced" in n for n in res2.notes)
    assert next(t.md for t in pr.wells["W-1"].resolved_tops() if t.name == "Seal") == 1010.0

    cp = _w(tmp_path, "c.csv", "well,md,porosity,perm_mD\nW-1,1000,12.5,50\nW-1,1100,20,300\n")
    prev = inspect("core", cp, pr)
    assert _found(prev)["Porosity columns"] == "porosity"
    dt = _param(prev, "depth_type")
    assert dt.value == "md" and dt.source == "file"
    with pytest.raises(ImportRejected, match="Porosity is in"):
        import_file("core", cp, scripted({"z_unit": "m"}), project=pr)
    import_file("core", cp, scripted({"z_unit": "m", "porosity_unit": "percent"}), project=pr)
    assert pr.wells["W-1"].resolved_core()["c"].columns["porosity"][0] == pytest.approx(0.125)
    assert _param(inspect("core", _w(tmp_path, "c2.csv", "well,depth,porosity\nW-1,1000,0.1\n"), pr), "depth_type").value is None

    # picks for a well that has no header or survey yet are kept, flagged, and placed later
    res3 = import_file("tops", _w(tmp_path, "t3.csv", "well,top,depth\nW-9,Base,2000\n"),
                       scripted({"z_unit": "m", "depth_type": "md"}), project=pr)
    assert any("unplaced" in n and "header" in n for n in res3.notes)
    with pytest.raises(Exception, match="needs a header"):
        pr.wells["W-9"].resolved_tops()


def test_picks_follow_a_corrected_survey(tmp_path):
    """Stale-coordinate regression: a pick given as TVD moves in MD when the survey changes."""
    pr = _project(tmp_path)
    R, th = 500.0 / (np.pi / 6), np.pi / 6
    tvd_at_1000 = float(R * np.sin(th) + 500.0 * np.cos(th))  # build section, then 500 m of tangent
    import_file("tops", _w(tmp_path, "t.csv", f"well,top,depth\nW-1,Seal,{tvd_at_1000!r}\n"),
                scripted({"z_unit": "m", "depth_type": "tvd"}), project=pr)
    assert pr.wells["W-1"].resolved_tops()[0].md == pytest.approx(1000.0, abs=1e-6)
    steeper = "well,md,incl,azim\nW-1,0,0,0\nW-1,100,0,0\nW-1,1500,0,0\n"             # now vertical
    import_file("deviation", _w(tmp_path, "d2.csv", steeper),
                scripted({"z_unit": "m", "azimuth_reference": "grid", "conflict_actions": {"W-1|survey|survey": REPL}}), project=pr)
    assert pr.wells["W-1"].resolved_tops()[0].md == pytest.approx(tvd_at_1000, abs=1e-6)


# -- linking -----------------------------------------------------------------------------------------------------
def test_similar_name_is_suggested_and_user_can_override(tmp_path):
    pr = _project(tmp_path)
    tp = _w(tmp_path, "t.csv", "well,top,depth\nW1,Seal,1000\n")
    prev = inspect("tops", tp, pr)
    sug, note = prev.links["W1"]
    assert sug == WellLink("existing", "W-1") and "similar name" in note
    vals = {"z_unit": "m", "depth_type": "md"}
    import_file("tops", tp, scripted({**vals, "well_links": {"W1": WellLink("new", "W1")}}), project=pr)
    assert "W1" in pr.wells and "Seal" in pr.wells["W1"].tops and "Seal" not in pr.wells["W-1"].tops
    pr2 = _project(tmp_path)
    import_file("tops", tp, scripted(vals), project=pr2)                      # accept the suggestion
    assert "Seal" in pr2.wells["W-1"].tops and "W1" not in pr2.wells


def test_unrelated_name_defaults_to_new_well(tmp_path):
    pr = _project(tmp_path)
    prev = inspect("tops", _w(tmp_path, "t.csv", "well,top,depth\nALPHA,Seal,1000\n"), pr)
    assert prev.links["ALPHA"][0] == WellLink("new", "ALPHA")


@pytest.mark.parametrize("links, msg", [
    ({"W1": ("existing", "NOPE")}, "not a well in the project"),
    ({"W1": ("new", "W-2")}, "already exists"),
    ({"W1": ("new", "  ")}, "needs a name"),
    ({"W1": ("sideways", "x")}, "choose 'existing' or 'new'"),
    ({}, "connect it to an existing well or import it as a new well"),
    ({"W1": ("existing", "W-1"), "W-1": ("existing", "W-1")}, "links given for wells not in the file"),
])
def test_link_validation(tmp_path, links, msg):
    pr = _project(tmp_path)
    tp = _w(tmp_path, "t.csv", "well,top,depth\nW1,Seal,1000\n")
    with pytest.raises(ImportRejected, match=msg):
        import_file("tops", tp, scripted({"z_unit": "m", "depth_type": "md", "well_links": links}), project=pr)
    assert "Seal" not in pr.wells["W-1"].tops


def test_two_file_wells_cannot_merge_into_one(tmp_path):
    pr = _project(tmp_path)
    tp = _w(tmp_path, "t.csv", "well,top,depth\nA,Seal,1000\nB,Res,1100\n")
    with pytest.raises(ImportRejected, match="would both become 'W-1'"):
        import_file("tops", tp, scripted({"z_unit": "m", "depth_type": "md",
                                          "well_links": {"A": ("existing", "W-1"), "B": ("existing", "W-1")}}), project=pr)


def test_failed_import_leaves_project_untouched(tmp_path):
    pr = _project(tmp_path)
    tp = _w(tmp_path, "t.csv", "well,top,depth\nW-1,Seal,1000\nW-1,Deep,99999\nNEW,X,5\n")
    with pytest.raises(WellDataError, match="outside surveyed"):
        import_file("tops", tp, scripted({"z_unit": "m", "depth_type": "md"}), project=pr)
    assert pr.wells["W-1"].tops == {} and "NEW" not in pr.wells


LAS = """~VERSION INFORMATION
 VERS.   2.0 : CWLS
 WRAP.   NO  :
~WELL INFORMATION
 STRT.{u}  1000.0 :
 STOP.{u}  1001.0 :
 STEP.{u}  0.5 :
 NULL.  -999.25 :
 WELL.  {w} :
~CURVE INFORMATION
 DEPT.{u}   : Depth
 GR  .GAPI  : Gamma
~ASCII
1000.0 55.0
1000.5 -999.25
1001.0 60.0
"""


def test_las_unit_from_file_or_required_and_well_link(tmp_path):
    pr = _project(tmp_path)
    m = _w(tmp_path, "m.las", LAS.format(u="M", w="W-1"))
    prev = inspect("las", m, pr)
    f = _found(prev)
    assert f["Well"] == "W-1" and f["Samples"] == "3" and "GR [GAPI]" in f["Curves"]
    assert (_param(prev, "depth_unit").value, _param(prev, "depth_unit").source) == ("m", "file")
    assert prev.file_wells == ["W-1"] and prev.links["W-1"][0] == WellLink("existing", "W-1")
    import_file("las", m, scripted({}), project=pr)
    assert pr.wells["W-1"].logs["m"].md[0] == 1000.0
    with pytest.raises(ImportRejected, match="W-1 curve GR: choose"):
        import_file("las", m, scripted({}), project=pr)              # same curve, same depths: must choose
    res = import_file("las", m, scripted({"conflict_actions": {"W-1|curve|GR": REPL}}), project=pr)
    assert any("curve GR replaced" in n for n in res.notes) and list(pr.wells["W-1"].logs) == ["m"]

    other = _w(tmp_path, "o.las", LAS.format(u="M", w="W-77"))
    import_file("las", other, scripted({}), project=pr)
    assert "W-77" in pr.wells and pr.wells["W-77"].header is None             # logs-only well is allowed

    ft = _w(tmp_path, "f.las", LAS.format(u="FT", w="W-1"))
    p2 = inspect("las", ft, pr)
    assert _param(p2, "depth_unit").value is None and "FT" in _param(p2, "depth_unit").help
    with pytest.raises(ImportRejected, match="Unit of the depth index"):
        import_file("las", ft, scripted({}), project=pr)


def test_every_kind_is_covered():
    assert set(imp._INSPECT) == {"surface", "polygon", "well_headers", "deviation", "tops", "core", "las", "variogram", "property_map", "fault_sticks"}
    with pytest.raises(ImportRejected, match="unknown import kind"):
        inspect("seismic", "x")


# -- the dialog ---------------------------------------------------------------------------------------------
qt = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


def test_dialog_blocks_ok_until_required_chosen(app, surf):
    from cosma.gui.import_dialog import ImportDialog

    d = ImportDialog(inspect("surface", surf))
    ok = d.buttons.button(qt.QDialogButtonBox.Ok)
    assert d.table.rowCount() == len(d.preview.found) and d.table.item(1, 1).text() == "5 x 4"
    assert not ok.isEnabled() and "must be chosen" in d.problem.text()
    d.widgets["xy_unit"].setCurrentText("m")
    assert not ok.isEnabled()
    d.widgets["z_unit"].setCurrentText("m")
    assert ok.isEnabled() and d.problem.text() == ""
    assert d.values() == {"xy_unit": "m", "z_unit": "m", "positive": "down"}
    assert "default" in d.rows["positive"].text()


def test_dialog_conditional_field_and_warnings(app, tmp_path):
    from cosma.gui.import_dialog import ImportDialog

    d = ImportDialog(inspect("deviation", _w(tmp_path, "d.csv", DEV)))
    d.widgets["z_unit"].setCurrentText("m")
    d.widgets["azimuth_reference"].setCurrentText("grid")
    assert d.widgets["convergence_deg"].isHidden()
    assert d.buttons.button(qt.QDialogButtonBox.Ok).isEnabled()
    d.widgets["azimuth_reference"].setCurrentText("true")
    assert not d.widgets["convergence_deg"].isHidden()
    assert not d.buttons.button(qt.QDialogButtonBox.Ok).isEnabled()
    d.widgets["convergence_deg"].setText("1.5")
    assert d.buttons.button(qt.QDialogButtonBox.Ok).isEnabled()
    assert finalize_ok(d)

    h = ImportDialog(inspect("well_headers", _w(tmp_path, "h.csv", HEADERS)))
    assert "ABOVE mean sea level" in h.warn.text()


def finalize_ok(d):
    return imp.finalize(d.preview, d.values())["convergence_deg"] == 1.5


def test_dialog_accept_and_cancel_through_import_file(app, surf, monkeypatch):
    from cosma.gui import import_dialog as idlg

    # drive the modal dialog without a user: pre-fill, then accept / reject
    def run(accept):
        def fake_exec(self):
            self.widgets["xy_unit"].setCurrentText("m")
            self.widgets["z_unit"].setCurrentText("m")
            if accept:
                self.accept()
                return qt.QDialog.Accepted
            self.reject()
            return qt.QDialog.Rejected
        monkeypatch.setattr(idlg.ImportDialog, "exec", fake_exec)
        return import_file("surface", surf, idlg.confirm_with_dialog)

    s = run(True)
    assert s.ncol == 5
    with pytest.raises(ImportCancelled):
        run(False)


def test_shapefile_without_prj_demands_source_crs(tmp_path):
    import shapefile

    base = tmp_path / "aoi"
    with shapefile.Writer(str(base), shapeType=shapefile.POLYGON) as w:
        w.field("NAME", "C")
        w.poly([[(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]])    # clockwise outer ring, as the spec wants
        w.record("A")
    prev = inspect("polygon", base.with_suffix(".shp"))
    assert _found(prev)["Coordinate system in file"] == "none found"
    assert _param(prev, "source_crs").required and _param(prev, "source_crs").source == "required"
    with pytest.raises(ImportRejected, match="Source CRS"):
        import_file("polygon", base.with_suffix(".shp"), scripted({"target_crs": "EPSG:32616"}))
    poly = import_file("polygon", base.with_suffix(".shp"),
                       scripted({"source_crs": "EPSG:32616", "target_crs": "EPSG:32616"}))
    assert poly.area == pytest.approx(100.0)


def test_dialog_well_link_table(app, tmp_path):
    from cosma.gui.import_dialog import _NEW, ImportDialog

    pr = _project(tmp_path)
    tp = _w(tmp_path, "t.csv", "well,top,depth\nW1,Seal,1000\nALPHA,Res,1100\nW-2,Base,1200\n")
    d = ImportDialog(inspect("tops", tp, pr))
    ok = d.buttons.button(qt.QDialogButtonBox.Ok)
    assert d.link_table.rowCount() == 3
    c1, n1 = d.link_rows["W1"]
    c2, n2 = d.link_rows["ALPHA"]
    c3, n3 = d.link_rows["W-2"]
    assert (c1.currentText(), c2.currentText(), c3.currentText()) == ("W-1", _NEW, "W-2")   # suggestions
    assert "similar name" in d.link_table.item(0, 3).text()
    assert not n1.isEnabled() and n2.isEnabled() and not n3.isEnabled()
    d.widgets["z_unit"].setCurrentText("m")
    d.widgets["depth_type"].setCurrentText("md")
    assert ok.isEnabled()
    # toggle W1 to a new well: its name field wakes up, and the same name as an existing well blocks Import
    c1.setCurrentText(_NEW)
    assert n1.isEnabled()
    c1.setCurrentText("W-2")
    assert not n1.isEnabled()                                      # connecting to an existing well locks the name
    c1.setCurrentText(_NEW)
    n1.setText("W-2")
    assert not ok.isEnabled() and "already exists" in d.problem.text()
    n1.setText("W-1B")
    assert ok.isEnabled()
    # two file wells onto one target blocks Import
    c2.setCurrentText("W-2")
    assert not ok.isEnabled() and "would both become" in d.problem.text()
    c2.setCurrentText(_NEW)
    n2.setText("")
    assert not ok.isEnabled() and "needs a name" in d.problem.text()
    n2.setText("ALPHA")
    assert d.values()["well_links"] == {"W1": WellLink("new", "W-1B"), "ALPHA": WellLink("new", "ALPHA"),
                                        "W-2": WellLink("existing", "W-2")}


def test_dialog_end_to_end_connects_to_existing_well(app, tmp_path, monkeypatch):
    from cosma.gui import import_dialog as idlg

    pr = _project(tmp_path)
    tp = _w(tmp_path, "t.csv", "well,top,depth\nSOMETHING,Seal,1000\n")

    def fake_exec(self):
        self.widgets["z_unit"].setCurrentText("m")
        self.widgets["depth_type"].setCurrentText("md")
        combo, _ = self.link_rows["SOMETHING"]
        assert combo.currentText() == idlg._NEW                    # nothing similar: default is a new well
        combo.setCurrentText("W-2")                                 # the user connects it by hand
        self.accept()
        return qt.QDialog.Accepted

    monkeypatch.setattr(idlg.ImportDialog, "exec", fake_exec)
    res = import_file("tops", tp, idlg.confirm_with_dialog, project=pr)
    assert "Seal" in pr.wells["W-2"].tops and "SOMETHING" not in pr.wells
    assert res.applied == ["SOMETHING -> W-2: 1 of 1 tops imported"]


def test_project_api_enforces_links_itself(tmp_path):
    """The Project does not rely on the dialog layer having validated the links."""
    from cosma.project import ProjectError
    from cosma.wells.model import RawTop

    pr = _project(tmp_path)
    pick = {"A": [RawTop("S", 1000.0, "md")], "B": [RawTop("S", 1100.0, "md")]}
    with pytest.raises(ProjectError, match="would both go to well"):
        pr.apply_tops(pick, {"A": WellLink("existing", "W-1"), "B": WellLink("existing", "W-1")})
    with pytest.raises(ProjectError, match="already exists"):
        pr.apply_tops({"A": pick["A"]}, {"A": WellLink("new", "W-2")})
    with pytest.raises(ProjectError, match="no such well"):
        pr.apply_tops({"A": pick["A"]}, {"A": WellLink("existing", "NOPE")})
    with pytest.raises(ProjectError, match="unknown link mode"):
        pr.apply_tops({"A": pick["A"]}, {"A": WellLink("maybe", "W-1")})
    assert pr.wells["W-1"].tops == {}


def test_las_batch_reuses_choices_and_asks_again_when_needed(tmp_path):
    from cosma.imports import import_las_batch
    pr = _project(tmp_path)
    a = _w(tmp_path, "a.las", LAS.format(u="M", w="W-1"))
    b = _w(tmp_path, "b.las", LAS.format(u="M", w="W-77"))
    c = _w(tmp_path, "c.las", LAS.format(u="FT", w="W-88"))                    # unit not declared in metres -> own question
    asked = []

    def confirm(prev):
        asked.append(prev.path.name)
        v = prev.initial_values()
        v["depth_unit"] = "ft_intl" if prev.path.name == "c.las" else "m"
        return v

    res = import_las_batch([a, b, c], confirm, project=pr)
    assert asked == ["a.las", "c.las"]
    assert [h for _, _, h in res.files] == ["asked", "same choices as a.las", "asked"]
    assert {"W-1", "W-77", "W-88"} <= set(pr.wells) and set(pr.wells["W-88"].logs) == {"c"}
    assert pr.wells["W-88"].logs["c"].md[0] == pytest.approx(1000.0 * 0.3048)


def test_las_batch_is_all_or_nothing(tmp_path):
    from cosma.imports import import_las_batch
    pr = _project(tmp_path)
    a = _w(tmp_path, "a.las", LAS.format(u="M", w="W-5"))
    b = _w(tmp_path, "b.las", LAS.format(u="FT", w="W-6"))
    before = set(pr.wells)
    with pytest.raises(ImportCancelled):
        import_las_batch([a, b], lambda prev: prev.initial_values() if prev.path.name == "a.las" else None, project=pr)
    assert set(pr.wells) == before                                         # a.las was rolled back
    with pytest.raises(ImportRejected, match="twice"):
        import_las_batch([a, a], scripted({}), project=pr)
    with pytest.raises(ImportRejected, match="no LAS"):
        import_las_batch([], scripted({}), project=pr)
    bad = _w(tmp_path, "bad.las", "not a las file")
    with pytest.raises(Exception):
        import_las_batch([a, bad], scripted({}), project=pr)
    assert set(pr.wells) == before


def test_las_batch_clash_with_existing_data_asks(tmp_path):
    from cosma.imports import import_las_batch
    pr = _project(tmp_path)
    a = _w(tmp_path, "a.las", LAS.format(u="M", w="W-1"))
    import_las_batch([a], scripted({}), project=pr)
    b = _w(tmp_path, "b.las", LAS.format(u="M", w="W-9"))
    asked = []
    def confirm(prev):
        asked.append(prev.path.name)
        v = prev.initial_values()
        if prev.path.name == "a.las":
            v["conflict_actions"] = {"W-1|curve|GR": REPL}
        return v
    import_las_batch([b, a], confirm, project=pr)
    assert asked == ["b.las", "a.las"]                                      # a.las clashes with its own earlier import
