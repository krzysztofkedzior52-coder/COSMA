"""Built-in CRS list, suggestions for polygon files, the CRS box in the import dialog, and the project's CRS."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cosma import crs_catalog as cc
from cosma.imports import ImportRejected, import_file, inspect, scripted
from cosma.persist import load_project, save_project
from cosma.project import Project
from test_main_window import app  # noqa: F401
from test_shapefile_kml import cw, write_shp

CHI = [[-87.70, 41.80], [-87.60, 41.80], [-87.60, 41.90], [-87.70, 41.90], [-87.70, 41.80]]


def _gj(tmp_path, ring=CHI, crs=None, name="a.geojson"):
    obj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                                                         "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    if crs:
        obj["crs"] = {"type": "name", "properties": {"name": crs}}
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def test_catalog_codes_all_exist_with_areas_and_no_duplicates():
    cat = cc.catalog()
    codes = [e.code for e in cat]
    assert len(codes) == len(set(codes)) and len(cat) > 200
    assert all(e.bounds is not None or e.code in {"EPSG:3857", "EPSG:3395"} for e in cat), [e.code for e in cat if e.bounds is None]
    for c in ("EPSG:4326", "EPSG:32616", "EPSG:32760", "EPSG:26916", "EPSG:25832", "EPSG:28992", "EPSG:7856", "EPSG:27700"):
        assert cc.entry(c) is not None, c
    assert "UTM zone 16N" in cc.entry("EPSG:32616").name


def test_normalise_accepts_codes_labels_and_rejects_nonsense():
    assert cc.normalise("32616") == cc.normalise("epsg:32616") == "EPSG:32616"
    assert cc.normalise(cc.entry("EPSG:32616").label + "   [suggested]") == "EPSG:32616"
    assert cc.normalise("  ") is None and cc.normalise(None) is None
    with pytest.raises(ValueError, match="not a coordinate system"):
        cc.normalise("banana")


def test_utm_zone_and_datum_family_codes():
    assert cc.utm_zone(-87.45, 42.0) == (16, "N") and cc.utm_zone(151.2, -33.9) == (56, "S") and cc.utm_zone(179.9, 1) == (60, "N")
    assert cc._utm_code("nad83", 16, "N") == 26916 and cc._utm_code("etrs89", 32, "N") == 25832
    assert cc._utm_code("gda2020", 56, "S") == 7856 and cc._utm_code("nad83", 30, "N") == 32630     # no NAD83 zone 30: WGS 84
    assert cc._utm_code("other", 33, "S") == 32733


def test_suggestions_for_lonlat_polygon_without_crs_are_listed_not_preselected():
    xy = np.array(CHI)
    sg, note = cc.source_suggestions(xy, None)
    assert [s.code for s in sg] == ["EPSG:4326", "EPSG:4269"] and not any(s.strong for s in sg) and "datum" in note
    t = cc.target_suggestions(xy, "EPSG:4326")
    assert t[0].code == "EPSG:32616" and "UTM zone 16N" in t[0].reason and not t[0].strong
    t2 = cc.target_suggestions(xy, "EPSG:4269")
    assert [s.code for s in t2] == ["EPSG:26916", "EPSG:32616"]                  # same datum first, WGS 84 as the alternative


def test_projected_numbers_cannot_name_a_zone():
    xy = np.array([[600000.0, 4700000.0], [601000.0, 4700500.0]])
    sg, note = cc.source_suggestions(xy, None)
    assert sg == [] and "cannot be read from the numbers" in note
    sg, _ = cc.source_suggestions(xy, None, "EPSG:32616")
    assert [s.code for s in sg] == ["EPSG:32616"] and not sg[0].strong
    far = np.array([[100000.0, 100000.0]])                                      # near the equator, far from zone 16N's area
    assert cc.source_suggestions(far, None, "EPSG:32616")[0] == []              # outside the project CRS's area of use: not offered


def test_target_suggestion_for_projected_file_and_for_project_crs():
    xy = np.array([[600000.0, 4700000.0]])
    t = cc.target_suggestions(xy, "EPSG:32616")
    assert [s.code for s in t] == ["EPSG:32616"] and "no reprojection" in t[0].reason and t[0].strong
    t = cc.target_suggestions(np.array(CHI), "EPSG:4326", "EPSG:26916")
    assert t[0].code == "EPSG:26916" and t[0].strong and t[1].code == "EPSG:32616"


def test_polygon_preview_carries_hints_and_validates_crs_text(tmp_path):
    p = _gj(tmp_path)
    prev = inspect("polygon", p, Project("x"))
    h = prev.crs_hints({"source_crs": "", "target_crs": ""})
    assert h["target_crs"][0][0].code == "EPSG:32616"                           # file is RFC 7946: WGS 84 stated
    assert h["source_crs"] == ([], "")
    pr = Project("x")
    import_file("polygon", p, scripted({"target_crs": "32616"}), project=pr)
    assert pr.crs == "EPSG:32616" and pr.boundary is not None
    with pytest.raises(ImportRejected, match="not a coordinate system"):
        import_file("polygon", p, scripted({"target_crs": "banana"}), project=Project("x"))


def test_old_style_geojson_crs_member_and_missing_source_is_required(tmp_path):
    p = _gj(tmp_path, crs="urn:ogc:def:crs:EPSG::4269", name="n83.geojson")
    prev = inspect("polygon", p, Project("x"))
    assert "4269" in dict(prev.found)["Coordinate system in file"]
    assert prev.crs_hints({})["target_crs"][0][0].code == "EPSG:26916"           # UTM 16N on NAD83 first


def test_project_crs_is_set_by_the_confirmed_target_kept_and_persisted(tmp_path):
    pr = Project("c")
    p = _gj(tmp_path)
    res = import_file("polygon", p, scripted({"target_crs": "EPSG:32616"}), project=pr)
    assert pr.crs == "EPSG:32616" and any("project's CRS is now" in n for n in res.notes)
    prev = inspect("polygon", p, pr)
    assert prev.crs_hints({})["target_crs"][0][0].strong
    assert prev.advisor({"target_crs": "EPSG:26916"}) and not prev.advisor({"target_crs": "EPSG:32616"})
    res2 = import_file("polygon", p, scripted({"target_crs": "EPSG:26916"}), project=pr)
    assert pr.crs == "EPSG:32616" and not any("now" in n for n in res2.notes)    # an existing project CRS is never overwritten
    f = tmp_path / "a.cosma"
    save_project(pr, f)
    assert load_project(f).crs == "EPSG:32616"
    pr0 = Project("old")
    save_project(pr0, tmp_path / "b.cosma")
    assert load_project(tmp_path / "b.cosma").crs is None


def test_dialog_lists_suggestions_first_and_prefills_only_the_stated(app, tmp_path):
    from cosma.gui.import_dialog import ImportDialog
    pr = Project("c")
    p = _gj(tmp_path)
    d = ImportDialog(inspect("polygon", p, pr))
    w = d.widgets["target_crs"]
    items = [w.itemText(i) for i in range(w.count())]
    assert items[0] == "" and items[1].startswith("EPSG:32616") and "[suggested]" in items[1] and len(items) > 200
    assert w.currentText() == ""                                              # a guess is listed, never silently filled in
    assert "UTM zone 16N" in d.crs_label.text()
    assert d.values()["target_crs"] == "" and "Model CRS" in d.problem.text()
    w.setCurrentText(items[1])
    assert d.values()["target_crs"].startswith("EPSG:32616") and d.problem.text() == ""
    w.setCurrentText("32616")                                                 # typed code works too
    assert d.problem.text() == ""
    w.setCurrentText("zzz")
    assert "not a coordinate system" in d.problem.text()
    pr.crs = "EPSG:32616"
    d2 = ImportDialog(inspect("polygon", p, pr))
    assert d2.widgets["target_crs"].currentText().startswith("EPSG:32616") and d2.problem.text() == ""
    d2.reject()
    d.reject()


def test_dialog_target_list_follows_the_typed_source_crs(app, tmp_path):
    from cosma.gui.import_dialog import ImportDialog
    ring = [[-87.70, 41.80], [-87.70, 41.90], [-87.60, 41.90], [-87.60, 41.80]]            # clockwise, shapefile outer ring
    p = write_shp(tmp_path / "noprj", [[ring]])                                            # no .prj: the CRS is not stated
    d = ImportDialog(inspect("polygon", p, Project("c")))
    src, tgt = d.widgets["source_crs"], d.widgets["target_crs"]
    assert src.currentText() == "" and "Choose the source CRS first" in d.crs_label.text()
    assert "[suggested]" in src.itemText(1) and src.itemText(1).startswith("EPSG:4326")
    src.setCurrentText("EPSG:4269")
    assert tgt.itemText(1).startswith("EPSG:26916") and tgt.currentText() == ""
    src.setCurrentText("EPSG:4326")
    assert tgt.itemText(1).startswith("EPSG:32616")
    d.reject()
