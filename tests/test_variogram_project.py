"""Collecting variogram samples per interval, storing variograms on intervals, importing them."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cosma.imports import ImportRejected, import_file, inspect, scripted
from cosma.project import Action, Project, ProjectError, WellRecord
from cosma.variogram import Structure, VariogramModel, experimental_variogram, from_gslib, to_dict
from cosma.variogram_data import collect
from cosma.wells.model import LogSet, RawTop, Survey


def build(tmp_path=None, incl=(0, 0, 0)):
    """Three wells; chart Seal / Res / (base Basement). Res top 100 +10 per well, thickness 50, 60, 70."""
    pr = Project("v")
    tops = {"W-1": (100, 150, 200), "W-2": (110, 170, 220), "W-3": (120, 190, 240)}
    names = ["Seal", "Res", "Basement"]
    for k, (w, ds) in enumerate(tops.items()):
        rec = WellRecord(w)
        rec.header = dict(x=500.0 * k, y=0.0, kb=0.0, td=400.0)
        rec.survey = Survey(np.array([0.0, 400.0]), np.array([incl[k]] * 2, float), np.array([90.0, 90.0]), "grid")
        for n, d in zip(names, ds):
            rec.tops[n] = RawTop(n, float(d), "md")
        md = np.arange(0.0, 400.0, 1.0)
        rec.logs["lg"] = LogSet(md, {"PHI": md % 2.0, "PERM": md * 0 + 5.0})
        pr.wells[w] = rec
    pr.create_chart(["Seal", "Res"], "Basement")
    return pr


def uid(pr, name):
    return pr.chart.match_name(name)


def test_collect_frame_top_and_bounds():
    pr = build()
    d = collect(pr, uid(pr, "Res"), "PHI")
    # Res spans 150-200, 170-220, 190-240 (z measured below the Res top)
    assert d.used == {"W-1": 51, "W-2": 51, "W-3": 51}
    assert d.thickness == {"W-1": 50, "W-2": 50, "W-3": 50}
    for w, x in (("W-1", 0.0), ("W-2", 500.0), ("W-3", 1000.0)):
        p = d.points[d.groups == w]
        assert np.allclose(p[:, 0], x) and p[:, 2].min() == 0 and p[:, 2].max() == 50
    assert np.allclose(d.values[d.groups == "W-1"][:3], [0, 1, 0])      # md 150 -> PHI 0


def test_collect_proportional_stretches_to_mean_thickness():
    pr = build()
    pr.wells["W-2"].tops["Basement"] = RawTop("Basement", 250.0, "md")      # Res in W-2: 170-250 = 80 thick
    d = collect(pr, uid(pr, "Res"), "PHI", frame="proportional")
    assert d.mean_thickness == pytest.approx((50 + 80 + 50) / 3)
    p = d.points[d.groups == "W-2"]
    assert p[:, 2].max() == pytest.approx(d.mean_thickness)
    p1 = d.points[d.groups == "W-1"]
    assert p1[:, 2].max() == pytest.approx(d.mean_thickness)
    d2 = collect(pr, uid(pr, "Res"), "PHI", frame="top")
    assert d2.points[d2.groups == "W-2"][:, 2].max() == pytest.approx(80)


def test_collect_reports_skips_not_silent():
    pr = build()
    del pr.wells["W-2"].tops["Basement"]
    pr.wells["W-3"].logs["lg"] = LogSet(np.arange(0.0, 400.0), {"OTHER": np.zeros(400)})
    d = collect(pr, uid(pr, "Res"), "PHI")
    assert set(d.used) == {"W-1"}
    assert "Basement" in d.skipped["W-2"] and "PHI" in d.skipped["W-3"]
    with pytest.raises(ProjectError, match="no well has"):
        collect(pr, uid(pr, "Res"), "NOPE")


def test_collect_needs_source_when_ambiguous():
    pr = build()
    md = np.arange(0.0, 400.0)
    pr.wells["W-1"].logs["lg2"] = LogSet(md, {"PHI": md * 0.0 + 0.3})
    with pytest.raises(ProjectError, match="several sources"):
        collect(pr, uid(pr, "Res"), "PHI")
    d = collect(pr, uid(pr, "Res"), "PHI", source={"W-1": "lg2"})
    assert np.allclose(d.values[d.groups == "W-1"], 0.3)
    d = collect(pr, uid(pr, "Res"), "PHI", source="lg", wells=["W-2"])
    assert set(d.used) == {"W-2"}


def test_collect_deviated_well_uses_true_vertical_depth_and_position():
    pr = build(incl=(0, 30, 0))
    d = collect(pr, uid(pr, "Res"), "PERM")
    p = d.points[d.groups == "W-2"]
    # a sample 10 m along hole below the top is 10*cos(30) m deeper; azimuth 90 = east
    base = 500.0 + 170.0 * np.sin(np.radians(30)) * 0   # position at the top is read from the trajectory, not assumed
    k = np.argmin(abs(p[:, 2] - 10 * np.cos(np.radians(30))))
    assert p[k, 2] == pytest.approx(10 * np.cos(np.radians(30)), abs=1e-9)
    assert p[1, 0] - p[0, 0] == pytest.approx(np.sin(np.radians(30)))      # 1 m along hole -> 0.5 m east
    assert np.allclose(p[:, 1], 0)


def test_collect_core_and_experimental_end_to_end():
    pr = build()
    d = collect(pr, uid(pr, "Res"), "PHI")
    e = experimental_variogram(d.points, d.values, "vertical", lag=1, nlags=3, groups=d.groups)
    assert e.gamma[0] == pytest.approx(0.5) and e.gamma[1] == pytest.approx(0.0)       # alternating 0/1 every metre
    assert e.npairs[0] == 3 * 50
    pr.wells["W-1"].core["c"] = __import__("cosma.wells.model", fromlist=["RawCore"]).RawCore(
        np.array([160.0, 170.0]), "md", {"POR": np.array([0.1, 0.2])})
    dc = collect(pr, uid(pr, "Res"), "POR", kind="core")
    assert dc.used == {"W-1": 2} and list(dc.points[:, 2]) == [10, 20]
    assert list(collect(pr, uid(pr, "Res"), "PHI").values[:3]) == [0, 1, 0]


def test_transformed():
    pr = build()
    d = collect(pr, uid(pr, "Res"), "PERM")
    assert d.transformed("none") is d.values
    assert d.transformed("normal_score").shape == d.values.shape
    with pytest.raises(ProjectError):
        d.transformed("log")


# -- storage and import ---------------------------------------------------------------------------------------------------------
GS = "1 0.1\n1 0.9 45 0 0\n1000 500 5\n"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


OK = {"variable": "PHI", "transform": "none", "range_unit": "m"}


def test_import_gslib_attaches_to_interval(tmp_path):
    pr = build()
    f = _write(tmp_path, "phi.var", GS)
    res = import_file("variogram", f, scripted({**OK, "interval": uid(pr, "Res")}), project=pr)
    m = pr.variogram_for(uid(pr, "Res"), "PHI")
    assert m.azimuth == 45 and m.nugget == 0.1 and m.structures[0].range_major == 1000
    assert m.variable == "PHI" and "phi -> Res / PHI" in res.applied[0]
    assert pr.variogram_for(uid(pr, "Seal"), "PHI") is None


def test_import_converts_range_unit_and_requires_choices(tmp_path):
    pr = build()
    f = _write(tmp_path, "p.var", GS)
    import_file("variogram", f, scripted({**OK, "range_unit": "km", "interval": uid(pr, "Res")}), project=pr)
    assert pr.variogram_for(uid(pr, "Res"), "PHI").structures[0].range_major == 1_000_000
    for missing in ("variable", "transform", "range_unit", "interval"):
        v = {**OK, "interval": uid(pr, "Res")}
        v[missing] = None
        with pytest.raises(ImportRejected):
            import_file("variogram", f, scripted(v), project=Project("x") if False else pr)
    prev = inspect("variogram", f, pr)
    assert next(p for p in prev.params if p.key == "transform").value is None             # GSLIB can't state it: no guess
    assert any("metres" in w for w in prev.warnings)


def test_import_json_prefills_from_file(tmp_path):
    pr = build()
    m = VariogramModel(0.1, (Structure("exponential", 0.9, 800, 400, 4),), 10, "normal_score", "PHI").check()
    f = _write(tmp_path, "m.json", json.dumps(to_dict(m)))
    prev = inspect("variogram", f, pr)
    vals = {p.key: p.value for p in prev.params}
    assert vals["variable"] == "PHI" and vals["transform"] == "normal_score" and vals["range_unit"] == "m"
    import_file("variogram", f, scripted({"interval": uid(pr, "Seal")}), project=pr)
    assert pr.variogram_for(uid(pr, "Seal"), "PHI").transform == "normal_score"


def test_import_needs_chart_and_valid_file(tmp_path):
    f = _write(tmp_path, "p.var", GS)
    with pytest.raises(ImportRejected, match="chart"):
        import_file("variogram", f, scripted({}), project=Project("x"))
    with pytest.raises(ImportRejected, match="chart"):
        import_file("variogram", f, scripted({}), project=None)
    pr = build()
    bad = _write(tmp_path, "bad.var", "1 0.1\n1 0.9 0 20 0\n100 50 5\n")
    with pytest.raises(ValueError, match="dip"):
        inspect("variogram", bad, pr)
    with pytest.raises(ImportRejected):
        import_file("variogram", f, scripted({**OK, "interval": "base"}), project=pr)           # base is not an interval


def test_conflicts_replace_skip_suffix(tmp_path):
    pr = build()
    u = uid(pr, "Res")
    f1 = _write(tmp_path, "a.var", GS)
    f2 = _write(tmp_path, "b.var", "1 0.2\n1 0.8 0 0 0\n300 200 3\n")
    import_file("variogram", f1, scripted({**OK, "interval": u}), project=pr)
    prev = inspect("variogram", f2, pr)
    with pytest.raises(ImportRejected, match="conflict|action|choose"):
        import_file("variogram", f2, scripted({**OK, "interval": u}), project=pr)
    key = f"chart|variogram|{u}|PHI"
    import_file("variogram", f2, scripted({**OK, "interval": u, "conflict_actions": {key: Action("skip")}}), project=pr)
    assert pr.variogram_for(u, "PHI").nugget == 0.1
    r = import_file("variogram", f2, scripted({**OK, "interval": u, "conflict_actions": {key: Action("suffix", "_alt")}}), project=pr)
    assert pr.variogram_for(u, "PHI").nugget == 0.1 and "b_alt" in pr.variograms[u]["PHI"] and "remains" in r.notes[0]
    with pytest.raises(ImportRejected):                                    # suffix colliding with an existing label
        import_file("variogram", f2, scripted({**OK, "interval": u, "conflict_actions": {key: Action("suffix", "_alt")}}), project=pr)
    import_file("variogram", f2, scripted({**OK, "interval": u, "conflict_actions": {key: Action("replace")}}), project=pr)
    assert pr.variogram_for(u, "PHI").nugget == 0.2 and set(pr.variograms[u]["PHI"]) == {"b", "b_alt"} | {"a"} - {"a"}


def test_other_property_is_not_a_conflict(tmp_path):
    pr = build()
    u = uid(pr, "Res")
    f = _write(tmp_path, "a.var", GS)
    import_file("variogram", f, scripted({**OK, "interval": u}), project=pr)
    import_file("variogram", f, scripted({**OK, "variable": "PERM", "interval": u}), project=pr)
    assert pr.variogram_for(u, "PERM") is not None and pr.variogram_for(u, "PHI") is not None


def test_variogram_blocks_interval_removal_and_attach_validates():
    pr = build()
    u = uid(pr, "Res")
    pr.attach_variogram(u, "PHI", from_gslib(GS), "v")
    assert "variogram PHI" in pr.horizon_uses(u)
    for w in pr.wells.values():
        w.tops.pop("Res", None)
    with pytest.raises(ProjectError, match="variogram PHI"):
        pr.remove_interval(u)
    with pytest.raises(ProjectError, match="property"):
        pr.attach_variogram(u, "  ", from_gslib(GS), "v2")
    with pytest.raises(ProjectError, match="interval"):
        pr.attach_variogram("base", "PHI", from_gslib(GS), "v3")
    bad = VariogramModel(-1.0, from_gslib(GS).structures)
    with pytest.raises(ProjectError, match="nugget"):
        pr.attach_variogram(u, "X", bad, "v4")
    assert "X" not in pr.variograms.get(u, {})


def test_collect_non_last_interval_uses_next_pick_as_base():
    pr = build()
    d = collect(pr, uid(pr, "Seal"), "PHI")                      # Seal: 100-150, 110-170, 120-190 (base = the Res pick)
    assert d.thickness == {"W-1": 50, "W-2": 60, "W-3": 70}
    assert d.points[d.groups == "W-3"][:, 2].max() == 70


def test_collect_skips_zero_or_negative_thickness():
    pr = build()
    pr.wells["W-2"].tops["Basement"] = RawTop("Basement", 170.0, "md")      # base at the Res top
    pr.wells["W-3"].tops["Basement"] = RawTop("Basement", 100.0, "md")      # base above the top
    d = collect(pr, uid(pr, "Res"), "PHI")
    assert set(d.used) == {"W-1"} and "no thickness" in d.skipped["W-2"] and "no thickness" in d.skipped["W-3"]


def test_range_unit_applies_to_all_three_ranges(tmp_path):
    pr = build()
    f = _write(tmp_path, "p.var", GS)
    import_file("variogram", f, scripted({**OK, "range_unit": "km", "interval": uid(pr, "Res")}), project=pr)
    s = pr.variogram_for(uid(pr, "Res"), "PHI").structures[0]
    assert (s.range_major, s.range_minor, s.range_vert) == (1e6, 5e5, 5e3)


def test_conflict_names_the_existing_primary(tmp_path):
    pr = build()
    u = uid(pr, "Res")
    pr.attach_variogram(u, "PHI", from_gslib(GS), "first")
    c, = pr.conflicts_variogram(u, "PHI", "second")
    assert c.existing == "first" and c.new == "second" and c.base == "second" and "first" in c.reserved


def test_meta_follows_the_stored_model_through_replace_and_suffix():
    pr = build()
    u = uid(pr, "Res")
    m = from_gslib(GS)
    pr.attach_variogram(u, "PHI", m, "a", meta={"curve": "PHI", "kind": "log"})
    assert pr.var_meta[(u, "PHI", "a")]["kind"] == "log"
    pr.attach_variogram(u, "PHI", m, "b", {f"chart|variogram|{u}|PHI": Action("suffix", "_2")}, meta={"curve": "PHI", "kind": "core"})
    assert pr.var_meta[(u, "PHI", "b_2")]["kind"] == "core" and pr.var_meta[(u, "PHI", "a")]["kind"] == "log"
    pr.attach_variogram(u, "PHI", m, "c", {f"chart|variogram|{u}|PHI": Action("replace")})        # replaces primary "a", no meta given
    assert (u, "PHI", "a") not in pr.var_meta and (u, "PHI", "c") not in pr.var_meta
