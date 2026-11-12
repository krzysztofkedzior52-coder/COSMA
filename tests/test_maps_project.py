import json
import zipfile

import numpy as np
import pytest

from cosma.imports import ImportRejected, import_file, inspect, advice_for, scripted
from cosma.maps import MapGrid, PropertyMap, krige_map
from cosma.persist import PersistError, load_project, save_project
from cosma.project import Action, Project, ProjectError
from cosma.variogram import Structure, VariogramModel
from synthetic_field import facies_project
from test_surfaces import write_irap
from test_variogram_project import build, uid
from test_persist import _doctored, same


def pm(variable="PHI", unit="fraction", method="imported", v=0.2, variance=False):
    g = MapGrid(0, 0, 100, 100, 3, 4, 10.0)
    vals = np.full((3, 4), v)
    vals[1, 1] = np.nan
    return PropertyMap(g, vals, variable, unit, method, np.full((3, 4), 0.01) if variance else None, dict(wells=["A"], n=np.int64(3) if False else 3))


# -- project storage ------------------------------------------------------------------------------------------------------------------
def test_attach_map_replace_skip_suffix_and_primary():
    pr = build()
    u = uid(pr, "Res")
    r = pr.attach_map(u, pm(), "m1")
    assert pr.map_for(u, "PHI").values[0, 0] == 0.2 and "m1 -> Res / PHI" in r.applied[0]
    key = f"chart|map|{u}|PHI"
    c, = pr.conflicts_map(u, "PHI", "m2")
    assert c.existing == "m1" and c.base == "m2" and "m1" in c.reserved
    with pytest.raises(ProjectError, match="choose"):
        pr.attach_map(u, pm(v=0.3), "m2")
    pr.attach_map(u, pm(v=0.3), "m2", {key: Action("skip")})
    assert pr.map_for(u, "PHI").values[0, 0] == 0.2
    r = pr.attach_map(u, pm(v=0.4), "m2", {key: Action("suffix", "_b")})
    assert "remains" in r.notes[0] and set(pr.maps[u]["PHI"]) == {"m1", "m2_b"} and pr.map_for(u, "PHI").values[0, 0] == 0.2
    with pytest.raises(ProjectError, match="already exists"):
        pr.attach_map(u, pm(v=0.5), "m2", {key: Action("suffix", "_b")})
    pr.attach_map(u, pm(v=0.6), "m3", {key: Action("replace")})
    assert pr.map_for(u, "PHI").values[0, 0] == 0.6 and set(pr.maps[u]["PHI"]) == {"m2_b", "m3"}
    pr.attach_map(u, pm("PERM", "mD"), "k")
    assert pr.map_for(u, "PERM") is not None


def test_attach_map_validates_and_blocks_interval_removal():
    pr = build()
    u = uid(pr, "Res")
    with pytest.raises(ProjectError, match="declared unit"):
        pr.attach_map(u, pm(unit=" "), "m")
    with pytest.raises(ProjectError, match="name of the property"):
        pr.attach_map(u, pm(variable=" "), "m")
    with pytest.raises(ProjectError, match="give the map a name"):
        pr.attach_map(u, pm(), " ")
    with pytest.raises(ProjectError, match="interval"):
        pr.attach_map("base", pm(), "m")
    assert u not in pr.maps
    pr.attach_map(u, pm(), "m")
    assert "map PHI" in pr.horizon_uses(u)


# -- import from file -------------------------------------------------------------------------------------------------------------------
def write_map(tmp_path, name="phi.irap", fn=lambda x, y: 0.2, **kw):
    p = tmp_path / name
    write_irap(p, 5, 4, 0.0, 0.0, 100.0, 100.0, 0.0, fn, **kw)
    return p


V = {"xy_unit": "m", "variable": "PHI", "value_unit": "fraction"}


def test_import_property_map(tmp_path):
    pr = build()
    u = uid(pr, "Res")
    f = write_map(tmp_path, fn=lambda x, y: 0.1 + x / 1000.0, undefined={(0, 0)})
    res = import_file("property_map", f, scripted({**V, "interval": u}), project=pr)
    m = pr.map_for(u, "PHI")
    assert m.method == "imported" and m.unit == "fraction" and m.variable == "PHI" and m.info["source"] == "phi.irap"
    assert m.grid.ncol == 5 and m.grid.nrow == 4 and m.grid.xinc == 100
    assert m.values[2, 1] == pytest.approx(0.3) and np.isnan(m.values[0, 0]) and "phi -> Res / PHI" in res.applied[0]


def test_import_percent_converted_and_unit_kept(tmp_path):
    pr = build()
    u = uid(pr, "Res")
    f = write_map(tmp_path, fn=lambda x, y: 25.0)
    import_file("property_map", f, scripted({**V, "value_unit": "percent", "interval": u}), project=pr)
    m = pr.map_for(u, "PHI")
    assert m.unit == "fraction" and m.values[0, 0] == pytest.approx(0.25) and m.info["declared_unit"] == "percent"
    f2 = write_map(tmp_path, "k.irap", fn=lambda x, y: 150.0)
    import_file("property_map", f2, scripted({**V, "variable": "PERM", "value_unit": "mD", "interval": u}), project=pr)
    assert pr.map_for(u, "PERM").unit == "mD" and pr.map_for(u, "PERM").values[0, 0] == 150.0


def test_import_units_are_never_assumed(tmp_path):
    pr = build()
    f = write_map(tmp_path)
    for missing in ("xy_unit", "variable", "value_unit", "interval"):
        v = {**V, "interval": uid(pr, "Res")}
        v[missing] = None
        with pytest.raises(ImportRejected):
            import_file("property_map", f, scripted(v), project=pr)
    prev = inspect("property_map", f, pr)
    assert {p.key: p.value for p in prev.params} == {"xy_unit": None, "variable": None, "value_unit": None, "interval": None}
    assert prev.kind == "property_map"


def test_import_advice_catches_percent_fraction_mixups(tmp_path):
    pr = build()
    u = uid(pr, "Res")
    pct = write_map(tmp_path, "p.irap", fn=lambda x, y: 22.0)
    frac = write_map(tmp_path, "f.irap", fn=lambda x, y: 0.22)
    a = advice_for(inspect("property_map", pct, pr), {**V, "interval": u})
    assert a and "looks like percent" in a[0]
    b = advice_for(inspect("property_map", frac, pr), {**V, "value_unit": "percent", "interval": u})
    assert b and "looks like a fraction" in b[0]
    assert advice_for(inspect("property_map", frac, pr), {**V, "interval": u}) == []


def test_import_needs_chart_and_conflicts(tmp_path):
    f = write_map(tmp_path)
    for pr in (Project("x"), None):
        with pytest.raises(ImportRejected, match="chart"):
            import_file("property_map", f, scripted({}), project=pr)
    pr = build()
    u = uid(pr, "Res")
    import_file("property_map", f, scripted({**V, "interval": u}), project=pr)
    with pytest.raises(ImportRejected, match="choose"):
        import_file("property_map", f, scripted({**V, "interval": u}), project=pr)
    key = f"chart|map|{u}|PHI"
    import_file("property_map", f, scripted({**V, "interval": u, "conflict_actions": {key: Action("suffix", "_2")}}), project=pr)
    assert set(pr.maps[u]["PHI"]) == {"phi", "phi_2"}


# -- persistence ---------------------------------------------------------------------------------------------------------------------------
def with_maps():
    pr = facies_project(n=3, spacing=100.0)
    u = pr.chart.match_name("Res")
    m = VariogramModel(0.0, (Structure("spherical", 1.0, 400.0, 200.0, 10.0),), 30.0).check()
    km, _ = krige_map(pr, u, "PHI", m, MapGrid(0, 0, 50, 50, 5, 5), unit="fraction")
    pr.attach_map(u, km, "kriged")
    pr.attach_map(u, pm(variance=True), "imp", {f"chart|map|{u}|PHI": Action("suffix", "2")})
    pr.attach_map(u, pm("PERM", "mD"), "k")
    return pr


def test_maps_round_trip_exactly(tmp_path):
    pr = with_maps()
    back = load_project(save_project(pr, tmp_path / "m.cosma"))
    u = pr.chart.match_name("Res")
    assert pr.map_primary == back.map_primary and list(pr.maps[u]) == list(back.maps[u])
    for var, b in pr.maps[u].items():
        for lab, m in b.items():
            q = back.maps[u][var][lab]
            assert m.grid == q.grid and (m.variable, m.unit, m.method) == (q.variable, q.unit, q.method) and m.info == q.info
            assert np.array_equal(m.values, q.values, equal_nan=True)
            assert (m.variance is None) == (q.variance is None) and (m.variance is None or np.array_equal(m.variance, q.variance, equal_nan=True))
    same(pr, back)                                                         # nothing else changed


def test_numpy_scalars_in_map_info_are_stored(tmp_path):
    pr = build()
    m = pm()
    object.__setattr__(m, "info", dict(n=np.int64(3), x=np.float64(1.5)))
    pr.attach_map(uid(pr, "Res"), m, "m")
    back = load_project(save_project(pr, tmp_path / "n.cosma"))
    assert back.map_for(uid(pr, "Res"), "PHI").info == {"n": 3, "x": 1.5}


@pytest.mark.parametrize("edit,msg", [
    (lambda d: d["maps"].update(zzz=next(iter(d["maps"].values()))), "unknown interval"),
    (lambda d: d.update(map_primary=[[k, "ghost"] for k, _ in d["map_primary"]]), "primary map"),
    (lambda d: d.update(map_primary=[]), "no primary"),
    (lambda d: next(iter(next(iter(d["maps"].values())).values()))["kriged"].update(method="guessed"), "method"),
    (lambda d: next(iter(next(iter(d["maps"].values())).values()))["kriged"]["grid"].update(ncol=9), "shape"),
])
def test_bad_map_files_are_refused(tmp_path, edit, msg):
    import hashlib
    from test_persist import _members, _rewrite
    p = save_project(with_maps(), tmp_path / "d.cosma")
    m = _members(p)
    doc = json.loads(m["project.json"])
    edit(doc)
    pj = json.dumps(doc).encode()
    man = json.loads(m["manifest.json"])
    man["sha256"]["project"] = hashlib.sha256(pj).hexdigest()
    _rewrite(p, **{"project.json": pj, "manifest.json": json.dumps(man).encode()})
    with pytest.raises(PersistError, match=msg):
        load_project(p)
