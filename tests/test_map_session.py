import numpy as np
import pytest

from cosma.map_session import AVG, MapSession
from cosma.maps import MapError, MapGrid, export_map
from cosma.project import Action, ProjectError
from cosma.surfaces import Surface
from cosma.surfaces import read_surface
from cosma.variogram import Structure, VariogramModel
from cosma.variogram_session import VariogramSession
from synthetic_field import facies_project, synthetic


def ready():
    pr = synthetic(n=4, spacing=150.0)
    u = pr.chart.match_name("Res")
    a = VariogramSession(pr, u, "PHI", average=True)
    a.set_azimuth(30)
    a.set_nugget(0.02); a.set_structure(0, contribution=0.2, major=400, minor=200, vertical=50)
    a.save("avg")
    p = VariogramSession(pr, u, "PHI")
    p.set_nugget(0.1); p.set_structure(0, contribution=0.9, major=400, minor=200, vertical=20)
    p.save("pt")
    return pr, u


def test_choices_prefer_layer_average_variograms_and_warn_on_point_scale():
    pr, u = ready()
    ms = MapSession(pr, u)
    ch = ms.variogram_choices("PHI")
    assert [(c.key, c.label, c.average, c.primary) for c in ch] == [("PHI" + AVG, "avg", True, True), ("PHI", "pt", False, True)]
    assert ms.warning_for(ch[0]) is None and "smoother" in ms.warning_for(ch[1]) and "point-scale" in ch[1].text
    assert ms.variogram_choices("FACIES", 2) == [] and ms.variogram_choices("PERM") == []
    assert ms.key("FACIES", 2) == "FACIES[2]" and ms.interval == "Res"
    assert set(ms.properties()) == {"PHI"}


def test_grids_from_surfaces_and_wells():
    pr, u = ready()
    ms = MapSession(pr, u)
    with pytest.raises(MapError, match="no surface"):
        ms.surface_grid("top")
    with pytest.raises(MapError):
        ms.surface_grid("middle")
    g = ms.wells_grid(50.0, 100.0)
    assert (g.xori, g.yori) == (-100.0, -100.0) and g.xinc == 50
    pr.attach_surface(u, Surface(5, 6, 20.0, 30.0, 0.0, 1, np.zeros((6, 7))), "t")
    g2 = ms.surface_grid("top")
    assert (g2.xori, g2.yori, g2.xinc, g2.yinc, g2.ncol, g2.nrow) == (5, 6, 20, 30, 6, 7)
    pr.wells.clear()
    with pytest.raises(MapError, match="no well"):
        ms.wells_grid(50.0, 10.0)


def test_compute_requires_unit_notes_provenance_and_save():
    pr, u = ready()
    ms = MapSession(pr, u)
    ch = ms.variogram_choices("PHI")
    g = ms.wells_grid(75.0, 75.0)
    with pytest.raises(MapError, match="none is assumed"):
        ms.compute("PHI", ch[0], g, "")
    with pytest.raises(ProjectError, match="compute a map first"):
        ms.save("x")
    assert ms.conflicts("x") == []
    pm = ms.compute("PHI", ch[0], g, "fraction")
    assert pm.method == "kriged" and pm.info["variogram_label"] == "avg" and pm.info["variogram_average"] is True
    assert not any("smoother" in n for n in ms.notes)
    pm2 = ms.compute("PHI", ch[1], g, "fraction")
    assert "smoother" in ms.notes[0] and pm2.info["variogram_average"] is False
    ms.save("first")
    assert pr.map_for(u, "PHI") is pm2 and ms.conflicts("second")
    with pytest.raises(ProjectError, match="choose"):
        ms.save("second")
    ms.save("second", {ms.conflicts("second")[0].key: Action("suffix", "_x")})
    assert set(pr.maps[u]["PHI"]) == {"first", "second_x"}


def test_thickness_session():
    pr, u = ready()
    ms = MapSession(pr, u)
    g = MapGrid(0, 0, 100, 100, 4, 4)
    with pytest.raises(MapError, match="top, base"):
        ms.thickness(g)
    pr.attach_surface(u, Surface(-100, -100, 100.0, 100.0, 0.0, 1, np.full((10, 10), 100.0)), "t")
    pr.attach_surface("base", Surface(-100, -100, 100.0, 100.0, 0.0, 1, np.full((10, 10), 150.0)), "b")
    pm = ms.thickness(g)
    assert np.all(pm.values == 50.0) and ms.notes == []
    pr.surfaces["base"]["b"].values[:, :] = 80.0
    ms.thickness(g)
    assert "16 node(s)" in ms.notes[0]
    ms.save("thk")
    assert pr.map_for(u, "thickness").unit == "m"


def test_facies_proportion_map():
    fp = facies_project(n=3, spacing=100.0)
    u = fp.chart.match_name("Res")
    a = VariogramSession(fp, u, "FACIES", indicator=2, average=True)
    a.set_nugget(0.01); a.set_structure(0, contribution=0.05, major=300, minor=150, vertical=50)
    a.save("avg")
    ms = MapSession(fp, u)
    ch = ms.variogram_choices("FACIES", 2)
    assert len(ch) == 1 and ch[0].key == "FACIES[2] (layer average)"
    pm = ms.compute("FACIES", ch[0], MapGrid(0, 0, 50, 50, 5, 5), "fraction", indicator=2, bounds=(0, 1))
    assert pm.variable == "FACIES[2]" and np.nanmin(pm.values) >= 0 and np.nanmax(pm.values) <= 1


def test_export_is_readable_by_the_surface_reader(tmp_path):
    from cosma.maps import PropertyMap
    g = MapGrid(1000.0, 2000.0, 50.0, 25.0, 4, 3, 20.0)
    v = np.arange(12.0).reshape(4, 3) / 10.0
    v[1, 1] = np.nan
    export_map(PropertyMap(g, v, "PHI", "fraction", "imported"), tmp_path / "m.irap")
    s = read_surface(tmp_path / "m.irap", xy_unit="m", z_unit="m")
    assert (s.xori, s.yori, s.xinc, s.yinc, s.rotation) == (1000.0, 2000.0, 50.0, 25.0, 20.0)
    assert np.allclose(s.values, v, equal_nan=True) and np.isnan(s.values[1, 1])
    X, Y = s.node_xy()
    assert np.allclose(X, g.node_xy()[0]) and np.allclose(Y, g.node_xy()[1])
