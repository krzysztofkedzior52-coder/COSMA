import io
import json
import os
import zipfile

import numpy as np
import pytest

from cosma.persist import (FORMAT_VERSION, PersistError, fingerprint, load_project, save_project)
from cosma.project import Action, Project, WellRecord
from cosma.surfaces import Surface
from cosma.variogram import from_gslib
from cosma.wells.model import LogSet, RawCore, RawTop, Survey
from synthetic_field import facies_project


def rich():
    """A project using every kind of state."""
    pr = facies_project(n=2, spacing=100.0)
    u = pr.chart.match_name("Res")
    pr.name = "Rich / test ✓"
    pr.type_well = "W0"
    pr.chart.intervals[1].contact_above = "unconformity"
    pr.rename_horizon(u, "Reservoir")                       # leaves an alias
    rng = np.random.default_rng(1)
    w = pr.wells["W1"]
    w.header = dict(x=1234567.891, y=7654321.123, kb=25.5, td=None)
    w.survey = Survey(np.array([0, 100.0, 400.0]), np.array([0, 12.5, 30.0]), np.array([0.0, 45.0, 50.0]), "true", 1.234)
    w.logs["lg"] = LogSet(np.arange(0, 10.0), {"GR": rng.normal(size=10), "FLAG": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0])},
                          {"GR": "gAPI"})
    w.logs["lg"].curves["GR"][3] = np.nan
    w.tops["Extra"] = RawTop("Extra", 123.456789012345, "tvd")
    w.core["c1"] = RawCore(np.array([160.0, 170.5]), "md", {"POR": np.array([0.1, 0.2]), "KH": np.array([5.0, np.nan])})
    v = rng.normal(size=(5, 4))
    v[2, 2] = np.nan
    pr.attach_surface(u, Surface(1000.5, 2000.25, 50.0, 25.0, 12.5, 1, v, "tilt"), "s1")
    pr.attach_surface(u, Surface(0, 0, 10.0, 10.0, 0.0, 1, np.ones((3, 3)) * 7, "alt"), "s1", {f"chart|surface|{u}": Action("suffix", "_b")})
    pr.attach_surface("base", Surface(0, 0, 10.0, 10.0, 0.0, 1, np.ones((3, 3)) * 9), "bs")
    m = from_gslib("2 0.05\n1 0.5 30 0 0\n400 200 10\n2 0.45 30 0 0\n900 450 20\n", variable="PHI")
    pr.attach_variogram(u, "PHI", m, "main", meta=dict(curve="PHI", kind="log", source={"W0": "lg"}, frame="top", indicator=None, wells=["W0", "W1"]))
    pr.attach_variogram(u, "PHI", m, "alt", {f"chart|variogram|{u}|PHI": Action("suffix", "2")}, meta=dict(curve="PHI", kind="log"))
    from cosma.faults import Fault
    st = lambda o, dz: np.array([[1000.123456789 + o, 2000.5, 0.1], [1010.5 + o, 2000.5 + dz, 417.123456789012]])
    pr.attach_faults([Fault("Main fault, N", (st(0, 5.5), st(250.25, 1.0), st(500.5, -3.0)), "sticks.csv", dict(z_is="depth", xy_unit="ft_us")),
                      Fault("Single", (st(0, 0.5),)), Fault("Third", (st(0, 0.5),)), Fault("Fourth", (st(0, 0.5),))])
    from cosma.fault_interactions import Interaction
    pr.fault_depths = {"Main fault, N": 200.5, "Single": 150.25, "Third": 100.0, "Fourth": 120.0}
    pr.fault_reach = 55.5
    from cosma.fault_transmissibility import FaultTransmissibility as FT
    from cosma.facies import CodeMap, Condition, Facies, FaciesSetup, Flag, Rule
    pr.facies_setup = FaciesSetup((Facies(1, "Sand"), Facies(3, "Mud, \u00e5")),
                                  (Rule(1, (Condition("GR", "<", 52.123456789), Condition("RHOB", ">=", 2.25))),),
                                  source={"W0": "lg"}, flags=(Flag("W1", 123.456789, 130.5, "tvdss", 3),), flag_step=0.25,
                                  min_coverage=0.6, conflict="longest")
    from cosma.datacheck import CoreSpec, DataCheckSetup, PropertySpec, Thresholds
    pr.data_checks = DataCheckSetup(Thresholds(5, 2, 0.5, 12, 30, 2.5, 250.5, 8, 10, 1500.5, 4.25, 4, 12, 0.35),
                                    (PropertySpec("PHI", "PHIE", "arithmetic"), PropertySpec("K", "KLOG", "geometric")),
                                    CoreSpec("CPOR", "CKH", 0.3, "core1"))
    from cosma.proportions import Curve, ProportionSetup, Settings
    pr.proportions = ProportionSetup(Settings("depth", bin_size=2.75, window=3, min_wells=2, min_cells=4),
                                     (("Top", "user"), ("Base", "wells")),
                                     (Curve("Top", "depth", 2.75, (1, 3), ((0.125, 0.875), (0.5, 0.5)), "user"),))
    pr.fault_transmissibility = {"Main fault, N": FT("calculated", method="slab", kf=0.0123456789, thickness_per_throw=0.0123),
                                 "Single": FT("fixed", multiplier=0.0)}
    from cosma.fault_relations import Method
    pr.add_sgr_method(Method("mine", "-3.5*SGR - 0.125*log10(D)*(1 - SGR)**4 + 0.25", "Own study, 2026; basin X", "valid 0.2-0.8"))
    pr.fault_transmissibility["Third"] = FT("calculated", method="sgr", relation="manzocchi1999", thickness_per_throw=1 / 66, min_throw=2.5)
    pr.fault_transmissibility["Fourth"] = FT("calculated", method="sgr", expression=pr.sgr_methods["mine"].expression,
                                              expression_name="mine", thickness=0.123456789, min_throw=0.75)
    pr.fault_interactions = [Interaction("Main fault, N", "Single", "abutting", 1010.123456789, 2000.5, ends="Single",
                                         suggestion="Main fault, N", answer="Main fault, N", note="n")]
    from cosma.grid.boundary_options import AquiferSpec, BoundaryOptions, PVMultiplier
    from cosma.grid.builder import Layering
    from cosma.grid_definition import GridDefinition
    from shapely.geometry import Polygon
    pr.set_boundary(Polygon([(0.1, 0.2), (1000.3, 0.2), (1000.3, 900.7), (0.1, 900.7)],
                            [[(100.5, 100.5), (200.5, 100.5), (200.5, 200.5), (100.5, 200.5)]]))
    opts = BoundaryOptions("fraction", 0.4, 250.5, PVMultiplier(3.5, 2),
                           AquiferSpec("carter_tracy", ("I-", "J+"), 1500.5, None, 1e-4, 0.8, permeability=210.5, porosity=0.21,
                                       external_radius=5000.5, thickness=61.5, influx_angle=270.0, table=1))
    pr.grid_def = GridDefinition(1000.5, 2000.25, 33.3, 12, 7, 55.5, 44.25, 0.25,
                                 {u: Layering("follow_top", dz=2.5), "other": Layering("fixed_count", n=4)}, {u: "onlap"}, opts)
    return pr


def same(a: Project, b: Project):
    assert a.name == b.name and a.type_well == b.type_well and list(a.wells) == list(b.wells)
    assert a.fault_transmissibility == b.fault_transmissibility and a.facies_setup == b.facies_setup and a.proportions == b.proportions
    assert a.data_checks == b.data_checks
    assert (a.chart is None) == (b.chart is None) and (a.chart is None or a.chart.to_state() == b.chart.to_state())
    for n, r in a.wells.items():
        q = b.wells[n]
        assert r.header == q.header and r.tops == q.tops
        if r.survey is None:
            assert q.survey is None
        else:
            for f in ("md", "incl", "azim"):
                assert np.array_equal(getattr(r.survey, f), getattr(q.survey, f))
            assert (r.survey.azimuth_reference, r.survey.convergence_deg) == (q.survey.azimuth_reference, q.survey.convergence_deg)
        assert list(r.logs) == list(q.logs) and list(r.core) == list(q.core)
        for s, l in r.logs.items():
            assert np.array_equal(l.md, q.logs[s].md) and l.units == q.logs[s].units and list(l.curves) == list(q.logs[s].curves)
            for c, v in l.curves.items():
                assert np.array_equal(v, q.logs[s].curves[c], equal_nan=True) and v.dtype == q.logs[s].curves[c].dtype
        for s, c in r.core.items():
            assert np.array_equal(c.depth, q.core[s].depth) and c.depth_type == q.core[s].depth_type
            for k, v in c.columns.items():
                assert np.array_equal(v, q.core[s].columns[k], equal_nan=True)
    assert list(a.surfaces) == list(b.surfaces) and a.primary == b.primary
    for uid, bk in a.surfaces.items():
        for lab, s in bk.items():
            t = b.surfaces[uid][lab]
            assert (s.xori, s.yori, s.xinc, s.yinc, s.rotation, s.yflip, s.name) == (t.xori, t.yori, t.xinc, t.yinc, t.rotation, t.yflip, t.name)
            assert np.array_equal(s.values, t.values, equal_nan=True)
    assert list(a.faults) == list(b.faults)
    for n, f in a.faults.items():
        g = b.faults[n]
        assert f.same_geometry(g) and (f.source, f.info) == (g.source, g.info) and all(x.dtype == y.dtype for x, y in zip(f.sticks, g.sticks))
    assert a.fault_depths == b.fault_depths and a.fault_reach == b.fault_reach and a.fault_interactions == b.fault_interactions
    assert (a.boundary is None) == (b.boundary is None) and (a.boundary is None or a.boundary.equals(b.boundary))
    assert (a.grid_def is None) == (b.grid_def is None) and (a.grid_def is None or a.grid_def.to_state() == b.grid_def.to_state())
    assert a.variograms == b.variograms and a.var_primary == b.var_primary and a.var_meta == b.var_meta


def test_round_trip_is_exact(tmp_path):
    pr = rich()
    p = save_project(pr, tmp_path / "p.cosma")
    assert p.name == "p.cosma"
    same(pr, load_project(p))


def test_derived_results_identical_after_reload(tmp_path):
    pr = rich()
    back = load_project(save_project(pr, tmp_path / "p"))                      # extension added
    assert (tmp_path / "p.cosma").exists()
    a, b = pr.wells["W1"].trajectory, back.wells["W1"].trajectory
    assert np.array_equal(a.x, b.x) and np.array_equal(a.tvdss, b.tvdss)
    assert [t.tvdss for t in pr.wells["W0"].resolved_tops()] == [t.tvdss for t in back.wells["W0"].resolved_tops()]
    u = back.chart.match_name("Res")
    assert back.chart.match_name("Reservoir") == u and back.chart.match_name("Res") == u
    assert back.variogram_for(u, "PHI") == pr.variogram_for(u, "PHI")


def test_empty_and_chartless_projects(tmp_path):
    same(Project("e"), load_project(save_project(Project("e"), tmp_path / "e.cosma")))
    pr = Project("w")
    pr.wells["A"] = WellRecord("A")
    same(pr, load_project(save_project(pr, tmp_path / "w.cosma")))


def test_chart_uid_counter_survives(tmp_path):
    pr = rich()
    back = load_project(save_project(pr, tmp_path / "p.cosma"))
    before = {i.uid for i in back.chart.intervals}
    back.chart.add_interval("New", len(back.chart.intervals))
    new = next(i.uid for i in back.chart.intervals if i.name == "New")
    assert new not in before


def test_fingerprint_detects_every_kind_of_change(tmp_path):
    pr = rich()
    f0 = fingerprint(pr)
    assert fingerprint(pr) == f0 and fingerprint(load_project(save_project(pr, tmp_path / "p.cosma"))) == f0
    u = pr.chart.match_name("Res")
    changes = [lambda: setattr(pr, "name", "x"),
               lambda: pr.wells["W0"].logs["lg"].curves["PHI"].__setitem__(60, 0.123456),
               lambda: pr.wells["W0"].tops.__setitem__("Q", RawTop("Q", 1.0, "md")),
               lambda: pr.surfaces[u]["s1"].values.__setitem__((0, 0), 1e-9),
               lambda: pr.faults["Single"].sticks[0].__setitem__((0, 0), 1000.123456790),
               lambda: pr.remove_fault("Single"),
               lambda: pr.chart.intervals[0].aliases.append("zz"),
               lambda: pr.variograms[u]["PHI"].pop("alt2") and pr.var_meta.pop((u, "PHI", "alt2"))]
    for c in changes:
        c()
        f1 = fingerprint(pr)
        assert f1 != f0
        f0 = f1


def test_write_is_atomic_and_failure_keeps_the_old_file(tmp_path, monkeypatch):
    pr = rich()
    p = save_project(pr, tmp_path / "p.cosma")
    old = p.read_bytes()
    pr.name = "changed"
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        save_project(pr, p)
    assert p.read_bytes() == old and [f.name for f in tmp_path.iterdir()] == ["p.cosma"]          # no temp file left behind
    monkeypatch.undo()
    pr.wells["W0"].logs["lg"].curves["BAD"] = np.array(["a"] * len(pr.wells["W0"].logs["lg"].md), dtype=object)
    with pytest.raises(PersistError, match="only numbers"):
        save_project(pr, p)
    assert p.read_bytes() == old


def _members(p):
    with zipfile.ZipFile(p) as z:
        return {n: z.read(n) for n in z.namelist()}


def _rewrite(p, **repl):
    m = _members(p)
    m.update(repl)
    with zipfile.ZipFile(p, "w") as z:
        for n, b in m.items():
            z.writestr(n, b)


def test_corruption_and_tampering_are_detected(tmp_path):
    p = save_project(rich(), tmp_path / "p.cosma")
    m = _members(p)
    pj = json.loads(m["project.json"])
    pj["name"] = "evil"
    _rewrite(p, **{"project.json": json.dumps(pj).encode()})
    with pytest.raises(PersistError, match="checksum"):
        load_project(p)
    p = save_project(rich(), tmp_path / "q.cosma")
    ab = bytearray(_members(p)["arrays.npz"])
    ab[len(ab) // 2] ^= 0xFF
    _rewrite(p, **{"arrays.npz": bytes(ab)})
    with pytest.raises(PersistError, match="checksum"):
        load_project(p)


def test_not_a_project_and_missing_files(tmp_path):
    (tmp_path / "x.cosma").write_text("hello")
    with pytest.raises(PersistError, match="not a COSMA project"):
        load_project(tmp_path / "x.cosma")
    with pytest.raises(PersistError, match="no such file"):
        load_project(tmp_path / "nope.cosma")
    with zipfile.ZipFile(tmp_path / "z.cosma", "w") as z:
        z.writestr("other.txt", "x")
    with pytest.raises(PersistError, match="missing"):
        load_project(tmp_path / "z.cosma")


def test_version_handling(tmp_path):
    import hashlib
    p = save_project(rich(), tmp_path / "p.cosma")
    man = json.loads(_members(p)["manifest.json"])
    man["format_version"] = FORMAT_VERSION + 1
    _rewrite(p, **{"manifest.json": json.dumps(man).encode()})
    with pytest.raises(PersistError, match="newer COSMA"):
        load_project(p)
    man["format_version"] = 0
    _rewrite(p, **{"manifest.json": json.dumps(man).encode()})
    with pytest.raises(PersistError, match="invalid format version"):
        load_project(p)
    man["format_version"], man["format"] = FORMAT_VERSION, "other"
    _rewrite(p, **{"manifest.json": json.dumps(man).encode()})
    with pytest.raises(PersistError, match="not a COSMA project"):
        load_project(p)


def _doctored(tmp_path, edit):
    """Rewrite project.json with `edit(doc)` and a matching manifest (a consistent but wrong file)."""
    import hashlib
    p = save_project(rich(), tmp_path / "d.cosma")
    m = _members(p)
    doc = json.loads(m["project.json"])
    edit(doc)
    pj = json.dumps(doc).encode()
    man = json.loads(m["manifest.json"])
    man["sha256"]["project"] = hashlib.sha256(pj).hexdigest()
    _rewrite(p, **{"project.json": pj, "manifest.json": json.dumps(man).encode()})
    return p


@pytest.mark.parametrize("edit,msg", [
    (lambda d: d.update(type_well="nope"), "type well"),
    (lambda d: d["surfaces"].update(zzz=d["surfaces"][next(iter(d["surfaces"]))]), "unknown horizon"),
    (lambda d: d["primary"].update({next(iter(d["primary"])): "ghost"}), "primary surface"),
    (lambda d: d["primary"].clear(), "no primary"),
    (lambda d: d["variograms"].update(zzz={}), "unknown interval"),
    (lambda d: d.update(var_primary=[[[k[0], k[1]], "ghost"] for k, _ in d["var_primary"]]), "primary variogram"),
    (lambda d: d.update(var_primary=[]), "no primary"),
    (lambda d: d["var_meta"].append([["h1", "X", "y"], {}]), "does not exist"),
    (lambda d: d["chart"]["intervals"][1].update(uid=d["chart"]["intervals"][0]["uid"]), "duplicate"),
    (lambda d: d["chart"]["intervals"][1].update(name=d["chart"]["intervals"][0]["name"]), "used twice"),
    (lambda d: d["chart"].update(n=0), "counter"),
    (lambda d: d["faults"]["Single"].update(sticks=[]), "fault 'Single': Single: a fault needs at least one stick"),
    (lambda d: d["faults"]["Single"]["sticks"].append({"$array": "a999"}), "missing array"),
    (lambda d: d["faults"].update({" ": d["faults"]["Single"]}), "needs a name"),
    (lambda d: d["fault_depths"].update(ghost=1.0), "does not exist"),
    (lambda d: d["fault_transmissibility"].update(ghost=dict(mode="open")), "does not exist"),
    (lambda d: d["fault_transmissibility"]["Single"].update(multiplier=-1.0), "multiplier"),
    (lambda d: d["fault_transmissibility"]["Single"].update(mode="sideways"), "mode"),
    (lambda d: d["fault_transmissibility"]["Main fault, N"].update(kf=0.0), "permeability"),
    (lambda d: d["fault_transmissibility"]["Main fault, N"].update(thickness=1.0), "not both"),
    (lambda d: d["fault_transmissibility"].update(Single={}), "bad fault transmissibility"),
    (lambda d: d["fault_transmissibility"]["Third"].update(relation="nope"), "unknown relation"),
    (lambda d: d["fault_transmissibility"]["Third"].update(expression="-1", expression_name="x"), "not both"),
    (lambda d: d["fault_transmissibility"]["Fourth"].update(expression="q"), "unknown name"),
    (lambda d: d["fault_transmissibility"]["Fourth"].update(expression_name=None), "needs the name"),
    (lambda d: d["sgr_methods"][0].__setitem__(1, "__import__('os')"), "only these functions"),
    (lambda d: d["sgr_methods"][0].__setitem__(2, ""), "where this relation comes from"),
    (lambda d: d["sgr_methods"].append(list(d["sgr_methods"][0])), "appears twice"),
    (lambda d: d["sgr_methods"].__setitem__(0, ["x"]), "bad stored SGR method"),
    (lambda d: d["fault_transmissibility"]["Third"].update(min_throw=0), "minimum throw"),
    (lambda d: d["fault_transmissibility"]["Third"].update(kf=1.0), "no fixed fault-rock permeability"),
    (lambda d: d["proportions"]["settings"].update(window=2), "odd"),
    (lambda d: d["proportions"]["curves"][0].update(values=[[0.5, 0.4], [0.5, 0.5]]), "not rescaled"),
    (lambda d: d["proportions"]["choices"].__setitem__(0, ["Top", "wells"]), "not 'user'"),
    (lambda d: d["proportions"].pop("settings"), "bad proportion setup"),
    (lambda d: d["facies"].update(scheme=[]), "at least one facies"),
    (lambda d: d["facies"].update(scheme=[[1, "A"], [1, "B"]]), "unique"),
    (lambda d: d["facies"]["rules"][0].update(facies=9), "not in the scheme"),
    (lambda d: d["facies"]["rules"][0]["conditions"][0].__setitem__(1, "=="), "operator"),
    (lambda d: d["facies"]["flags"][0].__setitem__(3, "elsewhere"), "depth type"),
    (lambda d: d["facies"].pop("scheme"), "bad facies setup"),
    (lambda d: d["facies"].update(min_coverage=0.0), "min_coverage"),
    (lambda d: d["data_checks"]["thresholds"].update(min_wells=0), "min_wells"),
    (lambda d: d["data_checks"]["thresholds"].update(min_r2=2.0), "min_r2"),
    (lambda d: d["data_checks"]["thresholds"].pop("lag_v"), "bad data-check setup"),
    (lambda d: d["data_checks"]["properties"][0].__setitem__(2, "median"), "averaging"),
    (lambda d: d["data_checks"]["properties"].append(d["data_checks"]["properties"][0]), "unique"),
    (lambda d: d["data_checks"]["core"].update(depth_tol=0.0), "depth_tol"),
    (lambda d: d["facies"].update(conflict="coin toss"), "conflict policy"),
    (lambda d: d["grid_def"].update(nx=0), "whole number"),
    (lambda d: d["grid_def"]["boundary"].update(rule="other"), "inside rule"),
    (lambda d: d["grid_def"]["boundary"]["aquifer"].update(faces=[]), "outer faces"),
    (lambda d: d["grid_def"]["boundary"]["aquifer"].update(porosity=3.0), "fraction"),
    (lambda d: d["grid_def"]["boundary"]["pv"].update(factor=-1), "positive"),
    (lambda d: d["boundary"].update(shell={"$array": "a999"}), "boundary polygon"),
    (lambda d: d["grid_def"].update(dx=-1.0), "positive cell size"),
    (lambda d: d["grid_def"]["layerings"][next(iter(d["grid_def"]["layerings"]))].update(mode="bogus"), "layering mode"),
    (lambda d: d["grid_def"]["unconformity"].update({"x": "sideways"}), "erosional"),
    (lambda d: d["grid_def"].pop("min_thickness"), "bad grid definition"),
    (lambda d: d["fault_depths"].update(Single=float("nan")), "bad trace depth"),
    (lambda d: d.update(fault_reach=-3.0), "fault reach"),
    (lambda d: d["fault_interactions"][0].update(a="ghost"), "bad interaction|do not exist"),
    (lambda d: d["fault_interactions"][0].update(answer="Q"), "not a fault of the pair"),
    (lambda d: d["fault_interactions"][0].update(kind="weird"), "bad interaction"),
    (lambda d: d["wells"]["W0"]["logs"]["lg"]["md"].update({"$array": "a999"}), "missing array"),
    (lambda d: d["wells"]["W0"].pop("tops"), "inconsistent"),
    (lambda d: next(iter(next(iter(d["variograms"].values())).values()))["main"].update(nugget=-1), "nugget"),
])
def test_consistent_but_wrong_files_are_refused_with_a_reason(tmp_path, edit, msg):
    with pytest.raises(PersistError, match=msg):
        load_project(_doctored(tmp_path, edit))


def test_pickled_arrays_are_never_executed(tmp_path):
    import hashlib
    p = save_project(rich(), tmp_path / "p.cosma")
    m = _members(p)
    buf = io.BytesIO()
    np.savez(buf, a0=np.array([object()], dtype=object))                       # a pickled object array with a "valid" checksum
    ab = buf.getvalue()
    man = json.loads(m["manifest.json"])
    man["sha256"]["arrays"] = hashlib.sha256(ab).hexdigest()
    _rewrite(p, **{"arrays.npz": ab, "manifest.json": json.dumps(man).encode()})
    with pytest.raises(PersistError, match="inconsistent"):
        load_project(p)


def test_chart_state_refuses_reserved_and_stale_ids():
    from cosma.strat import StratChart, StratError
    st = StratChart(["A", "B"], "Base").to_state()
    ok = StratChart.from_state(st)
    assert [i.name for i in ok.intervals] == ["A", "B"]
    bad = json.loads(json.dumps(st))
    bad["intervals"][0]["uid"] = "base"
    with pytest.raises(StratError, match="reserved"):
        StratChart.from_state(bad)
    bad = json.loads(json.dumps(st))
    bad["n"] = 1                                                    # counter behind existing ids: new ids would collide
    with pytest.raises(StratError, match="counter"):
        StratChart.from_state(bad)
    with pytest.raises(StratError, match="incomplete"):
        StratChart.from_state({"n": 0})


def test_version_1_files_without_faults_still_load(tmp_path):
    import hashlib
    pr = rich()
    pr.faults.clear()
    pr.fault_depths, pr.fault_reach, pr.fault_interactions, pr.grid_def, pr.boundary = {}, None, [], None, None
    pr.fault_transmissibility, pr.facies_setup, pr.data_checks, pr.proportions = {}, None, None, None
    p = save_project(pr, tmp_path / "old.cosma")
    m = _members(p)
    doc = json.loads(m["project.json"])
    doc.pop("faults")                                                        # what a version-1 file looks like
    for k in ("fault_depths", "fault_reach", "fault_interactions", "grid_def", "boundary", "fault_transmissibility", "facies", "data_checks"):
        doc.pop(k)
    pj = json.dumps(doc).encode()
    man = json.loads(m["manifest.json"])
    man["format_version"] = 1
    man["sha256"]["project"] = hashlib.sha256(pj).hexdigest()
    _rewrite(p, **{"project.json": pj, "manifest.json": json.dumps(man).encode()})
    back = load_project(p)
    assert back.faults == {}
    same(pr, back)
