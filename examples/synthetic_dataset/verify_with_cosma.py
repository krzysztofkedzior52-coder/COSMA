"""Imports every file of the synthetic dataset through cosma.imports.import_file, the code path behind the GUI's
confirmation dialogs, with the choices the README lists. Prints what each import did and any problem.

    python verify_with_cosma.py DATASET_DIR [--save project.cosma] [--pipeline]

--pipeline goes on after the imports: fault depths / interactions / transmissibility, grid build, facies from logs,
data checks, and both proportion-map images, with the choices the README suggests.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from cosma.imports import conflicts_for, import_file
from cosma.project import Action, Project

SKIP = Action("skip")
M = {"xy_unit": "m", "z_unit": "m"}


def run(root: Path, save=None):
    root = Path(root)
    pr = Project("Synthetic CO2 storage dataset")
    log = []

    def chosen(extra):
        def confirm(prev):
            v = prev.initial_values()
            v.update(extra)
            c = conflicts_for(prev, v)
            if c:
                v["conflict_actions"] = {x.key: SKIP for x in c}
                log.append(f"      {len(c)} collision(s) with data already there, all set to 'skip' "
                           f"({sum(x.identical for x in c)} identical)")
            return v
        return confirm

    def do(kind, rel, extra=None, label=None):
        path = root / rel
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = import_file(kind, path, chosen(extra or {}), project=pr)
        n = len(res.applied) if hasattr(res, "applied") else ""
        print(f"  ok  {kind:13s} {rel}  {n}")
        for line in log:
            print(line)
        log.clear()
        for x in w[:3]:
            print(f"      warning: {x.message}")
        return res

    print("Type well (header, deviation, tops) -> stratigraphic chart")
    do("well_headers", "2_type_well/typewell_header_W-01.csv", M)
    do("deviation", "2_type_well/typewell_deviation_W-01.csv", {"z_unit": "m", "azimuth_reference": "grid"})
    do("tops", "2_type_well/typewell_tops_W-01.csv", {"z_unit": "m", "depth_type": "md"})
    pr.build_chart("W-01")
    print("  chart:", [i.name for i in pr.chart.intervals], "base:", pr.chart.horizons()[-1].name)

    print("Boundary polygon (five formats of the same polygon)")
    areas = []
    for rel, extra in (("1_boundary/boundary_utm16n.csv", {}), ("1_boundary/boundary_irap.pol", {}),
                       ("1_boundary/boundary_wgs84.geojson", {"target_crs": "EPSG:32616"}),
                       ("1_boundary/boundary_wgs84.kml", {"target_crs": "EPSG:32616"}),
                       ("1_boundary/boundary_utm16n.shp", {"target_crs": "EPSG:32616"})):
        import_file("polygon", root / rel, chosen(extra), project=pr)      # the last one stays the model boundary
        areas.append(pr.boundary.area)
        print(f"  ok  polygon       {rel}  area {pr.boundary.area / 1e6:.4f} km2")
    spread = (max(areas) - min(areas)) / np.mean(areas)
    print(f"      area spread between formats: {spread:.2e}")

    print("Surfaces (IRAP ascii, ZMAP+, IRAP binary), attached to horizons")
    hz = {h.name: h.uid for h in pr.chart.horizons()}
    for rel, hname in (("Top_Seal.irap", "Seal"), ("Top_Res_A.irap", "Res_A"), ("Top_Res_B.zmap", "Res_B"),
                       ("Top_Basement.gri", "Basement")):
        do("surface", f"3_surfaces/{rel}", {**M, "positive": "down", "horizon": hz[hname]})

    print("All wells")
    do("well_headers", "4_wells/well_headers_all.csv", M)
    do("deviation", "4_wells/deviation_all.csv", {"z_unit": "m", "azimuth_reference": "grid"})
    do("tops", "4_wells/tops_14_wells.csv", {"z_unit": "m", "depth_type": "md", "top_map": {"Marker_X": "ignore"}})
    late = {"Caprock": hz["Seal"], "Upper_Res": hz["Res_A"], "Lower_Res": hz["Res_B"], "Crystalline": hz["Basement"]}
    do("tops", "4_wells/tops_W-15_W-16_other_names.csv", {"z_unit": "m", "depth_type": "md", "top_map": late})
    for las in sorted((root / "4_wells/logs").glob("*.las")):
        do("las", f"4_wells/logs/{las.name}", {"depth_unit": "m"})
    do("core", "4_wells/core_5_wells_md_percent.csv", {"z_unit": "m", "depth_type": "md", "porosity_unit": "percent"})
    do("core", "4_wells/core_W-14_tvd_fraction.csv", {"z_unit": "m", "depth_type": "tvd", "porosity_unit": "frac"})

    print("Fault sticks")
    do("fault_sticks", "5_fault_sticks/fault_sticks_F_Main_F_Minor_m.csv", {**M, "z_is": "depth"})
    do("fault_sticks", "5_fault_sticks/fault_F_East_headerless_feet.txt",
       {"xy_unit": "ft_intl", "z_unit": "ft_intl", "z_is": "depth", "fault_name": "F_East"})
    for n, f in pr.faults.items():
        fit = f.fit()
        print(f"      {n}: {f.n_sticks} sticks, dip {fit.dip_deg:.1f} deg, rms {fit.rms:.2f} m")

    print("Property maps")
    for rel, var, unit, iv in (("PHIT_Res_A_fraction.irap", "PHIT", "fraction", "Res_A"),
                               ("PHIT_Res_B_percent.irap", "PHIT", "percent", "Res_B"),
                               ("SAND_FRACTION_Res_B.irap", "SAND", "fraction", "Res_B"),
                               ("PERM_geomean_Res_B_mD.irap", "PERM", "mD", "Res_B")):
        do("property_map", f"6_property_maps/{rel}", {"xy_unit": "m", "variable": var, "value_unit": unit,
                                                      "interval": next(i.uid for i in pr.chart.intervals if i.name == iv)})

    print("Variograms")
    uid = {i.name: i.uid for i in pr.chart.intervals}
    do("variogram", "7_variograms/PHIT_normal_score_Res_A.json", {"interval": uid["Res_A"], "variable": "PHIT"})
    do("variogram", "7_variograms/FACIES_1_sand_indicator_Res_B.json", {"interval": uid["Res_B"], "variable": "FACIES[1]"})
    do("variogram", "7_variograms/PHIT_normal_score_Res_B.var", {"interval": uid["Res_B"], "variable": "PHIT",
                                                                 "transform": "normal_score", "range_unit": "m"})
    do("variogram", "7_variograms/VSHALE_normal_score_Res_B.json", {"interval": uid["Res_B"], "variable": "VSHALE"})

    print(f"Wells: {len(pr.wells)}; log sets: {sum(len(w.logs) for w in pr.wells.values())}; "
          f"core sets: {sum(len(w.core) for w in pr.wells.values())}")
    if save:
        from cosma.persist import save_project
        save_project(pr, save)
        print("saved", save)
    return pr


def pipeline(pr: Project, root: Path):
    import csv

    from cosma.datacheck import CoreSpec, DataCheckSetup, PropertySpec, Thresholds, run_checks
    from cosma.facies import Condition, Facies, FaciesSetup, Rule, hard_facies
    from cosma.fault_transmissibility import FaultTransmissibility
    from cosma.grid.boundary import grid_spec_from_cells
    from cosma.grid.builder import Layering
    from cosma.grid_definition import GridDefinition, build_model
    from cosma.proportion_maps import Legend, Recipe, load_raster, run_recipe

    root = Path(root)
    print("Faults: trace depth, interactions, transmissibility")
    for n in pr.faults:
        pr.set_fault_depth(n, 1150.0)
    found, _, problems = pr.refresh_fault_interactions(150.0)
    for i in found:
        print("  ", i.describe(), "- suggestion:", i.suggestion)
        pr.confirm_fault_interaction(i.key, i.suggestion)
    print("   age order, oldest first:", pr.fault_age_order())
    pr.set_fault_transmissibility("F_Main", FaultTransmissibility("fixed", multiplier=0.1))
    pr.set_fault_transmissibility("F_Minor", FaultTransmissibility("fixed", multiplier=0.5))
    pr.set_fault_transmissibility("F_East", FaultTransmissibility("open"))
    print("Grid")
    nl = {"Seal": 10, "Res_A": 34, "Res_B": 40}
    gd = GridDefinition(600000.0, 4700000.0, 0.0, 50, 40, 100.0, 100.0,
                        layerings={iv.uid: Layering("fixed_count", n=nl[iv.name]) for iv in pr.chart.intervals})
    mb = build_model(pr, gd)
    print("  " + mb.summary().replace("\n", "\n  "))
    print("Facies from logs (VSHALE rules), upscaled to cells")
    scheme = (Facies(1, "Sand"), Facies(2, "Silt"), Facies(3, "Shale"))
    rules = (Rule(1, (Condition("VSHALE", "<", 0.25),)),
             Rule(2, (Condition("VSHALE", ">=", 0.25), Condition("VSHALE", "<", 0.60))),
             Rule(3, (Condition("VSHALE", ">=", 0.60),)))
    fs = FaciesSetup(scheme, rules, min_coverage=0.5, conflict="longest")
    hard = hard_facies(pr, fs, mb.grid)
    pr.set_facies_setup(fs)
    cnt = {fs.name_of(int(c)): int(n) for c, n in zip(*np.unique(hard.cells[hard.cells > 0], return_counts=True))}
    print("  hard cells per facies:", cnt, "| conflicts between wells:", len(hard.conflicts))
    print("Data checks")
    th = Thresholds(30, 3, 2.0, 15, 100, 10.0, 400.0, 8, 30, 1200.0, 3.0, 4, 30)
    rep = run_checks(pr, mb, DataCheckSetup(th, (PropertySpec("PHIT", "PHIT", "arithmetic"),
                                                 PropertySpec("VSHALE", "VSHALE", "arithmetic")),
                                            CoreSpec("poro", "perm", 0.5)))
    ok = [r for r in rep.regression if r["ok"]]
    print(f"  core regressions passing {len(ok)} of {len(rep.regression)}; e.g. "
          + "; ".join(f"{r['level']} {r['interval'] or 'all'} f{r['facies'] or 'all'}: slope {r['slope']:.1f}, R2 {r['r2']:.2f}"
                      for r in ok[:3]))
    print("Proportion-map images")
    leg = Legend((((255, 215, 0), 1), ((205, 133, 63), 2), ((105, 105, 105), 3)), 10.0)
    sp = grid_spec_from_cells(600000.0, 4700000.0, 0.0, 50, 40, 100.0, 100.0)
    sp.active_xy = mb.active_columns
    cp = tuple(tuple(float(v) for v in r) for r in list(csv.reader(open(root / "8_proportion_map_images/facies_map_Res_B_control_points.csv")))[1:])
    pm = run_recipe(load_raster(root / "8_proportion_map_images/facies_map_Res_B_screenshot.png"),
                    Recipe("Res_B", (("crop", 0, 0, 800, 600),), cp, 8.0, leg, 5.0, 0.9, "png"), scheme, sp)
    print("  " + pm.report.replace("\n", "\n  "))
    r2 = load_raster(root / "8_proportion_map_images/facies_map_Res_A_geotiff.tif")
    a = r2.geo_affine
    h, w = r2.shape
    cps = tuple((u, v, a[2] + a[0] * u, a[5] + a[4] * v) for u, v in ((0.5, 0.5), (w - .5, 0.5), (0.5, h - .5), (w - .5, h - .5)))
    pm2 = run_recipe(r2, Recipe("Res_A", (), cps, 1.0, leg, 3.0, 0.9, "tif"), scheme, sp)
    print("  " + pm2.report.replace("\n", "\n  "))


if __name__ == "__main__":
    flags = {"--save", "--pipeline"}
    args = [a for a in sys.argv[1:] if a not in flags]
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    if save:
        args = [a for a in args if a != save]
    project = run(Path(args[0]), save)
    if "--pipeline" in sys.argv:
        pipeline(project, Path(args[0]))
