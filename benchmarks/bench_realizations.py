"""Synthetic benchmark of the realization engine: time per stage, peak memory, serial against parallel.

    python benchmarks/bench_realizations.py --nx 200 --ny 200 --nz 100 --workers 2 --count 2

The model is flat-layered (two intervals, nz/2 layers each, 100 m x 100 m cells, 1 m layers), three wells with facies logs, typed
proportion curves (sand 0.3-0.6), SIS facies, two properties (PORO, VSHALE) and permeability. It is a speed and memory test, not a
geological case: flat layers and stationary proportions are the cheapest geometry there is, and three wells mean almost no hard data.
Memory is the resident set size read from /proc (Linux only); the parallel figure is the sum over this process and its workers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


def rss_tree() -> float:
    """Resident memory (bytes) of this process and its direct children."""
    me = os.getpid()
    total = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = open(f"/proc/{d}/status").read()
        except OSError:
            continue
        f = dict(l.split(":", 1) for l in st.splitlines() if ":" in l)
        if int(d) == me or int(f.get("PPid", "0").strip()) == me:
            total += int(f.get("VmRSS", "0 kB").split()[0]) * 1024
    return total


class Peak:
    def __init__(self):
        self.peak, self.stop = 0.0, False
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop:
            self.peak = max(self.peak, rss_tree())
            time.sleep(0.2)

    def __enter__(self):
        self.peak = rss_tree()
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join()


def build(nx, ny, nz, rng_h, rng_v, neighbours, levels):
    from cosma.facies_sim import FaciesSimSetup
    from cosma.grid.boundary import grid_spec_from_cells
    from cosma.grid.builder import Layering, build_grid
    from cosma.perm_sim import PermSetup, Relation
    from cosma.property_sim import Distribution, PropertySetup
    from cosma.proportions import Curve, ProportionSetup, Settings
    from cosma.realizations import PropertyJob, RealizationPlan
    from cosma.sis import SisSettings
    from cosma.strat import StratChart
    from cosma.variogram import Structure, VariogramModel
    from test_datacheck import make_project
    half = nz // 2
    spec = grid_spec_from_cells(0, 0, 0, nx, ny, 100.0, 100.0)
    X, _ = np.meshgrid(spec.x_edges, spec.y_edges)
    hz = [np.full_like(X, z) for z in (1000.0, 1000.0 + half, 1000.0 + 2 * half)]
    grid, rep = build_grid(spec, hz, ["conformable", "conformable"], [Layering("fixed_count", n=half), Layering("fixed_count", n=half)])
    model = SimpleNamespace(grid=grid, report=rep, interval_names=["A", "B"])
    pr = make_project()
    pr.chart = StratChart(["A", "B"], "Base")
    curves = tuple(Curve(n, "relative", 0.2, (1, 2), tuple((p, 1 - p) for p in ps), "user")
                   for n, ps in (("A", (0.6, 0.5, 0.4, 0.3, 0.4)), ("B", (0.5, 0.6, 0.5, 0.4, 0.3))))
    pr.set_proportions(ProportionSetup(Settings("relative", nbins=5), (("A", "user"), ("B", "user")), curves))
    keys = []
    for iv in pr.chart.intervals:
        for f in pr.facies_setup.scheme:
            pr.attach_variogram(iv.uid, f.name, VariogramModel(0.0, (Structure("spherical", 1.0, rng_h, rng_h, rng_v),), 0.0, "indicator"), "v")
            pr.attach_variogram(iv.uid, f"NS {f.name}", VariogramModel(0.0, (Structure("spherical", 1.0, rng_h, rng_h, rng_v),), 0.0,
                                                                         "normal_score"), "p")
            keys.append((iv.name, f.code))
    pr.set_facies_sim(FaciesSimSetup("sis", SisSettings(neighbours, 1.0, levels), None, 1))

    def ps(name, mean, sd):
        return PropertySetup(name, "NS {facies}", tuple((k, "user") for k in keys), tuple((k, Distribution("normal", mean, sd)) for k in keys),
                             lower=0.0, upper=1.0, max_neighbours=neighbours, levels=levels)
    perm = PermSetup("PHI", tuple((k, Relation("user", intercept=0.0, slope=12.0, scatter=0.3)) for k in keys), tuple((k, 0.1) for k in keys),
                     1.0, 0.0, "allow")
    return pr, model, (PropertyJob(ps("PHI", 0.18, 0.04), "PORO"), PropertyJob(ps("VSH", 0.3, 0.1), "VSHALE", export=False)), perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--count", type=int, required=True, help="realizations in the parallel run (0 = serial part only)")
    ap.add_argument("--range-h", type=float, default=1500.0)
    ap.add_argument("--range-v", type=float, default=10.0)
    ap.add_argument("--neighbours", type=int, default=12)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    from cosma.progress import Tracker
    from cosma.realization_runner import check_memory, memory_estimate, run_parallel
    from cosma.realizations import RealizationManager, RealizationPlan
    out = dict(args=vars(a))
    t0 = time.monotonic()
    pr, model, jobs, perm = build(a.nx, a.ny, a.nz, a.range_h, a.range_v, a.neighbours, a.levels)
    cells = a.nx * a.ny * a.nz
    out.update(cells=cells, build_s=time.monotonic() - t0, rss_after_build=rss_tree())
    print(f"model {a.nx}x{a.ny}x{a.nz} = {cells:,} cells built in {out['build_s']:.1f} s, RSS {out['rss_after_build'] / 1e9:.2f} GB", flush=True)
    plan = RealizationPlan(jobs, perm, max(a.count, 1), 11)
    mgr = RealizationManager(pr, model, plan)
    def timed(attempt):
        stages, last = [], [time.monotonic(), "prepare (proportions, hard data; first call only)"]

        def cb(label, f):
            name = label.split(": ", 1)[1].split(",")[0]
            name = {"starting": last[1], "done": "target check"}.get(name, name)
            name = "target check" if name.startswith("checking") else name
            if name != last[1]:
                now = time.monotonic()
                stages.append((last[1], now - last[0]))
                last[0], last[1] = now, name
        with Peak() as pk:
            t = time.monotonic()
            _, _, _, check = mgr.generate(0, attempt, cb)
            total = time.monotonic() - t
            stages.append((last[1], time.monotonic() - last[0]))
        agg = {}
        for k, v in stages:
            agg[k] = agg.get(k, 0.0) + v
        return total, agg, pk.peak, check

    serial, agg, peak, check = timed(0)
    out.update(serial_first_s=serial, serial_first_peak_rss=peak, stages_first=agg, check=check["text"].splitlines()[0])
    print(f"serial realization, first (includes prepare and loading the kernels): {serial:.1f} s, peak RSS {peak / 1e9:.2f} GB", flush=True)
    second, agg, peak, _ = timed(1)
    out.update(serial_second_s=second, serial_second_peak_rss=peak, stages_second=agg, bytes_per_cell_serial=(peak - 3.5e8) / cells)
    print(f"serial realization, second (warm): {second:.1f} s, peak RSS {peak / 1e9:.2f} GB "
          f"= {(peak - 3.5e8) / cells:.0f} bytes/cell above 350 MB", flush=True)
    for k, v in agg.items():
        print(f"   {k:50s} {v:8.1f} s   {cells / v if v > 0.05 else float('nan'):12,.0f} cells/s", flush=True)
    print("check:", out["check"], flush=True)
    if a.count > 0:
        print(check_memory(cells, a.workers)[0], flush=True)
        tr = Tracker(a.count)
        with Peak() as pk:
            t = time.monotonic()
            res, _ = run_parallel(mgr, range(a.count), a.workers, tr)
            par = time.monotonic() - t
        out.update(parallel_s=par, parallel_peak_rss=pk.peak, parallel_count=a.count, workers=a.workers,
                   speedup=second * a.count / par, estimate=memory_estimate(cells, a.workers))
        print(f"parallel: {a.count} realizations, {a.workers} workers: {par:.1f} s wall (serial would be about {second * a.count:.1f} s) "
              f"-> speed-up {out['speedup']:.2f}; peak RSS of parent + workers {pk.peak / 1e9:.2f} GB "
              f"(estimate for the workers {out['estimate'] / 1e9:.2f} GB)", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
