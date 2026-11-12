import numpy as np
import pytest

from cosma.progress import Tracker
from cosma.realization_runner import BASE_BYTES, BYTES_PER_CELL, cpu_count, memory_estimate, run_parallel
from cosma.realizations import RealizationError, RealizationManager
from test_realizations import plan, world


def arrays(cache, n):
    z = np.load(cache / f"realization_{n:03d}.npz")
    return {k: z[k] for k in z.files if k != "meta"}


def test_parallel_equals_serial_whatever_the_worker_count(tmp_path):
    pr, model = world()
    p = plan(pr, count=4)
    ser, par = tmp_path / "s", tmp_path / "p"
    ms = RealizationManager(pr, model, p, cache_dir=ser)
    for n in range(4):
        ms.realize(n)
    mp_ = RealizationManager(pr, model, p, cache_dir=par)
    t = Tracker(4)
    got = []
    res, stopped = run_parallel(mp_, range(4), 2, tracker=t, on_result=got.append)
    assert not stopped and sorted(res) == [0, 1, 2, 3] and len(got) == 4
    assert all(r.facies is None and r.props == {} for r in res.values())      # light records only
    assert t.finished == 4 and t.fraction == 1.0
    for n in range(4):
        a, b = arrays(ser, n), arrays(par, n)
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=f"{n} {k}")
        assert res[n].status == ms.realize(n, keep_arrays=False).status


def test_cached_ones_are_read_not_recomputed_and_workers_are_capped(tmp_path):
    pr, model = world()
    p = plan(pr, count=3)
    m = RealizationManager(pr, model, p, cache_dir=tmp_path)
    m.realize(1)
    stamp = (tmp_path / "realization_001.npz").stat().st_mtime_ns
    res, _ = run_parallel(m, range(3), 8)                                     # more workers than realizations to make
    assert sorted(res) == [0, 1, 2] and res[1].from_cache and not res[0].from_cache
    assert (tmp_path / "realization_001.npz").stat().st_mtime_ns == stamp
    import cosma.realization_runner as RR
    real = RR.ProcessPoolExecutor
    RR.ProcessPoolExecutor = lambda *a, **k: (_ for _ in ()).throw(AssertionError("a process pool was started"))
    try:
        res, _ = run_parallel(m, range(3), 2)                                 # all cached: no process is started
    finally:
        RR.ProcessPoolExecutor = real
    assert all(r.from_cache for r in res.values())
    # with one realization missing the pool is sized by the work, not by the request
    (tmp_path / "realization_002.npz").unlink()
    sizes = []
    RR.ProcessPoolExecutor = lambda *a, **k: (sizes.append(k["max_workers"]), real(*a, **k))[1]
    try:
        run_parallel(m, range(3), 8)
    finally:
        RR.ProcessPoolExecutor = real
    assert sizes == [1]


def test_stop_keeps_finished_realizations_and_a_worker_error_is_raised(tmp_path):
    pr, model = world()
    p = plan(pr, count=6)
    m = RealizationManager(pr, model, p, cache_dir=tmp_path)
    seen = []
    res, stopped = run_parallel(m, range(6), 1, cancelled=lambda: len(seen) >= 1, on_result=seen.append, poll=0.05)
    assert stopped and 1 <= len(res) < 6 and len(list(tmp_path.glob("*.npz"))) == len(res)
    with pytest.raises(RealizationError, match="workers"):
        run_parallel(m, range(6), 0)


def test_worker_failure_is_reported(tmp_path):
    import dataclasses
    from cosma.realizations import PropertyJob
    from test_realizations import psetup
    pr, model = world()
    job = PropertyJob(dataclasses.replace(psetup(pr), variogram="NOT ATTACHED {facies}"), "PORO")
    m = RealizationManager(pr, model, plan(pr, count=2, jobs=(job,)), cache_dir=tmp_path)
    with pytest.raises(RealizationError, match="NOT ATTACHED"):
        run_parallel(m, range(2), 2)
    assert not list(tmp_path.glob("*.npz"))


def test_memory_estimate_and_cpu_count():
    from cosma.realization_runner import PARENT_BASE_BYTES, PARENT_BYTES_PER_CELL
    assert memory_estimate(1_000_000, 4) == PARENT_BASE_BYTES + PARENT_BYTES_PER_CELL * 1e6 + 4 * (BASE_BYTES + BYTES_PER_CELL * 1e6)
    # the estimate covers what the benchmark measured (parent + 2 workers: 1.33 GB at 1M cells, 2.85 GB at 4M cells), within 25%
    for cells, measured in ((1_000_000, 1.33e9), (4_000_000, 2.85e9)):
        assert measured <= memory_estimate(cells, 2) <= 1.25 * measured
    assert cpu_count() >= 1


def test_a_real_built_model_and_project_can_be_sent_to_worker_processes():
    import pickle
    from cosma.grid_definition import build_model
    from test_grid_definition import gd, model
    pr = model()
    mb = build_model(pr, gd(pr))
    back_pr, back_mb = pickle.loads(pickle.dumps((pr, mb)))
    np.testing.assert_array_equal(back_mb.grid.zcorn, mb.grid.zcorn)
    assert back_mb.report.layers_per_interval == mb.report.layers_per_interval and back_pr.chart is not None
