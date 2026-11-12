"""Realization manager: up to 100 realizations of facies, properties and permeability from seeds, checked against their targets.

A realization is fully determined by the plan (facies simulation setup, property jobs, permeability setup), the base seed, its number
and its attempt number; the seeds of every stage are derived from those with numpy's SeedSequence, so nothing depends on the order in
which realizations are made. Arrays are stored as float32 (properties are rounded to float32 BEFORE permeability is computed, so a
regenerated realization equals a cached one). A realization can be regenerated from its seeds within the same generator version
(``GENERATOR_VERSION`` and the numpy/numba results it relies on); a cache file written by another version is still used as written
and the mismatch is reported (the cache wins).

Target check. Every (interval, facies) domain of every property is compared with its target distribution in normal-score space (the
transform the simulation used), and every interval's facies fractions with the mean of the local target proportions. A cell-by-cell
test would reject everything, because neighbouring cells are strongly correlated; each test therefore uses an effective sample size
n_eff = min(n, V / A) where V is the domain volume and A the integral volume of the correlation function (spherical (pi/6) rM rm rV,
exponential (8 pi/27) rM rm rV, gaussian (pi/3)^1.5 rM rm rV, plus the nugget times the cell volume, all divided by the sill).
Tests: mean (z), variance (chi-square, n_eff - 1 degrees of freedom) and shape (Kolmogorov-Smirnov, asymptotic, n_eff). The
significance of each test is (1 - confidence) divided by the number of tests of the realization (Bonferroni). A domain with
n_eff < ``MIN_EFFECTIVE`` is reported as not testable (not as passed). The effective size ignores the shape of the domain and the
conditioning to wells, and for TGS uses the Gaussian variogram for the facies fractions (shorter true indicator ranges make this
lenient); treat the p-values as a screening, not as exact probabilities.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import chi2, kstwobign, norm

from . import facies_sim as fsim
from .progress import report
from .perm_sim import PermSetup, PermSimError, simulate_permeability
from .property_sim import PropertySetup, PropertySimError, simulate_property

GENERATOR_VERSION = "cosma-realizations-1"
MAX_REALIZATIONS = 100
MIN_EFFECTIVE = 5.0
ON_FAIL = ("keep", "discard", "redraw")
REDRAW = ("both", "properties")
RESERVED = ("PERMX", "PERMY", "PERMZ", "ACTNUM", "COORD", "ZCORN", "SPECGRID", "FAULTS", "MULTX", "MULTY", "MULTZ")


class RealizationError(ValueError):
    pass


@dataclass(frozen=True)
class PropertyJob:
    setup: PropertySetup
    keyword: str                    # GRDECL keyword of the property (PORO, VSHALE, ...)
    export: bool = True             # False: used for the fault transmissibility (e.g. VSHALE) but not written into the GRDECL
    data_name: str | None = None    # name of the data-check property that supplies the hard values (default: setup.name)

    def __post_init__(self):
        k = self.keyword
        if not (isinstance(k, str) and 1 <= len(k) <= 8 and k.isalnum() and k.upper() == k and k[0].isalpha()):
            raise RealizationError("the keyword must be 1-8 upper-case letters/digits starting with a letter")
        if k in RESERVED:
            raise RealizationError(f"{k} is written by the permeability step or the grid; choose another keyword")
        if not isinstance(self.setup, PropertySetup):
            raise RealizationError("a property job needs a PropertySetup")


@dataclass(frozen=True)
class RealizationPlan:
    jobs: tuple                     # PropertyJob, ...
    perm: PermSetup | None
    count: int
    base_seed: int
    confidence: float = 0.95
    on_fail: str = "keep"
    max_redraws: int = 0            # needed (>= 1) for on_fail == 'redraw'
    redraw: str = "both"            # what a redraw changes: 'both' facies and properties, or 'properties' only

    def __post_init__(self):
        if not (isinstance(self.count, int) and not isinstance(self.count, bool) and 1 <= self.count <= MAX_REALIZATIONS):
            raise RealizationError(f"the number of realizations must be a whole number from 1 to {MAX_REALIZATIONS}")
        if not (isinstance(self.base_seed, int) and not isinstance(self.base_seed, bool) and 0 <= self.base_seed < 2**31):
            raise RealizationError("the base seed must be a whole number between 0 and 2^31 - 1")
        if not (isinstance(self.confidence, (int, float)) and 0.5 <= self.confidence < 1):
            raise RealizationError("the confidence must be a number from 0.5 up to (not including) 1")
        if self.on_fail not in ON_FAIL:
            raise RealizationError(f"on_fail must be one of {', '.join(ON_FAIL)}")
        if self.redraw not in REDRAW:
            raise RealizationError(f"redraw must be one of {', '.join(REDRAW)}")
        if self.on_fail == "redraw" and not (isinstance(self.max_redraws, int) and self.max_redraws >= 1):
            raise RealizationError("redraw needs the largest number of redraws (a whole number >= 1)")
        if not self.jobs:
            raise RealizationError("define at least one property")
        kws = [j.keyword for j in self.jobs]
        if len(set(kws)) != len(kws):
            raise RealizationError("two properties have the same keyword")
        names = [j.setup.name for j in self.jobs]
        if len(set(names)) != len(names):
            raise RealizationError("two properties have the same name")
        for i, j in enumerate(self.jobs):
            cw = j.setup.correlate_with
            if cw is not None and cw not in names[:i]:
                raise RealizationError(f"{j.setup.name} is correlated with {cw!r}, which must be a property listed before it")
        if self.perm is not None:
            j = [j for j in self.jobs if j.setup.name == self.perm.porosity]
            if not j:
                raise RealizationError(f"the permeability step uses porosity {self.perm.porosity!r}, which is not one of the properties")


def seed_for(plan: RealizationPlan, n: int, attempt: int, stage: int) -> int:
    """Seed of one stage (0 facies, 1.. properties in job order, 100 permeability) of realization n at an attempt."""
    return int(np.random.SeedSequence([plan.base_seed, n, attempt, stage]).generate_state(1)[0]) & 0x7FFFFFFF


def plan_digest(plan: RealizationPlan, facies_setup) -> str:
    return hashlib.sha256((repr(plan) + "|" + repr(facies_setup)).encode()).hexdigest()[:16]


# --- effective sample size and tests --------------------------------------------------------------------------------------
def integral_volume(model, cell_volume: float) -> float:
    """Integral of the correlation function of a variogram model, in m^3 (nugget counts as one cell volume)."""
    sill = model.nugget + sum(s.contribution for s in model.structures)
    shape = {"spherical": math.pi / 6.0, "exponential": 8.0 * math.pi / 27.0, "gaussian": (math.pi / 3.0) ** 1.5}
    a = model.nugget * cell_volume
    for s in model.structures:
        a += s.contribution * shape[s.type] * s.range_major * s.range_minor * s.range_vert
    return a / sill


def _ess(n: int, cell_volume: float, model) -> float:
    return float(min(n, n * cell_volume / integral_volume(model, cell_volume)))


def _tests(s: np.ndarray, n_eff: float) -> dict:
    """p-values of the mean, variance and shape of normal scores ``s`` against N(0, 1) with n_eff independent samples."""
    s = np.sort(s)
    n = len(s)
    zm = float(s.mean()) * math.sqrt(n_eff)
    p_mean = float(2 * norm.sf(abs(zm)))
    v = float(s.var(ddof=1)) if n > 1 else float("nan")
    c = chi2.cdf((n_eff - 1) * v, n_eff - 1)
    p_var = float(2 * min(c, 1 - c))
    F = norm.cdf(s)
    i = np.arange(1, n + 1)
    D = float(max((i / n - F).max(), (F - (i - 1) / n).max()))
    p_ks = float(kstwobign.sf(D * math.sqrt(n_eff)))
    return dict(mean=float(s.mean()), variance=v, D=D, p_mean=p_mean, p_variance=p_var, p_shape=p_ks)


def check_realization(project, model, plan: RealizationPlan, facies, values: dict, per: dict, props, scheme) -> dict:
    """Checks one realization. values {keyword: array}, per {keyword: PropertyResult.per_domain}, props the local target facies
    proportions {interval: (nx, ny, nlayers, nfac)}. -> dict(rows, ntests, alpha, failed, text)."""
    li = np.asarray(model.report.layer_interval)
    rows = []
    by_name = {iv.name: iv for iv in project.chart.intervals}
    for job in plan.jobs:
        for (name, code), d in per[job.keyword].items():
            a = model.interval_names.index(name)
            ks = np.nonzero(li == a)[0]
            dom = facies[:, :, ks] == code
            vals = values[job.keyword][:, :, ks][dom].astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                sc = d["score"](vals)
            sc = sc[np.isfinite(sc)]
            n = len(sc)
            ne = _ess(n, d["cell_volume"], d["variogram"]) if n > 1 else 0.0
            row = dict(kind=job.keyword, interval=name, facies=code, n=n, n_eff=ne)
            if ne < MIN_EFFECTIVE:
                row["testable"] = False
            else:
                row.update(_tests(sc, ne), testable=True)
            rows.append(row)
    # facies fractions per interval against the mean local target proportion
    fac_codes = [f.code for f in scheme]
    for a, name in enumerate(model.interval_names):
        ks = np.nonzero(li == a)[0]
        P = props.get(name)
        if P is None:
            continue
        sub = facies[:, :, ks]
        act = sub > 0
        n = int(act.sum())
        if n == 0:
            continue
        for fi, code in enumerate(fac_codes):
            pi = float(P[..., fi][act].mean())
            ph = float((sub[act] == code).mean())
            m = project.variogram_for(by_name[name].uid, scheme[fi].name) if project.facies_sim.method == "sis" else \
                project.variogram_for(by_name[name].uid, project.facies_sim.tgs.variogram)
            row = dict(kind="facies", interval=name, facies=code, n=n, target=pi, observed=ph)
            if m is None or not (0 < pi < 1):
                row["testable"] = False
            else:
                cv = _cell_vol(model, ks)
                ne = _ess(n, cv, m)
                row["n_eff"] = ne
                if ne < MIN_EFFECTIVE:
                    row["testable"] = False
                else:
                    z = (ph - pi) / math.sqrt(pi * (1 - pi) / ne)
                    row.update(testable=True, p_proportion=float(2 * norm.sf(abs(z))))
            rows.append(row)
    pkeys = ("p_mean", "p_variance", "p_shape", "p_proportion")
    ntests = sum(1 for r in rows if r.get("testable") for k in pkeys if k in r)
    alpha = (1.0 - plan.confidence) / max(ntests, 1)
    failed = []
    for r in rows:
        for k in pkeys:
            if r.get("testable") and k in r and r[k] < alpha:
                failed.append((r["kind"], r["interval"], r["facies"], k[2:]))
    untested = [r for r in rows if not r.get("testable")]
    lines = [f"{ntests} tests, significance {alpha:.2g} each (confidence {plan.confidence:g}, Bonferroni); "
             f"{len(failed)} failed, {len(untested)} domain(s) not testable (fewer than {MIN_EFFECTIVE:g} effective samples)"]
    for kind, name, code, what in failed:
        lines.append(f"  FAILED {kind} / {name} / facies {code}: {what}")
    return dict(rows=rows, ntests=ntests, alpha=alpha, failed=failed, text="\n".join(lines))


def _cell_vol(model, ks) -> float:
    from .grid.builder import _cell_thickness
    g = model.grid
    th = _cell_thickness(g.zcorn, g.nx, g.ny, g.nz)
    th = np.where(np.isfinite(th) & (th > 0), th, np.nan)
    c = np.asarray(g.coord, float)
    dx = float(np.nanmean(np.hypot(np.diff(c[:, :, 0], axis=1), np.diff(c[:, :, 1], axis=1)))) if g.nx > 0 else 1.0
    dy = float(np.nanmean(np.hypot(np.diff(c[:, :, 0], axis=0), np.diff(c[:, :, 1], axis=0)))) if g.ny > 0 else 1.0
    return float(np.nanmean(th[:, :, ks])) * dx * dy


# --- realization ------------------------------------------------------------------------------------------------------------
@dataclass
class Realization:
    n: int
    attempt: int
    status: str                          # 'ok' | 'flagged' | 'discarded'
    facies: np.ndarray | None = None     # uint8 codes
    props: dict = field(default_factory=dict)      # {keyword: float32 (nx, ny, nz), NaN = no facies}
    permx: np.ndarray | None = None
    permy: np.ndarray | None = None
    permz: np.ndarray | None = None
    check: dict | None = None
    from_cache: bool = False
    cache_note: str = ""

    def light(self):
        """The same record without the arrays (status, attempt, check); ``export`` regenerates or reloads the arrays."""
        import dataclasses
        return dataclasses.replace(self, facies=None, props={}, permx=None, permy=None, permz=None)


class RealizationManager:
    def __init__(self, project, model, plan: RealizationPlan, regression=None, cache_dir=None):
        if project.facies_sim is None:
            raise RealizationError("define the facies simulation first (Facies > Simulate facies)")
        self.project, self.model, self.plan = project, model, plan
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.digest = plan_digest(plan, project.facies_sim)
        self._prepared = None
        self._hard: dict = {}
        self.regression = regression
        if plan.perm is not None and any(r.source == "core" for _, r in plan.perm.relations) and regression is None:
            from .datacheck import DataCheckError, run_checks
            try:
                self.regression = run_checks(project, model).regression
            except DataCheckError as exc:
                raise RealizationError(f"the core permeability relation needs the data checks: {exc}") from None
        self._facies0: dict = {}

    # -- generation --------------------------------------------------------------------------------------------------------
    def _prep(self):
        if self._prepared is None:
            try:
                self._prepared = fsim.prepare(self.project, self.model, self.project.facies_sim)
            except fsim.FaciesSimError as exc:
                raise RealizationError(str(exc)) from None
        return self._prepared

    def _hard_data(self, job):
        from .property_sim import hard_property_data
        key = job.data_name or job.setup.name
        if key not in self._hard:
            needs = any(s == "data" for _, s in job.setup.sources)
            self._hard[key] = hard_property_data(self.project, self.model, key) if needs else []
        return self._hard[key]

    def has_cache(self, n: int) -> bool:
        return self.cache_dir is not None and self._file(n).exists()

    def _active_cells(self) -> int:
        if not hasattr(self, "_nact"):
            from .grid.builder import _cell_thickness
            g = self.model.grid
            th = _cell_thickness(g.zcorn, g.nx, g.ny, g.nz)
            self._nact = int(((np.asarray(g.actnum) > 0) & np.isfinite(th) & (th > 0)).sum())
        return self._nact

    def generate(self, n: int, attempt: int = 0, progress=None):
        """-> (facies uint8, {keyword: float32}, (permx, permy, permz) float32 or None, check). No cache, no failure handling.
        ``progress(label, fraction)`` gets the fraction (0..1) of this attempt done, weighting facies and every property by the
        number of active cells, permeability by a tenth of that and the target check by a twentieth; it may raise
        progress.Cancelled, which leaves nothing stored."""
        plan = self.plan
        A = max(self._active_cells(), 1)
        weights = [("facies", 1.0)] + [(j.setup.name, 1.0) for j in plan.jobs] + ([("permeability", 1.0 if plan.perm.scatter_mode == "correlated" else 0.1)] if plan.perm else []) + [("target check", 0.05)]
        W = sum(w for _, w in weights)
        offs, acc = {}, 0.0
        for label, w in weights:
            offs[label] = (acc / W, w / W)
            acc += w

        def stage(label):
            o, w = offs[label]

            def cb(text, done=0.0, total=1.0):
                if progress is not None:
                    progress(f"realization {n}: {text}", o + w * (done / total if total else 0.0))
            return cb
        if not (isinstance(n, int) and 0 <= n < plan.count):
            raise RealizationError(f"realization number must be 0..{plan.count - 1}")
        if progress is not None:
            progress(f"realization {n}: starting", 0.0)
        fattempt = 0 if plan.redraw == "properties" else attempt
        prepared = self._prep()
        try:
            res, _ = fsim.run(self.project, self.model, self.project.facies_sim, prepared, seed_for(plan, n, fattempt, 0),
                              stage("facies"))
            facies = np.asarray(res.facies).astype(np.uint8)
            values, per, results = {}, {}, {}
            for j, job in enumerate(plan.jobs):
                cw = job.setup.correlate_with
                r = simulate_property(self.project, self.model, facies, job.setup, self._hard_data(job), seed_for(plan, n, attempt, 1 + j),
                                      stage(job.setup.name), primary=results[cw] if cw is not None else None)
                if any(q.setup.correlate_with == job.setup.name for q in plan.jobs[j + 1:]):
                    results[job.setup.name] = r                 # kept only while a later property needs it
                values[job.keyword] = r.values.astype(np.float32)
                per[job.keyword] = r.per_domain
            perm = None
            if plan.perm is not None:
                phi_job = next(j for j in plan.jobs if j.setup.name == plan.perm.porosity)
                pr = simulate_permeability(self.project, self.model, facies, values[phi_job.keyword].astype(float), plan.perm,
                                           self.regression, seed_for(plan, n, attempt, 100), stage("permeability"))
                perm = (pr.permx.astype(np.float32), pr.permy.astype(np.float32), pr.permz.astype(np.float32))
        except (fsim.FaciesSimError, PropertySimError, PermSimError) as exc:
            raise RealizationError(str(exc)) from None
        stage("target check")("checking against the targets")
        check = check_realization(self.project, self.model, plan, facies, values, per, prepared[0], self.project.facies_setup.scheme)
        if progress is not None:
            progress(f"realization {n}: done", 1.0)
        return facies, values, perm, check

    def realize(self, n: int, progress=None, keep_arrays: bool = True) -> Realization:
        """Realization n with the plan's failure handling; uses the cache when there is a valid entry."""
        cached = self._load(n)
        if cached is not None:
            return cached if keep_arrays else cached.light()
        plan = self.plan
        attempt = 0
        while True:
            facies, values, perm, check = self.generate(n, attempt, progress)
            if not check["failed"]:
                status = "ok"
                break
            if plan.on_fail == "discard":
                status = "discarded"
                break
            if plan.on_fail == "redraw" and attempt < plan.max_redraws:
                attempt += 1
                continue
            status = "flagged"
            break
        if status == "discarded":
            r = Realization(n, attempt, status, check=check)
        else:
            r = Realization(n, attempt, status, facies, values, *(perm or (None, None, None)), check=check)
        self._store(r)
        return r if keep_arrays else r.light()

    def run_all(self, progress=None) -> list:
        """Serial run; the records keep no arrays (memory), export reloads or regenerates them."""
        out = []
        for n in range(self.plan.count):
            out.append(self.realize(n, keep_arrays=False))
            if progress is not None:
                progress(n + 1, self.plan.count, out[-1])
        return out

    # -- cache -------------------------------------------------------------------------------------------------------------
    def _file(self, n):
        return self.cache_dir / f"realization_{n:03d}.npz"

    def _store(self, r: Realization):
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        arrs = {}
        if r.facies is not None:
            arrs["facies"] = r.facies
            for k, v in r.props.items():
                arrs["p_" + k] = v
            if r.permx is not None:
                arrs.update(permx=r.permx, permy=r.permy, permz=r.permz)
        failed = [list(map(str, f)) for f in (r.check or {}).get("failed", [])]
        meta = dict(version=GENERATOR_VERSION, digest=self.digest, n=r.n, attempt=r.attempt, status=r.status,
                    failed=failed, text=(r.check or {}).get("text", ""))
        import json
        buf = io.BytesIO()
        np.savez_compressed(buf, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrs)
        tmp = self._file(r.n).with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(self._file(r.n))

    def _load(self, n):
        if self.cache_dir is None or not self._file(n).exists():
            return None
        import json
        try:
            z = np.load(self._file(n))
            meta = json.loads(bytes(z["meta"]).decode())
            names = set(z.files)
        except Exception as exc:
            raise RealizationError(f"cache file for realization {n} is unreadable ({exc}); delete it to regenerate") from None
        if meta["digest"] != self.digest:
            raise RealizationError(f"the cache of realization {n} was made from a different plan or setup; clear the cache folder")
        note = ""
        if meta.get("version") != GENERATOR_VERSION:
            note = (f"cache written by generator {meta.get('version')}, this is {GENERATOR_VERSION}: the cached arrays are used as "
                    "written; regenerating from the seeds would not reproduce them")
        r = Realization(n, int(meta["attempt"]), meta["status"], check=dict(failed=[tuple(f) for f in meta["failed"]], text=meta["text"]),
                        from_cache=True, cache_note=note)
        if "facies" in names:
            r.facies = z["facies"]
            r.props = {k[2:]: z[k] for k in names if k.startswith("p_")}
            if "permx" in names:
                r.permx, r.permy, r.permz = z["permx"], z["permy"], z["permz"]
        return r

    # -- reports -----------------------------------------------------------------------------------------------------------
    def summary(self, rs: list) -> str:
        n = len(rs)
        c = {s: sum(1 for r in rs if r.status == s) for s in ("ok", "flagged", "discarded")}
        exp = (1 - self.plan.confidence) * 100
        lines = [f"{n} realization(s): {c['ok']} passed, {c['flagged']} flagged, {c['discarded']} discarded; "
                 f"base seed {self.plan.base_seed}, generator {GENERATOR_VERSION}"]
        bad = (c["flagged"] + c["discarded"]) / max(n, 1) * 100
        lines.append(f"Share failing: {bad:.0f}% (at the chosen confidence about {exp:.0f}% or fewer are expected to fail by chance)")
        if n >= 20 and bad > 3 * exp + 5:
            lines.append("WARNING: far more realizations fail than chance allows - the target or the setup does not match what the "
                         "simulation can deliver (typical causes: clustered well data used as the target, a variogram that is too long "
                         "for the domain, a SIS bias). Look at the failed tests.")
        for r in rs:
            if r.status != "ok":
                lines.append(f"Realization {r.n} ({r.status}, attempt {r.attempt}): " + (r.check or {}).get("text", "").replace("\n", "; "))
            if r.cache_note:
                lines.append(f"Realization {r.n}: {r.cache_note}")
        return "\n".join(lines)

    # -- export ------------------------------------------------------------------------------------------------------------
    def export(self, n: int, path, *, allow_flagged: bool = False, progress=None) -> list:
        """Writes realization n as a GRDECL (grid, all exported properties, PERMX/Y/Z) with the fault transmissibility computed
        from this realization's properties (SGR from VSHALE when a property with keyword VSHALE exists). Returns the files written."""
        from .grid_definition import write_model
        r = self.realize(n, progress)
        if r.status == "discarded":
            raise RealizationError(f"realization {n} was discarded by the target check; nothing to export")
        if r.status == "flagged" and not allow_flagged:
            raise RealizationError(f"realization {n} is flagged ({(r.check or {}).get('text', '')}); export it with allow_flagged "
                                   "after you have looked at the failed tests")
        props = {}
        everything = {}
        for job in self.plan.jobs:
            a = np.nan_to_num(r.props[job.keyword], nan=0.0)
            everything[job.keyword] = a
            if job.export:
                props[job.keyword] = a
        if r.permx is not None:
            for kw, arr in (("PERMX", r.permx), ("PERMY", r.permy), ("PERMZ", r.permz)):
                a = np.nan_to_num(arr, nan=0.0)
                props[kw] = a
                everything[kw] = a
        return write_model(self.model, path, props=props, fault_props=everything)


# --- persistence of the plan ------------------------------------------------------------------------------------------------
def to_state(plan):
    if plan is None:
        return None
    from .perm_sim import Relation
    from .property_sim import Distribution
    pj = []
    for j in plan.jobs:
        s = j.setup
        pj.append(dict(keyword=j.keyword, export=j.export, data_name=j.data_name,
                       setup=dict(name=s.name, variogram=s.variogram, sources=[[list(k), v] for k, v in s.sources],
                                  user=[[list(k), dict(kind=d.kind, mean=d.mean, std=d.std)] for k, d in s.user], min_data=s.min_data,
                                  conflict_probability=s.conflict_probability, lower=s.lower, upper=s.upper,
                                  max_neighbours=s.max_neighbours, search_factor=s.search_factor, levels=s.levels,
                                  correlate_with=s.correlate_with, correlation=[[list(k), v] for k, v in s.correlation])))
    pm = None
    if plan.perm is not None:
        p = plan.perm
        pm = dict(porosity=p.porosity, relations=[[list(k), {f: getattr(r, f) for f in r.__dataclass_fields__}] for k, r in p.relations],
                  kv_kh=[[list(k), v] for k, v in p.kv_kh], ky_kx=p.ky_kx, core_shift=p.core_shift, extrapolation=p.extrapolation,
                  lower=p.lower, upper=p.upper, scatter_mode=p.scatter_mode, scatter_variogram=p.scatter_variogram,
                  scatter_neighbours=p.scatter_neighbours, scatter_search_factor=p.scatter_search_factor, scatter_levels=p.scatter_levels)
    return dict(jobs=pj, perm=pm, count=plan.count, base_seed=plan.base_seed, confidence=plan.confidence, on_fail=plan.on_fail,
                max_redraws=plan.max_redraws, redraw=plan.redraw)


def from_state(d):
    if d is None:
        return None
    from .perm_sim import Relation
    from .property_sim import Distribution
    try:
        jobs = []
        for j in d["jobs"]:
            s = j["setup"]
            setup = PropertySetup(s["name"], s["variogram"], tuple((tuple(k), v) for k, v in s["sources"]),
                                  tuple((tuple(k), Distribution(**dd)) for k, dd in s["user"]), s["min_data"], s["conflict_probability"],
                                  s["lower"], s["upper"], s["max_neighbours"], s["search_factor"], s["levels"],
                                  s.get("correlate_with"), tuple((tuple(k), v) for k, v in s.get("correlation", ())))
            jobs.append(PropertyJob(setup, j["keyword"], j["export"], j["data_name"]))
        perm = None
        if d["perm"] is not None:
            p = d["perm"]
            perm = PermSetup(p["porosity"], tuple((tuple(k), Relation(**r)) for k, r in p["relations"]),
                             tuple((tuple(k), v) for k, v in p["kv_kh"]), p["ky_kx"], p["core_shift"], p["extrapolation"], p["lower"], p["upper"],
                             p.get("scatter_mode", "independent"), p.get("scatter_variogram"), p.get("scatter_neighbours"),
                             p.get("scatter_search_factor"), p.get("scatter_levels"))
        return RealizationPlan(tuple(jobs), perm, d["count"], d["base_seed"], d["confidence"], d["on_fail"], d["max_redraws"], d["redraw"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, (RealizationError, PropertySimError, PermSimError)):
            raise RealizationError(str(exc)) from None
        raise RealizationError(f"bad realization plan: {exc}") from None
