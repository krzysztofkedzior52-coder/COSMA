"""Fault transmissibility: what each fault does to flow across it. Every fault needs an explicit choice.

Modes
  open        no multiplier (1): the fault is a plain neighbour connection. The user says so; it is never assumed.
  fixed       one user-chosen multiplier m >= 0 on every connection of the fault (0 = sealing).
  calculated  only for faults whose throw is not negligible. Method ``slab``: the fault is a slab of thickness
              t and permeability kf between the two cells, in series with the matrix on each side:

                  TM = 1 / (1 + t (2/kf - 1/ki - 1/kj) / (Li/ki + Lj/kj))

              (ki, kj, Li, Lj: permeability and length across the fault of the two cells; derived from series
              resistance. Manzocchi et al. 1999 give the same quantity in terms of fault thickness and permeability;
              the equation number in their paper has not been checked here). t is either given, or a user
              ratio times the fault's median throw. kf is given by the user. One multiplier is written per fault: the
              transmissibility-weighted mean over its faces.
              Method ``sgr``: the same slab, but kf is evaluated per cell pair from the shale gouge ratio computed from
              the VSHALE cell property (cosma.fault_sgr) and the face's throw, with a published relation from
              cosma.fault_relations or one of the user's own stored methods (an expression for log10 kf). The slab thickness is the user's ratio times the
              face's throw, or a constant. Faces with throw below the user's minimum, or whose slipped interval reaches
              above the model, are left out of the weighted mean and counted in the export comments.

Only the multiplier is exported (OPM FAULTS + MULTFLT), so it applies to juxtaposed connections across the throw as
well as to neighbours. Calculated multipliers need the property step (PERMX, PERMY and, for sgr, VSHALE): until
those exist the grid exports without the fault include and says so.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .fault_relations import RelationError

MODES = ("open", "fixed", "calculated")
METHODS = ("slab", "sgr")


class TransmissibilityError(ValueError):
    pass


class PendingProperties(TransmissibilityError):
    """A calculated multiplier cannot be computed yet: cell properties it needs do not exist."""

    def __init__(self, fault, missing):
        self.fault, self.missing = fault, tuple(missing)
        super().__init__(f"fault {fault!r}: a calculated transmissibility needs {', '.join(missing)} on the grid, which "
                         "exist only after the property step; choose 'open' or 'fixed' for now, or build properties first")


def _pos(x, what):
    if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0):
        raise TransmissibilityError(f"{what} must be a number > 0")
    return float(x)


@dataclass(frozen=True)
class FaultTransmissibility:
    mode: str
    multiplier: float | None = None          # fixed
    method: str | None = None                # calculated
    kf: float | None = None                  # mD, fault-rock permeability (slab)
    thickness: float | None = None           # m, fault-rock thickness (slab) ...
    thickness_per_throw: float | None = None  # ... or this fraction of the fault's median throw (sgr: of the face's throw)
    relation: str | None = None              # sgr: the built-in library entry (cosma.fault_relations) ...
    expression: str | None = None            # ... or the user's own expression for log10 kf (a copy, so it cannot drift) ...
    expression_name: str | None = None       # ... and the name of the stored method it came from
    min_throw: float | None = None           # sgr: faces with less throw (m) are left out

    def __post_init__(self):
        if self.mode not in MODES:
            raise TransmissibilityError(f"transmissibility mode {self.mode!r} is not one of {', '.join(MODES)}")
        if self.mode == "fixed":
            m = self.multiplier
            if not (isinstance(m, (int, float)) and math.isfinite(m) and m >= 0):
                raise TransmissibilityError("a fixed multiplier must be a number >= 0 (0 = sealing)")
        elif self.multiplier is not None:
            raise TransmissibilityError(f"mode {self.mode!r} takes no multiplier")
        if self.mode == "calculated":
            if self.method not in METHODS:
                raise TransmissibilityError(f"calculation method {self.method!r} is not one of {', '.join(METHODS)}")
            if self.method == "slab":
                _pos(self.kf, "the fault-rock permeability")
                if any(v is not None for v in (self.relation, self.expression, self.expression_name, self.min_throw)):
                    raise TransmissibilityError("method 'slab' takes no SGR relation, expression or minimum throw")
            else:
                from .fault_relations import RelationError, resolve
                if self.kf is not None:
                    raise TransmissibilityError("method 'sgr' takes no fixed fault-rock permeability")
                try:
                    resolve(self.relation, self.expression)
                    if (self.expression is None) != (self.expression_name is None):
                        raise RelationError("a user expression needs the name of its method")
                except RelationError as exc:
                    raise TransmissibilityError(str(exc)) from None
                _pos(self.min_throw, "the minimum throw")
            if (self.thickness is None) == (self.thickness_per_throw is None):
                raise TransmissibilityError("give either a fault-rock thickness or a thickness/throw ratio, not both")
            if self.thickness is not None:
                _pos(self.thickness, "the fault-rock thickness")
            else:
                _pos(self.thickness_per_throw, "the thickness/throw ratio")
        elif any(v is not None for v in (self.method, self.kf, self.thickness, self.thickness_per_throw, self.relation,
                                         self.expression, self.expression_name, self.min_throw)):
            raise TransmissibilityError(f"mode {self.mode!r} takes no calculation settings")

    def describe(self) -> str:
        if self.mode == "open":
            return "open (no multiplier)"
        if self.mode == "fixed":
            return "sealing (0)" if self.multiplier == 0 else f"fixed multiplier {self.multiplier:g}"
        t = f"{self.thickness:g} m" if self.thickness is not None else f"{self.thickness_per_throw:g} x throw"
        if self.method == "sgr":
            from .fault_relations import resolve
            rel = f"built-in {self.relation}" if self.relation else f"own method {self.expression_name!r}"
            return (f"calculated (SGR from VSHALE, {rel}: {resolve(self.relation, self.expression).describe()}; thickness {t}; "
                    f"faces with throw < {self.min_throw:g} m left out)")
        return f"calculated (slab, kf {self.kf:g} mD, thickness {t})"


def slab_multiplier(ki, kj, Li, Lj, kf, t):
    """Transmissibility multiplier of a fault slab between two cells (arrays broadcast). Never above 1 unless the
    fault rock is more permeable than the matrix, in which case it exceeds 1."""
    ki, kj, Li, Lj = (np.asarray(a, float) for a in (ki, kj, Li, Lj))
    return 1.0 / (1.0 + t * (2.0 / kf - 1.0 / ki - 1.0 / kj) / (Li / ki + Lj / kj))


def to_state(d: dict) -> dict:
    return {n: dict(mode=v.mode, multiplier=v.multiplier, method=v.method, kf=v.kf, thickness=v.thickness,
                    thickness_per_throw=v.thickness_per_throw, relation=v.relation, expression=v.expression,
                    expression_name=v.expression_name, min_throw=v.min_throw) for n, v in d.items()}


def from_state(rows: dict) -> dict:
    out = {}
    try:
        for n, r in rows.items():
            r = dict(r)
            k = r.pop("constants", None)           # files written by the first SGR build stored four constants
            if k is not None and r.get("expression") is None:
                a, b, c, d = (float(x) for x in k)
                r["expression"] = f"{d!r} + {a!r}*SGR + {b!r}*log10(D)*(1 - SGR)**{c!r}"
                r["expression_name"] = "own constants (migrated)"
            out[str(n)] = FaultTransmissibility(r["mode"], r.get("multiplier"), r.get("method"), r.get("kf"),
                                                r.get("thickness"), r.get("thickness_per_throw"), r.get("relation"),
                                                r.get("expression"), r.get("expression_name"), r.get("min_throw"))
    except (KeyError, TypeError, AttributeError, RelationError) as exc:
        raise TransmissibilityError(f"bad fault transmissibility record: {exc}") from None
    return out


# -- export ------------------------------------------------------------------------------------------------------

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,7}$")


def deck_names(names) -> dict:
    """Name used in the deck for each fault: the fault's own name when it is a valid 8-character simulator name,
    otherwise FLT001, FLT002 ... (the mapping is written as comments)."""
    out, used = {}, set()
    for n in names:
        if _NAME.match(n) and n.upper() not in used:
            out[n] = n.upper()
            used.add(n.upper())
    k = 0
    for n in names:
        if n not in out:
            k += 1
            out[n] = f"FLT{k:03d}"
    return out


def _runs(flags):
    """(start, end) index pairs of consecutive True values."""
    out, s = [], None
    for k, f in enumerate(list(flags) + [False]):
        if f and s is None:
            s = k
        elif not f and s is not None:
            out.append((s, k - 1))
            s = None
    return out


def face_records(fi, fj, active_columns):
    """FAULTS records (i1, i2, j1, j2, face) with 1-based indices (all layers), merging runs of faces. An I-direction
    face between (i,j) and (i+1,j) is written on cell (i,j) as face 'X'; a J-direction face between (i,j) and
    (i,j+1) on cell (i,j) as 'Y'. Faces where neither column is active are skipped."""
    nx, ny = active_columns.shape
    a = active_columns
    recs = []
    keep_i = fi & (a[:-1, :] | a[1:, :])
    for i in range(nx - 1):
        for s, e in _runs(keep_i[i, :]):
            recs.append((i + 1, i + 1, s + 1, e + 1, "X"))
    keep_j = fj & (a[:, :-1] | a[:, 1:])
    for j in range(ny - 1):
        for s, e in _runs(keep_j[:, j]):
            recs.append((s + 1, e + 1, j + 1, j + 1, "Y"))
    return recs


def calculated_detail(spec: FaultTransmissibility, fi, fj, grid, report) -> dict:
    """Weighted-mean multiplier of one fault and how it was obtained. Needs PERMX and PERMY on the grid (and VSHALE for
    method sgr), else raises PendingProperties. The weight of each cell pair is its transmissibility without the fault.
    Returns dict(m, n, left_out={reason: count}, sgr=(min, mean, max) or None, kf=(min, max) or None)."""
    need = ["PERMX", "PERMY"] + (["VSHALE"] if spec.method == "sgr" else [])
    missing = [p for p in need if p not in grid.props]
    if missing:
        raise PendingProperties(report.name, missing)
    from .grid.builder import _cell_thickness
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    act = grid.actnum.astype(bool)
    thick = _cell_thickness(grid.zcorn, nx, ny, nz)
    dx = float(np.hypot(*(grid.coord[0, 1, :2] - grid.coord[0, 0, :2])))
    dy = float(np.hypot(*(grid.coord[1, 0, :2] - grid.coord[0, 0, :2])))
    sgr_mode = spec.method == "sgr"
    if sgr_mode:
        from .fault_relations import resolve
        from .fault_sgr import REASONS, SGRError, cell_centre_depth, check_vshale, window_sgr
        try:
            vsh = check_vshale(grid.props["VSHALE"])
        except SGRError as exc:
            raise TransmissibilityError(str(exc)) from None
        consts = resolve(spec.relation, spec.expression)
        zc = cell_centre_depth(grid.zcorn)
    elif spec.thickness is not None:
        t_fixed = spec.thickness
    else:
        thr = [abs(x) for x in report.median_throw if np.isfinite(x)]
        if not thr or max(thr) <= 0:
            raise TransmissibilityError(f"{report.name}: no measurable throw to take the thickness from")
        t_fixed = spec.thickness_per_throw * float(np.median(thr))
    num = den = 0.0
    n = 0
    left = {}
    sg, kfs = [], []
    w_sgr = 0.0
    for faces, perm, L, W, step in ((fi, grid.props["PERMX"], dx, dy, (1, 0)), (fj, grid.props["PERMY"], dy, dx, (0, 1))):
        idx = np.argwhere(faces)
        if not len(idx):
            continue
        ia, ja = idx[:, 0], idx[:, 1]
        ib, jb = ia + step[0], ja + step[1]
        ka, kb = perm[ia, ja, :], perm[ib, jb, :]
        both = act[ia, ja, :] & act[ib, jb, :]
        ok = both & (ka > 0) & (kb > 0)
        if sgr_mode:
            T, sgr, status = window_sgr(zc[ia, ja, :], zc[ib, jb, :], thick[ia, ja, :], thick[ib, jb, :],
                                        vsh[ia, ja, :], vsh[ib, jb, :], spec.min_throw)
            for code, why in REASONS.items():
                c = int((ok & (status == code)).sum())
                if c:
                    left[why] = left.get(why, 0) + c
            ok = ok & (status == 0)
        if not ok.any():
            continue
        area = W * 0.5 * (thick[ia, ja, :] + thick[ib, jb, :])
        t0 = (area / (L / (2 * np.where(ok, ka, 1.0)) + L / (2 * np.where(ok, kb, 1.0))))[ok]
        a_, b_ = ka[ok], kb[ok]
        if sgr_mode:
            kf = 10.0 ** consts.log10_kf(sgr[ok], T[ok])
            t = spec.thickness if spec.thickness is not None else spec.thickness_per_throw * T[ok]
            sg.append(sgr[ok])
            kfs.append(kf)
        else:
            kf, t = spec.kf, t_fixed
        tm = slab_multiplier(a_, b_, L, L, kf, t)
        num += float((t0 * tm).sum())
        den += float(t0.sum())
        n += int(ok.sum())
        if sgr_mode:
            w_sgr += float((t0 * sgr[ok]).sum())
    if den <= 0:
        why = "; ".join(f"{k}: {v}" for k, v in left.items())
        raise TransmissibilityError(f"{report.name}: no active cell pairs across the fault to calculate from"
                                    + (f" ({why})" if why else ""))
    out = dict(m=num / den, n=n, left_out=left, sgr=None, kf=None)
    if sgr_mode:
        s_all, k_all = np.concatenate(sg), np.concatenate(kfs)
        out["sgr"] = (float(s_all.min()), w_sgr / den, float(s_all.max()))
        out["kf"] = (float(k_all.min()), float(k_all.max()))
    return out


def calculated_multiplier(spec: FaultTransmissibility, fi, fj, grid, report) -> tuple:
    """(multiplier, number of cell pairs) of one calculated fault; see calculated_detail."""
    d = calculated_detail(spec, fi, fj, grid, report)
    return d["m"], d["n"]


def faults_text(model, choices: dict) -> str:
    """The FAULTS / MULTFLT include for every fault of ``model`` (a ModelBuild). ``choices``: name -> FaultTransmissibility."""
    reps = {r.name: r for r in model.fault_reports}
    names = deck_names(list(reps))
    act = model.active_columns
    lines = ["-- Fault transmissibility written by COSMA. One line per fault says how its multiplier was chosen."]
    faults_rows, mult_rows = [], []
    for n, rep in reps.items():
        c = choices.get(n)
        if c is None:
            raise TransmissibilityError(f"fault {n!r} has no transmissibility choice")
        recs = face_records(rep.faces_i, rep.faces_j, act)
        if c.mode == "calculated":
            if not rep.has_throw:
                raise TransmissibilityError(f"fault {n!r} has negligible throw: it cannot be calculated; "
                                            "choose open or fixed")
            d = calculated_detail(c, rep.faces_i, rep.faces_j, model.grid, rep)
            m, npairs = d["m"], d["n"]
            how = f"calculated from {npairs} cell pairs, {c.describe()}"
            if d["sgr"] is not None:
                how += (f"; SGR min/weighted mean/max {d['sgr'][0]:.3f}/{d['sgr'][1]:.3f}/{d['sgr'][2]:.3f}; "
                        f"kf {d['kf'][0]:.3g} to {d['kf'][1]:.3g} mD")
            if d["left_out"]:
                how += "; left out: " + ", ".join(f"{v} face-layers ({k})" for k, v in d["left_out"].items())
        else:
            m = 1.0 if c.mode == "open" else c.multiplier
            how = c.describe()
        lines.append(f"-- {names[n]}: {n!r} {how}; multiplier {m:.6g}; {len(recs)} face record(s)")
        if not recs:
            lines.append(f"-- {names[n]}: no cell faces lie on this fault in this grid, nothing written")
            continue
        for r in recs:
            faults_rows.append(f" '{names[n]}' {r[0]} {r[1]} {r[2]} {r[3]} 1 {model.grid.nz} '{r[4]}' /")
        mult_rows.append(f" '{names[n]}' {m:.6g} /")
    text = "\n".join(lines) + "\n"
    if faults_rows:
        text += "FAULTS\n" + "\n".join(faults_rows) + "\n/\n\nMULTFLT\n" + "\n".join(mult_rows) + "\n/\n"
    return text
