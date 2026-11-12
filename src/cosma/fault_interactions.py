"""Fault interactions: where fault traces meet, and which fault is older.

Geometry can say that two traces cross or that one ends on another. It cannot say which is
older for a crossing, so no suggestion is made then. A fault that ends against another is
suggested to be the younger one (it stopped growing at the older fault). Nothing counts until
the user answers: ``answer`` is None until confirmed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np



class InteractionError(ValueError):
    pass


SAME_AGE = "same_age"
UNRELATED = "unrelated"


@dataclass(frozen=True)
class Interaction:
    a: str                      # names sorted so that a < b
    b: str
    kind: str                   # "crossing" | "abutting" | "near"
    x: float                    # map position of the meeting point
    y: float
    ends: str | None = None     # for abutting/near: the fault whose tip meets the other
    suggestion: str | None = None   # older fault's name, SAME_AGE or UNRELATED; None = geometry cannot say
    answer: str | None = None       # the user's confirmed answer, same vocabulary
    note: str = ""

    @property
    def key(self) -> tuple:
        return (self.a, self.b)

    @property
    def confirmed(self) -> bool:
        return self.answer is not None

    def describe(self) -> str:
        where = f"at ({self.x:.0f}, {self.y:.0f})"
        if self.kind == "crossing":
            return f"{self.a} and {self.b} cross {where}"
        if self.kind == "abutting":
            other = self.b if self.ends == self.a else self.a
            return f"{self.ends} ends against {other} {where}"
        other = self.b if self.ends == self.a else self.a
        return f"{self.ends} ends {self.note or 'close to'} {other} {where}"

    def valid_answers(self) -> tuple:
        return (self.a, self.b, SAME_AGE, UNRELATED)


def _seg_intersection(p, p2, q, q2):
    r, s = p2 - p, q2 - q
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
    u = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / den
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return p + t * r
    return None


def _crossings(A, B):
    pts = []
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            x = _seg_intersection(A[i], A[i + 1], B[j], B[j + 1])
            if x is not None:
                pts.append(x)
    return pts


def _end_distance(trace, other):
    """Distance from each end of ``trace`` to the polyline ``other`` (ignoring the 'beside the end' rule)."""
    out = []
    for e in (trace[0], trace[-1]):
        d = np.min([_pt_seg(e, other[i], other[i + 1]) for i in range(len(other) - 1)])
        out.append(float(d))
    return out


def _pt_seg(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 == 0 else float(np.clip(((p - a) @ ab) / L2, 0, 1))
    return float(np.hypot(*(p - (a + t * ab))))


def find_interactions(traces: dict, tol: float) -> list:
    """All pairs of fault traces (name -> (m,2) map-coordinate polyline) that cross, or where one ends within
    ``tol`` metres of the other. ``tol`` has no default: use a cell diagonal of the intended grid."""
    if not (math.isfinite(tol) and tol > 0):
        raise InteractionError("the reach tolerance must be a positive distance")
    names = sorted(traces)
    out = []
    for ai, a in enumerate(names):
        for b in names[ai + 1:]:
            A, B = np.asarray(traces[a], float), np.asarray(traces[b], float)
            if len(A) < 2 or len(B) < 2:
                continue
            pts = _crossings(A, B)
            ea, eb = _end_distance(A, B), _end_distance(B, A)
            a_ends, b_ends = min(ea) <= tol, min(eb) <= tol
            if pts:
                x = np.mean(pts, axis=0)
                if a_ends and not b_ends:
                    out.append(_abut(a, b, a, x, ""))
                elif b_ends and not a_ends:
                    out.append(_abut(a, b, b, x, ""))
                else:
                    note = "" if len(pts) == 1 else f"traces meet {len(pts)} times"
                    out.append(Interaction(a, b, "crossing", float(x[0]), float(x[1]), note=note))
            elif a_ends or b_ends:
                # one end close to the other trace without touching: the tip that is nearer is the one reported
                if min(ea) <= min(eb):
                    k, trace, other, d = a, A, B, min(ea)
                else:
                    k, trace, other, d = b, B, A, min(eb)
                e = trace[0] if (ea if k == a else eb)[0] <= (ea if k == a else eb)[1] else trace[-1]
                out.append(Interaction(a, b, "near", float(e[0]), float(e[1]), ends=k,
                                       suggestion=(b if k == a else a),
                                       note=f"{d:.0f} m short of"))
    return out


def _abut(a, b, ender, x, note):
    older = b if ender == a else a
    return Interaction(a, b, "abutting", float(x[0]), float(x[1]), ends=ender, suggestion=older, note=note)


def carry_answers(new: list, old: list, tol: float) -> tuple:
    """Keep earlier confirmed answers where the same pair still meets in the same way near the same place.
    Returns (interactions, dropped) with ``dropped`` the answers that no longer apply."""
    olds = {i.key: i for i in old if i.confirmed}
    kept, used = [], set()
    for i in new:
        o = olds.get(i.key)
        if o is not None and o.kind == i.kind and math.hypot(o.x - i.x, o.y - i.y) <= 2 * tol:
            kept.append(replace(i, answer=o.answer))
            used.add(i.key)
        else:
            kept.append(i)
    dropped = [o for k, o in olds.items() if k not in used]
    return kept, dropped


def set_answer(interactions: list, key: tuple, answer: str) -> list:
    out, hit = [], False
    for i in interactions:
        if i.key == key:
            if answer not in i.valid_answers():
                raise InteractionError(f"{answer!r} is not one of {', '.join(i.valid_answers())}")
            out.append(replace(i, answer=answer))
            hit = True
        else:
            out.append(i)
    if not hit:
        raise InteractionError(f"no interaction between {key[0]} and {key[1]}")
    return out


def unconfirmed(interactions: list) -> list:
    return [i for i in interactions if not i.confirmed]


def age_order(names, interactions: list) -> list:
    """Fault names oldest first. Only confirmed 'X is older' answers order faults; the rest keep input order
    (stable). Raises on a cycle (A older than B older than A)."""
    names = list(names)
    edges = {n: set() for n in names}
    for i in interactions:
        if i.answer in (i.a, i.b):
            young = i.b if i.answer == i.a else i.a
            if i.answer in edges and young in edges:
                edges[i.answer].add(young)
    indeg = {n: 0 for n in names}
    for n, ys in edges.items():
        for y in ys:
            indeg[y] += 1
    order, ready = [], [n for n in names if indeg[n] == 0]
    while ready:
        n = ready.pop(0)
        order.append(n)
        for y in sorted(edges[n], key=names.index):
            indeg[y] -= 1
            if indeg[y] == 0:
                ready.append(y)
    if len(order) != len(names):
        stuck = [n for n in names if n not in order]
        raise InteractionError("the confirmed ages contradict each other (a cycle through "
                               + ", ".join(stuck) + ")")
    return order


def to_state(interactions: list) -> list:
    return [dict(a=i.a, b=i.b, kind=i.kind, x=i.x, y=i.y, ends=i.ends, suggestion=i.suggestion,
                 answer=i.answer, note=i.note) for i in interactions]


def from_state(rows: list) -> list:
    out = []
    for r in rows:
        try:
            i = Interaction(str(r["a"]), str(r["b"]), str(r["kind"]), float(r["x"]), float(r["y"]),
                            r.get("ends"), r.get("suggestion"), r.get("answer"), str(r.get("note", "")))
        except (KeyError, TypeError, ValueError) as exc:
            raise InteractionError(f"bad interaction record: {exc}") from None
        if i.kind not in ("crossing", "abutting", "near") or i.a >= i.b:
            raise InteractionError(f"bad interaction record for {i.a}/{i.b}")
        if i.answer is not None and i.answer not in i.valid_answers():
            raise InteractionError(f"interaction {i.a}/{i.b}: answer {i.answer!r} is not a fault of the pair")
        out.append(i)
    return out
