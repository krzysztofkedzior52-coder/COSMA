"""Relations between shale gouge ratio (SGR) and fault-rock permeability.

A relation is an expression for log10(kf) in two variables: ``SGR`` (a fraction, 0-1) and ``D`` (the face's throw, in
metres); kf is in mD. COSMA ships one built-in relation, read from its source paper. Users add and store their own
*methods*: a name, an expression, and where it comes from. Nothing else is assumed.

Expressions are evaluated by a small, closed evaluator (no Python ``eval``): numbers, ``SGR``, ``D``, ``+ - * / **``,
parentheses, and the functions ``log10 ln exp sqrt abs min max``. Anything else is refused with a message.

Examples (the first is the built-in Manzocchi et al. 1999 relation)::

    -4*SGR - 0.25*log10(D)*(1-SGR)**5
    -2 - 3*SGR                      # no dependence on throw
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass

import numpy as np


class RelationError(ValueError):
    pass


VARIABLES = ("SGR", "D")
_FUNCS = {"log10": np.log10, "ln": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "min": np.minimum,
          "max": np.maximum}
_ARITY = {"log10": 1, "ln": 1, "exp": 1, "sqrt": 1, "abs": 1, "min": 2, "max": 2}
MAX_LENGTH = 300


def _check(node, expr):
    if isinstance(node, ast.Expression):
        return _check(node.body, expr)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise RelationError(f"only numbers are allowed, not {node.value!r}")
        if not math.isfinite(node.value):
            raise RelationError("numbers must be finite")
        return
    if isinstance(node, ast.Name):
        if node.id not in VARIABLES:
            raise RelationError(f"unknown name {node.id!r}: use SGR or D, or a function "
                                f"({', '.join(_FUNCS)})" if node.id not in _FUNCS else
                                f"{node.id!r} is a function: write {node.id}(...)")
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _check(node.operand, expr)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        _check(node.left, expr)
        return _check(node.right, expr)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise RelationError(f"only these functions are allowed: {', '.join(_FUNCS)}")
        if len(node.args) != _ARITY[node.func.id]:
            raise RelationError(f"{node.func.id} takes {_ARITY[node.func.id]} argument(s)")
        for a in node.args:
            _check(a, expr)
        return
    raise RelationError(f"this part of the expression is not allowed ({type(node).__name__}); use numbers, SGR, D, "
                        f"+ - * / ** and the functions {', '.join(_FUNCS)}")


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        return {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
                ast.Pow: np.power}[type(node.op)](a, b)
    return _FUNCS[node.func.id](*[_eval(a, env) for a in node.args])


def compile_expression(expr: str):
    """A function f(sgr, throw) -> log10 kf (numpy arrays broadcast) for ``expr``; RelationError if it is not an
    allowed expression or is not finite over a probe grid (SGR 0.05-0.95, D 1-100 m)."""
    if not isinstance(expr, str) or not expr.strip():
        raise RelationError("the expression is empty")
    if len(expr) > MAX_LENGTH:
        raise RelationError(f"the expression is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise RelationError(f"not a valid expression: {exc.msg}") from None
    _check(tree, expr)

    def f(sgr, throw):
        with np.errstate(all="ignore"):
            out = _eval(tree, {"SGR": np.asarray(sgr, float), "D": np.asarray(throw, float)})
        return np.asarray(out, float) * np.ones(np.broadcast(np.asarray(sgr), np.asarray(throw)).shape)

    s, d = np.meshgrid([0.05, 0.25, 0.5, 0.75, 0.95], [1.0, 10.0, 100.0])
    v = f(s, d)
    if not np.isfinite(v).all():
        k = int(np.argmin(np.isfinite(v)))
        raise RelationError(f"the expression is not finite at SGR = {s.ravel()[k]:g}, D = {d.ravel()[k]:g} m")
    return f


@dataclass(frozen=True)
class Method:
    name: str
    expression: str
    citation: str = ""          # where the relation comes from (required for the user's own: they own the choice)
    note: str = ""              # caveats, calibration basin, range

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise RelationError("a method needs a name")
        compile_expression(self.expression)

    def log10_kf(self, sgr, throw):
        """log10 of the fault-rock permeability (mD) for SGR (fraction) and throw D (m)."""
        return compile_expression(self.expression)(sgr, throw)

    def describe(self) -> str:
        return f"log10 kf = {self.expression.strip()}"


LIBRARY = {
    "manzocchi1999": Method(
        "manzocchi1999", "-4*SGR - 0.25*log10(D)*(1 - SGR)**5",
        "Manzocchi, T., Walsh, J.J., Nell, P. and Yielding, G. 1999. Fault transmissibility multipliers for flow "
        "simulation models. Petroleum Geoscience 5, 53-63, Eq. 1: log kf = -4 SGR - (1/4) log(D) (1 - SGR)^5, kf in mD, "
        "D = fault displacement in m.",
        "An empirical prediction of fault-zone permeability from shale content and displacement. The value is the "
        "median of a log-normal permeability distribution covering about two orders of magnitude (D = 20 m, SGR = 0.2: "
        "0.012 mD < kf < 1.2 mD, median 0.12 mD). The authors state that the influence of displacement is not as great "
        "at very low shale content as their data suggest, and that the equation does not give a reliable estimate as "
        "SGR tends to 0."),
}


def library_keys():
    return tuple(LIBRARY)


def resolve(relation: str | None, expression: str | None) -> Method:
    """The method to use: a library entry (``relation``) or the user's expression, exactly one of the two."""
    if (relation is None) == (expression is None):
        raise RelationError("choose either the built-in relation or one of your own methods, not both and not neither")
    if relation is not None:
        if relation not in LIBRARY:
            raise RelationError(f"unknown relation {relation!r}; the library has {', '.join(LIBRARY)}")
        return LIBRARY[relation]
    return Method("expression", expression)


# -- the user's stored methods -----------------------------------------------------------------------------------------

def check_new_method(m: Method, existing) -> None:
    if not isinstance(m, Method):
        raise RelationError("a method must be a Method")
    if m.name in LIBRARY:
        raise RelationError(f"{m.name!r} is the name of the built-in relation; choose another name")
    if not m.citation.strip():
        raise RelationError("say where this relation comes from (paper, report, calibration): it is your responsibility")


def methods_to_json(methods: dict) -> str:
    return json.dumps(dict(cosma_sgr_methods=1, methods=[dict(name=m.name, expression=m.expression, citation=m.citation,
                                                              note=m.note) for m in methods.values()]), indent=1)


def methods_from_json(text: str) -> dict:
    try:
        doc = json.loads(text)
        rows = doc["methods"]
        if doc.get("cosma_sgr_methods") != 1:
            raise KeyError
        out = {}
        for r in rows:
            m = Method(r["name"], r["expression"], r.get("citation", ""), r.get("note", ""))
            check_new_method(m, out)
            if m.name in out:
                raise RelationError(f"method {m.name!r} appears twice")
            out[m.name] = m
        return out
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        if isinstance(exc, RelationError):
            raise
        raise RelationError("this is not a COSMA SGR methods file") from None


def methods_to_state(methods: dict):
    return [[m.name, m.expression, m.citation, m.note] for m in methods.values()]


def methods_from_state(rows) -> dict:
    out = {}
    try:
        for name, expr, cit, note in rows or []:
            m = Method(name, expr, cit, note)
            check_new_method(m, out)
            if m.name in out:
                raise RelationError(f"method {m.name!r} appears twice")
            out[m.name] = m
    except (ValueError, TypeError) as exc:
        if isinstance(exc, RelationError):
            raise
        raise RelationError(f"bad stored SGR method: {exc}") from None
    return out
