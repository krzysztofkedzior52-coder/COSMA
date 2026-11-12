"""Stratigraphic chart: the ordered intervals of the model and the horizons between them.

Horizons are the primary objects data attaches to. A horizon between interval A (above)
and interval B (below) is at the same time "base of A" and "top of B": one object, so a
depth grid attached to it serves both and cannot be attached twice. Horizon ids follow
the interval whose top they are, so reordering or renaming never detaches data.

    intervals (top -> bottom):   A        B        C
    horizons:                 top A   top B   top C   base (named, e.g. "Basement")

An interval is named after the pick at its top (the usual convention); the last horizon
is the base of the lowest interval and has its own name. Old names are kept as aliases so
later imports with the old spelling still find the right horizon.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CONTACTS = ("conformable", "unconformity")
BASE_UID = "base"


class StratError(ValueError):
    pass


def norm(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.casefold())


@dataclass
class Interval:
    uid: str
    name: str
    contact_above: str = "conformable"
    aliases: list = field(default_factory=list)


@dataclass(frozen=True)
class Horizon:
    uid: str
    name: str
    index: int                  # 0 = top of the first interval ... n = base of the last
    above: str | None           # name of the interval above (None for the first)
    below: str | None           # name of the interval below (None for the base)
    contact: str

    @property
    def label(self) -> str:
        if self.above and self.below:
            return f"Base of {self.above} / Top of {self.below}"
        if self.below:
            return f"Top of {self.below}"
        return f"Base of {self.above} ({self.name})"


class StratChart:
    def __init__(self, interval_names, base_name: str):
        self._n = 0
        self.intervals: list = []
        for nm in interval_names:
            self.intervals.append(Interval(self._new_uid(), nm))
        self.base_name = base_name
        self.base_aliases: list = []
        self.check()

    def _new_uid(self) -> str:
        self._n += 1
        return f"h{self._n}"

    # -- construction -------------------------------------------------------------------------
    @classmethod
    def from_picks(cls, picks) -> "StratChart":
        """picks: iterable of (name, depth) from one well. Ordered by depth; the deepest pick
        becomes the base of the lowest interval. Equal depths are refused (zero-thickness
        interval): the user must decide which pick goes first, by editing the picks."""
        picks = sorted(picks, key=lambda p: p[1])
        if len(picks) < 2:
            raise StratError("a chart needs at least two picks (a top and a base)")
        for (a, da), (b, db) in zip(picks, picks[1:]):
            if da == db:
                raise StratError(f"picks {a!r} and {b!r} are at the same depth ({da:g}): zero-thickness interval; "
                                 "remove or correct one in the well's tops before building the chart")
        return cls([n for n, _ in picks[:-1]], picks[-1][0])

    # -- persistence ----------------------------------------------------------------------------------
    def to_state(self) -> dict:
        return dict(n=self._n, base_name=self.base_name, base_aliases=list(self.base_aliases),
                    intervals=[dict(uid=i.uid, name=i.name, contact_above=i.contact_above, aliases=list(i.aliases))
                               for i in self.intervals])

    @classmethod
    def from_state(cls, st: dict) -> "StratChart":
        try:
            ch = object.__new__(cls)
            ch.base_name = st["base_name"]
            ch.intervals = [Interval(i["uid"], i["name"], i["contact_above"], list(i["aliases"])) for i in st["intervals"]]
            ch.base_aliases = list(st["base_aliases"])
            ch._n = int(st["n"])
        except (KeyError, TypeError) as exc:
            raise StratError(f"stratigraphic chart state is incomplete: {exc!r}") from None
        ch.check()
        if len({i.uid for i in ch.intervals}) != len(ch.intervals) or any(i.uid == BASE_UID for i in ch.intervals):
            raise StratError("chart has duplicate or reserved interval ids (corrupt)")
        used = [int(i.uid[1:]) for i in ch.intervals if i.uid[1:].isdigit()]
        if used and ch._n < max(used):
            raise StratError("chart uid counter is behind its intervals (corrupt)")
        return ch

    # -- queries ----------------------------------------------------------------------------------------
    def horizons(self) -> list:
        out = []
        n = len(self.intervals)
        for i, iv in enumerate(self.intervals):
            out.append(Horizon(iv.uid, iv.name, i, self.intervals[i - 1].name if i else None, iv.name,
                               iv.contact_above))
        out.append(Horizon(BASE_UID, self.base_name, n, self.intervals[-1].name if n else None, None, "conformable"))
        return out

    def horizon(self, uid: str) -> Horizon:
        for h in self.horizons():
            if h.uid == uid:
                return h
        raise StratError(f"no horizon {uid!r} in the chart")

    def interval(self, uid: str) -> Interval:
        for iv in self.intervals:
            if iv.uid == uid:
                return iv
        raise StratError(f"no interval {uid!r} in the chart")

    def interval_index(self, uid: str) -> int:
        return self.intervals.index(self.interval(uid))

    def names(self) -> list:
        return [h.name for h in self.horizons()]

    def match_name(self, name: str):
        """uid of the horizon whose name or alias equals `name` (ignoring case/punctuation)."""
        key = norm(name)
        if not key:
            return None
        for iv in self.intervals:
            if key in {norm(iv.name), *map(norm, iv.aliases)}:
                return iv.uid
        if key in {norm(self.base_name), *map(norm, self.base_aliases)}:
            return BASE_UID
        return None

    def check(self):
        seen = {}
        for nm in [iv.name for iv in self.intervals] + [self.base_name]:
            if not nm.strip():
                raise StratError("interval and horizon names cannot be empty")
            if norm(nm) in seen:
                raise StratError(f"name {nm!r} is used twice (also {seen[norm(nm)]!r})")
            seen[norm(nm)] = nm
        if not self.intervals:
            raise StratError("a chart needs at least one interval")
        for iv in self.intervals:
            if iv.contact_above not in CONTACTS:
                raise StratError(f"contact must be one of {CONTACTS}")

    # -- editing ------------------------------------------------------------------------------------------------
    def _owner_of_alias_only(self, new: str, uid: str):
        """Horizon (other than `uid`) that has `new` as an alias but not as its name, else None.
        Raises if `new` is another horizon's actual name."""
        key = norm(new)
        for iv in self.intervals:
            if iv.uid == uid:
                continue
            if norm(iv.name) == key:
                raise StratError(f"{new!r} is already the name or alias of another horizon")
            if key in map(norm, iv.aliases):
                return iv
        if uid != BASE_UID:
            if norm(self.base_name) == key:
                raise StratError(f"{new!r} is already the name or alias of another horizon")
            if key in map(norm, self.base_aliases):
                return BASE_UID
        return None

    def rename(self, uid: str, new: str, *, keep_alias: bool = True) -> str:
        """Rename an interval (= its top horizon) or the base; returns the old name. The old name
        stays as an alias (unless keep_alias is False). Naming a horizon deliberately takes the
        name over from another horizon's alias list."""
        new = new.strip()
        if not new:
            raise StratError("interval and horizon names cannot be empty")
        owner = self._owner_of_alias_only(new, uid)
        if owner is BASE_UID:
            self.base_aliases = [a for a in self.base_aliases if norm(a) != norm(new)]
        elif owner is not None:
            owner.aliases = [a for a in owner.aliases if norm(a) != norm(new)]
        if uid == BASE_UID:
            old, self.base_name = self.base_name, new
            if old != new and keep_alias:
                self.base_aliases.append(old)
        else:
            iv = self.interval(uid)
            old, iv.name = iv.name, new
            if old != new and keep_alias:
                iv.aliases.append(old)
        self.check()
        return old

    def add_alias(self, uid: str, alias: str):
        other = self.match_name(alias)
        if other is not None and other != uid:
            raise StratError(f"{alias!r} already belongs to another horizon")
        (self.base_aliases if uid == BASE_UID else self.interval(uid).aliases).append(alias)

    def move(self, uid: str, new_index: int):
        i = self.interval_index(uid)
        if not 0 <= new_index < len(self.intervals):
            raise StratError("position outside the chart")
        self.intervals.insert(new_index, self.intervals.pop(i))

    def add_interval(self, name: str, index: int, contact_above: str = "conformable") -> str:
        """Insert a new interval so that it is number `index` from the top (its top horizon is new)."""
        if self.match_name(name) is not None:
            raise StratError(f"{name!r} already exists in the chart")
        if not 0 <= index <= len(self.intervals):
            raise StratError("position outside the chart")
        iv = Interval(self._new_uid(), name.strip(), contact_above)
        self.intervals.insert(index, iv)
        self.check()
        return iv.uid

    def remove_interval(self, uid: str):
        iv = self.interval(uid)
        if len(self.intervals) == 1:
            raise StratError("a chart needs at least one interval")
        self.intervals.remove(iv)

    def set_contact(self, uid: str, contact: str):
        if contact not in CONTACTS:
            raise StratError(f"contact must be one of {CONTACTS}")
        self.interval(uid).contact_above = contact
