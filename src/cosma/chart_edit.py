"""Apply an edited stratigraphic chart to a project (headless; the Qt editor calls this).

The editor works on a plain description: ordered rows (uid or None for a new interval,
name, contact at top) plus the base horizon's name. This validates it and applies it all
or nothing: renames carry the picks in every well along (old names stay as aliases),
removals are refused while data is attached, order and contacts are set.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .project import Project, ProjectError
from .strat import BASE_UID, CONTACTS, StratChart, StratError, norm


@dataclass
class Row:
    uid: str | None            # None = new interval
    name: str
    contact: str = "conformable"


def rows_from_chart(chart: StratChart) -> list:
    return [Row(iv.uid, iv.name, iv.contact_above) for iv in chart.intervals]


def validate(rows, base_name: str) -> list:
    """Problems with a proposed chart (empty list = fine)."""
    out, seen = [], {}
    if not rows:
        out.append("a chart needs at least one interval")
    for r in rows:
        if not r.name.strip():
            out.append("an interval has no name")
        elif norm(r.name) in seen:
            out.append(f"name {r.name!r} is used twice")
        else:
            seen[norm(r.name)] = r.name
        if r.contact not in CONTACTS:
            out.append(f"{r.name}: contact must be one of {CONTACTS}")
    if not base_name.strip():
        out.append("the base horizon has no name")
    elif norm(base_name) in seen:
        out.append(f"name {base_name!r} is used twice")
    return out


def apply_chart_edit(project: Project, rows, base_name: str) -> None:
    chart = project.chart
    if chart is None:
        raise ProjectError("the project has no chart to edit")
    problems = validate(rows, base_name)
    if problems:
        raise ProjectError("; ".join(problems))
    keep = {r.uid for r in rows if r.uid}
    current = {iv.uid for iv in chart.intervals}
    unknown = keep - current
    if unknown:
        raise ProjectError(f"unknown intervals {sorted(unknown)}")
    for uid in current - keep:                      # removals first: refused while data hangs on them
        uses = project.horizon_uses(uid)
        if uses:
            raise ProjectError(f"cannot remove {chart.interval(uid).name!r}: {', '.join(uses)} attached to its top")
    snap = (copy.deepcopy(chart), {n: copy.deepcopy(r.tops) for n, r in project.wells.items()})
    try:
        for uid in current - keep:
            chart.remove_interval(uid)
        # renames go through two steps so that swapping two names does not collide
        todo = [(r.uid, r.name.strip()) for r in rows if r.uid and chart.interval(r.uid).name != r.name.strip()]
        base_todo = base_name.strip() != chart.base_name
        olds = {uid: chart.interval(uid).name for uid, _ in todo}
        for k, (uid, _) in enumerate(todo):
            project.rename_horizon(uid, f"\u0000tmp{k}", keep_alias=False)
        for k, (uid, new) in enumerate(todo):
            project.rename_horizon(uid, new, keep_alias=False)
        for uid, old in olds.items():               # old spellings stay findable unless another horizon now owns them
            if chart.match_name(old) is None:
                chart.add_alias(uid, old)
        if base_todo:
            project.rename_horizon(BASE_UID, base_name.strip())
        new_uids = {}
        for i, r in enumerate(rows):
            if r.uid is None:
                new_uids[i] = chart.add_interval(r.name.strip(), min(i, len(chart.intervals)), r.contact)
        order = [r.uid if r.uid else new_uids[i] for i, r in enumerate(rows)]
        by_uid = {iv.uid: iv for iv in chart.intervals}
        chart.intervals = [by_uid[u] for u in order]
        for u, r in zip(order, rows):
            chart.set_contact(u, r.contact)
        chart.check()
    except (StratError, ProjectError):
        project.chart = snap[0]
        for n, tops in snap[1].items():
            project.wells[n].tops = tops
        raise


# -- starting a chart without a type well ---------------------------------------------------------------------------------
def wells_with_tops(project: Project) -> dict:
    """{well: number of picks} for the wells a chart could be built from (at least two picks)."""
    return {n: len(r.tops) for n, r in project.wells.items() if len(r.tops) >= 2}


def pick_names(project: Project) -> list:
    """Distinct pick names over all wells, in the order first met (a hint when the chart is typed in by hand)."""
    seen = {}
    for r in project.wells.values():
        for t in r.tops.values():
            seen.setdefault(norm(t.name), t.name)
    return list(seen.values())


def parse_interval_text(text: str) -> list:
    """One interval name per line (blank lines ignored, spaces trimmed)."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def check_hand_chart(names, base_name: str) -> list:
    """Problems with a chart typed in by hand (empty list = fine)."""
    if not names:
        return ["type at least one interval name"]
    if not base_name.strip():
        return ["name the base: the pick below the lowest interval"]
    try:
        StratChart(list(names), base_name.strip())
    except StratError as exc:
        return [str(exc)]
    return []


def start_chart(project: Project, how: dict) -> StratChart:
    """Create the chart. how = {"mode": "well", "well": name} (built from that well's tops) or
    {"mode": "hand", "intervals": [...], "base": name}. Refuses if there already is one."""
    if project.chart is not None:
        raise ProjectError("the project already has a stratigraphic chart")
    mode = how.get("mode")
    if mode == "well":
        return project.build_chart(how.get("well"))
    if mode == "hand":
        problems = check_hand_chart(list(how.get("intervals", [])), how.get("base", ""))
        if problems:
            raise ProjectError("; ".join(problems))
        return project.create_chart(list(how["intervals"]), how["base"].strip())
    raise ProjectError("say how the chart is made: from a well's tops, or by hand")
