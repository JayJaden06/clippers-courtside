"""Short player names that never mix two players up.

The site calls players by their last name ("Garland", "Jones Jr.", "Konan Niederhäuser"): everything after the first
name. Two players on the same team can share it (Baba Miller and Jordan Miller), so a shared last name gets the first
initial ("B. Miller", "J. Miller"), and the full name if even that collides.

``set_team`` is called once per run with every name on the team (box scores plus today's roster), and ``short`` is
used everywhere a name is shortened. Without ``set_team`` it falls back to the plain last name.
"""

from __future__ import annotations

from collections.abc import Iterable

_SHORT: dict[str, str] = {}


def last_name(full: str) -> str:
    """Everything after the first name: 'Darius Garland' -> 'Garland', 'Derrick Jones Jr.' -> 'Jones Jr.'."""
    return full.split(" ", 1)[-1] if full else full


def short_names(names: Iterable[str]) -> dict[str, str]:
    """Full name -> the shortest name that's unique within ``names``."""
    names = sorted({n for n in names if isinstance(n, str) and n})
    by_last: dict[str, list[str]] = {}
    for n in names:
        by_last.setdefault(last_name(n), []).append(n)
    out = {}
    for last, group in by_last.items():
        if len(group) == 1:
            out[group[0]] = last
            continue
        initials = {n: f"{n[0]}. {last}" for n in group}
        clash = {v for v in initials.values() if list(initials.values()).count(v) > 1}
        out |= {n: (n if s in clash else s) for n, s in initials.items()}
    return out


def set_team(names: Iterable[str]) -> None:
    """Remember the team's names, so ``short`` can tell players with the same last name apart."""
    global _SHORT
    _SHORT = short_names(names)  # replaced in one step, so a reader never sees it half-built


def short(name: str) -> str:
    """The name the site uses for a player: the last name, or more when a teammate shares it."""
    if not name:
        return name
    return _SHORT.get(name) or last_name(name)


def distinct(name: str, *others: str) -> str:
    """``short(name)``, or the full name when another player in the same sentence has the same last name
    ('Kris Dunn allows the fewest...' next to Phoenix's Ryan Dunn)."""
    mine = last_name(name)
    return name if any(o and o != name and last_name(o) == mine for o in others) else short(name)
