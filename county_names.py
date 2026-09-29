"""
Resolve a county name written one way to the same county written another way.

The app reads five sources that spell the same places differently:

    dropdown (states/XX/config.yaml)   'Alexandria City'  'Acadia Parish'  'Capitol Planning Region'
    predictions CSV (county_id)        'ALEXANDRIA'       'ACADIA'         'CAPITOL'
    weather parquet (county)           'ALEXANDRIA CITY'  'ACADIA PARISH'  'CAPITOL PLANNING REGION'
    state map (counties.geojson)       'ALEXANDRIA CITY'  'ACADIA PARISH'  'CAPITOL PLANNING REGION'
    national map (properties._id)      'ALEXANDRIA_CITY'  'ACADIA_PARISH'  'CAPITOL'

The national map file also writes spaces as underscores (LOS_ANGELES) and drops
accents (DONA_ANA for Doña Ana), so matching ignores both. The maps fail
silently: Plotly just leaves a county blank when its name finds no shape.

Stripping " County" alone -- the old approach -- misses every Louisiana parish,
every Connecticut planning region, and most Virginia independent cities.

Stripping every suffix is worse. Virginia has counties and independent cities
that share a base name (Fairfax County and Fairfax City, Richmond County and
Richmond City), and counties whose real name contains "City" (Charles City,
James City). A blanket strip would silently return Fairfax County's numbers for
Fairfax City: a wrong answer rather than an error.

So matching tries the exact spelling first and only then a suffix-stripped one,
and a candidate's stripped spelling is indexed only when it does not collide
with another candidate's exact spelling. scripts/verify_county_names.py checks
that this resolves every county in every state, one to one.
"""
from __future__ import annotations

import unicodedata
from typing import Iterable, Optional

# Longest first, so ' PLANNING REGION' is removed before anything shorter could
# match inside it.
_SUFFIXES = (' PLANNING REGION', ' CENSUS AREA', ' MUNICIPALITY', ' BOROUGH',
             ' COUNTY', ' PARISH', ' CITY')


def _norm(name: str) -> str:
    """'Doña Ana' -> 'DONA ANA', 'LOS_ANGELES' -> 'LOS ANGELES'."""
    ascii_name = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
    return ' '.join(ascii_name.replace('_', ' ').upper().split())


def _strip_suffix(name: str) -> Optional[str]:
    for suffix in _SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)].strip()
    return None


def build_index(candidates: Iterable[str]) -> dict[str, str]:
    """Map lookup keys to the original candidate spellings.

    Every candidate is reachable by its exact normalized spelling. It is also
    reachable by its suffix-stripped spelling, unless that would collide with a
    different candidate's exact spelling (FAIRFAX CITY must not answer to
    FAIRFAX, because FAIRFAX is a county of its own).
    """
    cands = list(dict.fromkeys(candidates))
    index = {_norm(c): c for c in cands}
    exact = set(index)
    stripped: dict[str, list[str]] = {}
    for c in cands:
        s = _strip_suffix(_norm(c))
        if s:
            stripped.setdefault(s, []).append(c)
    for key, owners in stripped.items():
        if key not in exact and len(owners) == 1:
            index[key] = owners[0]
    return index


def resolve(name: str, index: dict[str, str]) -> Optional[str]:
    """Return the candidate that `name` refers to, or None if there is none."""
    n = _norm(name)
    if n in index:
        return index[n]
    s = _strip_suffix(n)
    while s:  # 'Charles City County' -> 'CHARLES CITY' (a hit) before 'CHARLES'
        if s in index:
            return index[s]
        s = _strip_suffix(s)
    return None


def display_name(name: str) -> str:
    """Human-readable name: 'King' -> 'King County'; keep names that already
    carry their own designation ('Acadia Parish', 'Capitol Planning Region',
    'Alexandria City')."""
    n = _norm(name)
    if any(n.endswith(s) for s in _SUFFIXES):
        return str(name).strip()
    return f'{str(name).strip()} County'


def census_label(namelsad: str) -> str:
    """A county's Census long name (NAMELSAD) as a label. The Census writes
    independent cities in lower case: 'Alexandria city' -> 'Alexandria City'.
    Prefer this to display_name() wherever NAMELSAD is available: a short name
    alone cannot tell Alexandria City from a county, or know that James City
    is a county."""
    name = str(namelsad).strip()
    return name[: -len(' city')] + ' City' if name.endswith(' city') else name
