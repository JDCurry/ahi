"""
Prove that county_names.resolve() maps every county correctly, in every state,
across every pair of sources the app actually joins.

    dropdown (config.yaml)   -> predictions CSV county_id   County Risk tab
    CSV county               -> weather parquet county      National tab weather drivers
    dropdown (config.yaml)   -> weather parquet county      v4 fallback inference
    CSV county_id            -> national map shape          National tab map
    CSV county               -> state map shape             State tab map
    dropdown (config.yaml)   -> state map shape             County Spotlight map

A map join that fails does not raise anything in the app: Plotly just leaves
that county blank, which is how 379 went unnoticed.

For each pair it checks three things:
    complete   every source name resolves to something
    injective  no two source names resolve to the same target (the Fairfax
               City / Fairfax County trap)
    onto       every target is reached (nothing is orphaned)

Exits non-zero on any failure, so it can gate a deploy.

Usage: python scripts/verify_county_names.py [MM]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from county_names import build_index, resolve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, sources: list[str], targets: list[str]) -> list[str]:
    idx = build_index(targets)
    hit = {s: resolve(s, idx) for s in sources}
    problems = [f"{label}: unresolved {s!r}" for s, t in hit.items() if t is None]
    dup = [t for t, n in Counter(t for t in hit.values() if t).items() if n > 1]
    for t in dup:
        who = [s for s, v in hit.items() if v == t]
        problems.append(f"{label}: {who} all resolve to {t!r}")
    orphan = set(targets) - {t for t in hit.values() if t}
    problems += [f"{label}: nothing resolves to {t!r}" for t in sorted(orphan)]
    return problems


def state_map_names(state: str) -> list[str]:
    """The state map's keys, built as app._add_geojson_name_norms builds NAME_NORM."""
    path = ROOT / f"states/{state}/counties.geojson"
    if not path.exists():
        return []
    names = []
    for feat in json.load(open(path, encoding="utf-8-sig"))["features"]:
        p = feat.get("properties", {})
        name = p.get("NAMELSAD") or p.get("NAME") or p.get("name")
        if name:
            names.append(name.replace(" County", "").strip().upper())
    return names


def national_map_names() -> dict[str, list[str]]:
    """{state: county part of each properties._id in the national map file}."""
    national: dict[str, list[str]] = {}
    geojson = json.load(open(ROOT / "data/national_counties.geojson", encoding="utf-8-sig"))
    for feat in geojson["features"]:
        state, _, name = feat["properties"]["_id"].partition("|")
        national.setdefault(state, []).append(name)
    return national


def main() -> int:
    month = sys.argv[1] if len(sys.argv) > 1 else "10"
    nat = pd.read_csv(ROOT / f"data/national_predictions_month{month}.csv")
    states = sorted(p.name for p in (ROOT / "states").iterdir()
                    if (p / "config.yaml").exists() and not p.name.startswith("_"))  # _template is a scaffold
    national = national_map_names()
    problems, n = [], 0
    for s in states:
        cfg = yaml.safe_load(open(ROOT / f"states/{s}/config.yaml", encoding="utf-8-sig"))
        dropdown = sorted(cfg.get("county_coords") or {})
        csv = nat[nat.state == s]
        single, folder = ROOT / f"states/{s}/inference_data.parquet", ROOT / f"states/{s}/inference_data"
        src = single if single.exists() else (folder if folder.is_dir() else None)  # GA and TX are folders
        parquet = sorted(set(pq.read_table(src, columns=["county"]).column("county").to_pylist())) if src else []
        if not parquet:
            problems.append(f"{s} has no inference data, so its parquet joins were not checked")
        n += len(dropdown)
        problems += [f"{s} {p}" for p in check("dropdown->CSV", dropdown, csv.county_id.tolist())]
        if parquet:
            problems += [f"{s} {p}" for p in check("CSV->parquet", csv.county.tolist(), parquet)]
            problems += [f"{s} {p}" for p in check("dropdown->parquet", dropdown, parquet)]
        problems += [f"{s} {p}" for p in check("CSV->national map", csv.county_id.tolist(), national.get(s, []))]
        shapes = state_map_names(s)
        if not shapes:
            problems.append(f"{s} has no states/{s}/counties.geojson, so its state map was not checked")
        else:
            problems += [f"{s} {p}" for p in check("CSV->state map", csv.county.tolist(), shapes)]
            problems += [f"{s} {p}" for p in check("dropdown->state map", dropdown, shapes)]
    stray = sorted(set(national) - set(states))
    problems += [f"national map has shapes for {s}, which has no state folder" for s in stray]
    labels = ROOT / "data/county_labels.csv"
    if labels.exists():
        labeled = set(pd.read_csv(labels, encoding="utf-8", keep_default_na=False)["_id"])
        ids = {f"{s}|{name}" for s, names in national.items() for name in names}
        problems += [f"{i} has no label; re-run scripts/build_county_labels.py" for i in sorted(ids - labeled)]
    else:
        problems.append("data/county_labels.csv is missing; run scripts/build_county_labels.py")
    print(f"{len(states)} states, {n:,} counties, 6 joins each")
    if problems:
        print(f"FAIL: {len(problems)} problems")
        for p in problems[:60]:
            print("  ", p)
        return 1
    print("PASS: every county resolves, one to one, in every join")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
