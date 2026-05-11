"""
StateContext — single source of truth for the active state's UI data.

Loads states/registry.yaml + states/<XX>/config.yaml on initialization and
exposes everything the app.py needs as attributes. The app.py reads the
sidebar dropdown, instantiates a StateContext for the chosen state, and
all state-specific UI content flows from that one object.

Usage in app.py:
    ctx = StateContext.load(state_code='CO')
    st.title(ctx.page_title)
    coords = ctx.county_coords[county_name]
    util   = ctx.county_utility[county_name]
    guidance = ctx.hazard_guidance['fire']['High']
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required. Add 'pyyaml>=6.0' to requirements.txt."
    ) from e

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / 'states' / 'registry.yaml'

_MONTH_TO_SEASON = {
    12: 'winter', 1: 'winter',  2: 'winter',
     3: 'spring', 4: 'spring',  5: 'spring',
     6: 'summer', 7: 'summer',  8: 'summer',
     9: 'fall',  10: 'fall',   11: 'fall',
}


def load_registry() -> Dict[str, Dict[str, Any]]:
    """Return {state_code: {region, name, deployed, default_county, ...}}."""
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f) or {}


def deployed_states() -> List[str]:
    """List of state codes whose `deployed: true` in the registry."""
    reg = load_registry()
    return [code for code, meta in reg.items() if meta.get('deployed', False)]


@dataclass
class StateContext:
    state_code: str
    state_name: str
    region: str
    default_county: str

    # UI content (from config.yaml)
    page_title: str = ''
    tagline: str = ''
    map_bbox: Dict[str, List[float]] = field(default_factory=dict)
    geojson_county_property: str = 'NAME'
    county_coords: Dict[str, List[float]] = field(default_factory=dict)
    county_utility: Dict[str, str] = field(default_factory=dict)
    nws_offices: List[Dict[str, str]] = field(default_factory=list)
    state_agencies: Dict[str, str] = field(default_factory=dict)
    hazard_guidance: Dict[str, Dict[str, str]] = field(default_factory=dict)
    season_notes: Dict[str, str] = field(default_factory=dict)
    audit_factors: Dict[str, Dict[str, str]] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)

    # Derived
    counties: List[str] = field(default_factory=list)

    # Paths
    state_dir: Path = field(default_factory=Path)
    parquet_path: Path = field(default_factory=Path)
    geojson_path: Path = field(default_factory=Path)

    @classmethod
    def load(cls, state_code: str) -> 'StateContext':
        registry = load_registry()
        if state_code not in registry:
            raise ValueError(f"State '{state_code}' not in registry. "
                             f"Available: {sorted(registry.keys())}")
        meta = registry[state_code]
        state_dir = ROOT / 'states' / state_code
        cfg_path = state_dir / 'config.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"No config.yaml at {cfg_path}. "
                f"Run scripts/onboard_state.py to bootstrap state {state_code}."
            )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

        ctx = cls(
            state_code=state_code,
            state_name=cfg.get('state_name', meta.get('name', state_code)),
            region=cfg.get('region', meta.get('region', state_code.lower())),
            default_county=cfg.get('default_county',
                                    meta.get('default_county', '')),
            page_title=cfg.get('page_title',
                                f"AHI — {cfg.get('state_name', state_code)}"),
            tagline=cfg.get('tagline', meta.get('tagline', '')),
            map_bbox=cfg.get('map_bbox', {}),
            geojson_county_property=cfg.get('geojson_county_property', 'NAME'),
            county_coords=cfg.get('county_coords', {}),
            county_utility=cfg.get('county_utility', {}),
            nws_offices=cfg.get('nws_offices', []),
            state_agencies=cfg.get('state_agencies', {}),
            hazard_guidance=cfg.get('hazard_guidance', {}),
            season_notes=cfg.get('season_notes', {}),
            audit_factors=cfg.get('audit_factors', {}),
            performance=cfg.get('performance', {}),
            versions=cfg.get('versions', {}),
            state_dir=state_dir,
            parquet_path=state_dir / 'inference_data.parquet',
            geojson_path=state_dir / 'counties.geojson',
        )
        ctx.counties = sorted(ctx.county_coords.keys())
        return ctx

    # ---- Convenience accessors ----

    @property
    def model_version(self) -> str:
        return self.versions.get('model', f'AHI v2.5 ({self.state_code})')

    @property
    def data_version(self) -> str:
        return self.versions.get('data',
                                  'NOAA GridMET / WFIGS / USGS / NOAA Storm Events, 2000–2025')

    def season_for_month(self, month: int) -> str:
        return _MONTH_TO_SEASON.get(month, 'spring')

    def season_note_for_month(self, month: int) -> str:
        return self.season_notes.get(self.season_for_month(month), '')

    def audit_factor(self, hazard: str, month: int) -> str:
        season = self.season_for_month(month)
        return self.audit_factors.get(hazard, {}).get(season, '')

    def guidance(self, hazard: str, level: str) -> str:
        return self.hazard_guidance.get(hazard, {}).get(level, '')

    def utility_for(self, county: str) -> str:
        return self.county_utility.get(county, 'Utility provider not on file')

    def coords_for(self, county: str) -> Optional[List[float]]:
        return self.county_coords.get(county)

    def nws_office_codes(self) -> List[str]:
        return [o.get('code', '') for o in self.nws_offices]

    def __repr__(self) -> str:
        return (f"StateContext(state_code={self.state_code!r}, "
                f"region={self.region!r}, counties={len(self.counties)})")
