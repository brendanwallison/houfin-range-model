"""Canonical model timeline — the single source of truth for what "a year" is.

Every stage (ingest → states → encoder → path features → model → visualization)
must agree on the meaning of a model year. That agreement lives here and in the
``timeline`` block of ``config/data_config.json``; nothing else should hardcode
start/end years or the invasion offset.

Definitions
-----------
model year T
    Indexed 0..N-1 over the contiguous calendar years ``first_year..end_year``.
climate = bio-year Aug(T-1) → Jul(T)
    The antecedent 12-month window ending at the ~June BBS breeding-season
    count, so weather *after* the count never leaks into predictor T. Climate
    (climr/CRU) observations begin Jan 1901, so the first year with a *complete*
    bio-year is 1902 (Aug 1901 → Jul 1902) — hence ``first_year = 1902``.
land use / soil = calendar-year-T state
    LUH-3 / HYDE are the annual land state as of calendar year T (soil is
    static). Only climate uses the bio-year window.
invasion
    The NYC House-Finch release (~1940). ``inv_timestep`` is *derived* as
    ``invasion_year - first_year`` (never hardcoded), so the release always
    fires in calendar 1940 regardless of where the timeline starts.
end_year
    The newest BBS field-season year (2026 release → 2025). Covariates that lag
    (e.g. LUH-3 ending 2024) are EMA/persistence-carried to end_year.
"""
from typing import Dict, List, Sequence, Union

import numpy as np

from src.config_utils import load_data_config

DEFAULTS = {
    "first_year": 1902,
    "end_year": 2025,
    "invasion_year": 1940,
    "bio_year_start_month": 8,  # August
}


def load_timeline(cfg: dict = None) -> Dict[str, int]:
    """Return the timeline block (with defaults) from data_config.json."""
    if cfg is None:
        cfg = load_data_config()
    t = dict(DEFAULTS)
    t.update({k: int(v) for k, v in (cfg.get("timeline") or {}).items() if k in DEFAULTS})
    if t["first_year"] > t["end_year"]:
        raise ValueError(f"first_year {t['first_year']} > end_year {t['end_year']}")
    return t


def model_years(tl: dict = None) -> List[int]:
    """Contiguous list of calendar years the model runs over."""
    tl = tl or load_timeline()
    return list(range(tl["first_year"], tl["end_year"] + 1))


def bio_year_months(year: int, start_month: int = None) -> List[tuple]:
    """(calendar_year, month) pairs composing bio-year T = Aug(T-1) → Jul(T).

    12 pairs: start_month..12 of year T-1, then 1..start_month-1 of year T.
    """
    if start_month is None:
        start_month = load_timeline()["bio_year_start_month"]
    prev = [(year - 1, m) for m in range(start_month, 13)]
    curr = [(year, m) for m in range(1, start_month)]
    return prev + curr


def assert_contiguous(years: Sequence[int]) -> None:
    """Raise if ``years`` is not a gap-free ascending run (the mapping assumes it)."""
    years = [int(y) for y in years]
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError(f"model years are not contiguous (gaps present): {years}")


def year_to_index(years: Sequence[int],
                  year: Union[int, Sequence[int]]) -> Union[int, np.ndarray]:
    """Gap-safe calendar-year → contiguous model index via lookup (not subtraction).

    ``years`` is the sorted model-year list; a year absent from it raises, which
    is the point — it surfaces a timeline desync instead of silently producing
    an off-by-N index. Accepts a scalar or an array of years.
    """
    idx = {int(y): i for i, y in enumerate(years)}
    if np.ndim(year) == 0:
        return idx[int(year)]
    missing = sorted({int(y) for y in year} - idx.keys())
    if missing:
        raise KeyError(f"years not in model timeline: {missing[:10]}"
                       f"{'...' if len(missing) > 10 else ''}")
    return np.array([idx[int(y)] for y in year], dtype=int)


def invasion_timestep(tl: dict = None, first_year: int = None) -> int:
    """Model index of the invasion pulse = invasion_year - first_year.

    Pass ``first_year`` to derive against a timeline actually realized on disk
    (e.g. the min year of the Z_disp files) rather than the config default.
    """
    tl = tl or load_timeline()
    fy = tl["first_year"] if first_year is None else int(first_year)
    return tl["invasion_year"] - fy
