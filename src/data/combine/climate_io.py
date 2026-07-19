"""Shared helpers for turning per-centroid climate CSVs into model-grid arrays.

The climate acquire step (climr) writes long-format CSVs — one row per
``(id, PERIOD)`` with monthly columns (``Tmax01..Tmax12``, ``PPT01..PPT12``, ...;
climr also emits an underscore form ``Tmax_01``). Two things consume that:

- ``scripts/viz/quicklook_grids.py`` (thumbnail QC), and
- ``src/data/preprocess/climate_grid.py`` (per-year model-grid rasters, A1).

Both need the same two operations, factored here so there is exactly one
implementation: scatter per-``id`` values onto the grid (``grid_from_centroids``)
and collapse the 12 monthly columns into a model **bio-year** value per base
variable (``bioyear_aggregate``). Both are pure functions (DataFrame/array in,
array/DataFrame out) so they unit-test without a cluster or rasterio.
"""
import re

import numpy as np
import pandas as pd

# Monthly column parser: "Tmax01" / "Tmax_01" / "PPT_1" -> base="Tmax", month=1.
# Base names are alphabetic in climr's Monthly set; the trailing 1-2 digits are
# the month. An optional single separator ('_' or '.') is tolerated.
_MONTH_COL = re.compile(r"^(?P<base>[A-Za-z][A-Za-z.]*?)[._]?(?P<mm>\d{1,2})$")

# Base variables that are extensive over the year (SUMMED across the 12 months,
# like HYDE population counts); everything else is intensive (AVERAGED), mirroring
# ``hyde.py:_resampling_for``. Matched by case-insensitive *prefix* on the base
# name — precipitation (PPT/PAS), radiation (RAD/SRAD), degree-days (DD*),
# frost-free-day counts (NFFD), and moisture fluxes (Eref/CMD) are totals;
# temperatures (Tmax/Tmin/Tave/Tdmean) and humidity (RH) are means.
_SUM_PREFIXES = ("ppt", "prec", "pas", "rad", "srad", "dd", "nffd", "eref", "cmd")


def _is_sum_base(base: str) -> bool:
    """True if a base variable is a total/flux (summed), else intensive (mean)."""
    return base.lower().startswith(_SUM_PREFIXES)


def parse_month_columns(columns, non_var=("id", "PERIOD", "row", "col", "DATASET")):
    """Group monthly columns by base variable.

    Returns ``{base: {month:int -> column_name}}`` for every column that parses as
    ``<base><mm>``; non-variable and unparseable columns are ignored.
    """
    groups = {}
    for c in columns:
        if c in non_var:
            continue
        m = _MONTH_COL.match(str(c))
        if not m:
            continue
        base, mm = m.group("base"), int(m.group("mm"))
        if not (1 <= mm <= 12):
            continue
        groups.setdefault(base, {})[mm] = c
    return {b: mm for b, mm in groups.items() if len(mm) == 12}


def bioyear_aggregate(df_level, year, start_month, month_groups=None):
    """Collapse monthly climate to one model **bio-year** value per base variable.

    ``df_level`` is a level CSV (``id, PERIOD, <monthly cols>``). Bio-year ``T``
    spans Aug(T-1)→Jul(T) (12 ``(calendar_year, month)`` pairs from
    ``temporal.bio_year_months``); for each base variable the 12 monthly values
    are **summed** (fluxes: PPT etc.) or **averaged** (intensive: temperatures),
    per ``_is_sum_base``. Returns a DataFrame indexed by ``id`` with one column
    per base variable. Rows missing any required month are dropped.
    """
    from src.temporal import bio_year_months

    if month_groups is None:
        month_groups = parse_month_columns(df_level.columns)
    if not month_groups:
        raise ValueError("no monthly variable columns parsed from climate CSV")

    pairs = bio_year_months(year, start_month)          # 12 (cal_year, month)
    by_period = {int(p): g for p, g in df_level.groupby("PERIOD")}
    # Restrict to ids present in every calendar year the bio-year touches, so a
    # bio-year straddling a data-gap year is dropped rather than half-filled.
    cal_years = sorted({cy for cy, _ in pairs})
    if any(cy not in by_period for cy in cal_years):
        return pd.DataFrame(columns=list(month_groups)).rename_axis("id")
    ids = set.intersection(*[set(by_period[cy]["id"]) for cy in cal_years])
    ids = np.array(sorted(ids))
    if ids.size == 0:
        return pd.DataFrame(columns=list(month_groups)).rename_axis("id")

    indexed = {cy: by_period[cy].set_index("id").reindex(ids) for cy in cal_years}
    out = {"id": ids}
    for base, months in month_groups.items():
        acc = np.zeros(ids.size, dtype="float64")
        for cal_year, month in pairs:
            acc += indexed[cal_year][months[month]].to_numpy(dtype="float64")
        out[base] = acc if _is_sum_base(base) else acc / 12.0
    return pd.DataFrame(out).set_index("id")


def grid_from_centroids(values, centroids, ny, nx, value_col=None):
    """Scatter per-``id`` values onto an ``(ny, nx)`` grid (NaN elsewhere).

    ``centroids`` maps ``id -> (row, col)`` (a DataFrame with ``id,row,col``).
    ``values`` is either a Series indexed by ``id`` or a DataFrame with an ``id``
    column plus ``value_col``. Cells with no value stay NaN.
    """
    grid = np.full((ny, nx), np.nan, dtype="float32")
    if isinstance(values, pd.Series):
        vdf = values.rename("value").reset_index()
        vdf.columns = ["id", "value"]
    else:
        vdf = values[["id", value_col]].rename(columns={value_col: "value"})
    merged = vdf.merge(centroids[["id", "row", "col"]], on="id", how="inner")
    rows = merged["row"].to_numpy(dtype=int)
    cols = merged["col"].to_numpy(dtype=int)
    grid[rows, cols] = merged["value"].to_numpy(dtype="float32")
    return grid
