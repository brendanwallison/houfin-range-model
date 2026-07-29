"""Shared helpers for turning per-centroid climate CSVs into model-grid arrays.

The climate acquire step (climr) writes long-format CSVs — one row per
``(id, PERIOD)`` with monthly columns (``Tmax01..Tmax12``, ``PPT01..PPT12``, ...;
climr also emits an underscore form ``Tmax_01``). Two things consume that:

- ``scripts/viz/quicklook_grids.py`` (thumbnail QC), and
- ``src/data/preprocess/climate_grid.py`` (per-year model-grid rasters, A1).

Both need the same operations, factored here so there is exactly one
implementation:

- ``grid_from_centroids`` — scatter per-``id`` values onto the model grid.
- ``bioyear_monthly`` — reshape the 12 monthly columns onto the model **bio-year**
  window (Aug(T-1)→Jul(T)) as 12 separate channels per base variable, values
  verbatim. This is what the model pipeline uses.
- ``bioyear_aggregate`` — the legacy annual collapse (12 months → 1 sum or mean
  per base). Kept for QC and comparison only; it destroys seasonality, which is
  why the pipeline moved to ``bioyear_monthly``.

All are pure functions (DataFrame/array in, array/DataFrame out) so they
unit-test without a cluster or rasterio.
"""
import re

import numpy as np
import pandas as pd

# Monthly column parser: "Tmax01" / "Tmax_01" / "PPT_1" -> base="Tmax", month=1.
# The trailing 1-2 digits are the month; an optional single separator ('_' or
# '.') before them is tolerated.
#
# The base is `.*?` rather than an alphabetic class because climr's Monthly set
# includes degree-day variables whose names CONTAIN digits and separators:
# "DD_0_01", "DD5_01", "DD_18_12", "DD18_12". An `[A-Za-z]`-only base silently
# dropped every one of them before aggregation ever ran -- even though
# ``_SUM_PREFIXES`` below explicitly lists "dd" and so was written expecting
# them. Laziness plus the end-anchored month keeps the split at the LAST digit
# group ("DD_18_12" -> base "DD_18", month 12), and the letter guard in
# ``parse_month_columns`` rejects a bare numeric column.
_MONTH_COL = re.compile(r"^(?P<base>.*?)[._]?(?P<mm>\d{1,2})$")
_HAS_LETTER = re.compile(r"[A-Za-z]")

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


def parse_month_columns(columns, non_var=("id", "PERIOD", "row", "col", "DATASET"),
                        warn=True):
    """Group monthly columns by base variable.

    Returns ``{base: {month:int -> column_name}}`` for every base with a COMPLETE
    set of 12 months; non-variable and unparseable columns are ignored. Bases
    with only some months are dropped (a partial bio-year window is not a valid
    covariate) but, with ``warn``, are reported rather than vanishing silently.
    """
    groups = {}
    for c in columns:
        if c in non_var:
            continue
        m = _MONTH_COL.match(str(c))
        if not m:
            continue
        base, mm = m.group("base").rstrip("._"), int(m.group("mm"))
        if not base or not _HAS_LETTER.search(base):
            continue
        if not (1 <= mm <= 12):
            continue
        # Two spellings of the same (base, month) -- e.g. both "Tmax01" and
        # "Tmax_01" -- would silently overwrite each other here, making which
        # column reached the model depend on CSV column order.
        if mm in groups.get(base, {}):
            raise ValueError(
                f"duplicate climate column for base {base!r} month {mm}: "
                f"{groups[base][mm]!r} and {c!r}")
        groups.setdefault(base, {})[mm] = c
    complete = {b: mm for b, mm in groups.items() if len(mm) == 12}
    if warn:
        partial = {b: len(mm) for b, mm in groups.items() if len(mm) != 12}
        if partial:
            print("[climate_io] dropping bases without all 12 months: "
                  + ", ".join(f"{b} ({n}/12)" for b, n in sorted(partial.items())),
                  flush=True)
    return complete


def annual_columns(columns, non_var=("id", "PERIOD", "row", "col", "DATASET"),
                   month_groups=None):
    """Value columns consumed by no monthly group — climr's annual/seasonal vars.

    ``parse_month_columns`` only ever returns complete-12 monthly bases, so any
    genuinely annual variable (what ``list_vars("Annual")`` would add) is
    invisible to it. This names those columns so a caller can carry them through
    instead of dropping them. Empty for a pure ``list_vars("Monthly")`` pull.
    """
    if month_groups is None:
        month_groups = parse_month_columns(columns, non_var=non_var, warn=False)
    consumed = {c for months in month_groups.values() for c in months.values()}
    return [c for c in columns if c not in non_var and c not in consumed]


def _bioyear_frame(df_level, year, start_month, month_groups):
    """Shared bio-year setup for the aggregate and monthly paths.

    Returns ``(ids, indexed, pairs)`` where ``pairs`` is the 12
    ``(calendar_year, month)`` sequence of bio-year ``T`` in window order and
    ``indexed`` maps each touched calendar year to that year's rows reindexed
    onto ``ids``. ``ids`` is empty when the bio-year straddles a data gap — both
    callers must treat that as "no output for this year".
    """
    from src.temporal import bio_year_months

    if not month_groups:
        raise ValueError("no monthly variable columns parsed from climate CSV")

    pairs = bio_year_months(year, start_month)          # 12 (cal_year, month)
    by_period = {int(p): g for p, g in df_level.groupby("PERIOD")}
    # Restrict to ids present in every calendar year the bio-year touches, so a
    # bio-year straddling a data-gap year is dropped rather than half-filled.
    cal_years = sorted({cy for cy, _ in pairs})
    if any(cy not in by_period for cy in cal_years):
        return np.array([], dtype="int64"), {}, pairs
    ids = set.intersection(*[set(by_period[cy]["id"]) for cy in cal_years])
    ids = np.array(sorted(ids))
    if ids.size == 0:
        return ids, {}, pairs

    indexed = {cy: by_period[cy].set_index("id").reindex(ids) for cy in cal_years}
    return ids, indexed, pairs


def bioyear_month_columns(base, start_month):
    """The 12 ``{base}_b{kk}m{MM}`` channel names for one base, in window order.

    ``b{kk}`` is the 1-based POSITION in the bio-year window (``b01`` is
    ``start_month``); ``m{MM}`` is the calendar month that position resolves to.
    Both are carried because the position is what makes channel adjacency
    meaningful (``b01..b12`` also sorts lexicographically into window order, and
    ``discover_variables``' ``sorted()`` is the de-facto channel order), while
    the calendar month keeps the filenames readable and makes a phase error
    visible by eye. Single source of truth for this naming.
    """
    from src.temporal import bio_year_months

    pairs = bio_year_months(2000, start_month)   # any year: only months are used
    return [f"{base}_b{k:02d}m{mm:02d}" for k, (_cy, mm) in enumerate(pairs, start=1)]


def bioyear_monthly(df_level, year, start_month, month_groups=None):
    """Reshape monthly climate to one column per (base, bio-year month position).

    Unlike ``bioyear_aggregate``, values pass through **verbatim** — no sum, no
    ``/12``. Bio-year ``T`` spans Aug(T-1)→Jul(T), so the 12 columns of a base
    are that window's months in order: ``{base}_b01m08`` is August of ``T-1``
    and ``{base}_b12m07`` is July of ``T`` (see ``bioyear_month_columns``).
    Returns a DataFrame indexed by ``id`` with ``12 * len(month_groups)``
    columns, or an empty frame when the bio-year straddles a data gap.

    This is the path that preserves within-year structure; ``bioyear_aggregate``
    destroys it by construction and is kept only for the legacy/QC annual view.
    """
    if month_groups is None:
        month_groups = parse_month_columns(df_level.columns)
    cols = [c for base in month_groups for c in bioyear_month_columns(base, start_month)]

    ids, indexed, pairs = _bioyear_frame(df_level, year, start_month, month_groups)
    if ids.size == 0:
        return pd.DataFrame(columns=cols).rename_axis("id")

    out = {"id": ids}
    for base, months in month_groups.items():
        names = bioyear_month_columns(base, start_month)
        for name, (cal_year, month) in zip(names, pairs):
            out[name] = indexed[cal_year][months[month]].to_numpy(dtype="float64")
    return pd.DataFrame(out).set_index("id")


def bioyear_aggregate(df_level, year, start_month, month_groups=None):
    """Collapse monthly climate to one model **bio-year** value per base variable.

    ``df_level`` is a level CSV (``id, PERIOD, <monthly cols>``). Bio-year ``T``
    spans Aug(T-1)→Jul(T) (12 ``(calendar_year, month)`` pairs from
    ``temporal.bio_year_months``); for each base variable the 12 monthly values
    are **summed** (fluxes: PPT etc.) or **averaged** (intensive: temperatures),
    per ``_is_sum_base``. Returns a DataFrame indexed by ``id`` with one column
    per base variable. Rows missing any required month are dropped.

    LEGACY / QC ONLY. This discards every within-year contrast — two cells with
    equal annual means but opposite seasonality are indistinguishable in its
    output. The model pipeline uses ``bioyear_monthly``; ``_is_sum_base`` (and
    hence the sum-vs-mean distinction) applies only to this function.
    """
    if month_groups is None:
        month_groups = parse_month_columns(df_level.columns)

    ids, indexed, pairs = _bioyear_frame(df_level, year, start_month, month_groups)
    if ids.size == 0:
        return pd.DataFrame(columns=list(month_groups)).rename_axis("id")

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
