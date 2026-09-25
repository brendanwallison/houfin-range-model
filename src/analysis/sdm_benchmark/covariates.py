"""Covariate design matrices for the SDM benchmark, sampled at route cells.

THREE TIERS, so the baseline cannot be called starved of information:

  standard  ~12-20 interpretable predictors (bioclim-style climate summaries,
            LUH-3 land use, HYDE population, elevation). What an ecologist would
            actually publish, and the only tier biolith's linear/spline form takes.
  full      the ~295 raw encoder channels -- everything the community encoder
            sees, from encoder/states/yearly_states/state_{year}.npz.
  latent    the 24-d learned basis Z, i.e. exactly what the dynamic model consumes.

AVAILABILITY IS CHECKED, NOT ASSUMED. Most of these grids are built on HPC and
are absent from a laptop checkout, so ``discover`` reports what each tier can
actually supply and the callers refuse a tier that is short rather than silently
fitting a baseline on three covariates and calling it a benchmark.

Sampling is nearest-cell at the route's (row, col) -- the same assignment the
dynamic model uses (src/data/preprocess/bbs.py map_routes_to_grid), so both
sides read the same cell.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import rasterio

from src.config_utils import load_data_config

_CFG = load_data_config()
_DR = _CFG["datasets_root"]
_PROC = os.path.join(_DR, "processed")


@dataclass
class Source:
    """One covariate source. ``annual`` sources carry a {year} in the template."""
    name: str
    template: str                 # absolute path; may contain {year} and {var}
    variables: tuple = ()         # for {var} templates; empty means a single file
    annual: bool = False
    transform: str = "none"       # none | log1p

    @property
    def n_features(self):
        return len(self.variables) if self.variables else 1

    def paths_for(self, year=None):
        vs = self.variables or (None,)
        out = {}
        for v in vs:
            p = self.template
            if v is not None:
                p = p.replace("{var}", v)
            if self.annual:
                p = p.replace("{year}", str(year))
            out[f"{self.name}:{v}" if v is not None else self.name] = p
        return out

    def available(self, years):
        """True when every file this source needs exists AND is on the model grid."""
        probe_years = years if self.annual else [None]
        for y in probe_years:
            for p in self.paths_for(y).values():
                if not _geometry_ok(p):
                    return False
        return True


def _hyde_vars():
    """HYDE variables present on disk, discovered from the filenames."""
    d = os.path.join(_DR, "hyde35_grid")
    if not os.path.isdir(d):
        return ()
    names = set()
    for f in glob.glob(os.path.join(d, "*_grid.tif")):
        b = os.path.basename(f)[: -len("_grid.tif")]
        parts = b.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            names.add(parts[0])
    return tuple(sorted(names))


def _soil_vars():
    d = os.path.join(_DR, "soilgrids_grid")
    if not os.path.isdir(d):
        return ()
    return tuple(sorted(os.path.basename(f)[: -len("_grid.tif")]
                        for f in glob.glob(os.path.join(d, "*_grid.tif"))))


# Climate channel token, fixed by src/data/preprocess/climate_grid.py:
# ``{base}_b{kk}m{MM}_{lvl}``, e.g. Tmax_b01m08_q50 -- b{kk} is the position in
# the bio-year window (b01 = bio_year_start_month), m{MM} the calendar month.
_CLIM_TOKEN = re.compile(r"^(?P<base>.+)_b(?P<b>\d{2})m(?P<m>\d{2})_(?P<lvl>q\d{2})$")

SUMMER_MONTHS = (6, 7, 8)
WINTER_MONTHS = (12, 1, 2)


@dataclass
class DerivedSource:
    """A source whose features are MEANS over groups of monthly rasters.

    ``groups`` maps a feature name to the raster {var} tokens averaged into it.
    This is how the 'standard' tier gets bioclim-analogue predictors (annual
    mean, summer mean, winter mean per base) out of a climate stream that stores
    12 separate monthly channels per base.
    """
    name: str
    template: str
    groups: dict
    annual: bool = True
    transform: str = "none"

    @property
    def n_features(self):
        return len(self.groups)

    def available(self, years):
        if not self.groups:
            return False
        probe = years if self.annual else [None]
        for y in probe:
            for members in self.groups.values():
                for v in members:
                    p = self.template.replace("{var}", v)
                    if self.annual:
                        p = p.replace("{year}", str(y))
                    if not _geometry_ok(p):
                        return False
        return True


def _discover(grid_dir, level=None):
    """Variable tokens in a grid dir, manifest-validated where one exists."""
    if not os.path.isdir(grid_dir):
        return []
    try:
        from src.data.combine.build_states import discover_variables
        return list(discover_variables(grid_dir, level=level))
    except SystemExit:
        raise
    except Exception:
        toks = set()
        for f in glob.glob(os.path.join(grid_dir, "*_????_grid.tif")):
            b = os.path.basename(f)[: -len("_grid.tif")]
            parts = b.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                toks.add(parts[0])
        if level:
            toks = {t for t in toks if t.endswith(f"_{level}")}
        return sorted(toks)


def climate_summary_groups(level="q50"):
    """Bioclim-analogue groupings of the monthly climate channels.

    Per base variable: annual mean, summer mean, winter mean. Discovered from
    what is on disk rather than hardcoded, because the channel names encode the
    bio-year position and are validated against manifest.json -- inventing them
    here would produce a job that fails on a missing file.
    """
    d = os.path.join(_DR, "climate_grid_monthly")
    by_base = {}
    for v in _discover(d, level=level):
        m = _CLIM_TOKEN.match(v)
        if m:
            by_base.setdefault(m.group("base"), []).append((int(m.group("m")), v))
    groups = {}
    for base, items in sorted(by_base.items()):
        months = {mm: v for mm, v in items}
        groups[f"{base}_ann"] = [v for _, v in sorted(items)]
        for label, sel in (("summer", SUMMER_MONTHS), ("winter", WINTER_MONTHS)):
            got = [months[mm] for mm in sel if mm in months]
            if got:
                groups[f"{base}_{label}"] = got
    return groups


def _luh3_vars():
    return tuple(_discover(os.path.join(_DR, "luh3_grid")))


def standard_sources():
    """The 'what an ecologist would publish' tier, discovered from disk."""
    return [
        Source("elev", os.path.join(_DR, "elevation", "{var}.tif"),
               variables=("elev_q10", "elev_q50", "elev_q90")),
        Source("hyde", os.path.join(_DR, "hyde35_grid", "{var}_{year}_grid.tif"),
               variables=_hyde_vars(), annual=True, transform="log1p"),
        Source("soil", os.path.join(_DR, "soilgrids_grid", "{var}_grid.tif"),
               variables=_soil_vars()),
        Source("luh3", os.path.join(_DR, "luh3_grid", "{var}_{year}_grid.tif"),
               variables=_luh3_vars(), annual=True),
        DerivedSource("climate",
                      os.path.join(_DR, "climate_grid_monthly", "{var}_{year}_grid.tif"),
                      groups=climate_summary_groups()),
    ]


@lru_cache(maxsize=1)
def _ref_geometry():
    """(shape, crs) of the model grid. Every covariate raster must match it."""
    with rasterio.open(_CFG["grid"]["ref_raster"]) as src:
        return src.shape, str(src.crs)


@lru_cache(maxsize=512)
def _read(path):
    """Read a covariate band, REFUSING anything off the model grid.

    This guard is not ceremonial. A laptop checkout still carries elevation,
    SoilGrids and HYDE grids from the retired 25 km ESRI:102039 lattice
    (117x185) alongside the current 27 km ESRI:102003 one (133x224). Indexing
    those with model-grid (row, col) does not raise wherever the indices happen
    to land in range -- it silently returns the covariate of some OTHER place,
    and the benchmark would fit cleanly on corrupted predictors. Compare against
    the reference grid and fail loudly instead.
    """
    ref_shape, ref_crs = _ref_geometry()
    with rasterio.open(path) as src:
        if src.shape != ref_shape or str(src.crs) != ref_crs:
            raise ValueError(
                f"{path} is on {src.shape} / {src.crs}, but the model grid is "
                f"{ref_shape} / {ref_crs}. This is almost certainly a leftover "
                f"from the retired 25 km ESRI:102039 lattice. Regrid it onto "
                f"ref_raster (regrid.reproject_to_ref) before use -- sampling it "
                f"as-is would silently mis-assign every covariate.")
        return src.read(1).astype(np.float32)


def _geometry_ok(path):
    """True when ``path`` exists AND sits on the model grid."""
    if not os.path.exists(path):
        return False
    try:
        ref_shape, ref_crs = _ref_geometry()
        with rasterio.open(path) as src:
            return src.shape == ref_shape and str(src.crs) == ref_crs
    except (OSError, rasterio.RasterioIOError):
        return False


def discover(years, sources=None):
    """Report which sources can supply every file they need for ``years``."""
    sources = sources or standard_sources()
    rep = []
    for s in sources:
        n_vars = s.n_features
        rep.append({"name": s.name, "n_vars": n_vars, "annual": s.annual,
                    "available": s.available(years) and n_vars > 0})
    return rep


def build_design(df, sources=None, require_all=True):
    """Sample covariates at each route-year's (row, col, Year).

    Returns ``(X, names)`` with X a float32 (n_rows, n_features) array aligned to
    ``df``. Annual sources are read per year; static ones once.
    """
    sources = sources or standard_sources()
    years = sorted(df["Year"].unique().tolist())
    avail = {r["name"]: r["available"] for r in discover(years, sources)}
    missing = [k for k, v in avail.items() if not v]
    if missing and require_all:
        raise FileNotFoundError(
            f"Covariate sources unavailable for {years[0]}-{years[-1]}: {missing}. "
            f"These grids are built on HPC; either run there or pass "
            f"require_all=False to fit on the reduced set (a smoke test, NOT a "
            f"benchmark).")
    usable = [s for s in sources if avail.get(s.name)]
    if not usable:
        raise FileNotFoundError("No covariate sources available at all.")

    r = df["row"].to_numpy()
    c = df["col"].to_numpy()
    yr = df["Year"].to_numpy()

    cols, names = [], []
    for s in usable:
        if isinstance(s, DerivedSource):
            per_feat = {k: np.full(len(df), np.nan, dtype=np.float32) for k in s.groups}
            for y in years:
                sel = yr == y
                if not sel.any():
                    continue
                rs, cs = r[sel], c[sel]
                for feat, members in s.groups.items():
                    acc = np.zeros(rs.shape, dtype=np.float64)
                    for v in members:
                        p_ = s.template.replace("{var}", v)
                        if s.annual:
                            p_ = p_.replace("{year}", str(y))
                        acc += _read(p_)[rs, cs]
                    per_feat[feat][sel] = acc / max(len(members), 1)
            for nm, v in per_feat.items():
                cols.append(np.log1p(v) if s.transform == "log1p" else v)
                names.append(f"{s.name}:{nm}")
            continue
        if not s.annual:
            for nm, p in s.paths_for().items():
                v = _read(p)[r, c]
                cols.append(np.log1p(v) if s.transform == "log1p" else v)
                names.append(nm)
            continue
        # annual: fill per year so each row reads its own year's raster
        per_var = {nm: np.full(len(df), np.nan, dtype=np.float32)
                   for nm in s.paths_for(years[0])}
        for y in years:
            sel = yr == y
            if not sel.any():
                continue
            rs, cs = r[sel], c[sel]
            for nm, p in s.paths_for(y).items():
                v = _read(p)[rs, cs]
                per_var[nm][sel] = np.log1p(v) if s.transform == "log1p" else v
        for nm, v in per_var.items():
            cols.append(v)
            names.append(nm)

    X = np.column_stack(cols).astype(np.float32)
    return X, names


def latent_z_design(df, z_dir=None, latent_dim=24):
    """The 'latent' tier: the same truncated basis Z the dynamic model consumes."""
    z_dir = z_dir or os.path.join(_PROC, "latent_avian_paths")
    years = sorted(df["Year"].unique().tolist())
    r, c, yr = df["row"].to_numpy(), df["col"].to_numpy(), df["Year"].to_numpy()
    X = np.full((len(df), latent_dim), np.nan, dtype=np.float32)
    for y in years:
        p = os.path.join(z_dir, f"Z_latent_{y}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. The latent cube is built on HPC; the 'latent' "
                f"tier cannot run from a laptop checkout.")
        z = np.load(p, mmap_mode="r")
        sel = yr == y
        X[sel] = np.asarray(z[r[sel], c[sel], :latent_dim], dtype=np.float32)
    return X, [f"z{i:02d}" for i in range(latent_dim)]


def full_states_design(df, states_dir=None):
    """The 'full' tier: the ~295 raw encoder channels from yearly_states."""
    states_dir = states_dir or os.path.join(_PROC, "encoder", "states", "yearly_states")
    years = sorted(df["Year"].unique().tolist())
    r, c, yr = df["row"].to_numpy(), df["col"].to_numpy(), df["Year"].to_numpy()

    first = os.path.join(states_dir, f"state_{years[0]}.npz")
    if not os.path.exists(first):
        raise FileNotFoundError(
            f"{first} not found. The encoder states are built on HPC; the 'full' "
            f"tier cannot run from a laptop checkout.")
    with np.load(first) as z0:
        stream_names = list(z0.files)
        widths = {k: (z0[k].shape[-1] if z0[k].ndim == 3 else 1) for k in stream_names}

    names = [f"{k}:{i:03d}" for k in stream_names for i in range(widths[k])]
    X = np.full((len(df), len(names)), np.nan, dtype=np.float32)
    for y in years:
        sel = yr == y
        if not sel.any():
            continue
        with np.load(os.path.join(states_dir, f"state_{y}.npz")) as z:
            blocks = []
            for k in stream_names:
                a = z[k]
                a = a[..., None] if a.ndim == 2 else a
                blocks.append(a[r[sel], c[sel], :])
        X[sel] = np.concatenate(blocks, axis=1).astype(np.float32)
    return X, names


def standardize(X, mu=None, sd=None):
    """Z-score columns, fitting stats on TRAIN only when mu/sd are passed back in.

    Constant columns get sd=1 so they become exact zeros rather than NaN -- the
    same convention covariate_io.fit_norm uses for indicator channels.
    """
    if mu is None:
        mu = np.nanmean(X, axis=0)
    if sd is None:
        sd = np.nanstd(X, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def probe_tier(years, tier):
    """Report whether a tier can supply covariates, and name the first gap.

    Cheap: touches file metadata only, never reads a band. This is what the
    ``preflight`` subcommand runs so a four-hour job is not the way you discover
    that a grid directory was never built.
    """
    if tier == "standard":
        srcs = standard_sources()
        # Go through discover() rather than re-deriving availability here: it is
        # the function build_design calls first, so preflight must exercise the
        # same path. A separate branch here once passed while build_design
        # crashed on the very same source list.
        rep = {r["name"]: r for r in discover(years, srcs)}
        parts, missing = [], []
        for sc in srcs:
            r = rep[sc.name]
            ok = bool(r["available"])
            parts.append({"source": sc.name, "n_features": r["n_vars"],
                          "available": ok})
            if not ok:
                missing.append(_first_missing(sc, years))
        return {"tier": tier,
                "available": all(p["available"] for p in parts) and bool(parts),
                "n_features": sum(p["n_features"] for p in parts if p["available"]),
                "sources": parts,
                "first_missing": [m for m in missing if m]}

    if tier == "full":
        d = os.path.join(_PROC, "encoder", "states", "yearly_states")
        gaps = [os.path.join(d, f"state_{y}.npz") for y in years
                if not os.path.exists(os.path.join(d, f"state_{y}.npz"))]
        n = 0
        if not gaps:
            with np.load(os.path.join(d, f"state_{years[0]}.npz")) as z:
                n = sum((z[k].shape[-1] if z[k].ndim == 3 else 1) for k in z.files)
        return {"tier": tier, "available": not gaps, "n_features": n,
                "dir": d, "first_missing": gaps[:3]}

    if tier == "latent":
        d = os.path.join(_PROC, "latent_avian_paths")
        gaps = [os.path.join(d, f"Z_latent_{y}.npy") for y in years
                if not os.path.exists(os.path.join(d, f"Z_latent_{y}.npy"))]
        return {"tier": tier, "available": not gaps, "n_features": 24,
                "dir": d, "first_missing": gaps[:3]}

    raise ValueError(f"unknown tier {tier!r}")


def _first_missing(source, years):
    """The first path a source needs but cannot use, for a legible error."""
    probe = years if getattr(source, "annual", False) else [None]
    if isinstance(source, DerivedSource):
        if not source.groups:
            return f"{source.name}: no channels discovered (grid dir empty or absent)"
        for y in probe:
            for members in source.groups.values():
                for v in members:
                    pth = source.template.replace("{var}", v)
                    if source.annual:
                        pth = pth.replace("{year}", str(y))
                    if not _geometry_ok(pth):
                        return _why(pth, source.name)
        return None
    if not source.variables:
        return f"{source.name}: no variables discovered"
    for y in probe:
        for pth in source.paths_for(y).values():
            if not _geometry_ok(pth):
                return _why(pth, source.name)
    return None


def _why(path, name):
    """Distinguish 'absent' from 'present but on the wrong grid'."""
    if not os.path.exists(path):
        return f"{name}: MISSING {path}"
    try:
        ref_shape, ref_crs = _ref_geometry()
        with rasterio.open(path) as src:
            return (f"{name}: OFF-GRID {path} is {src.shape}/{src.crs}, "
                    f"model grid is {ref_shape}/{ref_crs}")
    except Exception as e:
        return f"{name}: UNREADABLE {path} ({e})"
