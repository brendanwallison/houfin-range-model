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

from src.config_utils import load_age_model_config, load_config, load_data_config


def _load_esk_desk():
    return load_config("esk_desk_config.json")

_CFG = load_data_config()
_DR = _CFG["datasets_root"]
# NOT datasets_root/processed. On HPC the two roots are different filesystems --
# HOUFIN_DATA under $SCRATCH, HOUFIN_PROCESSED under $WORK -- so constructing the
# processed root from the data root silently points at a directory that does not
# exist. data_config.processed_root is the single source of truth.
_PROC = _CFG.get("processed_root") or os.path.join(_DR, "processed")


def _cfg_path(loader, *keys, default=None):
    """A path from a config, falling back when the config or key is absent."""
    try:
        cfg = loader()
    except Exception:
        return default
    for k in keys:
        if not isinstance(cfg, dict) or k not in cfg:
            return default
        cfg = cfg[k]
    return cfg if isinstance(cfg, str) else default


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
        probe = _sample_years(years) if self.annual else [None]
        checked_geometry = False
        for y in probe:
            for p in self.paths_for(y).values():
                if not _exists_ok(p):
                    return False
                if not checked_geometry:
                    if not _geometry_ok(p):
                        return False
                    checked_geometry = True
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

# Availability probing samples years rather than opening every file. The climate
# stream alone is ~144 monthly channels; times 26 years that is ~3,700 raster
# opens, each a Lustre metadata round-trip -- far too much I/O for the login-node
# check this is meant to be. Geometry is a property of how a source was BUILT,
# so it is verified once per source; existence is checked on sampled years.
# build_design still fails loudly on a genuinely missing intermediate year.
PROBE_YEARS = 3

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
        probe = _sample_years(years) if self.annual else [None]
        seen, checked_geometry = set(), False
        for y in probe:
            for members in self.groups.values():
                for v in members:
                    p = self.template.replace("{var}", v)
                    if self.annual:
                        p = p.replace("{year}", str(y))
                    if p in seen:
                        continue            # groups share monthly members
                    seen.add(p)
                    if not _exists_ok(p):
                        return False
                    if not checked_geometry:
                        if not _geometry_ok(p):
                            return False
                        checked_geometry = True
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


def _sample_years(years, k=PROBE_YEARS):
    """First, last and a middle year -- enough to catch an unbuilt stream."""
    ys = sorted(set(int(y) for y in years))
    if len(ys) <= k:
        return ys
    return sorted({ys[0], ys[len(ys) // 2], ys[-1]})


def _exists_ok(path):
    """Cheap existence check -- a stat, no header read."""
    return os.path.exists(path)


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


def _latent_year_path(z_dir, year):
    """Path and key for one year of the latent basis, preferring what the model reads.

    The pipeline writes Z_latent_{year}.npy as an INTERMEDIATE
    (build_final_z_cube), then generate_all_path_features consumes it and emits
    Z_disp_{year}.npz carrying both Z_raw and the path-integrated Z_disp. What
    the age model actually ingests is the .npz: model_inputs globs Z_disp_*.npz
    and reads peek['Z_raw']. So a tree with completed MAP runs can perfectly
    well have no .npy left -- looking for those reported the tier as unbuilt on
    a machine that had just finished fitting the model with it.

    Returns (path, key) where key is None for a bare .npy.
    """
    npz = os.path.join(z_dir, f"Z_disp_{year}.npz")
    if os.path.exists(npz):
        return npz, "Z_raw"
    return os.path.join(z_dir, f"Z_latent_{year}.npy"), None


def _default_latent_dim():
    v = _cfg_path(load_age_model_config, "latent_dim")
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(load_age_model_config().get("latent_dim", 24))
        except Exception:
            return 24


def latent_z_design(df, z_dir=None, latent_dim=None):
    """The 'latent' tier: the same truncated basis Z the dynamic model consumes."""
    z_dir = z_dir or _cfg_path(load_age_model_config, "raw_z_dir",
                               default=os.path.join(_PROC, "latent_avian_paths"))
    latent_dim = latent_dim or _default_latent_dim()
    years = sorted(df["Year"].unique().tolist())
    r, c, yr = df["row"].to_numpy(), df["col"].to_numpy(), df["Year"].to_numpy()
    X = np.full((len(df), latent_dim), np.nan, dtype=np.float32)
    for y in years:
        path, key = _latent_year_path(z_dir, y)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no Z_disp_{y}.npz or Z_latent_{y}.npy in {z_dir}. The latent "
                f"cube is built on HPC; this tier cannot run from a laptop "
                f"checkout.")
        sel = yr == y
        if key is None:
            z = np.load(path, mmap_mode="r")
            X[sel] = np.asarray(z[r[sel], c[sel], :latent_dim], dtype=np.float32)
        else:
            with np.load(path) as zf:
                z = zf[key]
                z = z[0] if z.ndim == 4 else z      # (1, Ny, Nx, M) -> (Ny, Nx, M)
                X[sel] = np.asarray(z[r[sel], c[sel], :latent_dim],
                                    dtype=np.float32)
    return X, [f"z{i:02d}" for i in range(latent_dim)]


def full_states_design(df, states_dir=None):
    """The 'full' tier: the ~295 raw encoder channels from yearly_states."""
    if states_dir is None:
        hist = _cfg_path(_load_esk_desk, "states", "hist_dir",
                         default=os.path.join(_PROC, "encoder", "states"))
        states_dir = os.path.join(hist, "yearly_states")
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
        hist = _cfg_path(_load_esk_desk, "states", "hist_dir",
                         default=os.path.join(_PROC, "encoder", "states"))
        d = os.path.join(hist, "yearly_states")
        gaps = [os.path.join(d, f"state_{y}.npz") for y in _sample_years(years)
                if not os.path.exists(os.path.join(d, f"state_{y}.npz"))]
        n = 0
        if not gaps:
            with np.load(os.path.join(d, f"state_{years[0]}.npz")) as z:
                n = sum((z[k].shape[-1] if z[k].ndim == 3 else 1) for k in z.files)
        return {"tier": tier, "available": not gaps, "n_features": n,
                "dir": d, "first_missing": gaps[:3]}

    if tier == "latent":
        d = _cfg_path(load_age_model_config, "raw_z_dir",
                      default=os.path.join(_PROC, "latent_avian_paths"))
        gaps = [_latent_year_path(d, y)[0] for y in _sample_years(years)
                if not os.path.exists(_latent_year_path(d, y)[0])]
        return {"tier": tier, "available": not gaps,
                "n_features": _default_latent_dim(),
                "dir": d, "first_missing": gaps[:3]}

    raise ValueError(f"unknown tier {tier!r}")


def _first_missing(source, years):
    """The first path a source needs but cannot use, for a legible error."""
    probe = _sample_years(years) if getattr(source, "annual", False) else [None]
    if isinstance(source, DerivedSource):
        if not source.groups:
            return f"{source.name}: no channels discovered (grid dir empty or absent)"
        for y in probe:
            for members in source.groups.values():
                for v in members:
                    pth = source.template.replace("{var}", v)
                    if source.annual:
                        pth = pth.replace("{year}", str(y))
                    if not _exists_ok(pth) or not _geometry_ok(pth):
                        return _why(pth, source.name)
        return None
    if not source.variables:
        return f"{source.name}: no variables discovered"
    for y in probe:
        for pth in source.paths_for(y).values():
            if not _exists_ok(pth) or not _geometry_ok(pth):
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


def last_complete_year(years, tier):
    """Greatest year in ``years`` for which ``tier`` has all its inputs.

    Covariate streams end before BBS does -- LUH-3's source netCDF runs to 2024
    and the climate downscaling lags similarly, while the BBS 2026 release
    carries the 2025 field season. Rather than hardcode a cutoff that will drift,
    walk back from the newest year until one is complete. Returns None when no
    year is.

    One representative file per source per year, so the cost is a few stats.
    """
    ys = sorted(set(int(y) for y in years))
    for y in reversed(ys):
        if _year_complete(y, tier):
            return y
    return None


def _year_complete(year, tier):
    if tier == "standard":
        for sc in standard_sources():
            if not sc.annual:
                continue
            if isinstance(sc, DerivedSource):
                if not sc.groups:
                    return False
                probe = next(iter(sc.groups.values()))[0]
            else:
                if not sc.variables:
                    return False
                probe = sc.variables[0]
            p = sc.template.replace("{var}", probe).replace("{year}", str(year))
            if not _exists_ok(p):
                return False
        return True
    if tier == "full":
        hist = _cfg_path(_load_esk_desk, "states", "hist_dir",
                         default=os.path.join(_PROC, "encoder", "states"))
        return _exists_ok(os.path.join(hist, "yearly_states", f"state_{year}.npz"))
    if tier == "latent":
        d = _cfg_path(load_age_model_config, "raw_z_dir",
                      default=os.path.join(_PROC, "latent_avian_paths"))
        return _exists_ok(_latent_year_path(d, year)[0])
    raise ValueError(f"unknown tier {tier!r}")


def pca_reduce(X, n_components=None, var_target=None, rel_tol=1e-8,
               mu=None, sd=None, basis=None, center=None):
    """Decorrelate a design matrix, fitting the rotation on TRAIN only.

    WHY THIS EXISTS. The encoder's covariate bank is 12 climate bases x 12
    bio-year months plus near-duplicate quantile levels, so adjacent channels
    are almost the same variable. A boosted tree is indifferent to that; a
    LINEAR model is not -- collinear columns make a ridge in the coefficient
    posterior, which is what leaves NUTS saturating max tree depth on every
    iteration and reports as slow, badly mixed sampling.

    PCA is the standard SDM remedy for correlated predictors, and it keeps the
    comparison honest in a way a hand-picked subset would not: the linear model
    still sees the SAME INFORMATION the encoder sees, merely rotated into a
    basis it can condition on. Nothing is discarded except variance below
    ``var_target``.

    By DEFAULT nothing is truncated beyond numerically degenerate directions:
    pass ``n_components`` or ``var_target`` to reduce deliberately, or use
    select_n_components_cv to choose the count by held-out prediction.

    Pass ``basis``/``center``/``mu``/``sd`` back in to apply a fitted rotation
    to held-out rows.
    """
    X = np.asarray(X, dtype=np.float64)
    if basis is None:
        Xs, mu, sd = standardize(X)
        Xs = np.nan_to_num(Xs)
        center = Xs.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xs - center, full_matrices=False)
        var = S ** 2
        ratio = var / max(var.sum(), 1e-300)
        if n_components is not None:
            k = max(1, min(int(n_components), Vt.shape[0]))
        elif var_target is not None:
            k = int(np.searchsorted(np.cumsum(ratio), float(var_target)) + 1)
            k = max(1, min(k, Vt.shape[0]))
        else:
            # DEFAULT: rotate, do not truncate. Keep every direction above the
            # numerical-noise floor. Rotation alone is what fixes conditioning
            # -- the components are standardized to unit variance before
            # fitting, so the design becomes orthonormal regardless of how many
            # are kept -- and truncating on VARIANCE can discard real signal,
            # because variance is not predictive relevance. Verified: with the
            # signal planted in a low-variance component, a 0.99 variance budget
            # drops it (CV AUC 0.49) while keeping all of them recovers it
            # (0.71). Cost is not a reason to truncate either: the SVD is 0.06 s.
            k = max(1, int((S > S[0] * float(rel_tol)).sum()))
        basis = Vt[:k]
        explained = float(np.cumsum(ratio)[k - 1])
    else:
        Xs, _, _ = standardize(X, mu, sd)
        Xs = np.nan_to_num(Xs)
        explained = float("nan")
    Z = (Xs - center) @ basis.T
    names = [f"pc{i:03d}" for i in range(basis.shape[0])]
    return Z.astype(np.float32), names, {
        "basis": basis, "center": center, "mu": mu, "sd": sd,
        "n_components": int(basis.shape[0]), "explained_variance": explained}


def condition_number(X):
    """Ratio of largest to smallest singular value of the standardized matrix.

    A large value is the quantitative statement of "these predictors are nearly
    duplicates", which is what a linear model chokes on.
    """
    Xs, _, _ = standardize(np.asarray(X, dtype=np.float64))
    sv = np.linalg.svd(np.nan_to_num(Xs), compute_uv=False)
    sv = sv[sv > 0]
    return float(sv[0] / sv[-1]) if sv.size else float("inf")


def stream_of(name):
    """The encoder stream a channel name belongs to ("climate:012" -> climate)."""
    return str(name).split(":", 1)[0] if ":" in str(name) else "_"


def pca_reduce_by_stream(X, names, var_target=None, min_components=1,
                         fitted=None):
    """PCA WITHIN each covariate stream, then concatenate.

    USE THIS ONLY WHEN TRUNCATING. Per-stream rotation orthogonalizes WITHIN
    each stream and does nothing across them, and the streams are physically
    correlated -- temperature with elevation through the lapse rate, HYDE
    population with HISDAC built-up and urban land use all measuring human
    footprint. Measured on streams sharing latent drivers, the concatenated
    per-stream components still carry a condition number of 31.6 and a worst
    cross-stream correlation of 0.997 (built-up vs population), where a single
    global rotation gives exactly 1.

    What per-stream buys is protection for the SMALL streams when components are
    being discarded: climate is ~80% of the bank, so a variance budget applied
    globally can squeeze the ~10 human-footprint channels out of the retained
    set invisibly -- a bad trade for a human-commensal species. That only
    matters if something is being dropped. The default path truncates nothing,
    so it uses the global rotation instead; reach for this one alongside
    ``var_target``.

    Returns ``(Z, names, info)``; pass ``info["fitted"]`` back as ``fitted`` to
    apply the same rotation to held-out rows.
    """
    X = np.asarray(X, dtype=np.float64)
    groups = {}
    for j, nm in enumerate(names):
        groups.setdefault(stream_of(nm), []).append(j)

    out_cols, out_names, per_stream = [], [], {}
    for stream in sorted(groups):
        idx = groups[stream]
        sub = X[:, idx]
        prev = (fitted or {}).get(stream)
        if prev is None:
            Z, _, info = pca_reduce(sub, var_target=var_target)
            if Z.shape[1] < min_components:
                Z, _, info = pca_reduce(sub, n_components=min(min_components,
                                                              sub.shape[1]))
        else:
            Z, _, info = pca_reduce(sub, basis=prev["basis"],
                                    center=prev["center"], mu=prev["mu"],
                                    sd=prev["sd"])
        out_cols.append(Z)
        out_names += [f"{stream}:pc{i:03d}" for i in range(Z.shape[1])]
        per_stream[stream] = {"n_in": len(idx), "n_out": int(Z.shape[1]),
                              "explained_variance": info["explained_variance"],
                              "basis": info["basis"], "center": info["center"],
                              "mu": info["mu"], "sd": info["sd"]}
    Z = np.hstack(out_cols).astype(np.float32)
    return Z, out_names, {
        "fitted": per_stream,
        "n_components": int(Z.shape[1]),
        "per_stream": {k: {"n_in": v["n_in"], "n_out": v["n_out"],
                           "explained_variance": v["explained_variance"]}
                       for k, v in per_stream.items()}}


def select_n_components_cv(Z, y, groups=None, candidates=None, n_folds=5,
                           seed=0):
    """Choose how many principal components to keep by HELD-OUT PREDICTION.

    WHY NOT A VARIANCE BUDGET. "Keep 99% of variance" is arbitrary, and variance
    is not predictive relevance: a low-variance direction can carry real signal
    and a high-variance one can be pure nuisance. That is the standard objection
    to principal-component regression, and truncating on variance quietly bakes
    it in.

    Cost is not the constraint either -- the SVD is 0.06 s on this design -- so
    the count can be chosen the way any other hyperparameter would be. This
    scores nested prefixes of the (variance-ordered) components with a cheap
    logistic ridge under grouped CV, and returns the smallest count within one
    standard error of the best: the usual one-SE rule, which prefers parsimony
    where the curve is flat.

    ``groups`` should be spatial blocks, so the choice is not made optimistic by
    spatial autocorrelation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, KFold

    Z = np.asarray(Z, dtype=np.float64)
    y = (np.asarray(y) > 0).astype(int)
    p = Z.shape[1]
    if candidates is None:
        candidates = sorted({max(1, int(round(p * f)))
                             for f in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)})

    splitter = (GroupKFold(n_splits=n_folds) if groups is not None
                else KFold(n_splits=n_folds, shuffle=True, random_state=seed))
    split_args = (Z, y, groups) if groups is not None else (Z, y)

    means, ses = [], []
    for k in candidates:
        scores = []
        for tr, va in splitter.split(*split_args):
            if y[tr].min() == y[tr].max():
                continue
            m = LogisticRegression(max_iter=2000, C=1.0)
            m.fit(Z[tr, :k], y[tr])
            scores.append(_auc_binary(m.predict_proba(Z[va, :k])[:, 1], y[va]))
        if not scores:
            means.append(np.nan); ses.append(np.nan); continue
        means.append(float(np.mean(scores)))
        ses.append(float(np.std(scores, ddof=1) / max(np.sqrt(len(scores)), 1)))

    means = np.asarray(means, dtype=float)
    if not np.isfinite(means).any():
        return p, {"error": "CV produced no usable folds", "candidates": candidates}
    best = int(np.nanargmax(means))
    threshold = means[best] - (ses[best] if np.isfinite(ses[best]) else 0.0)
    chosen = next(k for k, m in zip(candidates, means)
                  if np.isfinite(m) and m >= threshold)
    return int(chosen), {"candidates": list(map(int, candidates)),
                         "mean_auc": [None if not np.isfinite(v) else round(v, 5)
                                      for v in means],
                         "best_k": int(candidates[best]),
                         "chosen_k": int(chosen),
                         "rule": "smallest k within 1 SE of best CV AUC"}


def _auc_binary(scores, labels):
    """Mann-Whitney AUC, duplicated here to keep this module import-light."""
    from scipy.stats import rankdata
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def numerical_rank(X, rel_tol=1e-8):
    """Components above the numerical-noise floor.

    The floor matters because the components are standardized to unit variance
    before fitting: whitening a direction whose eigenvalue is numerically zero
    amplifies pure rounding error to the same scale as real signal.
    """
    Xs, _, _ = standardize(np.asarray(X, dtype=np.float64))
    sv = np.linalg.svd(np.nan_to_num(Xs), compute_uv=False)
    return int((sv > sv[0] * float(rel_tol)).sum()) if sv.size else 0
