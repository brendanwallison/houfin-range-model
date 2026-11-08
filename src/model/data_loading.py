"""Shared loader for the age-structured model's numpyro inputs.

One ``load_data`` for every inference entry point, so residency and dtype handling
cannot drift between them. Heavy inputs default to explicit device residency: a
JIT-compiled differentiable ``lax.scan`` cannot truly stream NumPy slices from host
memory, so calling these arrays "streaming" only obscured XLA's copies. Set
``HOUFIN_MODEL_INPUT_RESIDENCY=host`` as an experimental mode only.
"""
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

# Only the Z arrays are large enough to need explicit residency handling; the disease
# bases are two time-independent spatial bases of a few MB.
LARGE_INPUT_KEYS = {"Z_gathered", "Z_disp_gathered", "Z_disp_seg_gathered"}


def occupied_habitat_center(meta):
    """Mean Z feature vector over OCCUPIED cell-years (a positive BBS count), host-side.

    The location the habitat fields are centred at when population_model.habitat_centering
    is "occupied" (age_fields.project_and_scatter_age_structured): each rate's intercept
    then means "the rate at the habitat typical of occupied cells", the population the
    capacity anchor (median occupied cell) was measured on. Each occupied (year, cell)
    counts once, however many routes it has. None if the inputs needed are absent.
    """
    keys = ("Z_gathered", "land_rows", "land_cols", "obs_time_indices", "obs_rows",
            "obs_cols", "observed_results")
    if not all(isinstance(meta.get(k), np.ndarray) for k in keys):
        return None
    lookup = np.full((int(meta["Ny"]), int(meta["Nx"])), -1, dtype=np.int64)
    lookup[meta["land_rows"], meta["land_cols"]] = np.arange(meta["land_rows"].size)
    cell = lookup[meta["obs_rows"], meta["obs_cols"]]
    keep = (np.asarray(meta["observed_results"]) > 0) & (cell >= 0)
    pairs = np.unique(np.stack([meta["obs_time_indices"][keep], cell[keep]], axis=1), axis=0)
    if pairs.size == 0:
        return None
    return np.asarray(meta["Z_gathered"][pairs[:, 0], pairs[:, 1]], dtype=np.float64).mean(axis=0)


def _habitat_shrink_enabled():
    from src.config_utils import load_age_model_config
    spec = load_age_model_config().get("population_model", {}).get("habitat_change") or {}
    return spec.get("shrink") == "learned"


def _route_effects_enabled():
    from src.config_utils import load_age_model_config
    spec = load_age_model_config().get("population_model", {}).get("observation") or {}
    return bool(spec.get("route_effects", False))


def _observer_effect_enabled():
    from src.config_utils import load_age_model_config
    spec = (load_age_model_config().get("population_model", {}).get("observation") or {})
    return bool((spec.get("observer_effect") or {}).get("enabled", False))


OBSERVER_KEYS = {"obs_route_idx": -1, "obs_obsroute_idx": -1, "obs_first_year": 0}   # key -> pseudo-zero fill


def readout_spec():
    """population_model.readout from the active config ({} = the single-amplitude readout)."""
    from src.config_utils import load_age_model_config
    return dict(load_age_model_config().get("population_model", {}).get("readout") or {})


def covariate_drift_spec():
    """population_model.covariate_drift from the active config, or None (off, the default)."""
    from src.config_utils import load_age_model_config
    spec = load_age_model_config().get("population_model", {}).get("covariate_drift") or {}
    return dict(spec) if spec.get("form") else None


def covariate_drift_basis(years, obs_time_indices, n_basis, window_start=1966):
    """(n_basis, T) smooth time basis for population_model.covariate_drift.

    cos(m*pi*s), m = 1..n_basis, with s running 0 -> 1 over [window_start, last year] and held at 0
    before it: the drift is defined only where BBS observes it, so the unobserved early decades
    cannot use it to manufacture the invasion. Each row is centred by the OBSERVATION-weighted
    mean over the window, so the static coefficients keep meaning "the average relationship over
    the data" and the drift is the departure from it.
    """
    years = np.asarray(years, dtype=float)
    s = np.clip((years - window_start) / max(years[-1] - window_start, 1.0), 0.0, 1.0)
    w = np.bincount(np.asarray(obs_time_indices), minlength=years.size).astype(float)
    w = np.where(years >= window_start, w, 0.0)
    if w.sum() == 0:
        raise ValueError(f"covariate_drift: no observations at or after {window_start}")
    w = w / w.sum()
    rows = []
    for m in range(1, int(n_basis) + 1):
        b = np.cos(m * np.pi * s)
        rows.append(b - (w * b).sum())
    return np.stack(rows, axis=0)


def covariate_drift_ramp(years, obs_time_indices, start, end, window_start=1966):
    """(1, T) ramp shape for the covariate drift: 0 before ``start``, 1 after ``end``, linear between,
    held before ``window_start``, centred by the observation-weighted mean over the window (as the cosine
    basis). The default start/end 1979.5-2005 is the hinge-exponential fit to Plummer et al. 2019's
    UK bird-food product counts (log products flat to 1979.5, then +0.090/yr; reference/plummer2019...)."""
    years = np.asarray(years, dtype=float)
    g = np.clip((np.maximum(years, window_start) - start) / (end - start), 0.0, 1.0)
    w = np.bincount(np.asarray(obs_time_indices), minlength=years.size).astype(float)
    w = np.where(years >= window_start, w, 0.0); w = w / w.sum()
    return (g - (w * g).sum())[None, :]


def _extra_pseudo_zeros_enabled():
    from src.config_utils import load_age_model_config
    spec = load_age_model_config().get("population_model", {}).get("observation") or {}
    return bool(spec.get("extra_pseudo_zeros", False))


def _extra_pseudo_zeros_file():
    from src.config_utils import load_age_model_config
    v = (load_age_model_config().get("population_model", {}).get("observation") or {}).get("extra_pseudo_zeros")
    return v if isinstance(v, str) else "extra_pseudo_zeros.npz"


def _subcell_enabled():
    from src.config_utils import load_age_model_config
    spec = load_age_model_config().get("population_model", {}).get("subcell_strata") or {}
    return bool(spec.get("enabled", False))


def _subcell_moderator(input_dir, meta, spec, strata_u, strata_a):
    """Build a fixed (time, land-cell) habitat moderator for a sub-cell tilt.

    ``covariates`` maps named, already-standardised covariates to weights.  The
    special name ``builtup`` is the area-weighted mean of the sub-cell BUI
    slices; other names come from ``drift_covariates.npz``.  With
    ``positive_only`` (the experimental default), only the hypothesised deficit
    tail is active.  RMS scaling makes a coefficient the effect in a typical
    targeted cell while preserving exact zeros outside that tail.
    """
    weights = dict(spec.get("covariates") or {})
    if not weights:
        raise ValueError("subcell moderator requires a non-empty covariates mapping")
    panel = None
    raw = np.zeros((int(meta["time"]), int(meta["N_land"])), dtype=np.float64)
    for name, weight in weights.items():
        if name == "builtup":
            x = (np.asarray(strata_u) * np.asarray(strata_a)[None, None, :]).sum(-1)
        else:
            if panel is None:
                path = os.path.join(input_dir, "drift_covariates.npz")
                with np.load(path) as z:
                    if not np.array_equal(np.asarray(z["years"]), np.asarray(meta["years"]).astype(int)):
                        raise ValueError("drift_covariates.npz years do not match the model inputs")
                    names = [str(n.item() if hasattr(n, "item") else n) for n in z["names"]]
                    panel = {n: np.asarray(z["x"])[..., i] for i, n in enumerate(names)}
            if name not in panel:
                raise ValueError(f"subcell moderator: {name!r} not in drift_covariates.npz "
                                 f"({sorted(panel)})")
            x = panel[name]
        raw += float(weight) * np.asarray(x)
    if bool(spec.get("positive_only", True)):
        raw = np.maximum(raw, 0.0)
    elif bool(spec.get("center", True)):
        raw = raw - raw.mean()
    rms = float(np.sqrt(np.mean(raw * raw)))
    if not rms > 0:
        raise ValueError(f"subcell moderator has no variation: {weights}")
    return raw / rms


def _observation_weighting_spec():
    from src.config_utils import load_age_model_config
    spec = load_age_model_config().get("population_model", {}).get("observation_weighting")
    return dict(spec) if spec else None


def observation_weights(meta, spec):
    """DIAGNOSTIC likelihood weights (population_model.observation_weighting) -- never production:
    a weighted likelihood is not a posterior. Re-score fits UNWEIGHTED to compare them.

    {"plains": w}: CONUS Great Plains barrier-corridor observations (great_plains_zones code 2,
        lat < 49) get weight w.
    {"sparse": {"c": 1, "p": 1, "w_max": 5, "from_year": 1966}}: each observation gets
        clip(w_max * (c / (m + c))^p, 1, w_max), m = its cell's mean OBSERVED count since from_year
        (data-derived, so the weights cannot move with the fit): empty cells w_max, dense cells 1.
    """
    rows, cols = np.asarray(meta["obs_rows"]), np.asarray(meta["obs_cols"])
    w = np.ones(rows.shape, dtype=np.float64)
    if "plains" in spec:
        import rasterio
        from rasterio.warp import transform as crs_transform
        from src.config_utils import load_data_config
        dcfg = load_data_config()
        with rasterio.open(os.path.expandvars(dcfg["regions"]["great_plains_zones"])) as src:
            zones = src.read(1)
        with rasterio.open(os.path.expandvars(dcfg["grid"]["ref_raster"])) as src:
            xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
            _, lat = crs_transform(src.crs, "EPSG:4326", list(xs), list(ys))
        plains = (zones[rows, cols] == 2) & (np.asarray(lat) < 49.0)
        w = np.where(plains, float(spec["plains"]), w)
    if "sparse" in spec:
        sp = dict(spec["sparse"])
        c, pw, wmax = float(sp.get("c", 1.0)), float(sp.get("p", 1.0)), float(sp.get("w_max", 5.0))
        years = np.asarray(meta["years"])[np.asarray(meta["obs_time_indices"])]
        y = np.asarray(meta["observed_results"], dtype=np.float64)
        cell = rows.astype(np.int64) * 100000 + cols
        _, idx = np.unique(cell, return_inverse=True)
        era = years >= int(sp.get("from_year", 1966))
        s = np.bincount(idx[era], weights=y[era], minlength=idx.max() + 1)
        n = np.bincount(idx[era], minlength=idx.max() + 1)
        m = np.where(n > 0, s / np.maximum(n, 1), 0.0)[idx]
        w = w * np.clip(wmax * (c / (m + c)) ** pw, 1.0, wmax)
    return w


def _observation_mask_spec():
    """population_model.observation_mask from the active config, or None (the default)."""
    from src.config_utils import load_age_model_config
    spec = load_age_model_config().get("population_model", {}).get("observation_mask")
    return dict(spec) if spec else None


def holdout_observation_mask(meta, spec):
    """True per observation HELD OUT by population_model.observation_mask (E070 model comparison).

    ``holdout_years``: every observation in those years (the temporal forecast test).
    ``holdout_blocks`` {"block_cells", "fraction", "seed"}: square blocks of ``block_cells`` grid cells; a random
    ``fraction`` of the blocks that hold observations is withheld in ALL years (the spatial test). Deterministic in
    the seed, so every arm of a comparison withholds the same observations.
    ``holdout_blocks.pinned_blocks`` (path to a JSON {"block_km", "blocks": [[block_row, block_col], ...]}): withhold
    exactly those blocks instead of drawing them. For comparisons ACROSS grids: the draw depends on which blocks hold
    observations, which changes with the grid, so a coarser grid pins the reference grid's selection. Requires
    block_cells x cell_size_km == block_km (the blocks then nest exactly: a 54 km 4x4 block is the 27 km 8x8 block
    with the same index). fraction/seed are ignored when pinned.
    """
    years = np.asarray(meta["years"])[np.asarray(meta["obs_time_indices"])].astype(int)
    out = np.zeros(years.shape, dtype=bool)
    if spec.get("holdout_years"):
        out |= np.isin(years, np.asarray(spec["holdout_years"], dtype=int))
    blk = spec.get("holdout_blocks")
    if blk:
        b = int(blk.get("block_cells", 8))
        bid = (np.asarray(meta["obs_rows"]).astype(np.int64) // b) * 100000 + np.asarray(meta["obs_cols"]) // b
        if blk.get("pinned_blocks"):
            import json
            with open(os.path.expandvars(blk["pinned_blocks"]), encoding="utf-8") as fh:
                pinned = json.load(fh)
            cell_km = float((meta.get("path_feature_meta") or {}).get("cell_size_km", np.nan))
            if not np.isclose(b * cell_km, float(pinned["block_km"])):
                raise ValueError(f"pinned holdout blocks are {pinned['block_km']} km but block_cells={b} x "
                                 f"{cell_km} km cells = {b * cell_km} km; blocks would not nest")
            pick = np.array([int(r) * 100000 + int(c) for r, c in pinned["blocks"]], dtype=np.int64)
        else:
            ids = np.unique(bid)
            rng = np.random.default_rng(int(blk.get("seed", 70)))
            pick = rng.choice(ids, size=max(1, int(round(float(blk.get("fraction", 0.1)) * ids.size))), replace=False)
        out |= np.isin(bid, pick)
    return out


def process_noise_basis(meta, spec):
    """(N_land, nb) basis for population_model.process_noise and the (time,) 0/1 window it acts in.

    Land-centred cosines (model_inputs.generate_spatial_basis), scaled so that iid unit-variance coefficients give
    unit mean-square eps over land: sigma is then the land-RMS of the yearly log-abundance nudge. The window is
    [window_start, window_end], optionally restricted to the listed active_years (a few large state resets instead
    of yearly noise -- multiple shooting with a smooth regional jump per boundary); held-out forecast years (observation_mask.holdout_years) and every year after the
    first of them are switched off, so a forecast cannot lean on noise it would not have.
    """
    from src.data.combine.model_inputs import generate_spatial_basis
    S = generate_spatial_basis(int(meta["Ny"]), int(meta["Nx"]), np.asarray(meta["land_rows"]),
                               np.asarray(meta["land_cols"]), int(spec.get("n_freq", 3)), label=" (process_noise)").T
    S = S / np.sqrt(np.mean(np.sum(S ** 2, axis=1)))
    years = np.asarray(meta["years"]).astype(int)
    active = (years >= int(spec.get("window_start", years[0]))) & (years <= int(spec.get("window_end", years[-1])))
    if spec.get("active_years"):     # state resets (multiple shooting): noise only in these years, e.g. [1966]
        active &= np.isin(years, [int(y) for y in spec["active_years"]])
    hold = (_observation_mask_spec() or {}).get("holdout_years")
    if hold:
        active &= years < min(int(y) for y in hold)
    return S, active.astype(float)


def state_anchor_arrays(meta, spec):
    """Observed abundance map for population_model.state_anchor: per land cell, the mean TRAINING count over
    `obs_years` (held-out observations, zero-weight ones, excluded) divided by pop_scalar -> model density units.
    Returns anchor_t (index of `year`), anchor_A (N_land,), anchor_s (N_land,; 1 = surveyed)."""
    years = np.asarray(meta["years"]).astype(int)
    yr = int(spec.get("year", 1964))
    if yr not in years:
        raise ValueError(f"state_anchor.year {yr} outside the model years")
    oy = set(int(y) for y in spec.get("obs_years", [1966, 1967, 1968]))
    y = np.asarray(meta["observed_results"], float)
    t = np.asarray(meta["obs_time_indices"]).astype(int)
    w = np.asarray(meta.get("obs_weight", np.ones_like(y)))
    held = np.asarray(meta.get("obs_holdout", np.zeros(y.shape, bool)), bool)
    use = np.isin(years[t], list(oy)) & (w > 0) & ~held
    lr, lc = np.asarray(meta["land_rows"]), np.asarray(meta["land_cols"])
    lut = -np.ones((int(meta["Ny"]), int(meta["Nx"])), np.int64); lut[lr, lc] = np.arange(len(lr))
    cell = lut[np.asarray(meta["obs_rows"])[use], np.asarray(meta["obs_cols"])[use]]
    ok = cell >= 0
    tot = np.bincount(cell[ok], weights=y[use][ok], minlength=len(lr))
    n = np.bincount(cell[ok], minlength=len(lr))
    A = np.where(n > 0, tot / np.maximum(n, 1), 0.0) / float(meta["pop_scalar"])
    return {"anchor_t": int(np.flatnonzero(years == yr)[0]), "anchor_A": A, "anchor_s": (n > 0).astype(float)}


def process_noise_groups(meta, spec):
    """(N_land,) int group ids and their count for process_noise.conserve: "total" = one group; "regional" = square
    blocks of conserve_block cells on the model grid; "none" = every cell its own group (unused)."""
    rows, cols = np.asarray(meta["land_rows"]).astype(int), np.asarray(meta["land_cols"]).astype(int)
    mode = str(spec.get("conserve", "none"))
    if mode == "total":
        return np.zeros(len(rows), np.int32), 1
    if mode == "regional":
        b = int(spec.get("conserve_block", 8))
        key = (rows // b) * (int(meta["Nx"]) // b + 1) + cols // b
        _, gid = np.unique(key, return_inverse=True)
        return gid.astype(np.int32), int(gid.max()) + 1
    return np.arange(len(rows), dtype=np.int32), len(rows)


def process_noise_detrend(years, active, degree):
    """(T, T) projector removing a polynomial trend (degree 0 = mean, 1 = mean + linear, ...) from each noise
    coefficient's series over the active years, and zeroing inactive years (process_noise.detrend_degree, E070
    stage 2). Applied to the yearly coefficients, it leaves eps no sustained regional trend: any persistent change must
    come from habitat (the lambda(t) term) or from the dynamics."""
    a = np.asarray(active) > 0
    yrs = np.asarray(years, dtype=float)
    s = (yrs - yrs[a].min()) / max(yrs[a].max() - yrs[a].min(), 1.0)
    G = np.stack([s ** k for k in range(int(degree) + 1)], axis=1) * a[:, None]       # (T, degree+1)
    P = np.diag(a.astype(float)) - G @ np.linalg.pinv(G)
    return P


def transient_observation_weight(meta, window, censor_years=2):
    """1/0 likelihood weight per observation that DROPS each cell's colonization transient.

    DIAGNOSTIC ONLY (population_model.observation_mask, research/age_model F1): a cell's
    arrival is its first positive standard-tier count; observations within +-``window`` years
    of it are dropped. Cells whose first positive comes within ``censor_years`` of their first
    survey are LEFT-CENSORED (already occupied when surveying began, e.g. the New York area in
    1966) -- their first positive is not an arrival, so they are kept. Data-derived on purpose:
    the question is which observations force the fit, not a model of them.
    """
    years = np.asarray(meta["years"])[np.asarray(meta["obs_time_indices"])]
    cell = np.asarray(meta["obs_rows"]).astype(np.int64) * 100000 + np.asarray(meta["obs_cols"])
    y = np.asarray(meta["observed_results"])
    q = np.asarray(meta.get("obs_quality", np.zeros_like(y)))
    _, idx = np.unique(cell, return_inverse=True)
    big = np.iinfo(np.int64).max
    first_survey = np.full(idx.max() + 1, big)
    first_pos = np.full(idx.max() + 1, big)
    std = q == 0
    np.minimum.at(first_survey, idx[std], years[std])
    np.minimum.at(first_pos, idx[std & (y > 0)], years[std & (y > 0)])
    arrival = (first_pos < big) & (first_pos - first_survey > censor_years)
    drop = arrival[idx] & (np.abs(years - np.where(arrival, first_pos, 0)[idx]) <= window)
    return np.where(drop, 0.0, 1.0)


NEST_COVARIATES = ("building", "nestbox", "other_artificial", "substrate_unknown", "card", "mg_era", "suburban")


def nest_covariate_matrix(z, names):
    """(n, p) 0/1 design for the nest nuisance terms; reference class = natural substrate, modern NestWatch."""
    cols = {"building": z["substrate"] == 1, "nestbox": z["substrate"] == 2, "other_artificial": z["substrate"] == 3,
            "substrate_unknown": z["substrate"] == 4, "card": z["source"] == 1, "mg_era": z["mg_era"] == 1,
            "suburban": z["suburban"] == 1}
    bad = [n for n in names if n not in cols]
    if bad:
        raise ValueError(f"nest_likelihood.covariates: unknown {bad}; choose from {NEST_COVARIATES}")
    n = np.asarray(z["substrate"]).shape[0]
    return np.stack([np.asarray(cols[c], float) for c in names], -1) if names else np.zeros((n, 0))


def nest_arrays(nspec, meta, f_type=np.float32, verbose=True):
    """Nest-record arrays for population_model.nest_likelihood (research/age_model/exp/prepare_nestwatch.py rows).

    Keeps attempts inside year_range with location precision <= max_precision (0 exact, 1 county centroid, 2 state
    centroid) that inform the likelihood: a logistic-exposure row (has_exp) and/or a count row (success with
    fledged >= 1). The fecundity readout is gathered at the DISTINCT nest cells only (nest_rows / nest_cols; nest_ci
    indexes them per attempt).
    """
    with np.load(nspec["path"]) as zz:
        z = {k: np.asarray(zz[k]) for k in zz.files}
    years = np.asarray(meta["years"]).astype(int)
    if int(z["model_first_year"]) != int(years[0]):
        raise ValueError(f"{nspec['path']}: built for first year {int(z['model_first_year'])}, model {years[0]}")
    y0, y1 = nspec.get("year_range", [int(years[0]), int(years[-1])])
    cnt = (z["outcome"] == 1) & (z["fledged"] >= 1)
    keep = ((z["year"] >= int(y0)) & (z["year"] <= int(y1)) & (z["precision"] <= int(nspec.get("max_precision", 1)))
            & ((z["has_exp"] == 1) | cnt) & (z["t_idx"] >= 0) & (z["t_idx"] < int(meta["time"])))
    z = {k: (v[keep] if np.ndim(v) and v.shape[0] == keep.shape[0] else v) for k, v in z.items()}
    cells, ci = np.unique(z["land_idx"], return_inverse=True)
    lr, lc = np.asarray(meta["land_rows"]), np.asarray(meta["land_cols"])
    X = nest_covariate_matrix(z, list(nspec.get("covariates", [])))
    out = {
        "nest_rows": lr[cells].astype(np.int64), "nest_cols": lc[cells].astype(np.int64),
        "nest_ci": ci.astype(np.int64), "nest_t": z["t_idx"].astype(np.int64), "nest_region": z["region"].astype(np.int64),
        "nest_X": X.astype(f_type), "nest_outcome": z["outcome"].astype(np.int64),
        "nest_fledged": z["fledged"].astype(np.int64), "nest_e_pre": z["e_pre"].astype(f_type),
        "nest_d_last": z["d_last"].astype(f_type), "nest_w_exp": (z["has_exp"] == 1).astype(f_type),
        "nest_w_cnt": ((z["outcome"] == 1) & (z["fledged"] >= 1)).astype(f_type),
        "nest_suburban": z["suburban"].astype(f_type),
    }
    if verbose:
        print(f"  [nest_likelihood] {keep.sum()} attempts ({int(out['nest_w_exp'].sum())} exposure rows, "
              f"{int(out['nest_w_cnt'].sum())} count rows) in {cells.size} cells, years {y0}-{y1}, covariates "
              f"{list(nspec.get('covariates', []))}")
    return out


def load_data(input_dir, target_device=None, precision="float32", verbose=True):
    """Load model inputs, casting to ``precision`` and placing device arrays.

    If ``target_device`` is given, arrays are materialized on it unless explicit
    experimental host residency is requested.
    """
    meta_path = os.path.join(input_dir, "metadata.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    from src.config_utils import load_age_model_config
    from src.model.dispersal_spec import require_inputs_match_config
    require_inputs_match_config(meta.get("dispersal_spec"), load_age_model_config())

    f_type_cpu = np.float32 if precision == "float32" else np.float64
    f_type_target = jnp.float32 if precision == "float32" else jnp.float64
    i_type_target = jnp.int32 if precision == "float32" else jnp.int64

    z_shape = (meta["time"], meta["N_land"], meta["M"])
    z_mem = np.memmap(
        os.path.join(input_dir, meta["z_gathered_path"]),
        dtype="float32", mode="r", shape=z_shape,
    )

    z_disp_shape = (meta["time"], meta["N_land"], meta["K"], meta["M"])
    z_disp_mem = np.memmap(
        os.path.join(input_dir, meta["z_disp_gathered_path"]),
        dtype="float32", mode="r", shape=z_disp_shape,
    )

    meta["Z_gathered"] = np.asarray(z_mem, dtype=f_type_cpu)
    meta["Z_disp_gathered"] = np.asarray(z_disp_mem, dtype=f_type_cpu)
    rspec = readout_spec()
    columns = rspec.get("feature_columns")
    if columns is not None:
        from src.community_encoder.cube_contract import select_block_prefixes
        selected = select_block_prefixes(meta["z_kernel_contract"], columns)
        if rspec.get("rank") and int(rspec["rank"]) != len(columns):
            raise ValueError("readout.rank must match explicit feature_columns")
        meta["Z_gathered"] = np.ascontiguousarray(meta["Z_gathered"][..., columns])
        meta["Z_disp_gathered"] = np.ascontiguousarray(meta["Z_disp_gathered"][..., columns])
        meta["M"] = len(columns)
        meta["z_kernel_contract"] = selected
    elif rspec.get("rank"):
        # population_model.readout.rank: use the leading `rank` DESK features of a wider input
        # (features are ordered by eigenvalue, so this is the same top-eigenfeature truncation the
        # builder does), so ranks 24/32/64 share ONE input directory.
        r = int(rspec["rank"])
        if meta.get("z_kernel_contract", {}).get("kernel") == "ruzicka_sum" and r != meta["Z_gathered"].shape[-1]:
            raise ValueError("sum kernel requires explicit per-season feature_columns for truncation")
        if r > meta["Z_gathered"].shape[-1]:
            raise ValueError(f"readout.rank {r} > input width {meta['Z_gathered'].shape[-1]}")
        meta["Z_gathered"] = np.ascontiguousarray(meta["Z_gathered"][..., :r])
        meta["Z_disp_gathered"] = np.ascontiguousarray(meta["Z_disp_gathered"][..., :r])
        meta["M"] = r
        c = dict(meta.get("z_kernel_contract") or {})
        if c:
            c["latent_dim"] = r
            if int(c.get("source_latent_dim", r)) > r:
                c["truncation"] = "top_eigenfeatures"
            meta["z_kernel_contract"] = c
    if rspec.get("split"):
        # population_model.readout.split: the cell's long-run level z-bar (mean over the reference
        # window, default 1996-2025 -- the DESK E012/E059e choice; a recent window erases the gain)
        # and the within-cell deviation z(t) - z-bar get SEPARATE weights/amplitudes (split kernel).
        y0, y1 = rspec.get("ref_window", [1996, 2025])
        yrs = np.asarray(meta["years"]).astype(int)
        win = (yrs >= int(y0)) & (yrs <= int(y1))
        if not win.any():
            raise ValueError(f"readout.ref_window {y0}-{y1} has no model years")
        meta["Z_ref"] = meta["Z_gathered"][win].mean(axis=0)
        meta["Z_disp_ref"] = meta["Z_disp_gathered"][win].mean(axis=0)
    # A plain int, computed from the HOST array before device placement, so the model can
    # decide structurally (whether quality_conc_mult exists) without touching jnp.
    if isinstance(meta.get("obs_quality"), np.ndarray):
        meta["n_obs_quality_tiers"] = int(np.unique(meta["obs_quality"]).size)
    if _habitat_shrink_enabled():
        # per-cell long-run means for population_model.habitat_change (host, one pass)
        meta["Z_mean"] = meta["Z_gathered"].mean(axis=0)
        meta["Z_disp_mean"] = meta["Z_disp_gathered"].mean(axis=0)
    if _route_effects_enabled():
        with np.load(os.path.join(input_dir, "obs_route_ids.npz")) as z:
            if z["route"].shape[0] != np.asarray(meta["observed_results"]).shape[0]:
                raise ValueError("obs_route_ids.npz does not match the observations; rebuild it "
                                 "(research/age_model/exp/build_route_ids.py)")
            meta["obs_route"] = np.asarray(z["route"]).astype(np.int64)
            meta["n_routes"] = int(z["n_routes"])        # a plain int: it sets a sample shape
    if _observer_effect_enabled() and "obs_obsroute_idx" not in meta:
        # Inputs ingested before model_inputs carried the observer arrays: a verified side file
        # (research/age_model/exp/build_observer_ids.py), like obs_route_ids.npz above.
        with np.load(os.path.join(input_dir, "obs_observer_ids.npz")) as z:
            if z["obs_obsroute_idx"].shape[0] != np.asarray(meta["observed_results"]).shape[0]:
                raise ValueError("obs_observer_ids.npz does not match the observations; rebuild it "
                                 "(research/age_model/exp/build_observer_ids.py)")
            for k in OBSERVER_KEYS:
                meta[k] = np.asarray(z[k]).astype(np.int64)
    if _extra_pseudo_zeros_enabled():
        # Appended as standard-tier zero counts with no route (research/age_model/exp/
        # build_extra_pseudo_zeros.py): pre-BBS absences 1940-1965 far from the release.
        with np.load(os.path.join(input_dir, _extra_pseudo_zeros_file())) as z:
            n_new = int(z["rows"].shape[0])
            meta["obs_rows"] = np.concatenate([np.asarray(meta["obs_rows"]), z["rows"]])
            meta["obs_cols"] = np.concatenate([np.asarray(meta["obs_cols"]), z["cols"]])
            meta["obs_time_indices"] = np.concatenate([np.asarray(meta["obs_time_indices"]), z["time_idx"]])
            meta["observed_results"] = np.concatenate(
                [np.asarray(meta["observed_results"]), np.zeros(n_new, np.asarray(meta["observed_results"]).dtype)])
            if isinstance(meta.get("obs_quality"), np.ndarray):
                meta["obs_quality"] = np.concatenate([meta["obs_quality"], np.zeros(n_new, meta["obs_quality"].dtype)])
            if "obs_route" in meta:
                meta["obs_route"] = np.concatenate([meta["obs_route"], np.full(n_new, -1, np.int64)])
            for k, fill in OBSERVER_KEYS.items():   # no route, no observer
                if k in meta:
                    meta[k] = np.concatenate([np.asarray(meta[k]), np.full(n_new, fill, np.int64)])
    if _subcell_enabled():
        with np.load(os.path.join(input_dir, "subcell_strata.npz")) as z:
            if not np.array_equal(np.asarray(z["years"]), np.asarray(meta["years"]).astype(int)):
                raise ValueError("subcell_strata.npz years do not match the model inputs; rebuild it "
                                 "(python -m src.data.preprocess.subcell_strata)")
            meta["strata_u"] = np.asarray(z["u"], dtype=f_type_cpu)
            meta["strata_a"] = np.asarray(z["a"], dtype=f_type_cpu)
        from src.config_utils import load_age_model_config
        _sc = load_age_model_config().get("population_model", {}).get("subcell_strata") or {}
        if _sc.get("capacity_moderator"):
            meta["subcell_capacity_moderator"] = _subcell_moderator(
                input_dir, meta, _sc["capacity_moderator"], meta["strata_u"], meta["strata_a"]
            ).astype(f_type_cpu)
        if _sc.get("survival_moderator"):
            meta["subcell_survival_moderator"] = _subcell_moderator(
                input_dir, meta, _sc["survival_moderator"], meta["strata_u"], meta["strata_a"]
            ).astype(f_type_cpu)
        if float(_sc.get("sigma_s_drift_scale", 0.0)) > 0:
            # linear time shape for the town-survival-tilt drift (as covariate_drift's "linear")
            ws = int(_sc.get("drift_window_start", 1966))
            meta["subcell_drift_g"] = covariate_drift_ramp(
                meta["years"], meta["obs_time_indices"], ws, float(np.asarray(meta["years"])[-1]), ws)[0].astype(f_type_cpu)
    syn = os.environ.get("HOUFIN_SYNTHETIC_OBS")
    if syn:
        # Null-calibration / recovery runs (research/age_model/exp/simulate_observations.py): the
        # counts are replaced by a simulated set drawn at a known parameter point; everything else
        # (cells, years, quality tiers, routes, extra zeros) is the real design.
        with np.load(syn) as z:
            y_syn = np.asarray(z["y"])
        if y_syn.shape != np.asarray(meta["observed_results"]).shape:
            raise ValueError(f"HOUFIN_SYNTHETIC_OBS has {y_syn.shape} counts, the design has "
                             f"{np.asarray(meta['observed_results']).shape}")
        meta["observed_results"] = y_syn.astype(np.asarray(meta["observed_results"]).dtype)
        if verbose:
            print(f"  [synthetic] observed counts replaced from {syn}")
    from src.config_utils import load_age_model_config as _lamc
    stspec = dict((_lamc().get("population_model", {}) or {}).get("st_correction") or {})
    if stspec.get("enabled"):
        # E067 smooth spatiotemporal habitat correction: spatial basis + its path image
        # (src/data/preprocess/st_correction_basis.py) and a time basis held before window_start.
        with np.load(os.path.join(input_dir, "st_correction_basis.npz")) as z:
            meta["st_S"] = np.asarray(z["S"]).T.astype(f_type_cpu)                 # (N_land, nb)
            meta["st_S_disp"] = np.asarray(z["S_disp"]).astype(f_type_cpu)         # (N_land, K, nb)
        ws_st = int(stspec.get("window_start", 1966))
        if stspec.get("time", "linear") == "linear":
            meta["st_g"] = covariate_drift_ramp(meta["years"], meta["obs_time_indices"], ws_st,
                                                float(np.asarray(meta["years"])[-1]), ws_st).astype(f_type_cpu)
        else:
            meta["st_g"] = covariate_drift_basis(meta["years"], meta["obs_time_indices"],
                                                 int(stspec.get("n_time", 2)), ws_st).astype(f_type_cpu)
    cspec = dict((_lamc().get("population_model", {}) or {}).get("path_compounding") or {})
    if cspec.get("enabled"):
        # compounding path survival: segmented path features (src/data/preprocess/segmented_path_features.py), kept
        # float32 on the host (6.6 GB at 54 km), and each kernel's mean distance from the stored kernels.
        import json as _json
        n_seg = int(cspec.get("n_seg", 4))
        with open(os.path.join(input_dir, f"z_disp_seg{n_seg}_meta.json")) as fh:
            sinfo = _json.load(fh)
        if sinfo["labels"] != [str(x) for x in meta.get("juvenile_kernel_labels", sinfo["labels"])]:
            raise ValueError("segmented path features were built with other kernels than this input")
        smem = np.memmap(os.path.join(input_dir, f"z_disp_seg{n_seg}_gathered.dat"), dtype="float32", mode="r",
                         shape=tuple(sinfo["shape"]))
        if tuple(sinfo["shape"][:3]) != (int(meta["time"]), int(meta["N_land"]), int(meta["K"])):
            raise ValueError(f"segmented path features {sinfo['shape']} do not match the input")
        Mz = int(meta["Z_disp_gathered"].shape[-1])
        if columns is not None:
            meta["Z_disp_seg_gathered"] = np.ascontiguousarray(np.asarray(smem)[..., columns])
        else:
            meta["Z_disp_seg_gathered"] = np.ascontiguousarray(np.asarray(smem)[..., :Mz])
        _ks = np.real(np.fft.ifft2(np.asarray(meta["juvenile_fft_kernel_stack"])))
        _Ly, _Lx = _ks.shape[1:]
        _dy = np.minimum(np.arange(_Ly), _Ly - np.arange(_Ly)); _dx = np.minimum(np.arange(_Lx), _Lx - np.arange(_Lx))
        _r = float(cspec.get("cell_km", 54.0)) * np.sqrt(_dy[:, None] ** 2 + _dx[None, :] ** 2)
        meta["kernel_mean_km"] = ((_ks * _r).sum((1, 2)) / _ks.sum((1, 2))).astype(f_type_cpu)
        if verbose:
            print(f"  [path_compounding] segments {meta['Z_disp_seg_gathered'].shape}, kernel mean km "
                  f"{np.round(meta['kernel_mean_km'], 0).tolist()}")
    jspec = dict((_lamc().get("population_model", {}) or {}).get("journey") or {})
    if jspec.get("water_hazard"):
        # J2 journey water hazard: the water share along each cohort's paths, built from the land mask and this
        # input's own stored juvenile kernels (src/data/preprocess/journey_water_share.py). Loaded only when enabled.
        _wf = os.path.join(input_dir, jspec.get("water_share_file", "journey_water_share.npz"))
        with np.load(_wf) as z:
            _W = np.asarray(z["W"]).astype(f_type_cpu)
            _wl = [str(x) for x in z["labels"]]
        _kl = [str(x) for x in meta.get("juvenile_kernel_labels", [])]
        if _W.shape != (int(meta["N_land"]), int(meta["K"])) or (_kl and _wl != _kl):
            raise ValueError(f"{_wf}: W {_W.shape} / labels {_wl} do not match the input "
                             f"({meta['N_land']}, {meta['K']}) / {_kl}")
        if not (np.all(np.isfinite(_W)) and _W.min() >= 0.0 and _W.max() <= 1.0):
            raise ValueError(f"{_wf}: W outside [0, 1]")
        meta["journey_water_W"] = _W
        if verbose:
            print(f"  [journey] water share {_W.shape}: mean {_W.mean():.3f}, share > .1 {np.mean(_W > .1):.3f}")
    nspec = dict((_lamc().get("population_model", {}) or {}).get("nest_likelihood") or {})
    if nspec.get("enabled"):
        meta.update(nest_arrays(nspec, meta, f_type_cpu, verbose=verbose))
    pspec = dict((_lamc().get("population_model", {}) or {}).get("process_noise") or {})
    if pspec.get("enabled"):
        S_pn, active = process_noise_basis(meta, pspec)
        meta["pn_S"], meta["pn_active"] = S_pn.astype(f_type_cpu), active.astype(f_type_cpu)
        if pspec.get("detrend_degree") is not None:
            meta["pn_P"] = process_noise_detrend(meta["years"], active, int(pspec["detrend_degree"])).astype(f_type_cpu)
        meta["pn_groups"], meta["pn_n_groups"] = process_noise_groups(meta, pspec)
        yrs_all = np.asarray(meta["years"]).astype(int)
        hold = (_observation_mask_spec() or {}).get("holdout_years")
        last = min(int(y) for y in hold) - 1 if hold else int(yrs_all[-1])
        meta["pn_penalty_years"] = ((yrs_all >= int(pspec.get("window_start", yrs_all[0]))) & (yrs_all <= last)).astype(f_type_cpu)
        if verbose:
            yrs = np.asarray(meta["years"]).astype(int)[active > 0]
            print(f"  [process_noise] {S_pn.shape[1]} basis functions, sigma {pspec.get('sigma', 0.05)}, "
                  f"active {yrs.min()}-{yrs.max()} ({active.sum():.0f} y)")
    kspec = dict((_lamc().get("population_model", {}) or {}).get("dispersal_kernel") or {})
    if kspec.get("time_knots"):
        from src.model.age_priors import _hat_basis
        meta["band_time_B"] = _hat_basis(np.asarray(meta["years"]), kspec["time_knots"]).astype(f_type_cpu)
    wspec = dict((_lamc().get("population_model", {}) or {}).get("state_warp") or {})
    if wspec.get("enabled"):
        S_w, act_w = process_noise_basis(meta, {"n_freq": wspec.get("n_freq", 3), "window_start": wspec.get("window_start", 1940),
                                                "active_years": wspec.get("active_years")})
        meta["warp_S"], meta["warp_active"] = S_w.astype(f_type_cpu), act_w.astype(f_type_cpu)
        yrs_all = np.asarray(meta["years"]).astype(int)
        hold = (_observation_mask_spec() or {}).get("holdout_years")
        last = min(int(y) for y in hold) - 1 if hold else int(yrs_all[-1])
        meta["skel_penalty_years"] = ((yrs_all >= int(wspec.get("window_start", 1940))) & (yrs_all <= last)).astype(f_type_cpu)
        if verbose:
            print(f"  [state_warp] {S_w.shape[1]} basis functions, sigma {wspec.get('sigma_cells', 2.0)} cells, "
                  f"years {yrs_all[act_w > 0].tolist()}")
    ospec = dict((_lamc().get("population_model", {}) or {}).get("obs_region_effects") or {})
    if ospec.get("enabled"):
        S_o, act_o = process_noise_basis(meta, {"n_freq": ospec.get("n_freq", 3), "window_start": 1966})
        meta["ore_S"], meta["ore_active"] = S_o.astype(f_type_cpu), act_o.astype(f_type_cpu)
        if ospec.get("detrend_degree", 1) is not None:
            meta["ore_P"] = process_noise_detrend(meta["years"], act_o, int(ospec.get("detrend_degree", 1))).astype(f_type_cpu)
        lr, lc = np.asarray(meta["land_rows"]), np.asarray(meta["land_cols"])
        lut = -np.ones((int(meta["Ny"]), int(meta["Nx"])), np.int64)
        lut[lr, lc] = np.arange(len(lr))
        oli = lut[np.asarray(meta["obs_rows"]), np.asarray(meta["obs_cols"])]
        assert (oli >= 0).all(), "obs_region_effects: an observation sits off the land grid"
        meta["obs_land_idx"] = oli.astype(np.int32)
        if verbose:
            print(f"  [obs_region_effects] {S_o.shape[1]} basis functions, sigma {ospec.get('sigma', 0.2)}")
    dspec = covariate_drift_spec()
    if dspec:
        ws = int(dspec.get("window_start", 1966))
        shape = dspec.get("shape", "cosine")
        if shape in ("ramp", "linear"):
            # linear = a straight line in time over the whole observed window (the parsimonious default)
            s0, s1 = ([ws, float(np.asarray(meta["years"])[-1])] if shape == "linear"
                      else dspec.get("ramp", [1979.5, 2005.0]))
            meta["drift_basis"] = covariate_drift_ramp(
                meta["years"], meta["obs_time_indices"], float(s0), float(s1), ws).astype(f_type_cpu)
        else:
            meta["drift_basis"] = covariate_drift_basis(
                meta["years"], meta["obs_time_indices"], int(dspec.get("n_basis", 3)), ws).astype(f_type_cpu)
        if dspec["form"] == "covariate":
            # E057 P2: the drifting covariate. builtup = cell-mean standardised log1p(HISDAC-US built-up
            # intensity) from subcell_strata.npz (CONUS; 0 = the pooled mean, and outside CONUS).
            covs = dspec.get("covariates", [dspec.get("covariate", "builtup")])
            cols = []
            panel = None
            for c in covs:
                if c == "builtup":
                    with np.load(os.path.join(input_dir, "subcell_strata.npz")) as z:
                        if not np.array_equal(np.asarray(z["years"]), np.asarray(meta["years"]).astype(int)):
                            raise ValueError("subcell_strata.npz years do not match the model inputs")
                        cols.append((np.asarray(z["u"]) * np.asarray(z["a"])[None, None, :]).sum(-1))
                else:
                    # raw-covariate panel (src/data/preprocess/drift_covariates.py)
                    if panel is None:
                        with np.load(os.path.join(input_dir, "drift_covariates.npz")) as z:
                            if not np.array_equal(np.asarray(z["years"]), np.asarray(meta["years"]).astype(int)):
                                raise ValueError("drift_covariates.npz years do not match the model inputs")
                            panel = {str(n): np.asarray(z["x"])[..., i] for i, n in enumerate(z["names"])}
                    if c not in panel:
                        raise ValueError(f"covariate_drift: {c!r} not in drift_covariates.npz ({sorted(panel)})")
                    cols.append(panel[c])
            meta["drift_x"] = np.stack(cols, axis=-1).astype(f_type_cpu)          # (time, N_land, C)
    mask_spec = _observation_mask_spec()
    if mask_spec:
        w = np.ones(np.asarray(meta["observed_results"]).shape, dtype=f_type_cpu)
        if "transient_years" in mask_spec:
            w = w * transient_observation_weight(
                meta, int(mask_spec["transient_years"]), int(mask_spec.get("censor_years", 2)))
            if verbose:
                print(f"  [mask] transient +-{mask_spec['transient_years']} y: {int((w == 0).sum())} observations dropped")
        if mask_spec.get("holdout_years") or mask_spec.get("holdout_blocks"):
            held = holdout_observation_mask(meta, mask_spec)
            w = np.where(held, 0.0, w).astype(f_type_cpu)
            meta["obs_holdout"] = held
            if verbose:
                print(f"  [mask] holdout: {int(held.sum())} observations withheld ({held.mean():.1%})")
        meta["obs_weight"] = w
    wspec = _observation_weighting_spec()
    if wspec:
        w = observation_weights(meta, wspec)
        meta["obs_weight"] = meta.get("obs_weight", np.ones_like(w)) * w
        if verbose:
            print(f"  [weights] {wspec}: mean {w.mean():.2f}, max {w.max():.2f}, "
                  f"share of obs upweighted {np.mean(w > 1.0):.1%}")
    aspec = dict((_lamc().get("population_model", {}) or {}).get("state_anchor") or {})
    if aspec.get("enabled"):
        meta.update(state_anchor_arrays(meta, aspec))
        meta["anchor_A"] = meta["anchor_A"].astype(f_type_cpu); meta["anchor_s"] = meta["anchor_s"].astype(f_type_cpu)
        yrs_all = np.asarray(meta["years"]).astype(int)
        hold = (_observation_mask_spec() or {}).get("holdout_years")
        last = min(int(y) for y in hold) - 1 if hold else int(yrs_all[-1])
        meta["skel_penalty_years"] = ((yrs_all >= int(aspec.get("year", 1964))) & (yrs_all <= last)).astype(f_type_cpu)
        if verbose:
            print(f"  [state_anchor] year {aspec.get('year', 1964)}: {int(meta['anchor_s'].sum())} surveyed land cells "
                  f"from training counts of {aspec.get('obs_years', [1966, 1967, 1968])}")
    occupied_center = occupied_habitat_center(meta)
    if occupied_center is not None:
        meta["habitat_center_occupied"] = occupied_center
    # A plain int (the observer-effect vector length is a sample shape), from the final host array.
    if isinstance(meta.get("obs_obsroute_idx"), np.ndarray) and meta["obs_obsroute_idx"].size:
        meta["n_obsroute"] = int(meta["obs_obsroute_idx"].max()) + 1
    # Isolated joint-CBC branch; host arrays go through the standard dtype/device conversion.
    _cbc = dict((_lamc().get("population_model", {}) or {}).get("cbc_likelihood") or {})
    if _cbc.get("enabled"):
        with np.load(_cbc["file"]) as _z:
            meta.update({k: np.asarray(_z[k]) for k in _z.files})
        meta["cbc_n_circles"] = int(meta["cbc_w_local"].shape[0])
        _cbc_years = np.asarray(meta["years"])[meta["cbc_t"]]
        if (_cbc_years < int(_cbc["minimum_state_year"])).any():
            raise ValueError("pre-initialization CBC likelihood would be invalid")
    # population_model.kernel_ramp (ROBUST lever A, default off): a time-varying EASTERN juvenile kernel -- origin
    # cells east of east_col_min send a share s(t) of their dispersing juveniles through an ALTERNATE (longer) kernel
    # stack loaded from another input built with the same pipeline (same bands, labels, grid).
    _kr = dict((_lamc().get("population_model", {}) or {}).get("kernel_ramp") or {})
    if _kr.get("enabled"):
        with open(os.path.join(_kr["alt_input_dir"], "metadata.pkl"), "rb") as _fh:
            _alt = pickle.load(_fh)
        if [str(x) for x in _alt["juvenile_kernel_labels"]] != [str(x) for x in meta["juvenile_kernel_labels"]]:
            raise ValueError("kernel_ramp alternate input has different juvenile kernel labels")
        for _k in ("juvenile_fft_kernel_stack", "juvenile_edge_correction_stack"):
            if np.asarray(_alt[_k]).shape != np.asarray(meta[_k]).shape:
                raise ValueError(f"kernel_ramp alternate {_k} shape differs")
            meta["alt_" + _k] = np.asarray(_alt[_k])
        _col = np.arange(int(meta["Nx"]))[None, :] >= int(_kr["east_col_min"])
        meta["ramp_mask"] = (np.asarray(meta["land_mask"]).astype(bool) & _col).astype(f_type_cpu)
        if verbose:
            print(f"  [kernel_ramp] alternate stack from {_kr['alt_input_dir']}; east mask {int(meta['ramp_mask'].sum())} land cells")
    # population_model.trend_routing (FINAL ROUND 2026-11-06, default off): each region's yearly-mean Z change (West /
    # Plains / East by grid column) reaches the VITAL RATES only through a learned share eta; capacity reads it fully.
    # Here: the region of every land cell and the per-region yearly mean of Z minus its all-years mean (T, R, M).
    _tr = dict((_lamc().get("population_model", {}) or {}).get("trend_routing") or {})
    if _tr.get("enabled"):
        _brk = [int(b) for b in _tr.get("region_col_breaks", [48, 56])]       # cols < 48 West, 48-55 Plains, >= 56 East
        _reg = np.searchsorted(np.asarray(_brk), np.asarray(meta["land_cols"]), side="right").astype(np.int32)
        _Z = np.asarray(meta["Z_gathered"], dtype=np.float64)                   # (T, N_land, M)
        _R = len(_brk) + 1
        _zbar = np.stack([_Z[:, _reg == r].mean(axis=1) for r in range(_R)], axis=1)   # (T, R, M)
        meta["trend_region"] = _reg
        meta["trend_zdev"] = (_zbar - _zbar.mean(axis=0, keepdims=True)).astype(f_type_cpu)
        if _tr.get("smooth_km"):
            # SMOOTH routing (2026-11-07; user: the column bands leave a seam at ~100W): each cell's "regional" anomaly is
            # a Gaussian-weighted mean of Z over cells within ~smooth_km (grid distance x 54 km), minus its all-years mean;
            # trend_region becomes the cell index (one anomaly per cell), and eta varies smoothly along the grid columns.
            _rc = np.stack([np.asarray(meta["land_rows"]), np.asarray(meta["land_cols"])], 1).astype(np.float64) * 54.0
            _d2 = ((_rc[:, None, :] - _rc[None, :, :]) ** 2).sum(-1)
            _Wk = np.exp(-_d2 / (2.0 * float(_tr["smooth_km"]) ** 2)); _Wk /= _Wk.sum(1, keepdims=True)
            _zs = np.einsum('ij,tjm->tim', _Wk, _Z)
            meta["trend_zdev"] = (_zs - _zs.mean(axis=0, keepdims=True)).astype(f_type_cpu)
            meta["trend_region"] = np.arange(_Z.shape[1], dtype=np.int32)
            meta["trend_col"] = np.asarray(meta["land_cols"], dtype=f_type_cpu)
            if str(_tr.get("eta_coord", "column")) == "native_distance":
                # eta varies with distance (km) to the NATIVE seed range (initpop_latent > 0 west of col 48, as the halo
                # absences use) instead of grid column: follows the real range edge (user 2026-11-07, "the shape").
                _ip = np.asarray(meta["initpop_latent"]); _lr, _lc = np.asarray(meta["land_rows"]), np.asarray(meta["land_cols"])
                _ip = _ip[_lr, _lc] if _ip.ndim == 2 else _ip
                _nat = (_ip > 0) & (_lc < 48)
                _dn = np.sqrt(((_rc[:, None, :] - _rc[None, _nat, :]) ** 2).sum(-1)).min(1)
                meta["trend_col"] = _dn.astype(f_type_cpu)          # reused slot: the eta coordinate (km here)
                if verbose:
                    print(f"  [trend_routing] eta coordinate = distance to native seed ({int(_nat.sum())} cells); "
                          f"q10/50/90 km {np.percentile(_dn, [10, 50, 90]).round(0).tolist()}")
        if verbose:
            print(f"  [trend_routing] regions {np.bincount(_reg).tolist()} land cells; breaks {_brk}; smooth_km {_tr.get('smooth_km')}")
    _um = dict((_lamc().get("population_model", {}) or {}).get("urban_mating") or {})
    if _um.get("enabled"):
        with np.load(_um["file"]) as _z:
            if [int(y) for y in _z["years"]] != [int(y) for y in meta["years"]]:
                raise ValueError("urban_mating strata years do not match the model inputs")
            meta["um_u"] = np.asarray(_z["u"], dtype=f_type_cpu); meta["um_a"] = np.asarray(_z["a"], dtype=f_type_cpu)
        if meta["um_u"].shape[1] != len(meta["land_rows"]):
            raise ValueError("urban_mating strata land cells do not match the model inputs")
    residency = os.environ.get("HOUFIN_MODEL_INPUT_RESIDENCY", "device").lower()
    if residency not in {"device", "host"}:
        raise ValueError("HOUFIN_MODEL_INPUT_RESIDENCY must be 'device' or 'host'")

    if verbose:
        print(f"Iterating through metadata and casting to {precision} on device: {target_device}...")

    # HOUFIN_Z_FLOAT32=1: keep the large covariate inputs (Z, Z_disp) in float32 even in a float64
    # run. They are float32 on disk, so float64 storage adds memory and no information; the model's
    # dot products promote per-year slices to float64. Lets rank-48/64 float64 refines fit (E048).
    z32 = os.environ.get("HOUFIN_Z_FLOAT32") == "1"

    # HOUFIN_COMPLEX128=1 (opt-in; ROBUST diagnostic 2026-11-06): in a float64 run, promote the complex64 FFT kernel
    # stacks to complex128. Stored complex64, they make every dispersal convolution single-precision, which puts O(0.1)
    # evaluation noise into the objective (finite differences jump by ~0.16 under 1e-6 steps at short-kernel optima).
    c128 = os.environ.get("HOUFIN_COMPLEX128") == "1" and precision == "float64"

    def _to_device(value, key=None):
        if c128 and np.issubdtype(value.dtype, np.complexfloating):
            return jnp.array(np.asarray(value, dtype=np.complex128))
        if np.issubdtype(value.dtype, np.floating):
            if z32 and key in LARGE_INPUT_KEYS:
                # Cast on host before transfer: transferring a float64 staging
                # array first can OOM even though the final float32 array fits.
                return jnp.array(np.asarray(value, dtype=np.float32))
            return jnp.array(value).astype(f_type_target)
        elif np.issubdtype(value.dtype, np.integer):
            return jnp.array(value).astype(i_type_target)
        return jnp.array(value)

    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            if key in LARGE_INPUT_KEYS and residency == "host":
                meta[key] = value.astype(f_type_cpu)
                if verbose:
                    print(f"  [HOST, experimental] {key}: {meta[key].nbytes / 1e9:.2f} GB")
            else:
                if target_device is not None:
                    meta[key] = jax.device_put(_to_device(value, key), target_device)
                else:
                    meta[key] = _to_device(value, key)
                if verbose:
                    print(f"  [DEVICE] {key}: {meta[key].nbytes / 1e6:.1f} MB")

    if verbose:
        device_bytes = sum(
            x.nbytes for x in meta.values()
            if hasattr(x, "device") and not isinstance(x, np.ndarray)
        )
        host_bytes = sum(x.nbytes for x in meta.values() if isinstance(x, np.ndarray))
        print(f"--- Explicit input residency: device={device_bytes / 1e9:.2f} GB, "
              f"host={host_bytes / 1e9:.2f} GB (mode={residency}) ---")

    return meta


def load_data_to_gpu(input_dir, precision="float32"):
    """Backward-compatible alias used by the visualization/analysis scripts."""
    return load_data(input_dir, target_device=None, precision=precision, verbose=True)
