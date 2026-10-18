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
LARGE_INPUT_KEYS = {"Z_gathered", "Z_disp_gathered"}


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


def load_data(input_dir, target_device=None, precision="float32", verbose=True):
    """Load model inputs, casting to ``precision`` and placing device arrays.

    If ``target_device`` is given, arrays are materialized on it unless explicit
    experimental host residency is requested.
    """
    meta_path = os.path.join(input_dir, "metadata.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

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
    if _subcell_enabled():
        with np.load(os.path.join(input_dir, "subcell_strata.npz")) as z:
            if not np.array_equal(np.asarray(z["years"]), np.asarray(meta["years"]).astype(int)):
                raise ValueError("subcell_strata.npz years do not match the model inputs; rebuild it "
                                 "(python -m src.data.preprocess.subcell_strata)")
            meta["strata_u"] = np.asarray(z["u"], dtype=f_type_cpu)
            meta["strata_a"] = np.asarray(z["a"], dtype=f_type_cpu)
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
    dspec = covariate_drift_spec()
    if dspec:
        meta["drift_basis"] = covariate_drift_basis(
            meta["years"], meta["obs_time_indices"], int(dspec.get("n_basis", 3)),
            int(dspec.get("window_start", 1966))).astype(f_type_cpu)
    mask_spec = _observation_mask_spec()
    if mask_spec:
        meta["obs_weight"] = transient_observation_weight(
            meta, int(mask_spec["transient_years"]), int(mask_spec.get("censor_years", 2)))
        if verbose:
            print(f"  [mask] transient +-{mask_spec['transient_years']} y: "
                  f"{int((meta['obs_weight'] == 0).sum())} observations dropped")
    wspec = _observation_weighting_spec()
    if wspec:
        w = observation_weights(meta, wspec)
        meta["obs_weight"] = meta.get("obs_weight", np.ones_like(w)) * w
        if verbose:
            print(f"  [weights] {wspec}: mean {w.mean():.2f}, max {w.max():.2f}, "
                  f"share of obs upweighted {np.mean(w > 1.0):.1%}")
    occupied_center = occupied_habitat_center(meta)
    if occupied_center is not None:
        meta["habitat_center_occupied"] = occupied_center
    residency = os.environ.get("HOUFIN_MODEL_INPUT_RESIDENCY", "device").lower()
    if residency not in {"device", "host"}:
        raise ValueError("HOUFIN_MODEL_INPUT_RESIDENCY must be 'device' or 'host'")

    if verbose:
        print(f"Iterating through metadata and casting to {precision} on device: {target_device}...")

    # HOUFIN_Z_FLOAT32=1: keep the large covariate inputs (Z, Z_disp) in float32 even in a float64
    # run. They are float32 on disk, so float64 storage adds memory and no information; the model's
    # dot products promote per-year slices to float64. Lets rank-48/64 float64 refines fit (E048).
    z32 = os.environ.get("HOUFIN_Z_FLOAT32") == "1"

    def _to_device(value, key=None):
        if np.issubdtype(value.dtype, np.floating):
            if z32 and key in LARGE_INPUT_KEYS:
                return jnp.array(value).astype(jnp.float32)
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
