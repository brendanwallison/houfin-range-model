"""Covariate features for the rung-1 surrogates: standardized principal components at every key, cell means over
trained years, and the pooled / between / within-cell linear maps to ESK (E016, E017).

Pure functions over arrays a research cache already holds (F_cov.npy, keys.npy, split.npz, esk_annual.npy).
"""
import numpy as np


def cov_pcs(F, fit_rows, n_pcs=64, max_fit_rows=40000):
    """Standardize the covariate channels and project onto their top ``n_pcs`` principal components, both fitted on
    ``fit_rows`` (training rows). 302 channels are collinear (rank ~255); a few dozen PCs carry them."""
    F = np.asarray(F, "float64")
    mu, sd = F[fit_rows].mean(0), F[fit_rows].std(0) + 1e-9
    Fs = (F - mu) / sd
    sub = np.where(fit_rows)[0]
    sub = sub[:: max(1, len(sub) // max_fit_rows)]
    _, _, Vt = np.linalg.svd(Fs[sub] - Fs[sub].mean(0), full_matrices=False)
    return Fs @ Vt[:n_pcs].T


def multi_ema_rows(config, keys, half_lives):
    """The covariates DESK sees (its states, channel transforms and training mu/sd, from the run in ``config``) at each
    key, after a causal EMA at each of ``half_lives`` years (0 = none beyond the states' light input-side EMA).
    Returns ``{hl: (N, C) float32}``. The same loader as validate_gp_species.covariates_for_keys, which applies only
    DESK's own learned half-life (E027: which timescales of covariate history carry temporal signal?)."""
    import json
    import os
    from src.community_encoder.train_DESK import covariate_io as cio
    from src.community_encoder.train_DESK.desk_training import apply_output_ema
    from src.community_encoder.train_DESK.validate_gp_species import rows_from_stack
    run_dir = config["paths"]["desk_output_dir"]
    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    schema = json.loads(str(dm["schema"]))
    states_dir = os.path.join(config["paths"]["hist_dir"], "yearly_states")
    cio.assert_schema_compatible(schema, cio.load_schema(states_dir), context="covfeat")
    mu, sd = dm["mu"].astype("float32"), dm["sd"].astype("float32")
    warm = int(dm["ema_warmup_start"]) if "ema_warmup_start" in dm.files else 1940
    keys = np.asarray(keys)
    cells = np.unique(keys[:, :2], axis=0)
    years = list(range(min(warm, int(keys[:, 2].min())), int(keys[:, 2].max()) + 1))
    stack = valid = None
    for t, y in enumerate(years):
        try:
            covn, mask = cio.norm_grid(cio.load_state_stack(y, states_dir, schema), mu, sd)
        except FileNotFoundError:
            continue
        if stack is None:
            stack = np.full((len(years), len(cells), covn.shape[-1]), np.nan, "float32")
            valid = np.zeros((len(years), len(cells)), bool)
        stack[t] = covn[cells[:, 0], cells[:, 1]]
        valid[t] = mask[cells[:, 0], cells[:, 1]]
        stack[t][~valid[t]] = np.nan
    if stack is None:
        raise FileNotFoundError(f"no yearly states under {states_dir}")
    out = {}
    for hl in half_lives:
        s = stack if float(hl) <= 0 else apply_output_ema(stack, float(hl), valid=valid)
        out[float(hl)] = rows_from_stack(s, years, cells, keys)
    return out


def rff(X, n_features, lengthscale, seed=0):
    """Random Fourier features of an RBF kernel on the rows of ``X`` (a nonlinear covariate surrogate)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(X.shape[1], n_features)) / float(lengthscale)
    b = rng.uniform(0, 2 * np.pi, n_features)
    return np.sqrt(2.0 / n_features) * np.cos(np.asarray(X, "float64") @ W + b)


def cell_means(keys, X, rows_mask):
    """Each row's cell mean of ``X`` over that cell's rows in ``rows_mask`` (e.g. trained years)."""
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    uc, inv = np.unique(cid, return_inverse=True)
    w = np.asarray(rows_mask, float)
    sums = np.zeros((len(uc), X.shape[1]))
    np.add.at(sums, inv, X * w[:, None])
    cnt = np.bincount(inv, weights=w, minlength=len(uc))
    return (sums / np.maximum(cnt, 1)[:, None])[inv]


def fit_maps(P, Pbar, Z, train_rows, blr):
    """Linear covariate -> ESK maps fitted on ``train_rows``: pooled (one block) and Mundlak (between = cell-mean
    covariates, within = deviations). Returns coefficient matrices (k, r): pooled, between, within."""
    k = P.shape[1]
    m_pool = blr.fit(P[train_rows], Z[train_rows], [(0, k)])
    Xm = np.hstack([Pbar, P - Pbar])
    m_mund = blr.fit(Xm[train_rows], Z[train_rows], [(0, k), (k, 2 * k)])
    return m_pool["coef"].T, m_mund["coef"][:, :k].T, m_mund["coef"][:, k:].T
