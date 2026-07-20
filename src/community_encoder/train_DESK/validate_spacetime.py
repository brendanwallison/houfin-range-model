"""Validate: does the eBird-only DESK already predict BBS spatiotemporal structure?

The headline test of ``bbs_mode='validate'``. The eBird-only-trained DESK gives a
predicted latent ``z(s,t)`` from that point's own-year covariates. At **held-out
historical** ``(cell, year)`` points (which the model never trained on) we ask
whether its predicted **similarities** reproduce the BBS-observed community
similarities — comparing at the **kernel level** (``⟨z_i,z_j⟩`` vs
``Ruzicka(x_i,x_j)``), never raw coordinates, because Z is basis/rotation-arbitrary.
Both live on the same eBird-unit Ruzicka scale (``x = E·anomaly`` is in eBird units,
and ``true_kernel_loss`` calibrated ``⟨z,z⟩`` to eBird Ruzicka), so the comparison is
fair. Reported per period with MSE + basis-invariant CKA/Mantel.

Strong agreement (esp. degrading gracefully, not randomly, back in time) ⇒ the
spatial→spatiotemporal extrapolation holds and no BBS-in-training is needed; weak
agreement flags where ``enrich`` is warranted.
"""
import json
import os

import numpy as np


# ----------------------------- pure metrics -----------------------------

def ruzicka_similarity_matrix(X):
    """Pairwise Ruzicka similarity ``Σmin/Σmax`` over rows of ``X (n, d)`` → ``(n, n)``."""
    X = np.asarray(X, dtype="float64")
    n = X.shape[0]
    S = np.empty((n, n))
    for i in range(n):
        mn = np.minimum(X[i], X).sum(1)
        mx = np.maximum(X[i], X).sum(1)
        S[i] = mn / np.where(mx > 0, mx, 1.0)
    return S


def _center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(K, L):
    """Centered-kernel-alignment between two Gram/similarity matrices (rotation-invariant)."""
    Kc, Lc = _center(np.asarray(K, float)), _center(np.asarray(L, float))
    num = (Kc * Lc).sum()
    den = np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return float(num / den) if den > 0 else 0.0


def mantel_r(A, B):
    """Pearson correlation of the off-diagonal (upper-triangle) entries of two matrices."""
    iu = np.triu_indices_from(np.asarray(A), k=1)
    a, b = np.asarray(A)[iu], np.asarray(B)[iu]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pair_sims(Z, X, pairs):
    """Predicted ``⟨z_i,z_j⟩`` and observed ``Ruzicka(x_i,x_j)`` for index pairs ``(2, m)``."""
    i, j = pairs
    sim_pred = (Z[i] * Z[j]).sum(1)
    xi, xj = X[i], X[j]
    mn = np.minimum(xi, xj).sum(1)
    mx = np.maximum(xi, xj).sum(1)
    sim_obs = mn / np.where(mx > 0, mx, 1.0)
    return sim_pred, sim_obs


# --------------------- spatiotemporal (temporal-nuance) metrics ---------------------
# All basis-invariant: they compare SIMILARITIES (<z,z> vs Ruzicka) or GEOGRAPHIC
# quantities, never the rotation-arbitrary embedding coordinates.

def ruzicka_rect(A, B):
    """Pairwise Ruzicka Σmin/Σmax between rows of ``A (n,D)`` and ``B (m,D)`` → ``(n,m)``.

    Σmin=(sa+sb−L1)/2, Σmax=(sa+sb+L1)/2 ⇒ Ruzicka=(sa+sb−L1)/(sa+sb+L1). Uses torch
    (GPU) for the L1 block if available, else scipy — same result either way.
    """
    A = np.asarray(A, "float64"); B = np.asarray(B, "float64")
    sa, sb = A.sum(1), B.sum(1)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        L1 = torch.cdist(torch.tensor(A, device=dev), torch.tensor(B, device=dev),
                         p=1).cpu().numpy()
    except Exception:
        from scipy.spatial.distance import cdist
        L1 = cdist(A, B, "cityblock")
    denom = sa[:, None] + sb[None, :]
    return np.where((denom + L1) > 0, (denom - L1) / (denom + L1), 1.0)


def temporal_turnover_agreement(Z, X, pidx, recent_year, early_year=1995, window=5):
    """Per-site community turnover over a FIXED window (``early_year`` → recent), pred vs obs.

    turnover = 1 − self-similarity over time (``⟨z(s,t0),z(s,rec)⟩`` for pred,
    ``Ruzicka(x(s,t0),x(s,rec))`` for obs) — basis-invariant. Spearman of the two
    per-site turnover fields answers "do the models agree on WHERE communities changed
    most" (magnitude, direction-agnostic). Returns the fields + rho.

    Each site is anchored to its historical point nearest ``early_year`` (within
    ``±window`` yr), matched to its recent point — a COMMON window across sites, so
    turnover magnitudes are comparable (per-cell *earliest* would confound magnitude
    with window length, and tie coverage to the sparse early-BBS footprint).
    """
    rows, cols, yrs = pidx[:, 0], pidx[:, 1], pidx[:, 2]
    rec = yrs == recent_year
    rec_ix = {(int(r), int(c)): int(i) for r, c, i in
              zip(rows[rec], cols[rec], np.where(rec)[0])}
    best = {}                                    # (r,c) -> (|year-early|, idx): nearest to early_year
    for i in np.where(~rec)[0]:
        key = (int(rows[i]), int(cols[i]))
        dy = abs(int(yrs[i]) - int(early_year))
        if dy <= window and key in rec_ix and (key not in best or dy < best[key][0]):
            best[key] = (dy, int(i))
    keys = list(best)
    if len(keys) < 4:
        return {"n_sites": len(keys), "note": "too few paired sites"}
    hi = np.array([best[k][1] for k in keys])
    ri = np.array([rec_ix[k] for k in keys])
    sim_pred = (Z[hi] * Z[ri]).sum(1)
    mn = np.minimum(X[hi], X[ri]).sum(1); mx = np.maximum(X[hi], X[ri]).sum(1)
    sim_obs = np.where(mx > 0, mn / mx, 1.0)
    tp, to = 1.0 - sim_pred, 1.0 - sim_obs
    from scipy.stats import spearmanr
    rho = float(spearmanr(tp, to).correlation)
    return {"n_sites": len(keys), "spearman_turnover": rho,
            "rows": rows[hi], "cols": cols[hi], "hist_year": yrs[hi],
            "turnover_pred": tp.astype("float32"), "turnover_obs": to.astype("float32")}


def analog_displacement(Z, X, pidx, xy, recent_year, rng, n_hist=1500, n_present=4000, topk=15):
    """Direction each historical site's community "points" toward among present cells.

    For a historical point, rank present-day cells by similarity (``⟨z,z⟩`` / Ruzicka),
    take the top-``k`` analog cells, and their mean location = the analog centroid.
    Displacement Δ = centroid − site is a GEOGRAPHIC vector (Albers x=E–W, y=N–S), so
    Δ_pred and Δ_obs ARE directly comparable across models (no rotation problem). Top-k
    is rank-based → scale-invariant between the two similarity types. Tests the
    climate-analog hypothesis: do both models send past sites toward the same present
    communities / same compass direction (poleward warming, E–W precip)?
    """
    yrs = pidx[:, 2]
    pres, hist = np.where(yrs == recent_year)[0], np.where(yrs != recent_year)[0]
    if pres.size < topk + 1 or hist.size < 4:
        return {"note": "insufficient points"}
    if pres.size > n_present:
        pres = rng.choice(pres, n_present, replace=False)
    if hist.size > n_hist:
        hist = rng.choice(hist, n_hist, replace=False)
    xyc = xy[pres]
    P_pred = Z[hist] @ Z[pres].T                 # (nh, np) dot similarities
    P_obs = ruzicka_rect(X[hist], X[pres])       # (nh, np)

    def _centroid(P):
        idx = np.argpartition(-P, kth=topk - 1, axis=1)[:, :topk]   # top-k present/site
        return xyc[idx].mean(axis=1)                                # (nh, 2)

    d_pred = _centroid(P_pred) - xy[hist]
    d_obs = _centroid(P_obs) - xy[hist]
    nrm = np.linalg.norm(d_pred, axis=1) * np.linalg.norm(d_obs, axis=1) + 1e-12
    cos = (d_pred * d_obs).sum(1) / nrm
    from scipy.stats import pearsonr
    pc = np.array([np.corrcoef(P_pred[i], P_obs[i])[0, 1] for i in range(hist.size)])
    return {"n_hist": int(hist.size), "n_present": int(pres.size), "topk": topk,
            "mean_cos_displacement": float(np.nanmean(cos)),
            "corr_disp_EW": float(pearsonr(d_pred[:, 0], d_obs[:, 0])[0]),
            "corr_disp_NS": float(pearsonr(d_pred[:, 1], d_obs[:, 1])[0]),
            "mean_profile_corr": float(np.nanmean(pc)),
            "d_pred": d_pred.astype("float32"), "d_obs": d_obs.astype("float32"),
            "xy_hist": xy[hist].astype("float32"), "hist_year": yrs[hist]}


def cell_xy(rows, cols, ref_raster):
    """Cell-center (x, y) in the ref-grid CRS (Albers: x=easting/E–W, y=northing/N–S)."""
    import rasterio
    with rasterio.open(ref_raster) as src:
        t = src.transform
    r = np.asarray(rows) + 0.5; c = np.asarray(cols) + 0.5
    return np.stack([t.c + c * t.a + r * t.b, t.f + c * t.d + r * t.e], axis=1)


# ----------------------------- orchestration -----------------------------

def _load_model(config):
    import torch
    from .model_arch import MultiStreamAutoencoder
    dm = np.load(os.path.join(config["paths"]["desk_output_dir"], "desk_meta.npz"), allow_pickle=True)
    schema = json.loads(str(dm["schema"]))
    spatial_kernel = int(dm["spatial_kernel"]) if "spatial_kernel" in dm else 0
    model = MultiStreamAutoencoder([int(d) for d in dm["stream_dims"]], int(dm["latent_dim"]),
                                   spatial_kernel)
    model.load_state_dict(torch.load(
        os.path.join(config["paths"]["desk_output_dir"], "env_model_semisup.pth"),
        map_location="cpu"))
    model.eval()
    return model, dm["mu"].astype("float32"), dm["sd"].astype("float32"), schema, int(dm["latent_dim"])


def encode_points(config, point_index):
    """Encode each ``(row,col,year)`` point with the eBird-only DESK → ``(N, latent)``.

    Returns ``(Z, ok)`` where ``ok`` masks points whose covariates were finite.
    """
    import torch
    from . import covariate_io as cio
    model, mu, sd, schema, latent = _load_model(config)
    states_dir = os.path.join(config["paths"]["hist_dir"], "yearly_states")
    rows, cols, years = point_index[:, 0], point_index[:, 1], point_index[:, 2]
    Z = np.full((len(point_index), latent), np.nan, dtype="float32")
    # Grid-native: encode each year's WHOLE grid (so the spatial residual conv sees
    # neighbours -- the same function the cube applies) and gather the points from it.
    for y in np.unique(years):
        sel = np.where(years == y)[0]
        covn, valid = cio.norm_grid(cio.load_state_stack(int(y), states_dir, schema), mu, sd)
        xg = torch.tensor(covn[None], dtype=torch.float32)
        mg = torch.tensor(valid[None])
        with torch.no_grad():
            zz, _ = model(xg, mg)                        # (1, H, W, L)
        zc = zz[0].numpy()
        for k in sel:
            if valid[rows[k], cols[k]]:
                Z[k] = zc[rows[k], cols[k]]
    return Z, ~np.isnan(Z).any(1)


def run_validate(config=None, n_pairs=20000, cka_sample=800, seed=0):
    """Compare eBird-only DESK predictions to BBS structure per period; write a report."""
    from .config_utils import load_config
    config = load_config(config) if not isinstance(config, dict) else config
    bc = config["bbs"]
    rng = np.random.default_rng(seed)

    zt = bc["z_dir"]                              # spacetime point set from build_amplitude_points
    X = np.load(os.path.join(zt, "X_points.npy"))
    pidx = np.load(os.path.join(zt, "point_index.npy"))
    meta = json.load(open(os.path.join(zt, "points_meta.json")))
    recent_year = int(meta["recent_year"])

    Z, ok = encode_points(config, pidx)
    X, pidx, Z = X[ok], pidx[ok], Z[ok]
    years = pidx[:, 2]

    def _bucket_report(mask, label):
        idx = np.where(mask)[0]
        if idx.size < 4:
            return {"period": label, "n": int(idx.size), "note": "too few points"}
        pr = np.stack([rng.choice(idx, n_pairs), rng.choice(idx, n_pairs)])
        sp, so = pair_sims(Z, X, pr)
        r = float(np.corrcoef(sp, so)[0, 1]) if sp.std() > 0 and so.std() > 0 else 0.0
        samp = rng.choice(idx, min(cka_sample, idx.size), replace=False)
        Kz = Z[samp] @ Z[samp].T
        Lx = ruzicka_similarity_matrix(X[samp])
        return {"period": label, "n": int(idx.size),
                "mse": float(np.mean((sp - so) ** 2)), "pearson": r,
                "cka": linear_cka(Kz, Lx), "mantel": mantel_r(Kz, Lx)}

    report = {"recent_control": _bucket_report(years == recent_year, f"recent({recent_year})")}
    hist_years = sorted(set(int(y) for y in years if y != recent_year))
    if hist_years:
        lo, hi = min(hist_years), max(hist_years)
        for d0 in range(lo - lo % 10, hi + 1, 10):
            report[f"{d0}s"] = _bucket_report((years >= d0) & (years < d0 + 10)
                                              & (years != recent_year), f"{d0}s")
    report["all_historical"] = _bucket_report(years != recent_year, "all_historical")

    # --- temporal-nuance metrics (turnover magnitude + spatiotemporal analog direction) ---
    from src.config_utils import load_data_config
    ref_raster = load_data_config()["grid"]["ref_raster"]
    xy = cell_xy(pidx[:, 0], pidx[:, 1], ref_raster)
    turn = temporal_turnover_agreement(Z, X, pidx, recent_year,
                                       early_year=int(bc.get("turnover_early_year", 1995)),
                                       window=int(bc.get("turnover_window", 5)))
    analog = analog_displacement(Z, X, pidx, xy, recent_year, rng)
    report["temporal_turnover"] = {k: v for k, v in turn.items()
                                   if k in ("n_sites", "spearman_turnover", "note")}
    report["analog"] = {k: v for k, v in analog.items()
                        if k in ("n_hist", "n_present", "topk", "mean_cos_displacement",
                                 "corr_disp_EW", "corr_disp_NS", "mean_profile_corr", "note")}

    out_dir = config["paths"]["desk_output_dir"]
    out = os.path.join(out_dir, "validate_report.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    # Bundle the per-site/per-point arrays for visualization (turnover maps + analog arrows).
    viz = os.path.join(out_dir, "validate_spacetime.npz")
    np.savez_compressed(
        viz,
        turn_rows=turn.get("rows", np.array([])), turn_cols=turn.get("cols", np.array([])),
        turnover_pred=turn.get("turnover_pred", np.array([])),
        turnover_obs=turn.get("turnover_obs", np.array([])),
        d_pred=analog.get("d_pred", np.zeros((0, 2))), d_obs=analog.get("d_obs", np.zeros((0, 2))),
        xy_hist=analog.get("xy_hist", np.zeros((0, 2))),
        analog_hist_year=analog.get("hist_year", np.array([])),
        ref_raster=np.array(ref_raster))

    print("[validate] eBird-only vs BBS structure (higher CKA/Mantel/Pearson = extrapolation holds):")
    for k, v in report.items():
        if "cka" in v:
            print(f"  {v['period']:<16} n={v['n']:<7} pearson={v['pearson']:+.3f} "
                  f"cka={v['cka']:.3f} mantel={v['mantel']:+.3f} mse={v['mse']:.4f}")
    if "spearman_turnover" in report["temporal_turnover"]:
        print(f"[validate] temporal turnover agreement (Spearman, {turn['n_sites']} sites): "
              f"{report['temporal_turnover']['spearman_turnover']:+.3f}")
    if "mean_cos_displacement" in report["analog"]:
        a = report["analog"]
        print(f"[validate] analog displacement ({a['n_hist']} hist pts): "
              f"mean cos={a['mean_cos_displacement']:+.3f} | dir corr E-W={a['corr_disp_EW']:+.3f} "
              f"N-S={a['corr_disp_NS']:+.3f} | profile corr={a['mean_profile_corr']:+.3f}")
    print(f"[validate] report -> {out} ; viz arrays -> {viz}")
    return report


if __name__ == "__main__":
    run_validate()
