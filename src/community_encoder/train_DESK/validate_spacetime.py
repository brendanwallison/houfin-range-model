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


# ----------------------------- orchestration -----------------------------

def _load_model(config):
    import torch
    from .model_arch import MultiStreamAutoencoder
    dm = np.load(os.path.join(config["paths"]["desk_output_dir"], "desk_meta.npz"), allow_pickle=True)
    schema = json.loads(str(dm["schema"]))
    model = MultiStreamAutoencoder([int(d) for d in dm["stream_dims"]], int(dm["latent_dim"]))
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
    for y in np.unique(years):
        sel = np.where(years == y)[0]
        cov = cio.load_state_stack(int(y), states_dir, schema)[rows[sel], cols[sel]]
        finite = ~np.isnan(cov).any(1)
        if finite.any():
            cn = cio.apply_norm(cov[finite], mu, sd)
            streams = cio.split_streams(torch.tensor(cn, dtype=torch.float32), schema)
            with torch.no_grad():
                zz, _ = model(*streams)
            Z[sel[finite]] = zz.numpy()
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

    out = os.path.join(config["paths"]["desk_output_dir"], "validate_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("[validate] eBird-only vs BBS structure (higher CKA/Mantel/Pearson = extrapolation holds):")
    for k, v in report.items():
        if "cka" in v:
            print(f"  {v['period']:<16} n={v['n']:<7} pearson={v['pearson']:+.3f} "
                  f"cka={v['cka']:.3f} mantel={v['mantel']:+.3f} mse={v['mse']:.4f}")
    print(f"[validate] report -> {out}")
    return report


if __name__ == "__main__":
    run_validate()
