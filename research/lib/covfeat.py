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
