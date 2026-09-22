"""Boosted-regression-tree baselines: the naive correlative SDM.

This is the standard-practice benchmark -- no detection model, fitted to
occurrence. Two heads on one design matrix:

  occurrence -> P(occurrence), thresholded to the comparable designation
  count      -> expected route count under a Poisson objective, with the NB2
                concentration estimated afterwards so predictions score under
                the SAME likelihood the dynamic model uses (age_priors.py:743)

HYPERPARAMETERS ARE NOT DEFAULTS. The Elith, Leathwick & Hastie (2008) BRT
recipe -- slow learning rate, shallow trees, heavy bagging, n_trees chosen by
CV -- is what the SDM literature means by "a boosted regression tree", and
"you didn't tune the baseline" is the easiest objection to a benchmark like
this one. LightGBM rather than sklearn's HistGradientBoosting because it carries
the Poisson/Tweedie objectives natively.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from src.community_encoder.train_DESK.augment import _shift2d

# Elith, Leathwick & Hastie (2008), "A working guide to boosted regression
# trees": lr 0.001-0.01, tree complexity 3-5, bag fraction 0.5, n_trees by CV.
ELITH_BRT = {
    "learning_rate": 0.005,
    "max_depth": 5,
    "num_leaves": 31,
    "subsample": 0.5,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "n_estimators": 5000,          # upper bound; early stopping picks the count
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 200


def spatial_block_folds(valid, block_cells=6, n_folds=5, buffer_cells=1, seed=0):
    """K-fold spatial block CV over the grid, with a buffer ring per fold.

    Mirrors the geometry of augment.blocked_holdout (whole block_cells x
    block_cells tiles, Chebyshev-dilated buffer) so the baseline's CV matches the
    encoder's, but assigns every block to one of ``n_folds`` instead of making a
    single split. Roberts et al. (2017) is the citation for why random-row CV
    would be optimistic here.

    The buffer's rationale differs from the encoder's: a BRT has no receptive
    field, so the ring is guarding against residual spatial autocorrelation
    between neighbouring train and val cells, not kernel leakage.

    Yields ``(train_grid, val_grid)`` boolean arrays shaped like ``valid``.
    """
    valid = np.asarray(valid, dtype=bool)
    H, W = valid.shape
    b = max(1, int(block_cells))
    rng = np.random.default_rng(int(seed))
    nby, nbx = (H + b - 1) // b, (W + b - 1) // b
    assign = rng.integers(0, n_folds, size=(nby, nbx))
    full = np.repeat(np.repeat(assign, b, axis=0), b, axis=1)[:H, :W]

    r = max(0, int(buffer_cells))
    for k in range(n_folds):
        val = (full == k) & valid
        if r:
            near = np.zeros_like(val)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    near |= _shift2d(val, dy, dx)
            buf = near & valid & (~val)
        else:
            buf = np.zeros_like(val)
        yield (valid & ~val & ~buf), val


def rows_in(mask_grid, rows, cols):
    """Boolean selector over observation rows whose cell is set in ``mask_grid``."""
    return mask_grid[np.asarray(rows), np.asarray(cols)]


def _lgb():
    import lightgbm as lgb
    return lgb


def fit_occurrence(X_tr, y_tr, X_va, y_va, params=None, seed=0):
    """Binary BRT -> P(occurrence). Returns (model, best_iteration)."""
    lgb = _lgb()
    p = dict(ELITH_BRT); p.update(params or {})
    m = lgb.LGBMClassifier(objective="binary", random_state=seed, **p)
    m.fit(X_tr, (np.asarray(y_tr) > 0).astype(int),
          eval_set=[(X_va, (np.asarray(y_va) > 0).astype(int))],
          eval_metric="auc",
          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return m, int(m.best_iteration_ or p["n_estimators"])


def fit_count(X_tr, y_tr, X_va, y_va, params=None, seed=0):
    """Poisson-objective BRT -> expected count. Returns (model, best_iteration).

    The Poisson objective fits the MEAN function only; overdispersion is handled
    afterwards by nb2_concentration_mle, so scoring can use NB2 and match the
    dynamic model's likelihood rather than a Poisson that would be wildly
    overconfident on these counts.
    """
    lgb = _lgb()
    p = dict(ELITH_BRT); p.update(params or {})
    m = lgb.LGBMRegressor(objective="poisson", random_state=seed, **p)
    m.fit(X_tr, np.asarray(y_tr, dtype=float),
          eval_set=[(X_va, np.asarray(y_va, dtype=float))],
          eval_metric="poisson",
          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return m, int(m.best_iteration_ or p["n_estimators"])


def nb2_loglik(y, mu, phi):
    """NegativeBinomial2 log-likelihood, matching numpyro's parameterisation.

    var = mu + mu^2 / phi, so a LOWER phi is MORE overdispersion -- the same
    convention as the ``concentration`` site in src/model/age_priors.py.
    """
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    phi = float(phi)
    return (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1.0)
            + phi * np.log(phi / (phi + mu)) + y * np.log(mu / (phi + mu)))


def nb2_concentration_mle(y, mu, bounds=(1e-3, 1e4)):
    """MLE of the NB2 concentration given a fitted mean function."""
    def nll(log_phi):
        return -np.sum(nb2_loglik(y, mu, np.exp(log_phi)))
    res = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])),
                          method="bounded")
    return float(np.exp(res.x))


def nb2_deviance_explained(y, mu, phi):
    """1 - (model NB2 deviance / null NB2 deviance), the standard SDM report."""
    ll_model = float(np.sum(nb2_loglik(y, mu, phi)))
    ll_null = float(np.sum(nb2_loglik(y, np.full_like(np.asarray(y, float),
                                                      np.mean(y)), phi)))
    ll_sat = float(np.sum(nb2_loglik(y, np.maximum(np.asarray(y, float), 1e-9), phi)))
    dev_model = 2.0 * (ll_sat - ll_model)
    dev_null = 2.0 * (ll_sat - ll_null)
    return 1.0 - dev_model / dev_null if dev_null > 0 else float("nan")
