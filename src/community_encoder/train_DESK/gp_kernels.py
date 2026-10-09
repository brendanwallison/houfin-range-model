"""GP regression on a finite feature map: the DESK kernel ``s^2 z(x).z(x')`` used as a GP kernel.

A kernel that is a dot product of ``r`` features is exactly Bayesian linear regression on those
features. For DESK, ``r = latent_dim = 64``. So inference here is EXACT, with no inducing points
and no subsampling: one thin SVD of the training features serves every species, and every
per-species quantity is an ``r``-vector operation.

THE MODEL, per species, on log1p abundance ``y``::

    y = mu + (z - zbar) . w + eps,     w ~ N(0, s^2 I_r),     eps ~ N(0, sigma^2)

``mu`` has a flat prior and is integrated out (REML). Centering ``z`` at its training mean
``zbar`` is what makes that integration closed-form: with centered features the intercept is
orthogonal to ``w``, so its posterior mean is ``ybar`` and its variance ``sigma^2/n`` exactly.
Only ``s^2`` and ``sigma^2`` are fitted, by marginal likelihood. Two scalars per species and no
per-dimension weights is the point: it is the isotropic coefficient prior the stat model puts on
``w_env``, which is the kernel contract's shape. The prior's AMPLITUDE cannot be carried over,
because the stat model's is in vital-rate units and this is log1p abundance, so ``s^2`` is fitted.

A CONSEQUENCE TO READ RESULTS BY. With ~80k training rows and 64 features the prior barely binds
for any well-observed species: the posterior mean is close to OLS on ``z``, the posterior
variance of ``w`` is tiny, and the predictive spread is almost all ``sigma^2``. The GP framing
earns its keep for RARE species, where ``s^2`` actually shrinks, and in the probabilistic scores.

Rows are treated as exchangeable given ``z``. A cell surveyed in 40 years contributes 40 rows
whose residuals are correlated, so ``sigma^2`` is a marginal residual variance and the evidence is
overcounted. That affects how hard the prior binds, not the mean's consistency. The deployed stat
model makes the same assumption of the kernel, so it is not corrected here.
"""
import numpy as np
from scipy.optimize import minimize

#: Absolute floor, used only for species with no training variance (nothing to fit) and for
#: the intercept predictor.
VAR_FLOOR = 1e-6
#: Per-species floor on both variances, RELATIVE to that species' training variance. An absolute
#: floor is not scale-free: a species with a few detections among ~80k rows has total variance
#: ~1e-5 in log1p abundance, so an absolute 1e-6 floor on sigma^2 was a tenth of its variance and
#: "s^2 at the floor" read as a collapse when it was mostly that species' units.
REL_FLOOR = 1e-6
#: Singular values below this fraction of the largest are dropped from the basis. A collapsed
#: latent dimension otherwise contributes log(sigma^2) terms with no data behind them.
RANK_TOL = 1e-8


def feature_svd(Z):
    """Thin SVD of centered training features. ``{zbar, V, S, n, r}``. Pure.

    ``U`` is never formed: every quantity needs only ``U^T y``, which equals
    ``S^-1 V^T Zc^T y`` and costs one ``(r, n) @ (n, n_species)`` product instead of an
    ``(n, r)`` matrix per call.
    """
    Z = np.asarray(Z, "float64")
    zbar = Z.mean(0)
    Zc = Z - zbar
    # Eigendecomposition of the r x r Gram is cheaper than an SVD of the n x r matrix and exact
    # to float64 precision at r=64 (condition number squared is still far from 1e16 here).
    G = Zc.T @ Zc
    evals, V = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals, V = np.clip(evals[order], 0.0, None), V[:, order]
    keep = evals > RANK_TOL * max(float(evals[0]), 1e-300)
    return {"zbar": zbar, "V": V[:, keep], "S": np.sqrt(evals[keep]),
            "n": int(Z.shape[0]), "r": int(keep.sum())}


def sufficient_stats(basis, Z, Y):
    """Per-species statistics the likelihood needs. ``{ybar, yy, u}``. Pure.

    ``u = U^T yc`` is ``(r, n_species)``; ``yy = ||yc||^2`` is ``(n_species,)``. Everything
    downstream is a function of these alone, which is what makes fitting 600 species cheap.
    """
    Y = np.asarray(Y, "float64")
    ybar = Y.mean(0)
    Yc = Y - ybar
    Zc = np.asarray(Z, "float64") - basis["zbar"]
    u = (basis["V"].T @ (Zc.T @ Yc)) / basis["S"][:, None]
    return {"ybar": ybar, "yy": (Yc * Yc).sum(0), "u": u}


def neg_log_marginal(log_s2, log_n2, basis, stats):
    """REML negative log marginal likelihood and its gradient, per species. Pure.

    Returns ``(nll, d_log_s2, d_log_n2)``, each ``(n_species,)``, constants dropped. With
    ``d_k = s^2 S_k^2 + sigma^2`` and ``m = n - 1`` (one dimension spent on the intercept)::

        2 nll = (m - r) log sigma^2 + sum_k log d_k
                + (yy - ||u||^2) / sigma^2 + sum_k u_k^2 / d_k
    """
    s2, n2 = np.exp(log_s2), np.exp(log_n2)
    S2 = basis["S"][:, None] ** 2                         # (r, 1)
    u2 = stats["u"] ** 2                                  # (r, n_species)
    m, r = basis["n"] - 1, basis["r"]
    d = s2[None, :] * S2 + n2[None, :]                    # (r, n_species)
    resid = np.clip(stats["yy"] - u2.sum(0), 0.0, None)   # energy outside the feature span
    nll = 0.5 * ((m - r) * log_n2 + np.log(d).sum(0) + resid / n2 + (u2 / d).sum(0))
    a = s2[None, :] * S2                                  # d d / d log s2
    g_s = 0.5 * (a / d - u2 * a / d ** 2).sum(0)
    g_n = 0.5 * ((m - r) + (n2[None, :] / d).sum(0) - resid / n2
                 - (u2 * n2[None, :] / d ** 2).sum(0))
    return nll, g_s, g_n


def species_floor(var_y):
    """Per-species variance floor: REL_FLOOR of the species' own variance. Pure."""
    return np.maximum(REL_FLOOR * np.asarray(var_y, "float64"), 1e-14)


#: Starting signal shares for the per-species scale fit. The profiled likelihood in (s^2, sigma^2)
#: is MULTIMODAL: measured on a long-lengthscale spacetime kernel, a single start at share 0.5
#: left a species at s^2 ~ 0 (NLL -231.9 summed) where a start a hair away found s^2 = 140
#: (-340.7). A collapsed amplitude reads as "the kernel explains nothing" when it is an optimizer
#: artefact, so every start is run and each species keeps its own best.
SCALE_STARTS = (0.5, 0.1, 0.9)
#: Amplitude multipliers on top of each share. When a kernel's useful eigenvalues are tiny (long
#: lengthscales: K is nearly constant and the centred data live in its small eigenvalues), the optimal
#: s^2 sits orders of magnitude above share * var_y / kappa, and every share-only start converged to
#: the collapsed optimum (measured: -237.4 summed where the better optimum is -347.8).
SCALE_MULTIPLIERS = (1.0, 1e3)


def _fit_scales_core(nll_parts, var_y, kappa, live, floor, max_iter=500):
    """Per-species ``(s^2, sigma^2)`` minimizing ``nll_parts`` from every SCALE_STARTS share. Pure.

    ``nll_parts(log_s2, log_n2) -> (nll, g_s, g_n)`` over the LIVE species. Species are
    independent, so one L-BFGS run on the summed objective is many separate ones at a fraction of
    the overhead; the multi-start then picks per species. Returns ``(s2, n2, nll)`` for live ones.
    """
    kl = int(live.sum())
    lo = np.log(floor[live])
    bounds = [(v, 30.0) for v in np.concatenate([lo, lo])]

    def f(x):
        nll, gs, gn = nll_parts(x[:kl], x[kl:])
        return float(nll.sum()), np.concatenate([gs, gn])

    best_x, best_nll = None, None
    for share, mult in [(s, m) for m in SCALE_MULTIPLIERS for s in SCALE_STARTS]:
        s2_0 = np.minimum(np.maximum(mult * share * var_y[live] / max(kappa, 1e-12), floor[live]),
                          np.exp(29.0))
        n2_0 = np.maximum((1.0 - share) * var_y[live], floor[live])
        res = minimize(f, np.concatenate([np.log(s2_0), np.log(n2_0)]), jac=True,
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": int(max_iter)})
        nll = nll_parts(res.x[:kl], res.x[kl:])[0]
        if best_x is None:
            best_x, best_nll = res.x.copy(), nll
        else:
            better = nll < best_nll - 1e-9
            best_x[:kl][better] = res.x[:kl][better]
            best_x[kl:][better] = res.x[kl:][better]
            best_nll = np.where(better, nll, best_nll)
    return np.exp(best_x[:kl]), np.exp(best_x[kl:]), best_nll


def fit_hyperparameters(basis, stats, max_iter=500):
    """Fit ``(s^2, sigma^2)`` for every species jointly by L-BFGS. ``{s2, n2, nll, ok}``. Pure.

    Species are independent, so the summed objective has a block-diagonal Hessian and one
    L-BFGS run is equivalent to many separate ones, at a fraction of the overhead. Species with
    no training variance get the floors and ``ok=False``: there is nothing to fit. Every start in
    ``SCALE_STARTS`` is run and each species keeps its best (the likelihood is multimodal).
    """
    k = stats["yy"].shape[0]
    n = basis["n"]
    var_y = stats["yy"] / max(n - 1, 1)
    live = var_y > 1e-14
    floor = species_floor(var_y)
    sub = {"ybar": stats["ybar"][live], "yy": stats["yy"][live], "u": stats["u"][:, live]}
    s2 = np.full(k, VAR_FLOOR)
    n2 = np.full(k, VAR_FLOOR)
    nll = np.full(k, np.nan)
    if live.any():
        s2[live], n2[live], nll[live] = _fit_scales_core(
            lambda a, c: neg_log_marginal(a, c, basis, sub), var_y,
            float((basis["S"] ** 2).sum()) / n, live, floor, max_iter)
    return {"s2": s2, "n2": n2, "nll": nll, "ok": live}


# ----------------------------- pooled scales -----------------------------
#
# Per-species marginal likelihood collapses the kernel's amplitude to the floor for rare species
# (first run: 94% of species with 1-3 training detections, 38% with 4-10), and a collapsed species
# is predicted by its training mean: the kernel contributes nothing exactly where a data-poor
# advantage would have to show. The fix is a cross-species prior -- an empirical-Bayes estimate of
# the amplitude prior a NEW species (House Finch, downstream) would face.
#
# It is a prior on the AMPLITUDE, centred relative to each species' own noise: with kappa the
# kernel's mean variance on the training rows,
#
#     log s^2_s ~ N(log sigma^2_s - log kappa + a, tau^2)
#
# equivalently logit rho_s ~ N(a, tau^2) for the kernel's variance share rho = s^2 kappa / v,
# v = s^2 kappa + sigma^2 (flat prior on v). Centring on the noise rather than on an absolute
# value is what makes it comparable across species: absolute s^2 spans six orders of magnitude
# mostly because rare species have little variance in log1p abundance at all.
#
# INTERCEPT ONLY, deliberately. A slope on log detections was tried and it learned "rare species
# have a low share" FROM the rare species' own estimates -- re-encoding the very thing the prior
# was meant to correct. And measured on simulations, the premise needs stating: when rarity means
# mostly zeros, the kernel's true share of the OBSERVED variance is genuinely small and the
# likelihood says so firmly, so no amplitude prior moves those species much. This arm is a check
# of that on the real data, not an expected rescue.
#
# (a, tau) by EM with a Laplace E-step: per-species MAP under the current prior, its curvature in
# logit rho, then a = mean mode and tau^2 = mode variance plus mean posterior variance. Only the
# scales are pooled; a stationary kernel's SHAPE still comes from the unpooled shared-shape fit.

TAU_FLOOR = 0.05


def _kappa(basis):
    """Mean prior variance of the unit-amplitude kernel over the rows it was built on."""
    return float((basis["S"] ** 2).sum() / max(basis["n"] - 1, 1))


def _pooled_objective(lv, eta, basis, stats, kappa, mu, tau2, floor=1e-14):
    """Per-species negative log posterior in ``(log v, logit rho)`` and its gradient. Pure."""
    rho = 1.0 / (1.0 + np.exp(-eta))
    v = np.exp(lv)
    ls2 = np.log(np.maximum(v * rho / kappa, floor))
    ln2 = np.log(np.maximum(v * (1 - rho), floor))
    nll, gs, gn = neg_log_marginal(ls2, ln2, basis, stats)
    obj = nll + 0.5 * (eta - mu) ** 2 / tau2
    g_lv = gs + gn
    g_eta = gs * (1 - rho) - gn * rho + (eta - mu) / tau2
    return obj, g_lv, g_eta


def fit_pooled_scales(basis, stats, hp0, n_em=300, tol=1e-4):
    """Every species' (s^2, sigma^2) under the cross-species amplitude prior.

    ``hp0`` is the unpooled fit, used as the starting point. Species with no training variance
    (``hp0['ok']`` False) keep their floors and do not inform the prior. Returns
    ``{s2, n2, ok, prior}``.
    """
    ok = np.asarray(hp0["ok"], bool)
    kappa = _kappa(basis)
    sub = {"ybar": stats["ybar"][ok], "yy": stats["yy"][ok], "u": stats["u"][:, ok]}
    k = int(ok.sum())
    flo = species_floor(stats["yy"][ok] / max(basis["n"] - 1, 1))
    v0 = hp0["s2"][ok] * kappa + hp0["n2"][ok]
    lv = np.log(v0)
    eta = np.clip(np.log(hp0["s2"][ok] * kappa / hp0["n2"][ok]), -12, 12)
    X = np.ones((k, 1))
    coef = np.linalg.lstsq(X, eta, rcond=None)[0]
    tau2 = float(np.var(eta - X @ coef)) + 1.0
    trace = []
    for it in range(int(n_em)):
        mu = X @ coef

        def f(z):
            o, gl, ge = _pooled_objective(z[:k], z[k:], basis, sub, kappa, mu, tau2, flo)
            return float(o.sum()), np.concatenate([gl, ge])
        res = minimize(f, np.concatenate([lv, eta]), jac=True, method="L-BFGS-B",
                       bounds=[(np.log(2 * f_), 30.0) for f_ in flo] + [(-15.0, 15.0)] * k,
                       options={"maxiter": 500})
        lv, eta = res.x[:k], res.x[k:]
        # Laplace curvature: the 2x2 Hessian per species by central differences of the gradient.
        h = 1e-4
        _, gl_p, ge_p = _pooled_objective(lv + h, eta, basis, sub, kappa, mu, tau2, flo)
        _, gl_m, ge_m = _pooled_objective(lv - h, eta, basis, sub, kappa, mu, tau2, flo)
        _, gl_q, ge_q = _pooled_objective(lv, eta + h, basis, sub, kappa, mu, tau2, flo)
        _, gl_r, ge_r = _pooled_objective(lv, eta - h, basis, sub, kappa, mu, tau2, flo)
        H_vv = (gl_p - gl_m) / (2 * h)
        H_ee = (ge_q - ge_r) / (2 * h)
        H_ve = 0.5 * ((ge_p - ge_m) / (2 * h) + (gl_q - gl_r) / (2 * h))
        det = H_vv * H_ee - H_ve ** 2
        post_var = np.where(det > 0, H_vv / np.where(det > 0, det, 1.0), tau2)
        post_var = np.clip(post_var, 0.0, tau2)
        new_coef = np.linalg.lstsq(X, eta, rcond=None)[0]
        new_tau2 = max(float(np.mean((eta - X @ new_coef) ** 2 + post_var)), TAU_FLOOR ** 2)
        delta = max(np.max(np.abs(new_coef - coef)), abs(np.sqrt(new_tau2) - np.sqrt(tau2)))
        coef, tau2 = new_coef, new_tau2
        trace.append({"a": float(coef[0]), "tau": float(np.sqrt(tau2))})
        if delta < tol:
            break
    rho = 1.0 / (1.0 + np.exp(-eta))
    v = np.exp(lv)
    s2, n2 = hp0["s2"].copy(), hp0["n2"].copy()
    s2[ok] = np.maximum(v * rho / kappa, flo)
    n2[ok] = np.maximum(v * (1 - rho), flo)
    return {"s2": s2, "n2": n2, "ok": ok,
            "prior": {"a": float(coef[0]), "tau": float(np.sqrt(tau2)),
                      "kappa": kappa, "em_iterations": len(trace),
                      "converged": bool(trace and len(trace) < int(n_em))}}


def fit(Z, Y, pool=False):
    """Fit the GP for every column of ``Y`` on features ``Z``. Returns the fitted model. Pure.

    With ``pool`` the scales are fitted under the cross-species amplitude prior
    (``fit_pooled_scales``); otherwise each species alone.
    """
    basis = feature_svd(Z)
    stats = sufficient_stats(basis, Z, Y)
    hp = fit_hyperparameters(basis, stats)
    if pool:
        hp = {**hp, **fit_pooled_scales(basis, stats, hp)}
    S = basis["S"][:, None]
    lam = hp["n2"][None, :] / hp["s2"][None, :]
    coef = S * stats["u"] / (S ** 2 + lam)               # posterior mean of w, in the V basis
    # Posterior variance of w in the V basis, diagonal: 1 / (S^2/sigma^2 + 1/s^2).
    wvar = 1.0 / (S ** 2 / hp["n2"][None, :] + 1.0 / hp["s2"][None, :])
    return {"basis": basis, "ybar": stats["ybar"], "coef": coef, "wvar": wvar, **hp}


def predict(model, Z_new, y_self=None):
    """Predictive mean and variance of a NEW OBSERVATION at each row of ``Z_new``. Pure.

    Returns ``(mean, var)``, both ``(n_new, n_species)``. ``var`` includes the noise, because
    held-out rows are observations rather than the latent function, and it includes the
    intercept's ``sigma^2/n``.

    ``y_self`` (``(n_new, n_species)``, NaN rows where unused) marks rows that are THEMSELVES in
    the training set and returns their exact leave-one-out prediction instead: a model graded on a
    row it was fitted to is graded on its own noise draw. For ridge-form BLR the LOO residual is
    the fitted residual over ``1 - h_ii``, with ``h_ii = sum_k P_ik^2 / (S_k^2 + lambda) + 1/n``
    per species. At ~80k rows and r=64 ``h_ii`` is ~1e-3, so this barely moves desk; it exists so
    every predictor answers the same question on the ``time`` set's in-sample rows.
    """
    b = model["basis"]
    P = (np.asarray(Z_new, "float64") - b["zbar"]) @ b["V"]     # (n_new, r)
    mean = model["ybar"][None, :] + P @ model["coef"]
    var = (P ** 2) @ model["wvar"] + model["n2"][None, :] * (1.0 + 1.0 / b["n"])
    if y_self is not None:
        ys = np.asarray(y_self, "float64")
        rows = np.isfinite(ys).all(1)
        if rows.any():
            lam = model["n2"][None, :] / model["s2"][None, :]                   # (1, S)
            S2 = (b["S"] ** 2)[:, None]                                         # (r, 1)
            h = (P[rows] ** 2) @ (1.0 / (S2 + lam)) + 1.0 / b["n"]              # (m, S)
            h = np.minimum(h, 1.0 - 1e-9)
            mean[rows] = ys[rows] - (ys[rows] - mean[rows]) / (1.0 - h)
    return mean, var


def dense_reference(Z, Y, s2, n2, Z_new):
    """The same GP done the slow way, with an ``n x n`` covariance. For tests only. Pure.

    The intercept is handled as a flat-prior fixed effect via GLS, which is what REML
    integrates. Returns ``(nll, mean, var)`` for a single species (``Y`` 1-D).
    """
    Z, y, Zn = np.asarray(Z, "float64"), np.asarray(Y, "float64"), np.asarray(Z_new, "float64")
    n = len(y)
    zbar = Z.mean(0)
    Zc, Znc = Z - zbar, Zn - zbar
    K = s2 * Zc @ Zc.T + n2 * np.eye(n)
    Ki = np.linalg.inv(K)
    one = np.ones(n)
    mu = (one @ Ki @ y) / (one @ Ki @ one)
    res = y - mu
    # REML: project out the intercept. log|K| + log(1'K^-1 1) + res'K^-1 res, up to constants
    # that differ from the low-rank form only by terms independent of (s2, n2).
    _, logdet = np.linalg.slogdet(K)
    nll = 0.5 * (logdet + np.log(one @ Ki @ one) + res @ Ki @ res)
    kx = s2 * Znc @ Zc.T
    mean = mu + kx @ Ki @ res
    # Predictive variance with the GLS intercept's uncertainty folded in.
    a = 1.0 - kx @ Ki @ one
    var = (s2 * (Znc * Znc).sum(1) - np.einsum("ij,jk,ik->i", kx, Ki, kx)
           + a ** 2 / (one @ Ki @ one) + n2)
    return nll, mean, var


# =====================================================================================
# Stationary baselines: spacetime and covariate GPs with a SHARED kernel shape.
#
# The comparison is between kernel SHAPES, so every predictor gets the same per-species freedom:
# DESK's shape (z.z') is fixed and shared, and each species fits only an amplitude s^2 and a noise
# sigma^2. The baselines get the same structure -- one set of lengthscales shared by every
# evaluation species, plus a per-species (s^2, sigma^2). Their shape is fitted on the evaluation
# species' OWN training data, which DESK's never saw, so if anything this favours the baselines.
#
# Fitting the shared shape uses the envelope theorem. For fixed lengthscales theta, eigendecompose
# K once; every species' (s^2, sigma^2) then has the closed-form likelihood of the low-rank code
# above, with eigenvalues in place of S^2. At those per-species optima the gradient of the summed
# NLL in theta is exactly
#
#     0.5 tr(M dK/dtheta),   M = sum_s s_s^2 (C_s^-1 - a_s a_s^T),   C_s = s_s^2 K + sigma_s^2 I
#
# and M is cheap in K's eigenbasis. So theta is updated through K(theta) alone, never by
# differentiating an eigendecomposition (whose gradient is unstable at near-degenerate spectra).
#
# The mean is the training mean of each species, a plug-in rather than REML's integrated
# intercept: with a full-rank stationary kernel the intercept is not orthogonal to the features,
# and at thousands of rows the difference is negligible next to the noise.
# =====================================================================================

MATERN_SQRT3 = 3.0 ** 0.5


def spacetime_kernel(A, B, theta):
    """Matern-3/2 in space x exponential in time, unit amplitude. ``A, B`` are ``(n, 3)`` tensors
    ``[x_km, y_km, year]``; ``theta = [log l_space_km, log l_time_yr]``."""
    import torch
    ls, lt = torch.exp(theta[0]), torch.exp(theta[1])
    d = torch.cdist(A[:, :2], B[:, :2]) / ls
    dt = torch.abs(A[:, 2:3] - B[:, 2:3].T) / lt
    return (1.0 + MATERN_SQRT3 * d) * torch.exp(-MATERN_SQRT3 * d) * torch.exp(-dt)


def spacetime_sum_kernel(A, B, theta):
    """A PERSISTENT spatial field plus a spatiotemporal one, unit total amplitude::

        k = (1 - p) M32(d / l_persist) + p M32(d / l_change) exp(-|dt| / l_time),  p = sigmoid(th3)

    ``theta = [log l_persist_km, log l_change_km, log l_time_yr, logit p]``. The product kernel
    alone forgets a cell's level as |dt| grows and reverts to the continental mean, which is the
    wrong prior for backcasting a place whose present is observed: most of a species' map persists
    for decades. This is the honest temporal rival -- it keeps what does not change and
    extrapolates only what does."""
    import torch
    ls_p, ls_c, lt = torch.exp(theta[0]), torch.exp(theta[1]), torch.exp(theta[2])
    p = torch.sigmoid(theta[3])
    d = torch.cdist(A[:, :2], B[:, :2])
    m_p = (1.0 + MATERN_SQRT3 * d / ls_p) * torch.exp(-MATERN_SQRT3 * d / ls_p)
    m_c = (1.0 + MATERN_SQRT3 * d / ls_c) * torch.exp(-MATERN_SQRT3 * d / ls_c)
    dt = torch.abs(A[:, 2:3] - B[:, 2:3].T) / lt
    return (1.0 - p) * m_p + p * m_c * torch.exp(-dt)


def ard_rbf_kernel(A, B, theta):
    """Squared-exponential with one lengthscale per input dimension, unit amplitude.
    ``theta`` is the vector of log lengthscales."""
    import torch
    ell = torch.exp(theta)
    d2 = torch.cdist(A / ell, B / ell) ** 2
    return torch.exp(-0.5 * d2)


def scaled_coords(kind, F, theta):
    """Coordinates in which Euclidean distance tracks the kernel's decay, for neighbour search.

    Exact for the ARD kernel. For the spacetime kernel the space and time factors are combined
    as one Euclidean distance, which orders neighbours the way the product kernel does to within
    the difference between an L2 and an L1 combination -- adequate for CHOOSING a conditioning
    set, which is all it is used for.
    """
    F = np.asarray(F, "float64")
    th = np.asarray(theta, "float64")
    if kind == "spacetime":
        return np.column_stack([F[:, :2] / np.exp(th[0]), F[:, 2:3] / np.exp(th[1])])
    if kind == "spacetime_sum":
        # Time matters only through the changing share p: as p -> 0 the field is persistent and
        # a cell's other decades are as near as its own year.
        p = 1.0 / (1.0 + np.exp(-th[3]))
        ls = min(np.exp(th[0]), np.exp(th[1]))
        return np.column_stack([F[:, :2] / ls, F[:, 2:3] * max(p, 1e-3) / np.exp(th[2])])
    return F / np.exp(th)


KERNELS = {"spacetime": spacetime_kernel, "spacetime_sum": spacetime_sum_kernel,
           "covariate": ard_rbf_kernel}


def _eig_basis(lam, n):
    """A full-rank eigendecomposition expressed in the low-rank code's terms (``m - r = 0``)."""
    return {"S": np.sqrt(np.clip(lam, 0.0, None)), "n": int(n) + 1, "r": int(n)}


def _scales_given_shape(K, Yc):
    """Eigendecompose ``K`` and fit every species' (s^2, sigma^2). ``(Q, lam, U, hp)``."""
    lam, Q = np.linalg.eigh(K)
    U = Q.T @ Yc
    basis = _eig_basis(lam, K.shape[0])
    stats = {"ybar": np.zeros(Yc.shape[1]), "yy": (Yc * Yc).sum(0), "u": U}
    hp = fit_hyperparameters(basis, stats)
    return Q, np.clip(lam, 0.0, None), U, hp


def fit_hyperparameters_blocks(bases, statss, var_y, max_iter=500):
    """``(s^2, sigma^2)`` per species maximizing the SUM of per-block marginal likelihoods. Pure.

    The block-diagonal approximation of the full GP likelihood: blocks are spatially contiguous
    groups of training rows (all years of each cell together), so the near pairs that identify an
    amplitude are inside blocks and only far pairs, which carry little, are dropped. It is how
    every training row informs the scales without an n x n solve over all of them. ``var_y`` is
    each species' variance over ALL training rows, which sets the RELATIVE floors (an absolute
    floor is not scale-free; see REL_FLOOR) and decides who is fittable at all.
    """
    var_y = np.asarray(var_y, "float64")
    k = len(var_y)
    live = var_y > 1e-14
    floor = species_floor(var_y)
    kappa = float(np.mean([(b["S"] ** 2).sum() / max(b["n"] - 1, 1) for b in bases]))
    subs = [{"ybar": st["ybar"][live], "yy": st["yy"][live], "u": st["u"][:, live]}
            for st in statss]
    kl = int(live.sum())

    def parts(a, c):
        tot, gs, gn = np.zeros(kl), np.zeros(kl), np.zeros(kl)
        for b, st in zip(bases, subs):
            nll, g1, g2 = neg_log_marginal(a, c, b, st)
            tot, gs, gn = tot + nll, gs + g1, gn + g2
        return tot, gs, gn

    s2, n2 = floor.copy(), floor.copy()
    nll = np.full(k, np.nan)
    if kl:
        s2[live], n2[live], nll[live] = _fit_scales_core(parts, var_y, kappa, live, floor,
                                                         max_iter)
    return {"s2": s2, "n2": n2, "nll": nll, "ok": live}


def _block_eig(kern, F_blocks, Yc_blocks, theta_t):
    """Per block: ``(Q, lam, U, basis, stats)`` at the given (detached) lengthscales."""
    import torch
    out = []
    for Fb, Ycb in zip(F_blocks, Yc_blocks):
        with torch.no_grad():
            K = kern(Fb, Fb, theta_t).numpy()
        lam, Q = np.linalg.eigh(K)
        lam = np.clip(lam, 0.0, None)
        U = Q.T @ Ycb
        out.append((Q, lam, U, _eig_basis(lam, K.shape[0]),
                    {"ybar": np.zeros(Ycb.shape[1]), "yy": (Ycb * Ycb).sum(0), "u": U}))
    return out


def blocked_scales(kind, theta, F_blocks, Y_blocks, ybar, var_y):
    """Every species' (s^2, sigma^2) at fixed lengthscales, from ALL the given blocks. Pure.

    The shape is fitted on a subsample for speed, but the per-species scales must not be: the
    first runs took them from the same 3,000 random rows, and a species with no detection there
    got the absolute 1e-6 floor -- an unfitted signal-to-noise of 1 and a predictive variance of
    ~1e-6 -- exactly in the rare-species regime the data-poor claim is about."""
    import torch
    kern = KERNELS[kind]
    th = torch.as_tensor(np.asarray(theta, "float64"))
    Fb = [torch.as_tensor(np.asarray(f, "float64")) for f in F_blocks]
    Yc = [np.asarray(y, "float64") - np.asarray(ybar, "float64") for y in Y_blocks]
    eig = _block_eig(kern, Fb, Yc, th)
    return fit_hyperparameters_blocks([e[3] for e in eig], [e[4] for e in eig], var_y)


#: A lengthscale beyond this many times its input's own extent is indistinguishable from infinite,
#: and letting the optimizer wander there makes K nearly constant -- a degenerate inner problem whose
#: per-species optima jump between basins (the old bound was e^12 in every unit, ~160,000 km).
LENGTHSCALE_EXTENT_MULT = 20.0


def theta_bounds(kind, F, n_theta):
    """Log-lengthscale bounds from the data's extent: [-8, log(20 x extent)] per input. Pure."""
    F = np.asarray(F, "float64")
    rng = np.maximum(F.max(0) - F.min(0), 1e-6)
    up = lambda r: float(np.log(LENGTHSCALE_EXTENT_MULT * r))
    if kind == "spacetime":
        return [(-8.0, up(rng[:2].max())), (-8.0, up(rng[2]))]
    if kind == "spacetime_sum":
        s = up(rng[:2].max())
        return [(-8.0, s), (-8.0, s), (-8.0, up(rng[2])), (-8.0, 8.0)]
    return [(-8.0, up(rng[j])) for j in range(n_theta)]


def fit_shared_shape(kind, F, Y, theta0, n_iter=400, verbose=True, ybar=None, var_y=None):
    """Fit the shared lengthscales and every species' (s^2, sigma^2). Returns the shape fit.

    ``F (n, d)`` are the fitting rows' inputs (a subsample of training rows: this is an n x n
    eigendecomposition per evaluation), ``Y (n, S)`` log1p abundance. ``F`` and ``Y`` may instead
    be LISTS of blocks: the objective is then the sum of per-block NLLs with each species' scales
    profiled jointly over all blocks, and the envelope gradient is the sum of the per-block terms.
    Contiguous blocks put a cell's own other years and its neighbours in the fit, which a random
    subsample of a continent almost never does -- without them a spacetime kernel's short-range
    structure is unidentified. ``ybar``/``var_y`` default to the fitting rows' own.

    L-BFGS-B on the log-lengthscales, minimizing the summed NLL with every species' scales
    PROFILED out, with the envelope gradient described above. Not Adam: the first run used Adam
    at a fixed step for a fixed 80 iterations, and it creeps along flat ridges indefinitely (a
    lengthscale heading for infinity, an ARD dimension that barely matters), so the covariate GP's
    30-odd lengthscales were still moving when it stopped. L-BFGS-B has a real convergence test
    (projected gradient and relative objective change) and reports whether it met it, which
    ``converged`` records. ``n_iter`` caps L-BFGS iterations.
    """
    import torch
    from scipy.optimize import minimize as _minimize
    kern = KERNELS[kind]
    blocks = isinstance(F, (list, tuple))
    F_list = list(F) if blocks else [F]
    Y_list = [np.asarray(y, "float64") for y in (Y if blocks else [Y])]
    Y_all = np.concatenate(Y_list)
    ybar = Y_all.mean(0) if ybar is None else np.asarray(ybar, "float64")
    var_y = Y_all.var(0) * len(Y_all) / max(len(Y_all) - 1, 1) if var_y is None else var_y
    Ft = [torch.as_tensor(np.asarray(f, "float64")) for f in F_list]
    Yc = [y - ybar for y in Y_list]
    trace = []

    def f(th):
        theta = torch.tensor(th, requires_grad=True)
        eig = _block_eig(kern, Ft, Yc, theta.detach())
        hp = fit_hyperparameters_blocks([e[3] for e in eig], [e[4] for e in eig], var_y)
        live = hp["ok"]
        s2, n2 = hp["s2"][live], hp["n2"][live]
        obj = 0.0
        for (Q, lam, U, _b, _s), Fb in zip(eig, Ft):
            D = s2[None, :] * lam[:, None] + n2[None, :]                 # (n_b, S_live)
            A = U[:, live] / D                                           # a_s in the eigenbasis
            M = Q @ (np.diag((s2[None, :] / D).sum(1)) - (A * s2[None, :]) @ A.T) @ Q.T
            obj = obj + 0.5 * (torch.as_tensor(M) * kern(Fb, Fb, theta)).sum()
        obj.backward()
        total = float(np.nansum(hp["nll"]))
        trace.append(total)
        if verbose and len(trace) % 10 == 1:
            print(f"[gp-{kind}] eval {len(trace):3d}  sum nll {total:,.1f}  "
                  f"lengthscales {np.round(np.exp(th), 3)[:6]}", flush=True)
        return total, theta.grad.numpy().copy()

    bnds = theta_bounds(kind, np.concatenate([np.asarray(f, "float64") for f in F_list]),
                        len(theta0))
    x0 = np.clip(np.asarray(theta0, "float64"), [b[0] for b in bnds], [b[1] for b in bnds])
    res = _minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bnds,
                    options={"maxiter": int(n_iter), "maxfun": 2 * int(n_iter)})
    th = np.asarray(res.x, "float64")
    at_bound = np.array([np.isclose(t, lo) or np.isclose(t, hi) for t, (lo, hi) in zip(th, bnds)])
    if verbose:
        print(f"[gp-{kind}] {'converged' if res.success else 'NOT converged'} after "
              f"{res.nit} iterations ({res.message}); "
              f"{int(at_bound.sum())} lengthscale(s) at a bound", flush=True)
    eig = _block_eig(kern, Ft, Yc, torch.as_tensor(th))
    hp = fit_hyperparameters_blocks([e[3] for e in eig], [e[4] for e in eig], var_y)
    return {"kind": kind, "theta": th, "ybar": ybar, "trace": trace,
            "converged": bool(res.success), "message": str(res.message),
            "n_iterations": int(res.nit), "at_bound": at_bound.tolist(),
            "n_blocks": len(F_list), **hp}


def pool_shape_scales(shape, F_fit, Y_fit):
    """A copy of a stationary shape fit with its per-species scales re-fitted under the
    cross-species prior. The shape (lengthscales) is unchanged. Pure."""
    import torch
    K = KERNELS[shape["kind"]](torch.as_tensor(np.asarray(F_fit, "float64")),
                               torch.as_tensor(np.asarray(F_fit, "float64")),
                               torch.as_tensor(shape["theta"])).numpy()
    Y = np.asarray(Y_fit, "float64")
    Yc = Y - Y.mean(0)
    lam, Q = np.linalg.eigh(K)
    basis = _eig_basis(lam, K.shape[0])
    stats = {"ybar": np.zeros(Y.shape[1]), "yy": (Yc * Yc).sum(0), "u": Q.T @ Yc}
    # Only species that VARY in these rows can inform or receive the pooled prior here; the rest
    # keep the scales they came with (fitted on every training row), instead of being driven to
    # their floors by rows where they are never seen.
    ok = np.asarray(shape["ok"], bool) & (stats["yy"] > 1e-14)
    hp0 = {"s2": shape["s2"], "n2": shape["n2"], "ok": ok}
    pooled = fit_pooled_scales(basis, stats, hp0)
    return {**shape, "s2": pooled["s2"], "n2": pooled["n2"], "prior": pooled["prior"]}


def knn_union(Fs_train, Fs_test, k, k_max, device=None):
    """Indices of training rows forming the conditioning set for a group of test rows. Pure-ish.

    The union of each test row's ``k`` nearest training rows, in coordinates where distance tracks
    the kernel. Capped at ``k_max`` by keeping the rows nearest to ANY test row, so a large group
    keeps its closest support rather than an arbitrary subset.
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr = torch.as_tensor(np.asarray(Fs_train, "float32"), device=dev)
    best = torch.full((tr.shape[0],), float("inf"), device=dev)
    picked = set()
    te = np.asarray(Fs_test, "float32")
    for s in range(0, len(te), 512):
        d = torch.cdist(torch.as_tensor(te[s:s + 512], device=dev), tr)
        kk = min(int(k), tr.shape[0])
        picked.update(torch.topk(d, kk, dim=1, largest=False).indices.flatten().cpu().tolist())
        best = torch.minimum(best, d.min(0).values)
    idx = np.fromiter(picked, dtype="int64")
    if len(idx) > k_max:
        b = best.cpu().numpy()[idx]
        idx = idx[np.argsort(b)[:int(k_max)]]
    return np.sort(idx)


def predict_local(shape, F_train, Y_train, F_test, groups, k=32, k_max=4000, self_idx=None):
    """Predictive mean and variance of a new observation, conditioning each group of test rows
    on its own local training set. ``(mean, var)``, each ``(n_test, S)``.

    ``groups`` labels the test rows (held-out blocks): rows in one group share a conditioning set,
    so one eigendecomposition per group serves every species. The shared shape and per-species
    scales come from ``fit_shared_shape``; nothing is refitted on the local set.

    ``self_idx`` (``(n_test,)``, index into the training rows or -1) marks test rows that ARE
    training rows -- the ``time`` set's in-sample modern epoch. Such a row is its own nearest
    neighbour at distance zero, so an exact GP reproduces its observed value, noise included, and
    the "predicted" modern epoch then shares the target's noise draw. Those rows get the exact
    leave-one-out prediction from the same conditioning set instead:
    ``mean = y_i - [C^-1 (y - ybar)]_i / [C^-1]_ii``, ``var = 1 / [C^-1]_ii``, with
    ``C = s^2 K + sigma^2 I`` per species -- all from the eigendecomposition already in hand.
    """
    import torch
    kern = KERNELS[shape["kind"]]
    th = torch.as_tensor(shape["theta"])
    Fs_tr = scaled_coords(shape["kind"], F_train, shape["theta"])
    Fs_te = scaled_coords(shape["kind"], F_test, shape["theta"])
    Y_train = np.asarray(Y_train, "float64")
    S = Y_train.shape[1]
    s2, n2, ybar = shape["s2"], shape["n2"], shape["ybar"]
    mean = np.empty((len(F_test), S))
    var = np.empty((len(F_test), S))
    Ftr = np.asarray(F_train, "float64")
    Fte = np.asarray(F_test, "float64")
    sidx = None if self_idx is None else np.asarray(self_idx, "int64")
    loo_var = np.zeros((len(F_test), S), bool)
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        cidx = knn_union(Fs_tr, Fs_te[rows], k, k_max)
        Fc = torch.as_tensor(Ftr[cidx])
        Kcc = kern(Fc, Fc, th).numpy()
        Ksc = kern(torch.as_tensor(Fte[rows]), Fc, th).numpy()
        lam, Q = np.linalg.eigh(Kcc)
        lam = np.clip(lam, 0.0, None)
        D = s2[None, :] * lam[:, None] + n2[None, :]                     # (k, S)
        A = Q.T @ (Y_train[cidx] - ybar)                                 # (k, S)
        P = Ksc @ Q                                                      # (t, k)
        mean[rows] = ybar + P @ (s2[None, :] * A / D)
        var[rows] = s2[None, :] - (P ** 2) @ (s2[None, :] ** 2 / D) + n2[None, :]
        if sidx is not None:
            own = sidx[rows]
            pos = np.searchsorted(cidx, own)
            hit = (own >= 0) & (pos < len(cidx))
            hit[hit] = cidx[pos[hit]] == own[hit]
            if hit.any():
                Qi = Q[pos[hit]]                                         # (m, k)
                cinv_diag = (Qi ** 2) @ (1.0 / D)                        # (m, S)
                alpha = Qi @ (A / D)                                     # (m, S)
                r_hit = rows[hit]
                mean[r_hit] = Y_train[own[hit]] - alpha / cinv_diag
                var[r_hit] = 1.0 / cinv_diag
                loo_var[r_hit] = True
    return mean, np.where(loo_var, var, np.maximum(var, n2[None, :]))
