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

#: Floor on both variances. A species with zero variance in training (never detected) has no
#: likelihood to fit, and an unfloored sigma^2 runs to zero and takes the log-density with it.
VAR_FLOOR = 1e-6
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


def fit_hyperparameters(basis, stats, max_iter=500):
    """Fit ``(s^2, sigma^2)`` for every species jointly by L-BFGS. ``{s2, n2, nll, ok}``. Pure.

    Species are independent, so the summed objective has a block-diagonal Hessian and one
    L-BFGS run is equivalent to many separate ones, at a fraction of the overhead. Species with
    no training variance get the floors and ``ok=False``: there is nothing to fit.
    """
    k = stats["yy"].shape[0]
    n = basis["n"]
    var_y = stats["yy"] / max(n - 1, 1)
    live = var_y > VAR_FLOOR
    lo = np.log(VAR_FLOOR)
    # Start with half the variance explained by the features and half residual.
    s2_0 = np.maximum(0.5 * var_y / max(float((basis["S"] ** 2).sum()) / n, 1e-12), VAR_FLOOR)
    n2_0 = np.maximum(0.5 * var_y, VAR_FLOOR)
    x0 = np.concatenate([np.log(s2_0), np.log(n2_0)])
    sub = {"ybar": stats["ybar"][live], "yy": stats["yy"][live], "u": stats["u"][:, live]}
    kl = int(live.sum())

    def f(x):
        nll, gs, gn = neg_log_marginal(x[:kl], x[kl:], basis, sub)
        return float(nll.sum()), np.concatenate([gs, gn])

    s2, n2 = np.exp(x0[:k]), np.exp(x0[k:])
    nll = np.full(k, np.nan)
    if kl:
        x0l = np.concatenate([x0[:k][live], x0[k:][live]])
        res = minimize(f, x0l, jac=True, method="L-BFGS-B",
                       bounds=[(lo, 30.0)] * (2 * kl), options={"maxiter": int(max_iter)})
        s2[live], n2[live] = np.exp(res.x[:kl]), np.exp(res.x[kl:])
        nll[live] = neg_log_marginal(res.x[:kl], res.x[kl:], basis, sub)[0]
    s2[~live], n2[~live] = VAR_FLOOR, VAR_FLOOR
    return {"s2": s2, "n2": n2, "nll": nll, "ok": live}


def fit(Z, Y):
    """Fit the GP for every column of ``Y`` on features ``Z``. Returns the fitted model. Pure."""
    basis = feature_svd(Z)
    stats = sufficient_stats(basis, Z, Y)
    hp = fit_hyperparameters(basis, stats)
    S = basis["S"][:, None]
    lam = hp["n2"][None, :] / hp["s2"][None, :]
    coef = S * stats["u"] / (S ** 2 + lam)               # posterior mean of w, in the V basis
    # Posterior variance of w in the V basis, diagonal: 1 / (S^2/sigma^2 + 1/s^2).
    wvar = 1.0 / (S ** 2 / hp["n2"][None, :] + 1.0 / hp["s2"][None, :])
    return {"basis": basis, "ybar": stats["ybar"], "coef": coef, "wvar": wvar, **hp}


def predict(model, Z_new):
    """Predictive mean and variance of a NEW OBSERVATION at each row of ``Z_new``. Pure.

    Returns ``(mean, var)``, both ``(n_new, n_species)``. ``var`` includes the noise, because
    held-out rows are observations rather than the latent function, and it includes the
    intercept's ``sigma^2/n``.
    """
    b = model["basis"]
    P = (np.asarray(Z_new, "float64") - b["zbar"]) @ b["V"]     # (n_new, r)
    mean = model["ybar"][None, :] + P @ model["coef"]
    var = (P ** 2) @ model["wvar"] + model["n2"][None, :] * (1.0 + 1.0 / b["n"])
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
    return F / np.exp(th)


KERNELS = {"spacetime": spacetime_kernel, "covariate": ard_rbf_kernel}


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


def fit_shared_shape(kind, F, Y, theta0, n_iter=80, lr=0.05, verbose=True):
    """Fit the shared lengthscales and every species' (s^2, sigma^2). Returns the shape fit.

    ``F (n, d)`` are the fitting rows' inputs (a subsample of training rows: this is an n x n
    eigendecomposition per iteration), ``Y (n, S)`` log1p abundance. Adam on log-lengthscales
    with the envelope gradient described above.
    """
    import torch
    kern = KERNELS[kind]
    Ft = torch.as_tensor(np.asarray(F, "float64"))
    Y = np.asarray(Y, "float64")
    ybar = Y.mean(0)
    Yc = Y - ybar
    theta = torch.tensor(np.asarray(theta0, "float64"), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    trace = []
    for it in range(int(n_iter)):
        with torch.no_grad():
            K = kern(Ft, Ft, theta).numpy()
        Q, lam, U, hp = _scales_given_shape(K, Yc)
        live = hp["ok"]
        s2, n2 = hp["s2"][live], hp["n2"][live]
        D = s2[None, :] * lam[:, None] + n2[None, :]                     # (n, S_live)
        A = U[:, live] / D                                               # a_s in the eigenbasis
        diag = (s2[None, :] / D).sum(1)
        B = (A * s2[None, :]) @ A.T
        M = Q @ (np.diag(diag) - B) @ Q.T
        total = float(np.nansum(hp["nll"]))
        trace.append(total)
        opt.zero_grad()
        surrogate = 0.5 * (torch.as_tensor(M) * kern(Ft, Ft, theta)).sum()
        surrogate.backward()
        opt.step()
        with torch.no_grad():
            theta.clamp_(-8.0, 12.0)
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f"[gp-{kind}] iter {it:3d}  sum nll {total:,.1f}  "
                  f"lengthscales {np.round(np.exp(theta.detach().numpy()), 3)[:6]}", flush=True)
    th = theta.detach().numpy().copy()
    K = kern(Ft, Ft, torch.as_tensor(th)).numpy()
    _, _, _, hp = _scales_given_shape(K, Yc)
    return {"kind": kind, "theta": th, "ybar": ybar, "trace": trace, **hp}


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


def predict_local(shape, F_train, Y_train, F_test, groups, k=32, k_max=4000):
    """Predictive mean and variance of a new observation, conditioning each group of test rows
    on its own local training set. ``(mean, var)``, each ``(n_test, S)``.

    ``groups`` labels the test rows (held-out blocks): rows in one group share a conditioning set,
    so one eigendecomposition per group serves every species. The shared shape and per-species
    scales come from ``fit_shared_shape``; nothing is refitted on the local set.
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
    return mean, np.maximum(var, n2[None, :])
