"""Bayesian linear regression with BLOCK priors, per species, by marginal likelihood: the fast arm's engine.

    y_s = mu_s + sum_b X_b w_{s,b} + eps,    w_{s,b} ~ N(0, a_{s,b} I),    eps ~ N(0, v_s)

The equivalent GP kernel is ``sum_b a_{s,b} X_b X_b'``: one linear kernel per block, each with its own
amplitude. One block is the iid-across-features prior the age model puts on ``w_env`` (the uncentered
Ruzicka GP contract; ``train_DESK.gp_kernels.fit`` is that case). More blocks let an experiment ask
whether the data want a different amplitude for, e.g., a cell's mean z and its within-cell deviation
(B1), or for a continental time basis (B4) -- each such kernel is one the age model could adopt with an
iid prior per block.

Everything is in the p x p form (p = total features, small), from sufficient statistics X'X (shared),
X'y and y'y per species; n never enters after they are formed:

    A_s = X'X / v_s + diag(1 / a_s)
    2 nll_s = n log v_s + sum_j log a_sj + log|A_s| + (y'y - y'X A_s^-1 X'y / v_s) / v_s

All species at once (batched Cholesky in torch, float64), autograd for the gradient, L-BFGS-B on the
summed objective (species are independent), several starting signal shares with each species keeping
its best (the scale likelihood is multimodal -- see gp_kernels.SCALE_STARTS). The intercept is a
plug-in: y and X are centred at their training means.
"""
import numpy as np

REL_FLOOR = 1e-6
STARTS = (0.5, 0.1, 0.9)


def _torch():
    # Never torch.set_default_dtype here: it is process-wide, and setting float64 broke an unrelated
    # float32 model later in the same test run. Every tensor below takes its dtype explicitly.
    import torch
    return torch


def _nll(theta, XtX, Xty, yy, n, cols, torch):
    """Per-species nll. ``theta`` (S, B+1) = [log a_1..a_B, log v]; ``cols`` maps column -> block."""
    S = theta.shape[0]
    log_a = theta[:, :-1][:, cols]                                  # (S, p)
    log_v = theta[:, -1]                                            # (S,)
    v = torch.exp(log_v)
    A = XtX.unsqueeze(0) / v[:, None, None] + torch.diag_embed(torch.exp(-log_a))
    L = torch.linalg.cholesky(A)
    sol = torch.cholesky_solve(Xty.T.unsqueeze(-1), L).squeeze(-1)  # (S, p) = A^-1 X'y
    quad = (Xty.T * sol).sum(1)                                     # y'X A^-1 X'y
    logdetA = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
    return 0.5 * (n * log_v + log_a.sum(1) + logdetA + (yy - quad / v) / v), S


def fit(X, Y, blocks, max_iter=300, starts=STARTS):
    """Fit every species. ``X (n, p)``, ``Y (n, S)``, ``blocks`` = [(start, stop), ...] covering p.

    Returns ``{xbar, ybar, blocks, cols, a (S, B), v (S,), coef (S, p), Ainv (S, p, p), nll (S,), n}``.
    Species with no variance get floors and ``ok=False``.
    """
    from scipy.optimize import minimize
    torch = _torch()
    X = np.asarray(X, "float64")
    Y = np.asarray(Y, "float64")
    n, p = X.shape
    B = len(blocks)
    cols = np.concatenate([np.full(e - s, b) for b, (s, e) in enumerate(blocks)])
    assert len(cols) == p, "blocks must cover every column exactly once"
    xbar, ybar = X.mean(0), Y.mean(0)
    Xc, Yc = X - xbar, Y - ybar
    XtX = torch.as_tensor(Xc.T @ Xc)
    Xty = torch.as_tensor(Xc.T @ Yc)                                # (p, S)
    yy = torch.as_tensor((Yc * Yc).sum(0))
    var_y = (Yc * Yc).sum(0) / max(n - 1, 1)
    ok = var_y > 1e-14
    floor = np.maximum(REL_FLOOR * var_y, 1e-14)
    cols_t = torch.as_tensor(cols)
    S = Y.shape[1]
    # mean prior variance contributed per unit amplitude, per block (kappa_b)
    kap = np.array([np.trace((Xc[:, s:e].T @ Xc[:, s:e])) / max(n - 1, 1) for s, e in blocks])
    lo = np.log(floor)

    def run(x0):
        def f(x):
            th = torch.tensor(x.reshape(S, B + 1), requires_grad=True)
            nll, _ = _nll(th, XtX, Xty, yy, n, cols_t, torch)
            tot = nll[torch.as_tensor(ok)].sum()
            tot.backward()
            g = th.grad.numpy().copy()
            g[~ok] = 0.0
            return float(tot), g.reshape(-1)
        bnds = []
        for s in range(S):
            bnds += [(lo[s], 30.0)] * B + [(lo[s], 30.0)]
        res = minimize(f, x0.reshape(-1), jac=True, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": int(max_iter)})
        th = res.x.reshape(S, B + 1)
        with torch.no_grad():
            nll, _ = _nll(torch.as_tensor(th), XtX, Xty, yy, n, cols_t, torch)
        return th, nll.numpy()

    best_th, best_nll = None, None
    for share in starts:
        x0 = np.empty((S, B + 1))
        for b in range(B):
            x0[:, b] = np.log(np.maximum(share * var_y / B / max(kap[b], 1e-12), floor))
        x0[:, B] = np.log(np.maximum((1.0 - share) * var_y, floor))
        th, nll = run(x0)
        if best_th is None:
            best_th, best_nll = th, nll
        else:
            better = nll < best_nll - 1e-9
            best_th[better], best_nll[better] = th[better], nll[better]
    best_th[~ok] = np.log(floor[~ok])[:, None]
    th_t = torch.as_tensor(best_th)
    with torch.no_grad():
        v = torch.exp(th_t[:, -1])
        A = XtX.unsqueeze(0) / v[:, None, None] + torch.diag_embed(torch.exp(-th_t[:, :-1][:, cols_t]))
        Ainv = torch.linalg.inv(A)
        coef = (Ainv @ Xty.T.unsqueeze(-1)).squeeze(-1) / v[:, None]
    return {"xbar": xbar, "ybar": ybar, "blocks": list(blocks), "cols": cols,
            "a": np.exp(best_th[:, :B]), "v": np.exp(best_th[:, B]), "coef": coef.numpy(),
            "Ainv": Ainv.numpy(), "nll": np.where(ok, best_nll, np.nan), "ok": ok, "n": n}


def predict(model, Xn, y_self=None, chunk=16, device=None):
    """Predictive mean and variance of a new observation; exact LOO where ``y_self`` is finite.

    ``var = x A^-1 x' + v (1 + 1/n)``; the in-sample hat diagonal is ``h = x A^-1 x' / v + 1/n`` and the
    LOO mean is ``y - (y - yhat) / (1 - h)``. Species are processed in chunks so the
    (chunk, m, p) quadratic-form intermediate stays bounded.
    """
    torch = _torch()
    dev = device or "cpu"
    Xc = torch.as_tensor(np.asarray(Xn, "float64") - model["xbar"], device=dev)
    mean = Xc @ torch.as_tensor(model["coef"].T, device=dev) + torch.as_tensor(model["ybar"],
                                                                            device=dev)
    S = model["coef"].shape[0]
    quad = torch.empty((Xc.shape[0], S), device=dev, dtype=torch.float64)
    for s0 in range(0, S, chunk):
        Ai = torch.as_tensor(model["Ainv"][s0:s0 + chunk], device=dev)        # (c, p, p)
        quad[:, s0:s0 + chunk] = torch.einsum("mp,cpq,mq->mc", Xc, Ai, Xc)
    v = torch.as_tensor(model["v"], device=dev)
    var = quad + v * (1.0 + 1.0 / model["n"])
    mean, var = mean.cpu().numpy(), var.cpu().numpy()
    if y_self is not None:
        ys = np.asarray(y_self, "float64")
        rows = np.isfinite(ys).all(1)
        if rows.any():
            h = np.minimum(quad.cpu().numpy()[rows] / model["v"][None, :] + 1.0 / model["n"],
                           1.0 - 1e-9)
            mean[rows] = ys[rows] - (ys[rows] - mean[rows]) / (1.0 - h)
    return mean, var
