"""research/lib/blr.py: the block-prior BLR the research fast arm is built on.

Same entitlement as test_gp_species's algebra tests: the p x p form must agree with the dense GP, the
gradient with finite differences, and a planted difference between block amplitudes must be recovered
-- the fast arm's kernel experiments (B1, B4) are readings of exactly those amplitudes.
"""
import numpy as np
import pytest

from research.lib import blr


def _data(n=300, p1=4, p2=3, S=3, a=(1.0, 0.05), v=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p1 + p2)) + 0.5
    W1 = rng.normal(size=(p1, S)) * np.sqrt(a[0])
    W2 = rng.normal(size=(p2, S)) * np.sqrt(a[1])
    Y = 2.0 + X[:, :p1] @ W1 + X[:, p1:] @ W2 + np.sqrt(v) * rng.normal(size=(n, S))
    return X, Y, [(0, p1), (p1, p1 + p2)]


def test_one_block_predictive_equals_the_dense_gp():
    X, Y, _ = _data()
    m = blr.fit(X, Y, [(0, X.shape[1])])
    Xn = np.random.default_rng(3).normal(size=(6, X.shape[1]))
    mean, var = blr.predict(m, Xn)
    Xc, Xnc = X - X.mean(0), Xn - X.mean(0)
    for s in range(Y.shape[1]):
        a, v = m["a"][s, 0], m["v"][s]
        C = a * Xc @ Xc.T + v * np.eye(len(X))
        k = a * Xnc @ Xc.T
        mu = Y[:, s].mean() + k @ np.linalg.solve(C, Y[:, s] - Y[:, s].mean())
        vv = a * (Xnc * Xnc).sum(1) - np.einsum("ij,ji->i", k, np.linalg.solve(C, k.T)) \
            + v + v / len(X)
        np.testing.assert_allclose(mean[:, s], mu, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(var[:, s], vv, rtol=1e-6, atol=1e-8)


def test_nll_gradient_matches_finite_differences():
    torch = blr._torch()
    X, Y, blocks = _data()
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    XtX, Xty = torch.as_tensor(Xc.T @ Xc), torch.as_tensor(Xc.T @ Yc)
    yy = torch.as_tensor((Yc * Yc).sum(0))
    cols = torch.as_tensor(np.r_[np.zeros(4, int), np.ones(3, int)])
    th0 = np.log(np.array([[0.5, 0.1, 0.2], [1.0, 0.02, 0.3], [0.2, 0.2, 0.05]]))
    th = torch.tensor(th0, requires_grad=True)
    nll, _ = blr._nll(th, XtX, Xty, yy, len(X), cols, torch)
    nll.sum().backward()
    g = th.grad.numpy()
    h = 1e-6
    for s in range(3):
        for j in range(3):
            tp, tm = th0.copy(), th0.copy()
            tp[s, j] += h
            tm[s, j] -= h
            fp = blr._nll(torch.as_tensor(tp), XtX, Xty, yy, len(X), cols, torch)[0][s].item()
            fm = blr._nll(torch.as_tensor(tm), XtX, Xty, yy, len(X), cols, torch)[0][s].item()
            assert g[s, j] == pytest.approx((fp - fm) / (2 * h), rel=1e-4, abs=1e-6)


def test_block_amplitudes_are_recovered_and_their_ratio_is_the_reading():
    """The B1 reading: when one block's true amplitude is 20x the other's, the fit must say so."""
    X, Y, blocks = _data(n=4000, p1=6, p2=6, S=12, a=(1.0, 0.05), v=0.2, seed=1)
    m = blr.fit(X, Y, blocks)
    ratio = np.median(m["a"][:, 0] / m["a"][:, 1])
    assert 6.0 < ratio < 60.0, ratio
    np.testing.assert_allclose(np.median(m["v"]), 0.2, rtol=0.1)


def test_leave_one_out_matches_an_explicit_refit_without_the_row():
    X, Y, blocks = _data(n=120)
    m = blr.fit(X, Y, blocks)
    i = 33
    y_self = np.full_like(Y, np.nan)
    y_self[i] = Y[i]
    loo, _ = blr.predict(m, X, y_self=y_self)
    Xc = X - m["xbar"]
    keep = np.arange(len(X)) != i
    for s in range(Y.shape[1]):
        D = np.column_stack([np.ones(keep.sum()), Xc[keep]])
        prec = np.diag(np.r_[0.0, 1.0 / m["a"][s][m["cols"]]])
        beta = np.linalg.solve(D.T @ D / m["v"][s] + prec, D.T @ Y[keep, s] / m["v"][s])
        np.testing.assert_allclose(loo[i, s], np.r_[1.0, Xc[i]] @ beta, rtol=1e-7, atol=1e-8)


def test_a_species_without_variance_is_floored_not_fitted():
    X, Y, blocks = _data()
    Y = np.hstack([Y, np.zeros((len(Y), 1))])
    m = blr.fit(X, Y, blocks)
    assert not m["ok"][-1] and m["ok"][:-1].all() and np.isnan(m["nll"][-1])
    mean, var = blr.predict(m, X[:4])
    assert np.allclose(mean[:, -1], 0.0) and np.isfinite(var).all()


def test_numerically_indefinite_A_is_jittered_for_the_failing_species_only():
    # X'X with an eigenvalue spread far beyond float64 (near-collinear features) at a tiny v and large a:
    # A = X'X / v + diag(1/a) is numerically indefinite and Cholesky used to raise; a well-conditioned
    # species in the same batch must come out exactly as without the fallback
    torch = blr._torch()
    rng = np.random.default_rng(0)
    p = 40
    U, _ = np.linalg.qr(rng.normal(size=(p, p)))
    ev = np.r_[1e12, 1e11, np.full(p - 2, 1e-30)]
    XtX = torch.as_tensor((U * ev) @ U.T)
    Xty = torch.as_tensor(rng.normal(size=(p, 2)))
    yy = torch.as_tensor(np.array([10.0, 10.0]))
    cols = torch.as_tensor(np.zeros(p, int))
    th = torch.as_tensor(np.array([[np.log(1e10), np.log(1e-8)],      # ill-conditioned
                                   [np.log(1e-3), np.log(1e14)]]))    # well-conditioned
    with pytest.raises(Exception):
        torch.linalg.cholesky(XtX.unsqueeze(0) / 1e-8 + torch.eye(p, dtype=XtX.dtype) * 1e-10)
    nll, _ = blr._nll(th, XtX, Xty, yy, 100, cols, torch)
    assert torch.isfinite(nll).all()
    nll1, _ = blr._nll(th[1:], XtX, Xty[:, 1:], yy[1:], 100, cols, torch)
    assert nll[1].item() == nll1[0].item()
