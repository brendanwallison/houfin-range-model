"""Output-EMA DESK objective: the learned causal EMA over the year axis.

Verifies the scan is causal, differentiable, reproduces the closed-form exponential
response for a known half-life, and that the half-life reparam stays in bounds. Also
checks the numpy cube-side EMA (NaN-persisting) matches the torch scan on clean data.
"""
import numpy as np
import torch

from src.community_encoder.train_DESK.desk_training import OutputEMA, apply_output_ema


def test_half_life_reparam_in_bounds():
    assert abs(OutputEMA(1.0, 40.0, 8.0).half_life().item() - 8.0) < 1e-3   # interior inits exactly
    for init in (1.0, 8.0, 40.0):
        ema = OutputEMA(1.0, 40.0, init)
        assert 1.0 <= ema.half_life().item() <= 40.0
        assert abs(ema.half_life().item() - init) < 0.05      # inits ~at requested (sigmoid clamp at edges)
    # extreme theta still clamps within [hl_min, hl_max]
    ema = OutputEMA(1.0, 40.0, 8.0)
    with torch.no_grad():
        ema.theta.fill_(100.0)
    assert ema.half_life().item() <= 40.0 + 1e-4
    with torch.no_grad():
        ema.theta.fill_(-100.0)
    assert ema.half_life().item() >= 1.0 - 1e-4


def test_scan_is_causal():
    # z_ema[t] must not depend on any future z_raw[t'] (t' > t).
    ema = OutputEMA(1.0, 40.0, 5.0)
    z = torch.randn(10, 3, 4)
    base = ema(z)
    z2 = z.clone()
    z2[7:] += 100.0                                            # perturb the future
    out = ema(z2)
    assert torch.allclose(base[:7], out[:7], atol=1e-6)       # past unchanged
    assert not torch.allclose(base[7], out[7])                # present changes


def test_step_response_matches_closed_form():
    # Step input (0 then 1) -> z_ema[t] = 1 - (1-a)^t, a = 1 - 2^{-1/h}.
    h = 6.0
    ema = OutputEMA(1.0, 40.0, h)
    T = 30
    z = torch.ones(T, 1)
    z[0] = 0.0
    out = ema(z).squeeze(-1).detach().numpy()
    a = 1.0 - 2.0 ** (-1.0 / h)
    t = np.arange(T)
    closed = 1.0 - (1.0 - a) ** t
    assert np.allclose(out, closed, atol=1e-5)
    # half-life sanity: value at t=h is ~1/2 of the way from its t=0 residual
    assert abs((1.0 - out[int(h)]) - 0.5) < 0.05


def test_scan_is_differentiable_wrt_half_life():
    ema = OutputEMA(1.0, 40.0, 8.0)
    z = torch.randn(12, 2, 2)
    out = ema(z)
    out.pow(2).mean().backward()
    assert ema.theta.grad is not None and torch.isfinite(ema.theta.grad).all()
    assert ema.theta.grad.abs().item() > 0                    # half-life actually moves the loss


def test_numpy_ema_matches_torch():
    # apply_output_ema is the inference-side twin of the torch scan (validate_bbs_routes
    # grades z_ema, since that is what the trainer supervised). On all-valid data the two
    # must agree exactly -- this is the assertion that keeps them from drifting apart.
    h = 7.0
    T, H, W, L = 15, 3, 3, 4
    rng = np.random.default_rng(0)
    raws = np.stack([rng.standard_normal((H, W, L)).astype("float32") for _ in range(T)])

    np_out = apply_output_ema(raws, h)
    torch_out = OutputEMA(1.0, 40.0, h)(torch.tensor(raws)).detach().numpy()
    assert np.allclose(np_out, torch_out, atol=1e-5)


def test_numpy_ema_persists_state_through_invalid_years():
    # An invalid year must PERSIST the prior EMA, not overwrite it with the raw value and
    # not poison it with NaN. Without this, a single gap year would corrupt every later
    # year at that location.
    h = 5.0
    a = 1.0 - 2.0 ** (-1.0 / h)
    raw = np.array([[1.0], [99.0], [1.0]], dtype="float32")     # (T=3, L=1)
    valid = np.array([True, False, True])

    out = apply_output_ema(raw, h, valid=valid)
    assert out[0, 0] == 1.0                                     # first observed seeds the state
    assert out[1, 0] == 1.0                                     # invalid year ignored entirely
    assert abs(out[2, 0] - (a * 1.0 + (1.0 - a) * 1.0)) < 1e-6  # blends against the PERSISTED 1.0

    # The invalid value must never reach the output, even transiently.
    assert not np.isclose(out, 99.0).any()
    assert np.isfinite(out).all()


def test_numpy_ema_first_observed_year_seeds_state():
    # A location whose early years are all invalid must initialize from its first OBSERVED
    # value (z_ema = z_raw there), not stay NaN forever.
    raw = np.array([[5.0], [5.0], [7.0]], dtype="float32")
    valid = np.array([False, False, True])
    out = apply_output_ema(raw, 5.0, valid=valid)
    assert np.isnan(out[0, 0]) and np.isnan(out[1, 0])          # nothing observed yet
    assert out[2, 0] == 7.0                                     # seeds, does not blend
