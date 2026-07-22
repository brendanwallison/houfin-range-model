"""Output-EMA DESK objective: the learned causal EMA over the year axis.

Verifies the scan is causal, differentiable, reproduces the closed-form exponential
response for a known half-life, and that the half-life reparam stays in bounds. Also
checks the numpy cube-side EMA (NaN-persisting) matches the torch scan on clean data.
"""
import numpy as np
import torch

from src.community_encoder.train_DESK.desk_training import OutputEMA


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


def test_numpy_cube_ema_matches_torch():
    # The cube applies the EMA in numpy (NaN-persist); on all-valid data it must
    # match the torch scan exactly.
    h = 7.0
    a = 1.0 - 2.0 ** (-1.0 / h)
    T, H, W, L = 15, 3, 3, 4
    rng = np.random.default_rng(0)
    raws = [rng.standard_normal((H, W, L)).astype("float32") for _ in range(T)]

    z_ema = np.full((H, W, L), np.nan, dtype=np.float32)
    cube = []
    for raw in raws:
        seeded = ~np.isnan(z_ema).any(axis=-1)
        blend = np.where(seeded[..., None], a * raw + (1.0 - a) * z_ema, raw)
        z_ema = blend
        cube.append(z_ema.copy())
    cube = np.stack(cube)                                      # (T,H,W,L)

    torch_out = OutputEMA(1.0, 40.0, h)(torch.tensor(np.stack(raws))).detach().numpy()
    assert np.allclose(cube, torch_out, atol=1e-5)
