"""DESK grid autoencoder + spatial residual conv invariants.

Guards the two properties the design depends on:
  1. At gamma=0 (init) the grid model == the pure point-wise model, so the spatial
     residual cannot perturb the ESK-aligned baseline until it earns influence.
  2. The residual is a *masked* (partial) conv: invalid (masked) neighbours do not
     leak into a valid cell's output -- and a plain (non-partial) conv would.
"""
import numpy as np
import torch

from src.community_encoder.train_DESK.model_arch import MultiStreamAutoencoder


def _grid(B, H, W, dims, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = sum(dims)
    return torch.randn(B, H, W, C, generator=g)


def test_gamma_zero_is_pointwise():
    """z at init (gamma=0) equals the model with the spatial term removed."""
    dims = [3, 2]
    torch.manual_seed(0)
    m = MultiStreamAutoencoder(dims, latent_dim=4, spatial_kernel=3).eval()
    assert float(m.gamma.detach()) == 0.0
    x = _grid(1, 6, 5, dims)
    mask = torch.ones(1, 6, 5)
    with torch.no_grad():
        z_full, _ = m(x, mask)
        # Force gamma large: output MUST change (proves the residual path is wired).
        m.gamma.data.fill_(3.0)
        z_active, _ = m(x, mask)
    assert not torch.allclose(z_full, z_active), "spatial residual has no effect even at gamma!=0"
    # And with a k=0 (no-conv) model, the gamma=0 grid output matches it exactly.
    torch.manual_seed(0)
    m0 = MultiStreamAutoencoder(dims, latent_dim=4, spatial_kernel=0).eval()
    with torch.no_grad():
        z_pointwise, _ = m0(x, mask)
    # same init seed -> identical MLP weights; gamma=0 path adds nothing.
    torch.manual_seed(0)
    m_z = MultiStreamAutoencoder(dims, latent_dim=4, spatial_kernel=3).eval()
    with torch.no_grad():
        z_g0, _ = m_z(x, mask)
    assert torch.allclose(z_pointwise, z_g0, atol=1e-6), "gamma=0 must equal the point-wise model"


def test_partial_conv_ignores_masked_neighbours():
    """A valid cell surrounded by masked cells must get the same latent whether those
    neighbours are zero or arbitrary garbage -- i.e. the mask truly excludes them."""
    dims = [3, 2]
    torch.manual_seed(1)
    m = MultiStreamAutoencoder(dims, latent_dim=4, spatial_kernel=3).eval()
    m.gamma.data.fill_(2.0)  # activate the residual
    H = W = 5
    x = _grid(1, H, W, dims, seed=2)
    mask = torch.zeros(1, H, W)
    mask[0, 2, 2] = 1.0  # only the centre cell is valid

    with torch.no_grad():
        z_a, _ = m(x, mask)
        # scramble the (masked) neighbours; the valid centre must be unaffected.
        x2 = x.clone()
        x2[0, :2] += 100.0
        x2[0, 3:] -= 77.0
        z_b, _ = m(x2, mask)
    assert torch.allclose(z_a[0, 2, 2], z_b[0, 2, 2], atol=1e-5), \
        "masked neighbours leaked into the valid cell's latent"


def test_forward_shapes():
    dims = [7, 4, 1]
    m = MultiStreamAutoencoder(dims, latent_dim=8, spatial_kernel=3).eval()
    x = _grid(3, 9, 11, dims)
    mask = (torch.rand(3, 9, 11) > 0.3).float()
    with torch.no_grad():
        z, recon = m(x, mask)
    assert z.shape == (3, 9, 11, 8)
    assert recon.shape == (3, 9, 11, sum(dims))
