"""population_model.occupancy (E063): psi / conditional-density closure in forward_sim_age_structured."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(__file__))
from test_full_model_smoke import synthetic_inputs  # noqa: E402
from src.model.age_forward import forward_sim_age_structured  # noqa: E402

T = 30


def _run(seed_scale=1.0, Fmax=4.0, occupancy_b=None, n50_birds=None):
    d = synthetic_inputs(T=T)
    n = d["land_rows"].shape[0]
    Sa, Sj = 0.6, 0.3
    c = max(Fmax * Sa * Sj / (1 - Sa) - 1, 0.0)
    full = lambda v: jnp.full((T, n), v)
    gamma = jnp.log(2.0) / (0.02 if n50_birds is None else n50_birds)
    return forward_sim_age_structured(
        full(Sa), full(Sj), full(Fmax), full(1.0), full(c), jnp.ones((T, n, d["K"])),
        d["land_rows"], d["land_cols"], d["land_mask"],
        d["adult_fft_kernel"], d["juvenile_fft_kernel_stack"],
        d["adult_edge_correction"], d["juvenile_edge_correction_stack"],
        d["initpop_latent"] * seed_scale, jnp.zeros(T), jnp.zeros((1, 1)),
        T, d["inv_locations"][:1], 5, 4.0, 0.0, gamma, occupancy_b=occupancy_b)


def test_large_b_nests_the_mean_field_model():
    off = _run()
    on = _run(occupancy_b=1e16)   # pairing loses ~sqrt(2 / (pi b m)) of breeders at finite b
    np.testing.assert_allclose(np.asarray(on[0]), np.asarray(off[0]), rtol=1e-5, atol=1e-12)
    assert np.all(np.asarray(on[3])[-1][np.asarray(off[0])[-1] > 1e-6] > 0.999)


def test_psi_bounded_density_nonnegative_and_gradients_finite():
    tot, _, _, psi = _run(occupancy_b=50.0)
    psi = np.asarray(psi)
    assert np.all((psi >= 0) & (psi <= 1)) and np.all(np.asarray(tot) >= 0)
    g = jax.grad(lambda F: _run(Fmax=F, occupancy_b=50.0)[0].sum())(4.0)
    assert np.isfinite(float(g))


def test_sub_bird_seed_breeds_no_more_than_mean_field():
    # With independent (Poisson) arrivals, a seed of ~0.1-0.2 birds per cell leaves occupied
    # realisations holding mostly ONE bird, which cannot pair; the mean-field model lets fractional
    # birds breed. So occupancy must not establish such a seed faster (luck needs clumped arrivals).
    kw = dict(seed_scale=1e-3, n50_birds=0.05)
    off = float(_run(**kw)[0][-1].sum())
    on = float(_run(occupancy_b=200.0, **kw)[0][-1].sum())
    assert on <= off * 1.01


def test_pairing_fraction_limits():
    from src.model.age_forward import pairing_fraction
    f = np.asarray(pairing_fraction(jnp.array([1.0, 1.5, 3.0, 10.0, 1e4])))
    assert f[0] < 1e-3                       # a lone bird cannot breed
    assert np.all(np.diff(f) > 0)            # more birds, more pairing
    assert abs(f[-1] - (1 - np.sqrt(2 / (np.pi * 1e4)))) < 1e-3   # Skellam large-lam limit -> 1


def test_lone_birds_do_not_found_populations():
    # b = 20: the sparse seed is ~0.02 birds per cell -- realisations hold single birds, which must not breed.
    kw = dict(seed_scale=1e-3, n50_birds=0.05)
    seed = float(_run(**kw)[0][0].sum())
    assert float(_run(occupancy_b=20.0, **kw)[0][-1].sum()) < seed
