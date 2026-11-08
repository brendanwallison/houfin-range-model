"""population_model.growth_kernel (research prototype): basis, learned forward update, readouts.

The flag-off equality on real data (traced expected_obs at the cp3 params to 1e-10) needs the grid54 inputs; it runs
when GK_REFERENCE_NPZ points at a reference trace written BEFORE the change (research/exp/invasion_sandbox/
growth_kernel_check.py --write_reference) and the process was started with the matching overlay.
"""
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(__file__))
from test_full_model_smoke import synthetic_inputs  # noqa: E402
from src.model import growth_kernel as gkm  # noqa: E402
from src.model.age_forward import forward_sim_age_structured  # noqa: E402

T = 25


def _spec(**kw):
    return gkm.spec_from_config(dict(enabled=True, **kw))


def test_basis_zero_at_floor_and_partition_of_unity():
    spec = _spec(n_basis=8)
    u = jnp.linspace(spec["u_min"] - 1.0, spec["u_max"] + 1.0, 301)
    full = np.asarray(gkm.bspline_basis(u, spec["u_min"], spec["u_max"], 8, drop_first=False))
    assert full.shape == (301, 8) and np.all(full >= -1e-12)
    np.testing.assert_allclose(full.sum(-1), 1.0, atol=1e-12)
    phi = np.asarray(gkm.bspline_basis(jnp.asarray([spec["u_min"]]), spec["u_min"], spec["u_max"], 8))
    np.testing.assert_allclose(phi, 0.0, atol=1e-12)
    # density_basis: empty cell -> exactly the floor -> all density terms 0 (g = low-density rate)
    for di in ("stage", "total"):
        s2 = _spec(density_input=di)
        p = np.asarray(gkm.density_basis(jnp.zeros(3), jnp.zeros(3), s2))
        assert p.shape == (3, gkm.n_density_terms(s2))
        np.testing.assert_allclose(p, 0.0, atol=1e-12)


def test_basis_gradient_finite():
    spec = _spec()
    f = lambda n: jnp.sum(gkm.density_basis(n, 0.5 * n, spec) * jnp.arange(1.0, 15.0))
    for n in (0.0, 1e-9, 1e-3, 0.3, 4.0, 50.0):
        assert np.isfinite(float(jax.grad(f)(jnp.asarray(n))))


def _forward(gk=None, Sa=0.55, Sj=0.3, Fmax=3.0, gamma=1e20):
    d = synthetic_inputs(T=T)
    n = d["land_rows"].shape[0]
    full = lambda v: jnp.full((T, n), v)
    return forward_sim_age_structured(
        full(Sa), full(Sj), full(Fmax), full(1.0), full(0.0), jnp.ones((T, n, d["K"])),
        d["land_rows"], d["land_cols"], d["land_mask"],
        d["adult_fft_kernel"], d["juvenile_fft_kernel_stack"],
        d["adult_edge_correction"], d["juvenile_edge_correction_stack"],
        d["initpop_latent"], jnp.zeros(T), jnp.zeros((1, 1)),
        T, d["inv_locations"][:1], 5, 4.0, 0.0, gamma, yearling_breeding=True, growth_kernel=gk), n


def test_density_independent_kernel_matches_mechanistic_without_dd_or_allee():
    # c = 0 (no Beverton-Holt brake), no survival brake, Allee saturated (gamma huge): the mechanistic update is
    # N_a' = Sa N_a + Sj J_s + J_a, N_j' = Fmax N_a'. A learned kernel with A = 0 and eta = (logit Sa, log Fmax) is the same.
    Sa, Fmax = 0.55, 3.0
    mech, n = _forward(Sa=Sa, Fmax=Fmax)
    spec = _spec(rho_cap=1e4)   # cap far above Fmax: the softcap is exact to ~1e-20
    P = gkm.n_density_terms(spec)
    gk = (jnp.full((T, n), math.log(Sa / (1 - Sa))), jnp.zeros((T, n, P)),
          jnp.full((T, n), math.log(Fmax)), jnp.zeros((T, n, P)), None, None, None, spec)
    learned, _ = _forward(gk=gk, Sa=0.1, Fmax=0.1)   # mechanistic Sa / Fmax inputs must be ignored
    occupied = np.asarray(mech[0]) > 1e-8
    np.testing.assert_allclose(np.asarray(learned[0])[occupied], np.asarray(mech[0])[occupied], rtol=1e-6)
    s_real, rho_real = np.asarray(learned[3]), np.asarray(learned[4])
    assert s_real.shape == (T, n) and rho_real.shape == (T, n)
    np.testing.assert_allclose(s_real, Sa, rtol=1e-12)
    np.testing.assert_allclose(rho_real, Fmax, rtol=1e-12)


def test_interaction_changes_dynamics_and_gradients_finite():
    spec = _spec()
    P = gkm.n_density_terms(spec)

    def total(scale):
        d = synthetic_inputs(T=T)
        n = d["land_rows"].shape[0]
        A_r = jnp.zeros((T, n, P)).at[..., P // 2 - 1].set(-scale)     # crowding: births fall at high adult density
        gk = (jnp.full((T, n), 0.2), jnp.zeros((T, n, P)), jnp.full((T, n), 1.0), A_r, None, None, None, spec)
        return _forward(gk=gk)[0][0][-1].sum()
    assert float(total(3.0)) < float(total(0.0))
    g = jax.grad(total)(1.0)
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_relative_axis_shifts_the_equilibrium_with_k():
    # relative axis only (A = 0): the same births-vs-(N / e^k) shape, so raising k by log 4 raises N* ~4-fold.
    spec = _spec(density_input="total", relative_axis=True)
    P, Q = gkm.n_density_terms(spec), gkm.n_relative_terms(spec)
    d_r = -jnp.linspace(0.0, 8.0, Q)

    def gk_t(k):
        return (jnp.zeros(1), jnp.zeros((1, P)), jnp.asarray([math.log(2.0)]), jnp.zeros((1, P)), jnp.asarray([k]),
                jnp.zeros(Q), d_r, spec)
    n1 = float(gkm.equilibrium_readout(gk_t(-2.0), jnp.asarray([0.5]), n_grid=400)["n_star"][0])
    n2 = float(gkm.equilibrium_readout(gk_t(-2.0 + math.log(4.0)), jnp.asarray([0.5]), n_grid=400)["n_star"][0])
    np.testing.assert_allclose(n2 / n1, 4.0, rtol=0.05)
    # gradient through the forward with respect to k
    d = synthetic_inputs(T=T)
    n = d["land_rows"].shape[0]

    def total(k):
        gk = (jnp.zeros((T, n)), jnp.zeros((T, n, P)), jnp.full((T, n), 1.0), jnp.zeros((T, n, P)),
              jnp.full((T, n), k), jnp.zeros(Q), d_r, spec)
        return _forward(gk=gk)[0][0][-1].sum()
    assert float(total(0.5)) > float(total(-0.5))
    assert np.isfinite(float(jax.grad(total)(0.0)))


def test_equilibrium_readout_on_known_kernel():
    # total-density input, s fixed, births fall linearly in u through the basis: one stable equilibrium.
    spec = _spec(density_input="total")
    P = gkm.n_density_terms(spec)
    eta_s, eta_r = jnp.asarray([0.0, 0.0]), jnp.asarray([math.log(2.0), math.log(0.1)])
    A_r = jnp.broadcast_to(-jnp.linspace(0.5, 6.0, P), (2, P))
    out = gkm.equilibrium_readout((eta_s, jnp.zeros((2, P)), eta_r, A_r, None, None, None, spec), jnp.asarray([0.5, 0.5]))
    lam0 = np.asarray(out["lambda0"])
    np.testing.assert_allclose(lam0, [0.5 + 0.5 * 2.0, 0.5 + 0.5 * 0.1], rtol=1e-6)
    st = np.asarray(out["status"])
    assert st[0] == 0 and st[1] == 1                      # source with an equilibrium; sink everywhere
    ns = float(out["n_star"][0])
    assert np.isfinite(ns) and np.isnan(float(out["n_star"][1]))
    # check: at N* the growth on the ray is ~1
    Bs = jnp.exp(jnp.linspace(spec["u_min"], spec["u_max"], 4000))
    lam, _, rho = gkm.lambda_on_ray((eta_s[:1], jnp.zeros((1, P)), eta_r[:1], A_r[:1], None, None, None, spec),
                                    jnp.asarray([0.5]), Bs)
    tot = np.asarray(Bs * (1 + rho[0]))
    np.testing.assert_allclose(np.interp(np.log(ns), np.log(tot), np.asarray(lam[0])), 1.0, atol=0.02)


@pytest.mark.skipif(not os.environ.get("GK_REFERENCE_NPZ"), reason="needs grid54 inputs + a pre-change reference trace")
def test_flag_off_matches_reference_on_real_data():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "exp", "invasion_sandbox"))
    from growth_kernel_check import trace_expected_obs
    ref = np.load(os.environ["GK_REFERENCE_NPZ"])
    got = trace_expected_obs(os.environ["GK_PARAMS_NPZ"])
    np.testing.assert_allclose(got["expected_obs"], ref["expected_obs"], rtol=1e-10, atol=0)
    np.testing.assert_allclose(got["Na"], ref["Na"], rtol=1e-10, atol=1e-300)
