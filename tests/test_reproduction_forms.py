"""reproduction_age_structured: adult-only (default), yearling_breeding, and Veit & Lewis forms."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from src.model.age_forward import reproduction_age_structured as repro  # noqa: E402

Sa, Sj, Fmax, K, gamma = 0.6, 0.4, 5.0, 10.0, 2.0


def _c(yearling):
    return max(Fmax * (Sj if yearling else Sa * Sj) / (1.0 - Sa) - 1.0, 0.0)


def _iterate(n_years=400, **kw):
    Na, Nj = jnp.array(0.5), jnp.array(0.5)
    for _ in range(n_years):
        Na, Nj = repro(Na, Nj, jnp.array(0.0), Sa, Sj, Fmax, K, kw["c"],
                       gamma, allee_off=jnp.array(1.0),
                       yearling_breeding=kw.get("yearling_breeding", False), veit=kw.get("veit", False))
    return float(Na), float(Nj)


def test_default_is_adult_only_breeding():
    Na, Nj = repro(jnp.array(1e-6), jnp.array(1e-6), jnp.array(0.0), Sa, Sj, Fmax, K, _c(False), gamma,
                   allee_off=jnp.array(1.0))
    np.testing.assert_allclose(float(Nj), 1e-6 * Sa * Fmax, rtol=1e-4)


def test_yearlings_breed_and_low_density_lambda():
    # births from all winter survivors: lambda0 = Sa + Sj*Fmax on the census of breeders
    Na, Nj = repro(jnp.array(1e-6), jnp.array(1e-6), jnp.array(0.0), Sa, Sj, Fmax, K, _c(True), gamma,
                   allee_off=jnp.array(1.0), yearling_breeding=True)
    np.testing.assert_allclose(float(Na), 1e-6 * (Sa + Sj), rtol=1e-9)
    np.testing.assert_allclose(float(Nj), float(Na) * Fmax, rtol=1e-4)


def test_yearling_equilibrium_is_K():
    # c re-solved for yearling breeding puts the no-dispersal equilibrium at K (pre-survival census)
    Na, Nj = _iterate(c=_c(True), yearling_breeding=True)
    np.testing.assert_allclose(Na + Nj, K, rtol=1e-6)


def test_veit_limits():
    # mates not limiting (exemption): P = B/2; few breeders -> sites not limiting, births ~ Fmax*B
    B_small = 1e-4
    Na, Nj = repro(jnp.array(B_small / Sa), jnp.array(0.0), jnp.array(0.0), Sa, Sj, Fmax, K, 0.0, gamma,
                   allee_off=jnp.array(1.0), veit=True)
    np.testing.assert_allclose(float(Nj), Fmax * float(Na), rtol=1e-3)
    # many breeders -> nest sites saturate at K/2 pairs, births -> Fmax*K
    Na, Nj = repro(jnp.array(1e6), jnp.array(0.0), jnp.array(0.0), Sa, Sj, Fmax, K, 0.0, gamma,
                   veit=True)
    np.testing.assert_allclose(float(Nj), Fmax * K, rtol=1e-3)


def test_veit_mate_finding_half_at_n50():
    # r = 2*n50: half of breeders are mated at B = n50 = ln2/gamma
    n50 = np.log(2.0) / gamma
    big_K = 1e12
    Na, Nj = repro(jnp.array(n50 / Sa), jnp.array(0.0), jnp.array(0.0), Sa, Sj, Fmax, big_K, 0.0, gamma,
                   veit=True)
    np.testing.assert_allclose(float(Nj) / (Fmax * float(Na)), 0.5, rtol=1e-6)


def test_veit_equilibrium_emergent():
    # equilibrium solves Sa + Sj * 2*Fmax*H(B)/B = 1 with no mate limitation
    Na, Nj = _iterate(c=0.0, veit=True)
    B = Sa * Na + Sj * Nj
    H = (B / 2) / (1 + B / K)
    np.testing.assert_allclose(Sa + Sj * 2 * Fmax * H / B, 1.0, rtol=1e-6)
