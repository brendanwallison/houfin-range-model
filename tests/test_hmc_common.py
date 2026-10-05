"""The posterior-inference machinery must be exact where it claims to be.

Toy models only; no data or GPU. What each test pins:

* chunked sampling that is interrupted and resumed in a NEW kernel/MCMC object (i.e. a
  new SLURM job) draws exactly what an uninterrupted run draws;
* deterministic sites never reach stored samples (the age model's are full-grid arrays);
* a MAP ``_auto_loc`` dict becomes a valid unconstrained init, and jittered starts
  respect constrained supports;
* our flattening order is numpyro's dense-mass order, so a Laplace matrix lands on the
  right coordinates;
* the chunked, resumable dense Hessian gives the Gaussian's precision exactly;
* NeuTra can be rebuilt from a saved low-rank guide of any rank.
"""
import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("numpyro")
pytest.importorskip("jax")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as np                                                # noqa: E402
import numpyro                                                    # noqa: E402
import numpyro.distributions as dist                              # noqa: E402
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO             # noqa: E402
from numpyro.infer.autoguide import AutoDelta, AutoLowRankMultivariateNormal  # noqa: E402

from src.model import hmc_common as hc                           # noqa: E402

Y = jnp.array([0.1, -0.3, 0.7, 0.2, 1.1, -0.4])


def toy(data, prior_scale=1.0):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0 * prior_scale))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([2]))
    numpyro.deterministic("big", jnp.ones((50, 50)) * mu)
    numpyro.sample("obs", dist.Normal(mu + b.sum(), sigma), obs=data["y"])


KW = {"data": {"y": Y}, "prior_scale": 1.0}


def _chunks(out_dir):
    samples, extra, _, n = hc.load_chain(out_dir)
    return samples, extra, n


def test_chunked_resume_matches_uninterrupted_run(tmp_path):
    model = hc.hide_deterministics(toy)
    z0, _, _ = hc.unconstrained_setup(model, KW["data"])
    key = jax.random.PRNGKey(3)

    ref = MCMC(NUTS(model), num_warmup=30, num_samples=20, progress_bar=False)
    ref.run(key, init_params=z0, **KW)
    expected = ref.get_samples()

    common = dict(out_dir=str(tmp_path), run_fingerprint="fp", payload={}, num_warmup=30,
                  chunk=10, rng_key=key, kwargs=KW, init_params=z0, log=lambda *_: None)
    # "job 1": warmup + one chunk, then the wall clock ends it.
    hc.run_chunked_mcmc(NUTS(model), num_samples=10, **common)
    # "job 2": a fresh kernel and MCMC object resume from the pickled state.
    hc.run_chunked_mcmc(NUTS(model), num_samples=20, **common)

    got, extra, n = _chunks(str(tmp_path))
    assert n == 2
    for k in expected:
        np.testing.assert_allclose(got[k], np.asarray(expected[k]), rtol=1e-5, atol=1e-6)
    assert set(extra) >= {"num_steps", "diverging", "energy", "adapt_state.step_size"}


def test_resume_refuses_a_different_fingerprint(tmp_path):
    model = hc.hide_deterministics(toy)
    z0, _, _ = hc.unconstrained_setup(model, KW["data"])
    common = dict(out_dir=str(tmp_path), payload={}, num_warmup=5, num_samples=5, chunk=5,
                  rng_key=jax.random.PRNGKey(0), kwargs=KW, init_params=z0,
                  log=lambda *_: None)
    hc.run_chunked_mcmc(NUTS(model), run_fingerprint="a", **common)
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        hc.run_chunked_mcmc(NUTS(model), run_fingerprint="b", **common)


def test_deterministics_are_not_collected():
    mcmc = MCMC(NUTS(hc.hide_deterministics(toy)), num_warmup=5, num_samples=5,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), **KW)
    assert set(mcmc.get_samples()) == {"mu", "sigma", "b"}


def test_map_params_become_a_valid_unconstrained_init_and_jitter_respects_supports():
    svi = SVI(toy, AutoDelta(toy), numpyro.optim.Adam(0.05), loss=Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(0), **KW)
    for _ in range(50):
        state, _ = svi.update(state, **KW)
    from src.model.checkpoints import auto_delta_params_to_latents
    latents = auto_delta_params_to_latents(svi.get_params(state))

    model = hc.hide_deterministics(toy)
    z, potential_fn, postprocess = hc.unconstrained_setup(model, KW["data"], latents)
    back = postprocess(z)
    for k, v in latents.items():
        np.testing.assert_allclose(back[k], v, rtol=1e-5)
    assert np.isfinite(potential_fn(z))

    zj = hc.jitter_unconstrained(z, jax.random.PRNGKey(1), 3.0, None)
    assert float(postprocess(zj)["sigma"]) > 0
    assert not np.allclose(zj["mu"], z["mu"])


def test_flatten_order_is_numpyros_dense_mass_order():
    model = hc.hide_deterministics(toy)
    z, _, _ = hc.unconstrained_setup(model, KW["data"])
    names, x, _ = hc.flatten_sorted(z)
    assert names == ("b", "mu", "sigma") and x.size == 4
    M = np.diag([1.0, 2.0, 3.0, 4.0])
    kernel = NUTS(model, dense_mass=True, inverse_mass_matrix={names: jnp.asarray(M)},
                  adapt_mass_matrix=False)
    mcmc = MCMC(kernel, num_warmup=3, num_samples=2, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), init_params=z, **KW)
    used = mcmc.last_state.adapt_state.inverse_mass_matrix
    np.testing.assert_allclose(np.asarray(used[names]), M)


def test_dense_hessian_recovers_gaussian_precision_and_resumes(tmp_path):
    cov = np.array([[2.0, 0.6, 0.0], [0.6, 1.0, -0.3], [0.0, -0.3, 0.5]])
    s = np.array([0.4])

    def gauss(data=None, prior_scale=1.0):
        numpyro.sample("x", dist.MultivariateNormal(jnp.zeros(3), jnp.asarray(cov)))
        numpyro.sample("y", dist.Normal(0.0, jnp.asarray(s)))

    z, potential_fn, _ = hc.unconstrained_setup(gauss, None)
    part = str(tmp_path / "h.npz")
    names, _, H1 = hc.dense_hessian(potential_fn, z, chunk=3, partial_path=part, save_every=1,
                                    log=lambda *_: None)
    # Simulate a wall-clock kill after the first column block: the partial file says 3 of 4
    # columns are done and the 4th is still unknown. The rerun must fill only that column.
    saved = dict(np.load(part))
    saved["H"][:, 3:] = np.nan
    np.savez(part, H=saved["H"], x0=saved["x0"], done=3)
    logs = []
    names, _, H = hc.dense_hessian(potential_fn, z, chunk=3, partial_path=part, save_every=1,
                                   log=logs.append)
    assert any("resuming at column 3/4" in m for m in logs)
    expected = np.zeros((4, 4))
    expected[:3, :3] = np.linalg.inv(cov)
    expected[3, 3] = 1 / s[0] ** 2
    assert names == ("x", "y")
    np.testing.assert_allclose(H, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(H1, H)
    inv_mass, _, _, report = hc.laplace_inverse_mass(H, eig_floor=1e-3)
    np.testing.assert_allclose(inv_mass[:3, :3], cov, rtol=1e-4, atol=1e-5)
    assert report["n_negative"] == 0 and report["n_below_floor"] == 0


def test_laplace_softabs_keeps_the_scale_of_negative_curvature():
    """A saddle axis with curvature -1 gets width 1 (|lambda|), not the floor's 10."""
    H = np.diag([4.0, -1.0, 1e-6])
    inv_mass, evals, _, report = hc.laplace_inverse_mass(H, eig_floor=0.01)
    assert report["n_negative"] == 1 and report["n_below_floor"] == 1
    np.testing.assert_allclose(np.diag(inv_mass), [0.25, 1.0, 100.0])
    assert report["condition_number_clipped"] == pytest.approx(4.0 / 0.01)


@pytest.mark.parametrize("rank", [1, 2])
def test_neutra_rebuilds_from_a_saved_guide_of_any_rank(rank):
    from src.model.age_run_hmc import build_neutra
    guide = AutoLowRankMultivariateNormal(toy, rank=rank)
    svi = SVI(toy, guide, numpyro.optim.Adam(0.01), loss=Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(0), **KW)
    for _ in range(20):
        state, _ = svi.update(state, **KW)
    vi = {"format_version": 2, "rank": rank, "params": svi.get_params(state)}
    vi = pickle.loads(pickle.dumps(jax.device_get(vi)))  # as loaded from disk

    neutra, warped, to_constrained = build_neutra(toy, vi, KW)
    mcmc = MCMC(NUTS(warped), num_warmup=10, num_samples=5, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(1), init_params={"auto_shared_latent": jnp.zeros(4)}, **KW)
    s = mcmc.get_samples()
    assert set(s) == {"auto_shared_latent"}
    back = to_constrained(s["auto_shared_latent"])
    assert set(back) == {"mu", "sigma", "b"} and (np.asarray(back["sigma"]) > 0).all()
    assert back["b"].shape == (5, 2)
    # Same values numpyro's own (model-replaying) transform gives, minus deterministics.
    ref = neutra.transform_sample(s["auto_shared_latent"])
    for k in back:
        np.testing.assert_allclose(back[k], ref[k], rtol=1e-5, atol=1e-6)


def test_neutra_refuses_a_guide_from_a_different_model():
    from src.model.age_run_hmc import build_neutra
    vi = {"format_version": 2, "rank": 1,
          "params": {"auto_loc": jnp.zeros(7), "auto_cov_factor": jnp.zeros((7, 1)),
                     "auto_scale": jnp.ones(7)}}
    with pytest.raises(RuntimeError, match="latents"):
        build_neutra(toy, vi, KW)



def test_release_cells_skip_the_allee_factor_only_while_exempt():
    """allee_off=1 restores full fecundity; partial values interpolate; None is legacy."""
    from src.model.age_forward import reproduction_age_structured
    N = jnp.array([1e-3, 1e-3])
    args = (N * 0.5, N * 0.25, N * 0.25, 0.6, 0.4, 3.0, 1.0, 0.5, 2.0)
    _, juv_allee = reproduction_age_structured(*args)
    _, juv_off = reproduction_age_structured(*args, allee_off=jnp.array([1.0, 0.0]))
    _, juv_legacy = reproduction_age_structured(*args, allee_off=None)
    np.testing.assert_allclose(juv_legacy, juv_allee)
    np.testing.assert_allclose(juv_off[1], juv_allee[1])           # not exempt: unchanged
    assert juv_off[0] > 100 * juv_allee[0]                          # exempt: mates not limiting


def test_quality_branch_is_static_under_jit():
    """build_model_2d's site-existence decision must not use jnp (TACC job 3480304)."""
    import inspect
    from src.model import age_priors
    src = inspect.getsource(age_priors.build_model_2d)
    assert "int(jnp.max(obs_quality))" not in src and "n_obs_quality_tiers" in src


def static_shape_model(data, prior_scale=1.0):
    """Mixes the two kinds of data the age model has: arrays, and Python ints used as shapes."""
    w = numpyro.sample("w", dist.Normal(0.0, 1.0).expand([data["n"]]))
    numpyro.sample("obs", dist.Normal(data["x"] @ w, 1.0), obs=data["y"])


def _static_shape_data():
    x = jax.random.normal(jax.random.PRNGKey(5), (200, 3))
    return {"x": x, "y": x @ jnp.array([1.0, -0.5, 0.2]), "n": 3, "label": "toy"}


def test_split_data_keeps_python_metadata_static():
    arrays, static = hc.split_data(_static_shape_data())
    assert set(arrays) == {"x", "y"} and static == {"n": 3, "label": "toy"}


def test_sampler_takes_data_as_arguments_not_compiled_constants(tmp_path):
    """TACC job 3486435: closing over the data compiled 2.68 GB of constants into every
    chunk's recompile, until host RAM ran out. JAX warns when lowering captures constants
    above jax_captured_constants_warn_bytes; with the data passed as arguments it must not."""
    import warnings
    data = _static_shape_data()
    arrays, static = hc.split_data(data)
    model = hc.hide_deterministics(hc.array_model(static_shape_model, static))
    kwargs = {"arrays": arrays}
    z, _, _ = hc.unconstrained_setup(model, None, kwargs=kwargs)
    old = jax.config.jax_captured_constants_warn_bytes
    jax.config.update("jax_captured_constants_warn_bytes", 1024)    # x alone is 2.4 kB
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hc.run_chunked_mcmc(NUTS(model), out_dir=str(tmp_path), run_fingerprint="fp",
                                payload={}, num_warmup=10, num_samples=10, chunk=5,
                                rng_key=jax.random.PRNGKey(0), kwargs=kwargs, init_params=z,
                                log=lambda *_: None)
        captured = [w for w in caught if "constants were captured" in str(w.message)]
        assert not captured, captured[0].message
        # ...and the same sampler with the data closed over DOES trip it (the bug).
        closed = hc.hide_deterministics(lambda: static_shape_model(data))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MCMC(NUTS(closed), num_warmup=5, num_samples=5, progress_bar=False).run(
                jax.random.PRNGKey(0))
        assert any("constants were captured" in str(w.message) for w in caught)
    finally:
        jax.config.update("jax_captured_constants_warn_bytes", old)
    samples, _, _ = hc.load_chain(str(tmp_path))[:3]
    assert samples["w"].shape == (10, 3)


def test_noise_floor_separates_a_smooth_potential_from_a_noisy_one():
    """The roughness diagnostic's core: a smooth U has residuals ~delta^2/2 (float64 here),
    while U carrying per-point rounding noise shows a floor that does not shrink."""
    from src.model.age_hmc_roughness import noise_floor
    x0 = np.zeros(3); v = np.array([1.0, 0.0, 0.0])
    smooth = lambda x: 0.5 * float(np.sum(np.asarray(x) ** 2)) + 2e5      # noqa: E731
    noisy = lambda x: smooth(x) + 0.3 * np.sin(1e12 * float(np.asarray(x)[0]) + 1.0)  # noqa: E731
    _, f_smooth = noise_floor(smooth, x0, v, 0.0)
    _, f_noisy = noise_floor(noisy, x0, v, 0.0)
    assert f_smooth < 1e-6 and f_noisy > 1e-2


def test_float64_inference_from_a_float32_map_gets_its_own_directories(monkeypatch):
    """A float64 sampler started from the float32 MAP must not write into float32 runs' dirs."""
    monkeypatch.setattr(hc, "PRECISION", "float64")
    monkeypatch.setattr(hc, "MAP_PRECISION", "float32")
    monkeypatch.setattr(hc, "map_dir", lambda: "/r/age_map_float32_run_18_new_z_quick90")
    pcfg = {"run_names": {"hmc": "hmc_{variant}__{map_run}", "probe": "probe__{map_run}"}}
    assert hc.posterior_dir(pcfg, "hmc", variant="x").endswith(
        "hmc_x__age_map_float32_run_18_new_z_quick90__float64")
    assert hc.posterior_dir(pcfg, "probe", at_map_precision=True).endswith(
        "probe__age_map_float32_run_18_new_z_quick90")


def test_map_latents_take_the_run_precision(monkeypatch):
    """A float32 MAP must not leave a float64 run with float32 parameters."""
    lat = {"a": np.float32([1.0, 2.0]), "b": np.float32(0.5)}
    monkeypatch.setattr(hc, "PRECISION", "float32")
    assert all(v.dtype == jnp.float32 for v in hc.cast_latents(lat).values())
    if jax.config.jax_enable_x64:
        monkeypatch.setattr(hc, "PRECISION", "float64")
        assert all(v.dtype == jnp.float64 for v in hc.cast_latents(lat).values())


def _quad():
    A = jnp.diag(jnp.array([1e6, 1.0, 1e-2, 5.0]))
    b = jnp.array([1.0, 2.0, 3.0, -1.0])
    return (lambda x, kw: 0.5 * x @ (kw["A"] @ x) - kw["b"] @ x), {"A": A, "b": b}, b / jnp.diag(A)


_REFINE = {"max_iter": 200, "memory": 10, "tol_du": 1e-10, "tol_window": 5, "tol_grad": 1e-8,
           "ckpt_every": 3, "log_every": 1000}


def test_lbfgs_refine_reaches_the_optimum_of_an_ill_conditioned_potential(tmp_path):
    from src.model.age_refine_map import lbfgs_minimize
    f, kw, exact = _quad()
    x, hist, reason = lbfgs_minimize(f, jnp.zeros(4), kw, _REFINE, str(tmp_path / "s.pkl"), "fp",
                                     log=lambda *_: None)
    np.testing.assert_allclose(np.asarray(x), np.asarray(exact), rtol=1e-4, atol=1e-8)
    assert reason and "max_iter" not in reason


def test_lbfgs_refine_resumes_exactly(tmp_path):
    from src.model.age_refine_map import lbfgs_minimize
    f, kw, _ = _quad()
    full, h_full, _ = lbfgs_minimize(f, jnp.zeros(4), kw, {**_REFINE, "max_iter": 12},
                                     str(tmp_path / "a.pkl"), "fp", log=lambda *_: None)
    lbfgs_minimize(f, jnp.zeros(4), kw, {**_REFINE, "max_iter": 6}, str(tmp_path / "b.pkl"), "fp",
                   log=lambda *_: None)                      # "job 1" stops at 6
    part, h_part, _ = lbfgs_minimize(f, jnp.zeros(4), kw, {**_REFINE, "max_iter": 12},
                                     str(tmp_path / "b.pkl"), "fp", log=lambda *_: None)
    np.testing.assert_allclose(np.asarray(part), np.asarray(full), rtol=1e-12)
    assert [h[1] for h in h_part] == pytest.approx([h[1] for h in h_full])


def test_lbfgs_refine_never_ends_on_a_non_finite_point(tmp_path):
    """U = -x below x = 2 and nan beyond: the minimizer walks toward the hole. Either the
    zoom linesearch backs off on its own or the loop's guard stops it; both must leave a
    finite point."""
    from src.model.age_refine_map import lbfgs_minimize
    f = lambda x, kw: jnp.where(x[0] < 2.0, -x[0] + 0.0 * kw["c"], jnp.nan)  # noqa: E731
    x, hist, reason = lbfgs_minimize(f, jnp.zeros(1), {"c": jnp.ones(())}, _REFINE,
                                     str(tmp_path / "s.pkl"), "fp", log=lambda *_: None)
    assert float(x[0]) < 2.0 and np.isfinite(float(f(x, {"c": 1.0}))) and np.isfinite(hist[-1][1])
