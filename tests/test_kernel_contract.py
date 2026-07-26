"""Uncentered Ružička feature-map and downstream GP contract tests."""
import numpy as np
from numpyro import handlers

from src.community_encoder.train_DESK.esk_kernel import compute_kernel_diagnostics_ruzicka
from src.config_utils import load_age_model_config, load_config
from src.model.age_priors import sample_priors, validate_environment_kernel_contract


def _ruzicka(X):
    sums = X.sum(1, keepdims=True)
    l1 = np.abs(X[:, None, :] - X[None, :, :]).sum(2)
    sp = sums + sums.T
    den = 0.5 * (sp + l1)
    return np.divide(0.5 * (sp - l1), den, out=np.zeros_like(den), where=den > 1e-8)


def test_diagnostics_separate_exact_truncation_from_landmark_error():
    rng = np.random.default_rng(4)
    X = rng.lognormal(size=(40, 6)).astype("float32")
    K = _ruzicka(X)
    vals, vecs = np.linalg.eigh(K)
    order = np.argsort(vals)[::-1][:3]
    Z = vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))

    d = compute_kernel_diagnostics_ruzicka(
        Z.astype("float32"), X, n_species=6, n_weeks=1,
        max_samples=len(X), seed=0)

    assert d["rank"] == 3
    assert set(d) >= {"uncentered", "centered", "effective_rank", "rmse_norm"}
    # Z is the exact best rank-3 feature map, so its discrepancy from that optimum
    # is numerical only; all remaining target error is rank truncation.
    assert d["uncentered"]["landmark_at_rank_rmse_norm"] < 1e-5
    assert np.isclose(d["uncentered"]["combined_rmse_norm"],
                      d["uncentered"]["truncation_only_rmse_norm"], atol=1e-5)


def test_latent_width_and_gp_contract_agree_across_configs():
    encoder = load_config()
    age = load_age_model_config()
    width = encoder["esk"]["spacetime"]["latent_dim"]
    assert width == encoder["desk"]["latent_dim"] == age["source_latent_dim"] == 64
    # latent_dim is the supported, config-driven VRAM tradeoff (16 -> 24 -> 64 are
    # all legitimate), so assert the CONTRACT -- a positive top-eigenfeature
    # truncation of the 64-D source -- rather than any one chosen width.
    assert 0 < age["latent_dim"] <= age["source_latent_dim"]
    assert encoder["esk"]["spacetime"]["landmark_mode"] == "random"
    assert age["kernel_contract"]["kernel"] == "ruzicka"
    assert age["kernel_contract"]["centered"] is False
    assert age["kernel_contract"]["feature_prior"] == "isotropic"

    model_width = age["latent_dim"]
    data = {
        "Z_gathered": np.zeros((2, 3, model_width), dtype="float32"),
        "z_kernel_contract": {
            "kernel": "ruzicka", "centered": False,
            "feature_prior": "isotropic", "latent_dim": model_width,
            "source_latent_dim": width, "truncation": "top_eigenfeatures",
        },
    }
    validate_environment_kernel_contract(data)


def test_contract_rejects_centered_or_truncated_features():
    base = {"kernel": "ruzicka", "centered": False,
            "feature_prior": "isotropic", "latent_dim": 64}
    data = {"Z_gathered": np.zeros((1, 2, 64)), "z_kernel_contract": dict(base)}
    data["z_kernel_contract"]["centered"] = True
    try:
        validate_environment_kernel_contract(data)
    except ValueError as exc:
        assert "uncentered" in str(exc)
    else:
        raise AssertionError("centered features were accepted")

    data["z_kernel_contract"] = dict(base, latent_dim=16)
    try:
        validate_environment_kernel_contract(data)
    except ValueError as exc:
        assert "latent_dim" in str(exc)
    else:
        raise AssertionError("truncated features were accepted")


def _trace_priors(seed=3, M=64):
    return handlers.trace(handlers.seed(sample_priors, seed)).get_trace(
        prior_scale=1.0, M_features=M, time=2, N_sev_basis=2, N_lag_basis=2)


def test_environment_weight_prior_is_iid_across_features():
    """The GP contract: iid across features with a shared 3x3 output covariance.

    ``w_env`` is now a one-factor construction (beta_j = s_j*(h_j*f + sqrt(1-h_j^2)*eps_j))
    rather than a direct MultivariateNormal draw, so this asserts the property the
    Ružička identity actually needs -- the per-feature plate introduces no
    feature-specific scale or covariance -- instead of inspecting a distribution
    object that no longer exists.
    """
    tr = _trace_priors()
    for site in ("manifold_factor", "manifold_idio"):
        d = tr[site]["fn"]
        # Feature plate sits at dim=-2, so the feature axis is the row axis and the
        # three manifolds occupy the rightmost axis.
        assert d.batch_shape[0] == 64, f"{site} is not iid over the 64 features"
    assert np.asarray(tr["manifold_factor"]["value"]).shape == (64, 1)
    assert np.asarray(tr["manifold_idio"]["value"]).shape == (64, 3)
    assert np.asarray(tr["w_env"]["value"]).shape == (64, 3)

    # Every feature must share ONE output covariance: Cov_jk = s_j*s_k*h_j*h_k
    # off-diagonal, s_j^2 on the diagonal.
    s = np.asarray(tr["w_scale"]["value"])
    h = np.asarray(tr["manifold_loadings"]["value"])
    corr = np.outer(h, h)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(s, s)
    assert np.all(np.linalg.eigvalsh(cov) > 0), "implied output covariance is not PD"
    L = np.asarray(tr["L_corr"]["value"])
    assert np.allclose(L @ L.T, corr, atol=1e-5)
    np.testing.assert_allclose(np.asarray(tr["environment_kernel_variance"]["value"]),
                               s ** 2, rtol=1e-6)


def test_capacity_couples_to_fecundity_more_tightly_than_survival_does():
    """Ordered prior correlations: Corr(Fmax, K) > Corr(Fmax, survival).

    Fecundity and capacity are both productivity/resource axes, so they are
    prior-coupled more tightly than either is to survival. This ordering is the
    whole reason K gets its own manifold rather than either reusing H_r (implicit
    correlation 1.0, which forced all K-vs-Fmax disagreement through the disease
    term) or floating free.
    """
    rho_sr, rho_rk, rho_sk = [], [], []
    for s in range(400):
        tr = _trace_priors(seed=s, M=8)
        rho_sr.append(float(np.asarray(tr["rho"]["value"])))
        rho_rk.append(float(np.asarray(tr["env_corr_repro_capacity"]["value"])))
        rho_sk.append(float(np.asarray(tr["env_corr_survival_capacity"]["value"])))
    rho_sr, rho_rk = np.array(rho_sr), np.array(rho_rk)
    # Config targets 0.70 / 0.85; allow slack for the logit-normal's skew.
    assert 0.60 < np.median(rho_sr) < 0.78
    assert 0.78 < np.median(rho_rk) < 0.90
    assert np.median(rho_rk) > np.median(rho_sr) + 0.08
    # A strong belief, not a hard constraint: the data may overturn the ordering.
    assert 0.85 < (rho_rk > rho_sr).mean() < 1.0
    # Correlations must stay inside [-1, 1] for every draw -- the logit-normal
    # loading is what guarantees that without a positive-definiteness guard.
    for arr in (rho_sr, rho_rk, np.array(rho_sk)):
        assert np.all(np.abs(arr) <= 1.0)


def test_k_trend_is_continental_and_time_centered():
    """The capacity time trend must not be able to act spatially."""
    from src.data.combine.model_inputs import generate_k_trend_basis

    basis = generate_k_trend_basis(124, 3)
    assert basis.shape == (3, 124)                       # (n_basis, time): no space axis
    assert np.abs(basis.mean(axis=1)).max() < 1e-5       # alpha_k keeps the level
    tr = _trace_priors()
    assert np.asarray(tr["w_k_trend"]["value"]).shape == (3,)


def test_k_trend_prior_is_concentrated_on_zero():
    """The K time trend is a SAFETY VALVE, not a modeled mechanism.

    Its prior must be tight enough that the term carries no meaningful signal on
    its own: a fitted trend escaping this range is a failure signal saying some real
    temporal pattern is going unexplained by the model's actual mechanisms. This
    test exists so the budget cannot be quietly loosened to accommodate such a
    signal -- which is exactly the wrong response, and was tried once (budget=0.3
    allowed a 44% capacity swing at the 95th percentile).
    """
    from src.config_utils import load_age_model_config
    from src.data.combine.model_inputs import generate_k_trend_basis

    spec = load_age_model_config()["population_model"]["k_trend"]
    n, budget = int(spec["n_basis"]), float(spec["budget"])
    basis = generate_k_trend_basis(124, n)
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, budget / np.sqrt(n), (100_000, n))
    # K sits near softplus's exponential regime, so the trend acts on K as exp(trend).
    dev = np.abs(np.exp(w @ basis) - 1.0)
    assert np.median(dev) < 0.05, "prior median K drift should be a few percent"
    assert np.percentile(dev, 95) < 0.10, "95% of prior draws must stay under 10%"
    assert np.percentile(dev, 99.9) < 0.20, "even the far tail must not reach 20%"


def test_deterministic_sites_the_viz_depends_on_exist():
    """Sites the diagnostics read must actually be emitted by the model.

    This test exists because a real failure slipped through: `w_env` was changed
    from a SAMPLED site to a numpyro.deterministic (it is now built from the
    one-factor manifold prior), but `map_diagnostics` read it out of
    `auto_delta_params_to_latents`, which returns sampled sites ONLY. The viz job
    ran for a minute on a GPU and then died with KeyError: 'w_env'. Unit-testing the
    site inventory is far cheaper than discovering it from a SLURM error file.
    """
    tr = _trace_priors()
    deterministic = {k for k, v in tr.items() if v["type"] == "deterministic"}
    sampled = {k for k, v in tr.items() if v["type"] == "sample"}

    # Read by scripts/viz/map_diagnostics.py and src/vis/age_model_math.py.
    for name in ("w_env", "k_level", "k_level_route_counts", "w_scale", "L_corr",
                 "rho", "env_corr_repro_capacity", "env_corr_survival_capacity",
                 "manifold_loadings", "disease_tau", "disease_rec",
                 "disease_tau_rec", "disease_k_half_route_counts", "disease_hill_n"):
        assert name in deterministic, f"{name} is no longer a deterministic site"

    # Read as raw latents (i.e. must stay SAMPLED, or checkpoint restore breaks).
    for name in ("alpha_a", "alpha_j", "alpha_f", "gamma_a_raw", "gamma_j_diff",
                 "gamma_f_raw", "gamma_k_raw", "log_k_level_counts", "n50_raw",
                 "disease_mu_sev", "disease_b_late", "disease_w_sev",
                 "disease_lag0", "disease_w_lag", "w_k_trend"):
        assert name in sampled, f"{name} is no longer a sampled site"

    # w_env must be (M, 3): survival, reproduction, capacity.
    assert np.asarray(tr["w_env"]["value"]).shape[1] == 3


def test_seeds_start_below_carrying_capacity_by_construction():
    """The native seed and the invasion pulse must be fractions of capacity, not counts.

    An absolute seed has to be re-checked against the capacity level every time that
    level moves -- and when the level fell 97x (from an implied ~205 route counts to
    2.1), the seeds were not rechecked. A 9.5-count seed against a 2.1-count level
    starts the native range at 4.5x its own carrying capacity: the density brake slams
    on at t=0, the population crashes, and the fit opens at a much worse loss. Stated
    as a FRACTION the pathology is unrepresentable, which is the point.
    """
    pop = load_age_model_config()["population_model"]
    core_frac = float(pop["initpop_seed"]["core_fraction_of_capacity"])
    pulse_frac = float(pop["invasion_pulse_prior"]["median_fraction_of_capacity"])

    assert 0.0 < core_frac < 1.0, "the native seed must start BELOW capacity"
    assert 0.0 < pulse_frac < 1.0, "the invasion pulse must start BELOW capacity"
    # An invasion model should grow into capacity, not start at it. Both values
    # reproduce the ratios the pre-run_11 model used, which behaved.
    assert core_frac <= 0.5, f"core seed at {core_frac:.0%} of capacity is too high"
    assert pulse_frac <= 0.5, f"pulse at {pulse_frac:.0%} of capacity is too high"

    # No absolute-scale seed keys may survive: their presence means someone
    # reintroduced a value that silently needs rechecking against the level.
    for stale in ("core_route_counts", "margin_route_counts"):
        assert stale not in pop["initpop_seed"], f"{stale} is level-dependent"
    assert "median_route_counts" not in pop["invasion_pulse_prior"]
