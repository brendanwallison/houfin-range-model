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
    """The GP contract: iid across features with a shared 4x4 output covariance.

    ``w_env`` is a RANK-2 angular construction --
    ``beta_j = s_j*(a_j*f1 + b_j*f2 + sqrt(1-a_j^2-b_j^2)*eps_j)`` -- rather than a direct
    MultivariateNormal draw, so this asserts the property the Ružička identity actually needs:
    the per-feature plate introduces no feature-specific scale or covariance. Adding a second
    FACTOR preserves that; a covariance over FEATURES would have destroyed it.
    """
    tr = _trace_priors()
    for site in ("manifold_factor", "manifold_idio"):
        d = tr[site]["fn"]
        # Feature plate sits at dim=-2, so the feature axis is the row axis and the
        # four manifolds occupy the rightmost axis.
        assert d.batch_shape[0] == 64, f"{site} is not iid over the 64 features"
    assert np.asarray(tr["manifold_factor"]["value"]).shape == (64, 2)   # TWO factors
    assert np.asarray(tr["manifold_idio"]["value"]).shape == (64, 4)
    assert np.asarray(tr["w_env"]["value"]).shape == (64, 4)

    # Every feature must share ONE output covariance: Cov_jk = s_j*s_k*(L L^T)_jk
    # off-diagonal, s_j^2 on the diagonal.
    s = np.asarray(tr["w_scale"]["value"])
    L_load = np.asarray(tr["manifold_loadings"]["value"])
    assert L_load.shape == (4, 2)
    corr = L_load @ L_load.T
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(s, s)
    assert np.all(np.linalg.eigvalsh(cov) > 0), "implied output covariance is not PD"
    L = np.asarray(tr["L_corr"]["value"])
    assert np.allclose(L @ L.T, corr, atol=1e-5)
    np.testing.assert_allclose(np.asarray(tr["environment_kernel_variance"]["value"]),
                               s ** 2, rtol=1e-6)
    # Communality must stay in (0,1) so sqrt(1 - r^2) is real -- what makes the
    # construction PD without any Cholesky guard.
    r = np.asarray(tr["manifold_communality"]["value"])
    assert r.shape == (4,) and np.all((r > 0) & (r < 1))


def test_beta_variance_equals_w_scale_squared_at_rank_two():
    """Var(beta_j) == w_scale_j^2 exactly -- the assertion that catches a broken GP identity.

    ``H_j = Z.beta_j`` is a GP with kernel ``w_scale_j^2 * Z(x).Z(x')`` ONLY if beta_j is iid
    over features with variance exactly w_scale_j^2. The rank-2 construction achieves that
    because ``r^2 + (1 - r^2) = 1`` for any communality. Checked at large M so the sample
    variance is actually informative -- at M=64 its own sd is ~0.18 and would hide a real error.
    """
    tr = _trace_priors(seed=3, M=40000)
    w = np.asarray(tr["w_env"]["value"])
    s = np.asarray(tr["w_scale"]["value"])
    np.testing.assert_allclose(w.var(axis=0) / s ** 2, np.ones(4), atol=0.03)

    # and the empirical cross-field correlation must match the analytic L L^T
    L_load = np.asarray(tr["manifold_loadings"]["value"])
    analytic = L_load @ L_load.T
    np.fill_diagonal(analytic, 1.0)
    assert np.abs(np.corrcoef(w.T) - analytic).max() < 0.02
    # iid over features: adjacent feature rows must be uncorrelated
    assert abs(np.corrcoef(w[:-1, 0], w[1:, 0])[0, 1]) < 0.02


def test_juvenile_survival_couples_to_adult_survival_as_tightly_as_capacity_to_fecundity():
    """The pair structure this rank-2 extension exists to create.

    Juvenile survival used to BE adult survival (same manifold, scalar shift and scale only), so
    there was no correlation to state. It now has its own weights, prior-coupled to adult
    survival at the same target as fecundity-capacity (0.85), and both pairs coupled to each
    other more weakly (0.70).

    RANK 1 COULD NOT DO THIS. With Corr(j,k) = h_j*h_k the tetrad identity
    rho(Sa,Sj)*rho(F,K) == rho(Sa,F)*rho(Sj,K) is forced; at rho(F,K)=0.85 and cross 0.70 it
    caps rho(Sa,Sj) at 0.576 (and demanding 0.85 needs a loading of 1.12, i.e. an imaginary
    idiosyncratic weight). This test is what would fail if anyone collapsed it back to one factor.
    """
    sj, fk, cross = [], [], []
    for s in range(300):
        tr = _trace_priors(seed=s, M=8)
        L_load = np.asarray(tr["manifold_loadings"]["value"])
        C = L_load @ L_load.T
        sj.append(C[0, 3])          # adult vs juvenile survival
        fk.append(C[1, 2])          # fecundity vs capacity
        cross.append(C[0, 1])       # adult survival vs fecundity (across groups)
        # the dedicated site must agree with the matrix
        assert abs(float(np.asarray(tr["env_corr_survival_adult_juv"]["value"])) - C[0, 3]) < 1e-5
    sj, fk, cross = np.array(sj), np.array(fk), np.array(cross)

    assert 0.78 < np.median(sj) < 0.90, np.median(sj)
    assert 0.78 < np.median(fk) < 0.90, np.median(fk)
    assert 0.60 < np.median(cross) < 0.78, np.median(cross)
    # both pairs must sit ABOVE the cross-group coupling -- a strong belief, not a constraint
    assert (sj > cross).mean() > 0.85
    assert (fk > cross).mean() > 0.85
    # within-pair is also LESS uncertain than cross-group, for free: cos is flat at zero
    # angular separation, so a single angle_scale produces both properties.
    assert sj.std() < cross.std()
    # rank-1 would have forced the tetrad identity; rank-2 must break it
    assert abs(np.median(sj) * np.median(fk) - np.median(cross) ** 2) > 0.15


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
    # Config targets 0.70 cross-group / 0.85 within-pair; slack for the logit-normal's skew.
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
    # gamma_*_raw moved from SAMPLED to DETERMINISTIC when the slopes were fixed at 1
    # (the amplitude they carried now lives in w_scale). They are still emitted, as
    # softplus^-1(1), so existing readers that apply softplus keep working.
    for name in ("gamma_a_raw", "gamma_f_raw", "gamma_k_raw", "gamma_a", "gamma_j",
                 "gamma_f", "gamma_k",
                 "w_env", "k_level", "k_level_route_counts", "w_scale", "L_corr",
                 "rho", "env_corr_repro_capacity", "env_corr_survival_capacity",
                 "manifold_loadings", "manifold_communality",
                 "env_corr_survival_adult_juv", "gamma_j_diff",
                 "disease_tau", "disease_rec",
                 "disease_tau_rec", "disease_k_half_route_counts", "disease_hill_n"):
        assert name in deterministic, f"{name} is no longer a deterministic site"

    # Read as raw latents (i.e. must stay SAMPLED, or checkpoint restore breaks).
    for name in ("alpha_a", "alpha_j", "alpha_f", "alpha_k", "n50_raw",
                 "disease_mu_sev", "disease_b_late", "disease_w_sev",
                 "disease_lag0", "disease_w_lag", "w_k_trend"):
        assert name in sampled, f"{name} is no longer a sampled site"

    # w_env must be (M, 4): survival_adult, reproduction, capacity, survival_juv. Juvenile
    # survival is APPENDED, not inserted -- eight viz/analysis modules index columns 0..2
    # positionally, so inserting would silently relabel reproduction and capacity.
    assert np.asarray(tr["w_env"]["value"]).shape[1] == 4


def test_native_seed_starts_at_local_capacity_by_construction():
    """The native seed must be a fraction of LOCAL capacity.

    Absolute values have to be re-checked against the capacity level every time it
    moves, and when the level fell 97x they were not. 9.5 route counts is ~33% of
    LOCAL capacity in the native core and entirely sensible -- but at initialization
    AutoDelta puts beta_k at its prior median, so H_k ~ 0, every cell has
    K = k_level = 2.1 counts, and the seed is 4.5x capacity: the density brake slams
    on and step 0 of the fit scores badly. As a fraction that is unrepresentable.

    The native seed is a fraction of LOCAL K_base at t=0 (so it tracks the capacity
    the covariates imply for those specific cells) and equals 1.0, because a range at
    equilibrium sits at its capacity.
    """
    pop = load_age_model_config()["population_model"]
    core_frac = float(pop["initpop_seed"]["core_fraction_of_local_capacity"])
    target = float(pop["dispersal_target_capacity_fraction"])

    assert 0.0 < core_frac <= 1.0, "the native seed must not exceed local capacity"
    # The native range starts AT capacity (1.0), which is above the emigration logit's
    # centre, so it is a net exporter in year one -- see the config's _initpop_comment
    # for why that is deliberate. The contract is the ORDERING (a range at equilibrium
    # is at least as dense as the dispersal target), not equality: an earlier version
    # pinned core_frac == target to make the seed migration-neutral.
    assert core_frac >= target, (
        f"core seed fraction {core_frac} should be at least "
        f"dispersal_target_capacity_fraction {target}: a native range at equilibrium "
        f"is not a net importer")

    # No absolute or continental-level seed key may return: either needs manual
    # rechecking whenever the capacity level moves.
    for stale in ("core_route_counts", "margin_route_counts",
                  "core_fraction_of_capacity"):
        assert stale not in pop["initpop_seed"], f"{stale} is level-dependent"


def test_invasion_pulse_prior_is_fixed_and_k_independent():
    """The invasion pulse must NOT reference k_level or local K_base at all.

    An earlier version scaled it against the continental k_level, which diluted the
    founder badly at the actual (unusually high-K) release site -- see config's
    _invasion_pulse_comment. It is now a fraction of a FIXED, empirically observed
    quantity (global_q50_route_counts), so no capacity-level or K_base key may appear
    in this block; that is the property that keeps it from silently changing meaning
    whenever K moves.
    """
    pop = load_age_model_config()["population_model"]
    inv = pop["invasion_pulse_prior"]

    assert float(inv["global_q50_route_counts"]) > 0.0
    assert 0.0 < float(inv["median_fraction_of_global_q50"]) < 1.0
    assert float(inv["log_budget"]) > 0.0

    for stale in ("median_fraction_of_capacity", "median_route_counts",
                  "capacity_level", "k_level"):
        assert stale not in inv, f"{stale} would re-tie the pulse to a moving K"

    assert int(pop["invasion_sites"]) >= 1


def test_invasion_pulse_budget_tightens_per_coefficient_as_sites_grow():
    """More candidate (site, year) coefficients must not mean more TOTAL freedom.

    Same guard this project applies to disease_prior's sev_field_budget/
    lag_field_budget: the shared budget is divided by sqrt(n_coefficients), so
    growing the parameterization (more sites) tightens each individual coefficient's
    prior SD rather than leaving it fixed -- which would let the term act as a free,
    high-dimensional escape hatch (the exact failure mode of the old 967-coefficient
    K-correction field).
    """
    budget = 3.0
    sd_small = budget / np.sqrt(1 * 10)
    sd_large = budget / np.sqrt(9 * 10)
    assert sd_large < sd_small
    assert np.isclose(sd_small, budget / np.sqrt(10))
    assert np.isclose(sd_large, budget / np.sqrt(90))


def test_juvenile_survival_and_Q_read_beta_sj_not_beta_s():
    """S_j and Q must respond to beta_sj and be INVARIANT to beta_s.

    This is the assertion that proves the untying happened. Before this change S_j was
    ``sigmoid(alpha_j + gamma_j * Z.beta_s)`` -- literally adult survival's field with a scalar
    shift and scale -- and Q was the same field on the dispersal block. If anyone reverts
    age_fields to read H_s_local for S_j, the correlation prior in age_priors would still look
    right while juvenile survival silently had zero spatial degrees of freedom again. Only a
    perturbation test catches that.
    """
    import jax.numpy as jnp
    from src.model.age_fields import project_and_scatter_age_structured as project
    from tests.test_disease_depression import N_LAND, _disease

    T, M, N, K_kern = 3, 4, N_LAND, 2
    rng = np.random.default_rng(0)
    Z = jnp.array(rng.normal(size=(T, N, M)))
    Zd = jnp.array(rng.normal(size=(T, N, K_kern, M)))
    idx = jnp.arange(N)
    k_trend = jnp.array(np.cos(np.pi * np.linspace(0, 1, T))[None, :])
    b = jnp.ones(M) * 0.1

    def run(beta_s, beta_sj):
        # signature: (time, Ny, Nx, land_rows, land_cols, Z, Z_disp, disease_timestep, disease,
        #             beta_s, beta_r, beta_k, beta_sj, k_trend_basis, w_k_trend, alpha/gamma...)
        return project(T, 40, 60, idx, idx, Z, Zd, T + 1, _disease(mu_sev=-30.0),
                       beta_s, b, b, beta_sj,
                       k_trend, jnp.zeros(1),
                       0.5, 1.0, -0.5, 1.0, 2.0, 1.0, -2.128295, 1.0)

    Sa0, Sj0, F0, K0, Q0, _ = run(b, b)
    # perturbing beta_sj must move S_j and Q, and leave S_a / Fmax / K untouched
    Sa1, Sj1, F1, K1, Q1, _ = run(b, b * 3.0)
    assert not np.allclose(Sj0, Sj1), "S_j does not respond to beta_sj -- still tied to beta_s?"
    assert not np.allclose(Q0, Q1), "Q does not respond to beta_sj"
    np.testing.assert_allclose(np.asarray(Sa0), np.asarray(Sa1), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(F0), np.asarray(F1), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(K0), np.asarray(K1), rtol=1e-6)

    # perturbing beta_s must move S_a and leave S_j / Q untouched -- the converse, and the
    # direction that actually failed before this change.
    Sa2, Sj2, F2, K2, Q2, _ = run(b * 3.0, b)
    assert not np.allclose(Sa0, Sa2), "S_a does not respond to beta_s"
    np.testing.assert_allclose(np.asarray(Sj0), np.asarray(Sj2), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(Q0), np.asarray(Q2), rtol=1e-6)


def test_juvenile_survival_field_is_not_a_monotone_transform_of_adult():
    """Prior-predictive: S_j maps must no longer be a deterministic function of S_a.

    Under the old parameterization S_j = sigmoid(alpha_j + gamma_j*H_s) and
    S_a = sigmoid(alpha_a + gamma_a*H_s) shared H_s, so S_j was an exact monotone transform of
    S_a and their Spearman correlation across cells was identically 1.0 for every draw. With
    separate manifolds it must fall below 1 while staying high (the prior still couples them).
    """
    import jax.numpy as jnp
    from scipy.stats import spearmanr
    from src.model.age_fields import project_and_scatter_age_structured as project
    from tests.test_disease_depression import N_LAND, _disease

    T, M, N, K_kern = 2, 24, N_LAND, 2
    rng = np.random.default_rng(1)
    Z = jnp.array(rng.normal(size=(T, N, M)))
    Zd = jnp.array(rng.normal(size=(T, N, K_kern, M)))
    idx = jnp.arange(N)
    k_trend = jnp.array(np.cos(np.pi * np.linspace(0, 1, T))[None, :])

    rhos = []
    for s in range(20):
        tr = _trace_priors(seed=s, M=M)
        w = np.asarray(tr["w_env"]["value"])
        Sa, Sj, _, _, _, _ = project(
            T, 40, 60, idx, idx, Z, Zd, T + 1, _disease(mu_sev=-30.0),
            jnp.array(w[:, 0]), jnp.array(w[:, 1]), jnp.array(w[:, 2]), jnp.array(w[:, 3]),
            k_trend, jnp.zeros(1),
            0.5, 1.0, -0.5, 1.0, 2.0, 1.0, -2.128295, 1.0)
        rhos.append(spearmanr(np.asarray(Sa[0]), np.asarray(Sj[0])).statistic)
    rhos = np.array(rhos)
    assert (np.abs(rhos) < 0.999).all(), f"S_j still a monotone transform of S_a: {rhos}"
    # but the prior coupling should keep them broadly aligned rather than independent
    assert np.median(np.abs(rhos)) > 0.4, f"S_j and S_a decoupled too far: {np.median(rhos)}"


def test_encoder_identity_propagates_cube_to_path_meta_to_metadata():
    """The DESK checkpoint's identity must survive the whole chain to metadata.pkl.

    The guards along this chain check the kernel FAMILY, the centering convention and the
    latent width -- none of which distinguishes two cubes encoded by DIFFERENT DESK
    checkpoints. ``config/overlays/map_new_z.json`` documents the consequence in its own
    comment: "Nothing hashes the DESK checkpoint into cube_meta.json, so if those are
    skipped this silently fits the OLD encoder and the A/B is vacuous -- both points would
    be the same run."

    So the identity is now recorded at the source and copied forward:

        build_final_z_cube      -> cube_meta["desk_checkpoint"]  (content sha256)
        generate_all_path_features -> path_meta["kernel_contract"] = cube_meta, verbatim
        model_inputs            -> metadata["z_kernel_contract"]["desk_checkpoint"]

    This test pins the two ends and the copy in the middle, reading the source rather than
    re-implementing it, so a future edit that drops a link fails here.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    # 1. build_final_z_cube writes it, hashing CONTENT (not mtime, so a re-copied file
    #    still compares equal).
    cube_src = (root / "src" / "community_encoder" / "build_final_z_cube.py").read_text()
    assert '"desk_checkpoint": _file_identity(model_path)' in cube_src
    assert "content_sha256" in cube_src, "identity must be a content hash, not a timestamp"

    # 2. generate_all_path_features copies cube_meta in verbatim, which is what carries it
    #    across without naming the key again.
    path_src = (root / "src" / "processing" / "generate_all_path_features.py").read_text()
    assert '"kernel_contract": cube_meta' in path_src

    # 3. model_inputs reads it back off path_meta into the metadata the model is fit against.
    mi_src = (root / "src" / "data" / "combine" / "model_inputs.py").read_text()
    assert 'source_contract.get("desk_checkpoint")' in mi_src
    assert re.search(r'source_contract\s*=\s*path_meta\.get\("kernel_contract"\)', mi_src), \
        "model_inputs must take the contract from path_meta, which is where cube_meta landed"
