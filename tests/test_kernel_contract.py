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
    assert age["latent_dim"] == 16
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


def test_environment_weight_prior_is_iid_across_features():
    tr = handlers.trace(handlers.seed(sample_priors, 3)).get_trace(
        anneal=1.0, M_features=64, N_basis=2, time=2)
    w_dist = tr["w_env"]["fn"]
    # The feature plate expands one shared bivariate distribution to 64 IID
    # coordinates; it does not introduce feature-specific scales/covariances.
    assert w_dist.batch_shape == (64,)
    assert w_dist.event_shape == (2,)
    cov = np.asarray(w_dist.base_dist.covariance_matrix)
    scale = np.asarray(tr["w_scale"]["value"])
    rho = float(np.asarray(tr["rho"]["value"]))
    expected = np.array([[scale[0] ** 2, rho * scale[0] * scale[1]],
                         [rho * scale[0] * scale[1], scale[1] ** 2]])
    assert np.allclose(cov, expected, rtol=1e-5, atol=1e-6)
