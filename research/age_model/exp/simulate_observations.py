"""Draw a synthetic BBS count set from the model at a known parameter point (null calibration / recovery).

    AGE_MODEL_CONFIG=<overlay WITH the structure under test> HOUFIN_MODEL_PRECISION=float64 \\
      python research/age_model/exp/simulate_observations.py --run-dir <fitted point> --out <y.npz> \\
        [--set covariate_drift_w=0] [--set covariate_drift_w=<json array>] [--set site=@values.npy] [--seed 0]

The fitted point supplies every latent it has; ``--set`` fixes the others (e.g. the drift weights
for a structure the point was not fitted with) -- a missing latent is an error, never a prior draw.
Counts are NB2(mean = expected_obs, concentration as the likelihood builds it), on the REAL design
(cells, years, quality tiers, routes, extra zeros). Fit them by exporting HOUFIN_SYNTHETIC_OBS=<y.npz>.

Writes <y.npz> with y (counts), mu (the expected counts) and the --set values, so the truth is
recorded next to the data it generated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="site=value (scalar or JSON array)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.environ.get("HOUFIN_SYNTHETIC_OBS"):
        raise SystemExit("unset HOUFIN_SYNTHETIC_OBS: the simulator must see the real design")

    os.environ.setdefault("HOUFIN_MODEL_PRECISION", "float64")
    from src.model import hmc_common  # noqa: F401  (sets x64 from the precision)
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from numpyro import handlers
    from numpyro.infer import Predictive
    from src.config_utils import load_age_model_config
    from src.model.age_priors import build_model_2d
    from src.model.checkpoints import auto_delta_params_to_latents, load_map_params
    from src.model.data_loading import load_data
    from src.model.runtime_diagnostics import require_gpu

    device = require_gpu("synthetic observations")
    data = load_data(load_age_model_config()["input_dir"], target_device=device,
                     precision=os.environ["HOUFIN_MODEL_PRECISION"], verbose=False)
    params, _ = load_map_params(a.run_dir)
    lat = {k: np.asarray(v) for k, v in auto_delta_params_to_latents(params).items()}
    fixed = {}
    for s in a.set:
        k, v = s.split("=", 1)
        fixed[k] = np.asarray(np.load(v[1:]) if v.startswith("@") else json.loads(v), dtype=float)

    # Every latent of THIS model must come from the point or --set; broadcast scalar --set values.
    tr = handlers.trace(handlers.seed(build_model_2d, 0)).get_trace(data=data, prior_scale=1.0)
    latent_sites = {n: s for n, s in tr.items()
                    if s["type"] == "sample" and not s["is_observed"] and not s.get("infer", {}).get("is_auxiliary")}
    post = {}
    for n, s in latent_sites.items():
        if n in fixed:
            post[n] = np.broadcast_to(fixed[n], np.shape(s["value"])).astype(float)
        elif n in lat:
            post[n] = lat[n]
        else:
            raise SystemExit(f"latent {n!r} is in neither the fitted point nor --set")
    post = {k: jnp.asarray(v)[None] for k, v in post.items()}
    out = Predictive(build_model_2d, posterior_samples=post,
                     return_sites=["expected_obs", "concentration", "quality_conc_mult", "obs_conc_kappa",
                                   "route_occupancy_log_m"])(
        jax.random.PRNGKey(0), data=data, prior_scale=1.0)

    # Same NB2 construction as the likelihood (and research/age_model/exp/loglik_decompose.py).
    mu = np.asarray(out["expected_obs"][0], dtype=np.float64)
    conc = float(np.asarray(out["concentration"][0]))
    q = float(np.asarray(out["quality_conc_mult"][0])) if "quality_conc_mult" in out else 1.0
    quality = np.asarray(data["obs_quality"]) if data.get("obs_quality") is not None else np.zeros_like(mu)
    conc_obs = conc * q ** quality
    if "obs_conc_kappa" in out:
        conc_obs = conc_obs * np.maximum(mu, 1e-3) ** float(np.asarray(out["obs_conc_kappa"][0]))
    if "route_occupancy_log_m" in out:      # observation.route_occupancy (E072): same ZINB as the likelihood
        from src.model.age_priors import route_occupancy_parts
        _, mu_occ, logits = route_occupancy_parts(jnp.asarray(mu), jnp.exp(out["route_occupancy_log_m"][0]))
        obs_dist = dist.ZeroInflatedNegativeBinomial2(mean=mu_occ, concentration=jnp.asarray(conc_obs),
                                                      gate_logits=logits)
    else:
        obs_dist = dist.NegativeBinomial2(mean=jnp.asarray(mu), concentration=jnp.asarray(conc_obs))
    y = np.asarray(obs_dist.sample(jax.random.PRNGKey(a.seed)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, y=y, mu=mu, seed=a.seed, run_dir=a.run_dir,
             **{f"set_{k}": v for k, v in fixed.items()})
    real = np.asarray(data["observed_results"], dtype=float)
    print(f"[simulate] {y.size} counts -> {a.out}; mean y {y.mean():.3f} (real {real.mean():.3f}), "
          f"zero share {np.mean(y == 0):.3f} (real {np.mean(real == 0):.3f})")


if __name__ == "__main__":
    main()
