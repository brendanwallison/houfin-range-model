"""Arm B of the HMC trial: NUTS in the NeuTra space of the MAP-initialized VI guide.

``age_resume_svi_from_map`` fits a low-rank MVN guide q; NeuTraReparam samples the
model through q's transform, so NUTS works on ``auto_shared_latent`` where the
posterior should look roughly standard normal.

WHAT THIS DOES AND DOES NOT BUY. q here is Gaussian, so its transform is AFFINE:
NeuTra over it is equivalent to NUTS in the original space with a low-rank-plus-
diagonal mass matrix and an init at q's mean. Expect it to behave like arm A's
laplace_* metrics (different covariance estimate, same idea). It can only beat them
where the curvature varies across the posterior, and that needs a nonlinear (flow)
guide -- a follow-up only worth building if arm A and this arm show divergences /
saturated trees concentrated in particular sites.

Chains, chunking and resume are as in arm A (``hmc_common.run_chunked_mcmc``). Each
chunk is saved both as warped latents and transformed back to constrained latent sites.
"""
from __future__ import annotations

import os
import pickle
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import hmc_common as hc                            # noqa: E402  (sets x64)
import jax                                                         # noqa: E402
import numpy as np                                                 # noqa: E402
import numpyro                                                     # noqa: E402
from numpyro.handlers import reparam                               # noqa: E402
from numpyro.infer import NUTS, SVI, Trace_ELBO                    # noqa: E402
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal  # noqa: E402
from numpyro.distributions.transforms import biject_to            # noqa: E402
from numpyro.infer.reparam import NeuTraReparam                    # noqa: E402

from src.config_utils import load_age_model_config                # noqa: E402
from src.model.age_priors import build_model_2d                    # noqa: E402
from src.model.age_resume_hmc import hmc_settings                  # noqa: E402
from src.model.age_resume_svi_from_map import vi_dir, vi_settings  # noqa: E402
from src.model.data_loading import load_data                       # noqa: E402
from src.model.runtime_diagnostics import memory_snapshot, require_gpu  # noqa: E402


def load_vi(path: str) -> dict:
    with open(path, "rb") as fh:
        vi = pickle.load(fh)
    if vi.get("format_version") != 2 or "rank" not in vi:
        raise RuntimeError(f"{path} is not a v2 VI output of age_resume_svi_from_map; refit VI")
    return vi


def build_neutra(model, vi: dict, kwargs: dict):
    """(neutra, warped_model, to_constrained) for a saved low-rank guide of any rank.

    The guide must be traced once (svi.init) before NeuTraReparam can read its
    prototype trace and latent layout; the optimizer is a placeholder.
    """
    rank = int(np.asarray(vi["params"]["auto_cov_factor"]).shape[-1])
    if rank != int(vi["rank"]):
        raise RuntimeError(f"saved rank {vi['rank']} disagrees with cov_factor rank {rank}")
    guide = AutoLowRankMultivariateNormal(model, rank=rank)
    SVI(model, guide, numpyro.optim.Adam(1e-3), loss=Trace_ELBO()).init(
        jax.random.PRNGKey(0), **kwargs)
    n_latent = int(np.asarray(vi["params"]["auto_loc"]).size)
    if n_latent != guide.latent_dim:
        raise RuntimeError(f"VI guide has {n_latent} latents but the current model has "
                           f"{guide.latent_dim}; refit VI")
    neutra = NeuTraReparam(guide, vi["params"])

    def config(site):
        if (site["type"] == "sample" and not site.get("is_observed", False)
                and not site.get("infer", {}).get("is_auxiliary", False)
                and site["name"] in guide.prototype_trace):
            return neutra
        return None

    names = [k for k, site in guide.prototype_trace.items()
             if site["type"] == "sample" and not site.get("is_observed", False)
             and not site.get("infer", {}).get("is_auxiliary", False)]
    bijectors = {k: biject_to(guide.prototype_trace[k]["fn"].support) for k in names}

    @jax.jit
    def to_constrained(warped_batch):
        """Warped draws -> constrained LATENT sites, without running the model.

        NeuTraReparam.transform_sample replays the model through its postprocess_fn,
        which for the age model reruns the whole simulation per draw and returns every
        full-grid deterministic with it; this applies only q's transform and each site's
        support bijection.
        """
        def one(w):
            unc = guide._unpack_latent(neutra.transform(w))
            return {k: bijectors[k](unc[k]) for k in names}
        return jax.vmap(one)(warped_batch)

    return neutra, hc.hide_deterministics(reparam(model, config=config)), to_constrained


def run_neutra_hmc():
    chain = hc.chain_index()
    pcfg = hc.load_posterior_config()
    s = hmc_settings(pcfg, "neutra")
    vs = vi_settings(pcfg)
    vi_path = os.path.join(vi_dir(pcfg, vs["rank"]), "vi_posterior_params.pkl")
    variant = f"neutra_rank{vs['rank']}"
    out_dir = os.path.join(hc.posterior_dir(pcfg, "hmc", variant=variant), f"chain_{chain:02d}")
    hc.archive_if_fresh(out_dir)
    print(f"--- NeuTra NUTS | {variant} chain={chain} {hc.PRECISION} | {s} ---")

    device = require_gpu("age-model NeuTra NUTS")
    vi = load_vi(vi_path)
    _, map_ckpt = hc.load_map()
    if vi["map_fingerprint"] != map_ckpt["fingerprint"]:
        raise RuntimeError(f"{vi_path} was fitted from a different MAP than {hc.map_dir()}; refit VI")
    hc.check_model_drift(vi["map_fingerprint_payload"], what="VI (via its MAP)")
    data = load_data(load_age_model_config()["input_dir"], target_device=device,
                     precision=hc.PRECISION)
    memory_snapshot("neutra-inputs-loaded", device)
    arrays, static = hc.split_data(data)
    kwargs = {"arrays": arrays}
    _, warped, to_constrained = build_neutra(hc.array_model(build_model_2d, static), vi, kwargs)
    kernel = NUTS(warped, target_accept_prob=s["target_accept"],
                  max_tree_depth=s["max_tree_depth"], step_size=s["step_size"],
                  dense_mass=False, adapt_mass_matrix=True)

    # The warped space is q's standard-normal base: 0 is q's mean, and N(0, jitter^2)
    # draws are q-shaped overdispersed starts.
    n_latent = int(np.asarray(vi["params"]["auto_loc"]).size)
    rng_jitter, rng_run = jax.random.split(jax.random.PRNGKey(2000 + chain))
    jitter = 0.0 if chain == 0 else s["init_jitter"]
    init = {"auto_shared_latent": jitter * jax.random.normal(rng_jitter, (n_latent,))}

    def postprocess(samples):
        warped_z = samples["auto_shared_latent"]
        out = {k: np.asarray(v) for k, v in jax.device_get(to_constrained(warped_z)).items()}
        out["auto_shared_latent"] = warped_z
        return out

    payload = {
        "arm": "neutra", "chain": chain, "precision": hc.PRECISION, "settings": s,
        "vi_fingerprint": vi["fingerprint"], "map_fingerprint": map_ckpt["fingerprint"],
        "versions": hc.versions(), "sources": hc.source_identities([__file__]),
    }
    hc.run_chunked_mcmc(
        kernel, out_dir=out_dir, run_fingerprint=hc.fingerprint(payload), payload=payload,
        num_warmup=s["num_warmup"], num_samples=s["num_samples"], chunk=s["chunk"],
        rng_key=rng_run, kwargs=kwargs, init_params=init, postprocess=postprocess,
    )
    memory_snapshot("neutra-done", device)


if __name__ == "__main__":
    run_neutra_hmc()
