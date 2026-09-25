"""N-mixture with the latent abundance summed out directly (Royle 2004).

WHY THIS EXISTS. biolith's nmixture draws N_i from a Categorical over
0..max_abundance and asks NumPyro to enumerate it. Parallel enumeration adds an
axis of size (M+1), and Categorical.log_prob then broadcasts its (M+1)-wide
logits against that axis, so the term is (M+1) x (M+1) x n_sites -- QUADRATIC in
the ceiling. With House Finch route counts reaching 490 in the western native
range the honest ceiling is ~1960, which asks an A100 for 55.2 GiB and dies.

That cost is an artifact of the enumerate-a-Categorical formulation, not of the
model. The textbook marginal likelihood -- and what unmarked::pcount computes --
sums over N directly:

    L_i = SUM_N  Poisson(N | lambda_i) * PROD_t Binomial(y_it | N, p_it)

which is LINEAR in the ceiling: n_sites x (M+1) x n_replicates. For the same
problem that is ~300 MB rather than 55 GiB, so the ceiling the data actually
implies becomes affordable and nothing has to be truncated or thrown away.

This is the same model, computed the standard way -- not a different or
simplified one. Verified against biolith's nmixture on simulated data where the
enumerated version is still affordable (see tests).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import gammaln, logsumexp


def _log_binom_pmf(y, n, p, eps=1e-9):
    """log Binomial(y | n, p), -inf where y > n. Broadcasts over all axes."""
    p = jnp.clip(p, eps, 1.0 - eps)
    coef = gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
    lp = coef + y * jnp.log(p) + (n - y) * jnp.log1p(-p)
    return jnp.where(n >= y, lp, -jnp.inf)


def marginal_log_likelihood(y, mask, lam, p, max_abundance):
    """Per-site log L, summing the latent N out over 0..max_abundance.

    y     (n_sites, n_rep)  counts, any value where mask is False
    mask  (n_sites, n_rep)  True where the site was actually surveyed
    lam   (n_sites,)        expected true abundance
    p     (n_sites, n_rep)  per-visit detection probability
    """
    support = jnp.arange(max_abundance + 1, dtype=jnp.float32)        # (M+1,)
    log_pois = dist.Poisson(lam[:, None]).log_prob(support[None, :])  # (S, M+1)

    yy = jnp.where(mask, y, 0.0)[:, :, None]                          # (S, T, 1)
    pp = p[:, :, None]
    nn = support[None, None, :]
    log_det = _log_binom_pmf(yy, nn, pp)                              # (S, T, M+1)
    log_det = jnp.where(mask[:, :, None], log_det, 0.0)               # skip unsurveyed
    return logsumexp(log_pois + log_det.sum(axis=1), axis=-1)         # (S,)


def nmixture(site_covs, obs_covs, obs=None, max_abundance=100,
             site_random_effects=True, prior_scale=2.0):
    """Royle (2004) N-mixture, latent N marginalized by direct summation.

    Shapes follow biolith's contract so the two are interchangeable:
      site_covs (n_sites, n_site_covs)
      obs_covs  (n_sites, n_periods, n_replicates, n_obs_covs)
      obs       (n_species, n_sites, n_periods, n_replicates), NaN = not surveyed
    Only n_species = n_periods = 1 is supported; the benchmark treats its window
    as one closed period.
    """
    site_covs = jnp.asarray(site_covs)
    obs_covs = jnp.asarray(obs_covs)
    n_sites, n_site_covs = site_covs.shape
    n_rep = obs_covs.shape[2]
    n_obs_covs = obs_covs.shape[-1]

    beta0 = numpyro.sample("beta0", dist.Normal(0.0, prior_scale))
    beta = numpyro.sample("beta", dist.Normal(0.0, prior_scale).expand([n_site_covs]).to_event(1))
    alpha0 = numpyro.sample("alpha0", dist.Normal(0.0, prior_scale))
    alpha = numpyro.sample("alpha", dist.Normal(0.0, prior_scale).expand([n_obs_covs]).to_event(1))

    log_lam = beta0 + site_covs @ beta
    if site_random_effects:
        # Poisson-lognormal: the overdispersion stand-in for the NB2 the dynamic
        # model uses, since the N-mixture latent is Poisson by construction.
        re_sd = numpyro.sample("site_re_sd", dist.HalfNormal(1.0))
        re = numpyro.sample("site_re", dist.Normal(0.0, 1.0).expand([n_sites]).to_event(1))
        log_lam = log_lam + re_sd * re
    lam = numpyro.deterministic("abundance", jnp.exp(jnp.clip(log_lam, -20.0, 20.0)))

    oc = obs_covs[:, 0, :, :]                                   # (S, T, n_obs_covs)
    p = numpyro.deterministic("prob_detection",
                              jax.nn.sigmoid(alpha0 + oc @ alpha))

    y_raw = jnp.asarray(obs)[0, :, 0, :]                        # (S, T)
    mask = ~jnp.isnan(y_raw)
    y = jnp.nan_to_num(y_raw)
    numpyro.factor("marginal",
                   marginal_log_likelihood(y, mask, lam, p, max_abundance).sum())


def enumeration_free_bytes(n_sites, max_abundance, n_replicates, dtype_bytes=4):
    """Peak bytes for the direct sum: LINEAR in the ceiling, not quadratic."""
    return dtype_bytes * int(n_sites) * int(n_replicates) * (int(max_abundance) + 1)
