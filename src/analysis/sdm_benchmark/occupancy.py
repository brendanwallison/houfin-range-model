"""biolith occupancy / N-mixture baselines, and the abundance->occupancy step.

Two detection-corrected models, both reduced to an occupancy probability so they
are comparable with the BRT and with the dynamic model's lam >= 1 designation:

  occu()      -> psi directly. The ROBUST member of the pair.
  nmixture()  -> abundance lambda, converted to P(N>0).

WHY BOTH. N-mixture abundance is known to be sensitive to the assumed mixing
distribution and often weakly identified (Barker et al. 2018; Link et al. 2018),
and any occupancy derived from it inherits that. Running occu alongside is a
sensitivity analysis on exactly the fragile step: agreement over the Great Plains
means the designation is solid, divergence bounds the claim.

REPLICATES ARE YEARS, not stop bins. The 10-stop columns in the BBS States files
are contiguous along a single ~40 km route and are not independent replicates;
years within a closed window are the defensible choice.

THE CONVERSION. biolith's nmixture marginalizes the latent N out by enumeration,
so there are NO posterior draws of N to count -- ``abundance`` (the Poisson rate)
is what comes back. The conversion is therefore per-draw

    P(N>0 | lambda_draw) = 1 - exp(-lambda_draw)

averaged over draws. That is not a Poisson-only approximation standing in for
something better: biolith's latent IS Poisson given lambda, and when
``site_random_effects=True`` the random effect sits INSIDE lambda, so averaging
the per-draw probability marginalizes the overdispersion correctly. Applying
``1 - exp(-mean lambda)`` to the posterior MEAN instead would be wrong by
Jensen's inequality, which is the mistake this function exists to prevent.
"""
from __future__ import annotations

import numpy as np

# Detection can drift across the window; a centred year index is the obs-level
# covariate. NOTE the confound: within-window abundance trend and detection
# drift are not separable by this design, so a nonzero year effect on detection
# should be read alongside data.closure_check, not instead of it.
OBS_COV_YEAR = "year_centered"


def build_inputs(sites, counts, years, X_site, binary=False, add_year_obs_cov=True):
    """Assemble biolith's (site_covs, obs_covs, obs) from the replicate matrix.

    ``counts`` is (n_sites, n_replicates) with NaN where a route was not run
    that year -- biolith masks those natively (mask_missing_obs). ``binary=True``
    reduces counts to detections for occu().

    Shapes follow biolith's contract:
      site_covs (n_sites, n_site_covs)
      obs_covs  (n_sites, n_periods=1, n_replicates, n_obs_covs)
      obs       (n_species=1, n_sites, n_periods=1, n_replicates)
    """
    counts = np.asarray(counts, dtype=float)
    n_sites, n_rep = counts.shape
    if X_site.shape[0] != n_sites:
        raise ValueError(f"X_site has {X_site.shape[0]} rows for {n_sites} sites.")
    if np.isnan(X_site).any():
        raise ValueError(
            "NaN in site_covs would mask EVERY observation at that site "
            "(biolith propagates covariate NaN into the obs mask). Impute or "
            "drop those sites before fitting.")

    y = (counts > 0).astype(float) if binary else counts
    y = np.where(np.isnan(counts), np.nan, y)
    obs = y[None, :, None, :]                         # (1, n_sites, 1, n_rep)

    if add_year_obs_cov:
        yc = np.asarray(years, dtype=float)
        yc = (yc - yc.mean()) / max(yc.std(), 1.0)
        oc = np.broadcast_to(yc[None, None, :, None], (n_sites, 1, n_rep, 1))
    else:
        oc = np.zeros((n_sites, 1, n_rep, 1), dtype=float)
    return {"site_covs": np.asarray(X_site, dtype=float),
            "obs_covs": np.asarray(oc, dtype=float),
            "obs": obs}


def suggest_max_abundance(counts, margin=100, ceiling=2000, detection_floor=None):
    """Size the latent-N summation ceiling, following unmarked::pcount.

    unmarked's convention is K = max(y) + 100: N cannot plausibly exceed the
    largest observed count by a wide margin, because detection is not that poor.
    The earlier rule here was max(y) / detection_floor with a 0.25 floor, which
    for House Finch gave 1960 against a 490-bird western route -- over 3x larger
    than the convention, and cost is linear in it, so it made every NUTS
    gradient step over 3x more expensive for no gain.

    ``detection_floor`` keeps the old behaviour when passed explicitly.

    THE CEILING IS AN ASSUMPTION, not a free parameter: the right check is that
    the answer does not move when it is raised. Use max_abundance_sensitivity.
    """
    m = float(np.nanmax(counts)) if np.isfinite(np.nanmax(counts)) else 0.0
    if detection_floor:
        need = int(np.ceil(m / max(detection_floor, 1e-3)))
    else:
        need = int(np.ceil(m)) + int(margin)
    return int(min(max(need, 10), ceiling)), need


def max_abundance_sensitivity(psi_low, psi_high):
    """How much a raised ceiling moved the answer. Near zero = ceiling is fine."""
    a, b = np.asarray(psi_low, float), np.asarray(psi_high, float)
    return {"max_abs_shift": float(np.nanmax(np.abs(a - b))),
            "mean_abs_shift": float(np.nanmean(np.abs(a - b))),
            "correlation": float(np.corrcoef(a, b)[0, 1])}


def fit_occu(inputs, coords=None, num_samples=1000, num_warmup=1000,
             num_chains=4, seed=0, **kwargs):
    """Fit biolith occu(). Returns the FitResult.

    ``extra_fields`` is requested so occu gets the SAME divergence and
    tree-depth diagnostics as the abundance fit. It was previously omitted, so
    the only saturation warning the benchmark ever printed was nmixture's and
    occu's sampler behaviour was simply unknown -- a hole in the diagnostic that
    was meant to say which fits to trust. biolith's fit() forwards **kwargs to
    mcmc.run, so this reaches NumPyro unchanged.
    """
    from biolith.models import occu
    from biolith.utils import fit
    kw = dict(inputs)
    if coords is not None:
        kw["coords"] = np.asarray(coords, dtype=float)
    kwargs.setdefault("extra_fields", ("diverging", "num_steps", "accept_prob",
                                       "adapt_state.step_size"))
    return fit(occu, num_samples=num_samples, num_warmup=num_warmup,
               num_chains=num_chains, random_seed=seed, **kw, **kwargs)


def enumeration_bytes(n_sites, max_abundance, dtype_bytes=4):
    """Peak bytes for nmixture's latent-N enumeration.

    biolith enumerates N over 0..max_abundance and the Categorical log_prob
    broadcast materializes an (n_sites, M+1, M+1)-shaped term, so cost is
    QUADRATIC in max_abundance. Verified against an observed failure: 3853 sites
    at max_abundance=1960 requested exactly 55.20 GiB.
    """
    return dtype_bytes * int(n_sites) * (int(max_abundance) + 1) ** 2


def max_abundance_for_budget(n_sites, budget_bytes, dtype_bytes=4):
    """Largest max_abundance whose enumeration fits in ``budget_bytes``."""
    import math
    return max(1, int(math.isqrt(int(budget_bytes) //
                                 max(dtype_bytes * int(n_sites), 1))) - 1)


def check_enumeration_budget(n_sites, max_abundance, budget_gib=8.0):
    """Refuse an nmixture fit that cannot fit, BEFORE the GPU allocates.

    An OOM deep inside NUTS initialization loses whatever else the job had
    already computed; this turns it into an actionable message up front.
    """
    need = enumeration_bytes(n_sites, max_abundance)
    budget = budget_gib * 2 ** 30
    if need <= budget:
        return need
    fits = max_abundance_for_budget(n_sites, budget)
    raise MemoryError(
        f"nmixture enumeration needs {need / 2**30:.1f} GiB for {n_sites} sites "
        f"at max_abundance={max_abundance} (budget {budget_gib:.1f} GiB). Cost is "
        f"QUADRATIC in max_abundance.\n"
        f"Options: restrict to a region whose counts are smaller (Great Plains "
        f"tops out near 79 and the East near 115, while the western native range "
        f"reaches 490); pass max_abundance<={fits}, ACCEPTING that abundances "
        f"above it are truncated and biased low; or fit occu() only, which is "
        f"the robust member of the pair and needs no enumeration.")


def fit_nmixture_marginal(inputs, max_abundance, mixture="NB",
                          site_random_effects=False,
                          num_samples=1000, num_warmup=1000, num_chains=4,
                          seed=0, budget_gib=8.0, **kwargs):
    """N-mixture with the latent N summed out directly (Royle 2004).

    Overdispersion comes from a NEGATIVE-BINOMIAL latent (unmarked::pcount's
    mixture="NB"), not from per-site random effects: one dispersion parameter
    instead of one latent per site. Per-site effects both wreck the sampler
    geometry and absorb the variation the benchmark wants attributed to
    environment, competing with the covariate coefficients that generate the
    suitability surface.

    The DEFAULT abundance fit. biolith's nmixture enumerates a Categorical and
    is quadratic in the ceiling (55 GiB at the ceiling this data implies); the
    direct sum is linear (~0.3 GiB) and agrees with it to MCMC noise. Same
    model, standard formulation -- see nmixture_marginal's module docstring.
    """
    import jax
    from numpyro.infer import MCMC, NUTS

    from src.analysis.sdm_benchmark import nmixture_marginal as nm

    n_sites = np.asarray(inputs["obs"]).shape[1]
    n_rep = np.asarray(inputs["obs"]).shape[-1]
    need = nm.enumeration_free_bytes(n_sites, max_abundance, n_rep)
    if need > budget_gib * 2 ** 30:
        fits = int(budget_gib * 2 ** 30 //
                   max(4 * n_sites * n_rep, 1)) - 1
        raise MemoryError(
            f"direct-sum N-mixture needs {need / 2**30:.1f} GiB for {n_sites} "
            f"sites x {n_rep} visits at max_abundance={max_abundance} "
            f"(budget {budget_gib:.1f} GiB). Cost is linear in the ceiling; "
            f"max_abundance<={fits} would fit.")

    kernel = NUTS(nm.nmixture)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(int(seed)),
             site_covs=inputs["site_covs"], obs_covs=inputs["obs_covs"],
             obs=inputs["obs"], max_abundance=int(max_abundance),
             mixture=str(mixture),
             site_random_effects=bool(site_random_effects),
             extra_fields=("diverging", "num_steps", "accept_prob",
                      "adapt_state.step_size"), **kwargs)

    class _R:                       # same surface as biolith's FitResult
        samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        diagnostics = sampler_diagnostics(mcmc)
    _R.mcmc = mcmc                  # needed for PER-CHAIN R-hat
    return _R


def fit_nmixture(inputs, max_abundance, coords=None, site_random_effects=True,
                 num_samples=1000, num_warmup=1000, num_chains=4, seed=0,
                 budget_gib=8.0, **kwargs):
    """Fit biolith nmixture().

    ``site_random_effects=True`` makes the latent abundance Poisson-lognormal,
    i.e. overdispersed -- the stand-in for the NB2 the dynamic model uses, since
    nmixture itself offers only a Poisson latent.

    NOTE this is biolith's ENUMERATED implementation, quadratic in
    max_abundance. Prefer fit_nmixture_marginal unless you specifically want to
    cross-check against biolith; this one exists for that comparison.
    """
    from biolith.models import nmixture
    from biolith.utils import fit
    check_enumeration_budget(np.asarray(inputs["obs"]).shape[1], max_abundance,
                             budget_gib=budget_gib)
    kw = dict(inputs)
    if coords is not None:
        kw["coords"] = np.asarray(coords, dtype=float)
    return fit(nmixture, max_abundance=int(max_abundance),
               site_random_effects=bool(site_random_effects),
               num_samples=num_samples, num_warmup=num_warmup,
               num_chains=num_chains, random_seed=seed, **kw, **kwargs)


def _site_axis(arr):
    """Collapse a posterior site array to (n_draws, n_sites)."""
    a = np.asarray(arr)
    return a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a


def psi_from_occu(result):
    """Posterior-mean occupancy probability per site, from occu()."""
    return _site_axis(result.samples["psi"]).mean(axis=0)


def psi_from_nmixture(result, max_abundance=None, concentration=None):
    """Posterior-mean P(N>0) per site, converted PER DRAW from abundance.

    THE ZERO PROBABILITY MUST MATCH THE LATENT. For a Poisson latent
    P(N=0) = exp(-lambda); for the NEGATIVE BINOMIAL latent this model now uses
    by default it is (phi/(phi+lambda))^phi, which is strictly LARGER -- an NB
    puts more mass on zero at the same mean. Applying the Poisson formula to an
    NB fit therefore OVERSTATES occupancy, and by a lot when phi is small. The
    concentration is picked up automatically from the fit when present.

    Averaging per draw is the other half: applying either formula to the
    posterior-mean lambda is biased by Jensen's inequality.
    """
    lam = _site_axis(result.samples["abundance"])
    if concentration is None:
        c = (result.samples or {}).get("concentration")
        if c is not None:
            c = np.asarray(c)
            concentration = c.reshape(c.shape[0], *([1] * (lam.ndim - 1)))

    if concentration is None:                      # Poisson latent
        p0 = np.exp(-lam)
    else:                                          # NB2 latent
        phi = np.asarray(concentration, dtype=float)
        p0 = np.exp(phi * np.log(phi / (phi + lam)))

    if max_abundance is not None and concentration is None:
        from scipy.stats import poisson
        norm = poisson.cdf(int(max_abundance), lam)
        return float_clip(1.0 - p0 / np.maximum(norm, 1e-12)).mean(axis=0)
    return float_clip(1.0 - p0).mean(axis=0)


def float_clip(a):
    return np.clip(a, 0.0, 1.0)


def analytic_poisson_check(result):
    """Cross-check: 1 - exp(-mean lambda) vs the correct per-draw mean.

    A large gap is evidence of real posterior spread in lambda (and, with site
    random effects on, of the overdispersion being estimated) -- report it rather
    than quietly picking one.
    """
    lam = _site_axis(result.samples["abundance"])
    per_draw = (1.0 - np.exp(-lam)).mean(axis=0)
    naive = 1.0 - np.exp(-lam.mean(axis=0))
    return {"per_draw_mean": per_draw, "naive_on_mean": naive,
            "max_abs_gap": float(np.nanmax(np.abs(per_draw - naive)))}


def sampler_diagnostics(mcmc, max_tree_depth=10):
    """Divergences and max-treedepth saturation from a raw numpyro MCMC.

    A run pinned at 2^max_tree_depth - 1 leapfrog steps every iteration is
    hitting the depth limit, which means a badly conditioned posterior -- slow
    AND poorly mixed. Reporting it beats inferring it from a progress bar after
    the fact. The threshold is derived from the kernel's depth rather than
    hardcoded, so it stays right if the depth is ever raised.
    """
    try:
        extra = mcmc.get_extra_fields()
    except Exception:
        return {}
    out = {}
    if "diverging" in extra:
        out["divergences"] = int(np.asarray(extra["diverging"]).sum())
    for key, name in (("accept_prob", "mean_accept_prob"),):
        if key in extra:
            out[name] = float(np.asarray(extra[key]).mean())
    if "adapt_state.step_size" in extra:
        out["step_size"] = float(np.asarray(extra["adapt_state.step_size"]).mean())
    if "num_steps" in extra:
        steps = np.asarray(extra["num_steps"])
        limit = 2 ** int(max_tree_depth) - 1
        out["mean_steps"] = float(steps.mean())
        out["max_treedepth_steps"] = int(limit)
        out["frac_at_max_treedepth"] = float((steps >= limit).mean())
    return out


def diagnostics_of(result, max_tree_depth=10):
    """Sampler diagnostics from any fit result carrying an .mcmc object."""
    mcmc = getattr(result, "mcmc", None)
    if mcmc is None:
        return {}
    return sampler_diagnostics(mcmc, max_tree_depth=max_tree_depth)


def _grouped_samples(result):
    """Per-chain posterior draws, shaped (n_chains, n_draws, ...).

    get_samples() CONCATENATES chains, so feeding it to Gelman-Rubin measures a
    single chain against itself and yields nan. biolith's FitResult carries the
    numpyro MCMC object as .mcmc, and the direct-sum fitter exposes the same, so
    ask it for group_by_chain=True.
    """
    mcmc = getattr(result, "mcmc", None)
    if mcmc is None:
        return None
    try:
        return mcmc.get_samples(group_by_chain=True)
    except Exception:
        return None


def convergence(result, max_params=None):
    """R-hat / ESS per parameter, from PER-CHAIN draws.

    Returns a dict with a ``worst`` summary so a caller does not have to scan.
    Deterministic site-level arrays (psi, abundance, prob_detection) are skipped:
    they are thousands of derived quantities whose R-hat says little that the
    parameters driving them do not.
    """
    import numpyro.diagnostics as diag

    grouped = _grouped_samples(result)
    if not grouped:
        return {"error": "no per-chain draws available; R-hat undefined"}

    skip = {"psi", "abundance", "prob_detection", "prob_detection_fp",
            "site_re", "N_i"}
    out = {}
    for k, v in grouped.items():
        if k in skip:
            continue
        a = np.asarray(v)
        if a.ndim < 2 or a.shape[0] < 2:
            continue                      # need >= 2 chains for R-hat
        try:
            rh = np.asarray(diag.split_gelman_rubin(a))
            ess = np.asarray(diag.effective_sample_size(a))
        except Exception:
            continue
        out[k] = {"r_hat_max": float(np.nanmax(rh)),
                  "ess_min": float(np.nanmin(ess))}
    if not out:
        return {"error": "R-hat needs at least 2 chains"}
    worst = max(out.items(), key=lambda kv: kv[1]["r_hat_max"])
    out["worst"] = {"param": worst[0], **worst[1]}
    return out


def worst_r_hat(conv):
    """The single number to look at, or nan when it could not be computed."""
    if not conv or "worst" not in conv:
        return float("nan")
    return float(conv["worst"]["r_hat_max"])


def identifiability(result, pairs=(("beta0", "alpha0"),)):
    """Posterior correlation between abundance and detection intercepts.

    THE DECISIVE TEST for why an N-mixture samples badly. lambda and p are only
    weakly separable -- their product is well determined while the split between
    them is not -- and that ridge shows up directly as a near +-1 posterior
    correlation between the abundance intercept and the detection intercept.

    This distinguishes the two competing explanations for tree-depth saturation:
    a correlation near +-1 means the ridge is INTRINSIC to the N-mixture and no
    amount of covariate rotation will fix it, while a modest correlation means
    the problem was the design matrix and PCA should have helped. The
    literature's identifiability warnings (Barker 2018, Link 2018) are about
    exactly this quantity.
    """
    samples = getattr(result, "samples", {}) or {}
    out = {}
    for a, b in pairs:
        if a in samples and b in samples:
            x = np.asarray(samples[a]).ravel()
            y = np.asarray(samples[b]).ravel()
            if x.size == y.size and x.size > 2:
                out[f"corr({a},{b})"] = float(np.corrcoef(x, y)[0, 1])
    return out


def detection_summary(result):
    """Posterior detection probability. Degenerate near 0 or 1."""
    samples = getattr(result, "samples", {}) or {}
    p = samples.get("prob_detection")
    if p is None and "alpha0" in samples:
        a = np.asarray(samples["alpha0"]).ravel()
        p = 1.0 / (1.0 + np.exp(-a))
    if p is None:
        return {}
    p = np.asarray(p, dtype=float)
    return {"mean": float(np.nanmean(p)),
            "q05": float(np.nanpercentile(p, 5)),
            "q95": float(np.nanpercentile(p, 95)),
            "degenerate": bool(np.nanmean(p) > 0.98 or np.nanmean(p) < 0.02)}
