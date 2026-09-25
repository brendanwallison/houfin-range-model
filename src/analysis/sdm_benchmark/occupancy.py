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


def suggest_max_abundance(counts, detection_floor=0.25, ceiling=2000):
    """Size nmixture's latent-N enumeration from the data.

    nmixture enumerates N over 0..max_abundance and observes Binomial(N, p), so
    max_abundance must exceed the largest plausible TRUE abundance -- not the
    largest observed count. With detection p, N ~ count / p, hence the floor.
    The default 100 is far too small for House Finch (route counts reach ~490 in
    the western native range), and silently truncating the support would bias
    every abundance downward.
    """
    m = float(np.nanmax(counts)) if np.isfinite(np.nanmax(counts)) else 0.0
    need = int(np.ceil(m / max(detection_floor, 1e-3)))
    return int(min(max(need, 10), ceiling)), need


def fit_occu(inputs, coords=None, num_samples=1000, num_warmup=1000,
             num_chains=4, seed=0, **kwargs):
    """Fit biolith occu(). Returns the FitResult."""
    from biolith.models import occu
    from biolith.utils import fit
    kw = dict(inputs)
    if coords is not None:
        kw["coords"] = np.asarray(coords, dtype=float)
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


def fit_nmixture(inputs, max_abundance, coords=None, site_random_effects=True,
                 num_samples=1000, num_warmup=1000, num_chains=4, seed=0,
                 budget_gib=8.0, **kwargs):
    """Fit biolith nmixture().

    ``site_random_effects=True`` makes the latent abundance Poisson-lognormal,
    i.e. overdispersed -- the stand-in for the NB2 the dynamic model uses, since
    nmixture itself offers only a Poisson latent.
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


def psi_from_nmixture(result, max_abundance=None):
    """Posterior-mean P(N>0) per site, converted PER DRAW from abundance.

    Averaging ``1 - exp(-lambda)`` over draws is the correct marginalization;
    applying the formula to the posterior-mean lambda is biased by Jensen. When
    ``max_abundance`` is given, the truncation is corrected exactly -- negligible
    unless the enumeration ceiling is close to the fitted abundances, which is
    itself a sign the ceiling is too low.
    """
    lam = _site_axis(result.samples["abundance"])
    p0 = np.exp(-lam)
    if max_abundance is not None:
        from scipy.stats import poisson
        norm = poisson.cdf(int(max_abundance), lam)
        return float_clip(1.0 - p0 / np.maximum(norm, 1e-12)).mean(axis=0)
    return (1.0 - p0).mean(axis=0)


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


def convergence(result, sites=("psi", "abundance")):
    """R-hat / ESS summary for the fitted parameters that matter."""
    import numpyro.diagnostics as diag
    out = {}
    for k, v in result.samples.items():
        if k.startswith("cov_") or k in sites or k.endswith("_sd"):
            a = np.asarray(v)
            try:
                out[k] = {"r_hat_max": float(np.nanmax(diag.gelman_rubin(a[None]))),
                          "ess_min": float(np.nanmin(diag.effective_sample_size(a[None])))}
            except Exception:
                continue
    return out
