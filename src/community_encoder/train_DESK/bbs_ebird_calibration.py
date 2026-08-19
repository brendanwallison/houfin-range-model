"""Put BBS counts and eBird abundances on a common scale, as a latent-abundance model.

SHELVED. The pipeline trains on raw BBS alone (``target.ebird_window.enabled`` is false);
see "Why this is shelved". Kept as the reference implementation for integrating the two.

## Model

One latent abundance per cell-year, seen by two surveys with different observation processes:

    t_si                                    true log abundance, never observed
    B_si ~ Poisson(exp(t_si))               BBS: a count
    y_si = log k_s + d_s (t_si - lB0) + e   eBird: a continuous modelled index

with ``d_s = exp(delta_s)``, ``lB0`` the pooled mean of ``log B`` so ``k_s`` is eBird's value
at a typical count, and the hierarchy

    delta_s ~ Normal(mu_delta, tau_delta^2)     mu_delta ~ Normal(log 1, pop_delta^2)
    log k_s ~ Normal(mu_k,     tau_k^2)         mu_k     ~ Normal(0,     pop_k^2)

## Both noise levels are derived, not fitted

This is what identifies the model without any search, cap, or invented bound.

**BBS is Poisson**, so its log-scale variance is not free: by the delta method a count with
mean ``lam`` has log-variance about ``1/lam``, giving ``sx2_s = 1/mean(B_s)`` straight from the
counts. Rare species are automatically noisier, which a single shared noise level could not
express. Everything else then follows in closed form:

    w2_s  = Var(x_s) - sx2_s          the shared latent's variance
    d_s   = Cov_s / w2_s              the slope, corrected for BBS noise
    sy2_s = Var(y_s) - d_s^2 w2_s     eBird's noise falls out, it is not assumed

An ordinary regression of eBird on BBS would treat the count as exact; it is not, so every
slope comes back shrunk toward zero and the calibrated values compressed. Subtracting the
Poisson variance from the denominator is the correction.

**eBird's trustworthiness is a CHECK, not an input.** Supplying ``TRUST_RATIO`` as a second
assumption would over-determine the model, so it is only compared against the implied value.
When the implied eBird noise sits far above it, the two surveys share less variance than one
latent can account for, and the surplus is not measurement error but structure one survey sees
and the other does not.

## Applying it

``E = k_s (B/B0)^(d_s)``. A zero maps to a zero for any ``k`` and ``d``, so a surveyed
cell-year where the species went unrecorded cannot become a small presence -- ``k`` is
multiplicative and cannot create a floor the way an additive intercept would.

## Why this is shelved

Fitted on the real products (84 of 96 species, 379,246 paired cell-years) the two surveys
correlate a median 0.405 per species, and ~80% of eBird's spread is unexplained by BBS --
against the ~1% a product ten times more trustworthy would show. Some of that gap is real:
eBird resolves spatial structure BBS cannot see at 27 km, and the two have different sampling
emphases. But it means they are not close measurements of one quantity, and putting both in one
similarity kernel asks the encoder to reproduce both. Training on raw BBS alone avoids the
question. This model is what to come back to if the eBird half is wanted.
"""
import numpy as np

_TINY = 1e-12

#: eBird's error as a fraction of its own spread, relative to BBS's. eBird Status & Trends is a
#: modelled product built from far more observation effort than a single roadside morning, so it
#: is taken to be about an order of magnitude more trustworthy -- relative to its own scale,
#: since the two products are not on a common one.
TRUST_RATIO = 0.1


def species_moments(pairs_by_species, n_species):
    """Per-species sufficient statistics from log1p inputs.

    Returns ``(n, mean_x, mean_y, var_x, var_y, cov, poisson_var_x, log_B0)``, where
    ``poisson_var_x`` is ``1/mean(B)`` -- the log-scale variance a Poisson count carries,
    derived rather than fitted.
    """
    S = int(n_species)
    n = np.zeros(S, dtype="int64")
    mx = np.zeros(S); my = np.zeros(S)
    vx = np.zeros(S); vy = np.zeros(S); cxy = np.zeros(S); pvx = np.zeros(S)
    xs_all = []
    for s, (x1p, y1p) in pairs_by_species.items():
        s = int(s)
        if not (0 <= s < S):
            continue
        B = np.maximum(np.expm1(np.asarray(x1p, "float64")), 1e-9)
        x = np.log(B)
        y = np.log(np.maximum(np.expm1(np.asarray(y1p, "float64")), 1e-9))
        if x.size < 3:
            continue
        n[s] = x.size; mx[s] = x.mean(); my[s] = y.mean()
        vx[s] = x.var(); vy[s] = y.var(); cxy[s] = float(((x - mx[s]) * (y - my[s])).mean())
        # Poisson log-variance by the delta method: Var(log B) ~= Var(B)/E[B]^2 = 1/lambda,
        # so the plug-in is 1/mean(B). NOT mean(1/B), which is biased upward by Jensen and
        # badly so here, because conditioning on B > 0 fills it with 1/1 terms -- measured, it
        # over-corrected a true slope of 1.0 to 1.50.
        pvx[s] = float(1.0 / max(B.mean(), 1e-9))
        xs_all.append(x)
    log_B0 = float(np.concatenate(xs_all).mean()) if xs_all else 0.0
    return n, mx, my, vx, vy, cxy, pvx, log_B0


def corrected_slope(vx, vy, cxy, sx2):
    """Slope with the BBS measurement noise taken out of the denominator.

    ``d = Cov / (Var(x) - sx2)``. With ``sx2`` known this is the maximum-likelihood slope, and
    it needs nothing about the noise in y -- that only affects how precisely ``d`` is known.
    """
    return cxy / np.maximum(vx - sx2, _TINY)


def fit_hierarchical_calibration(pairs_by_species, n_species, prior_exponent=1.0,
                                 prior_log_exponent_sd=0.05, prior_log_scale_sd=0.10,
                                 population_log_exponent_sd=0.01,
                                 population_log_scale_sd=5.0,
                                 trust_ratio=TRUST_RATIO, n_iter=60, verbose=True):
    """Latent-abundance calibration of BBS onto the eBird scale, partially pooled.

    ``pairs_by_species``: ``{species_index: (x_bbs_log1p, y_ebird_log1p)}`` over cell-years
    where both products recorded the species.
    """
    S = int(n_species)
    n, mx, my, vx, vy, cxy, pvx, log_B0 = species_moments(pairs_by_species, S)
    tau_d = float(prior_log_exponent_sd); tau_k = float(prior_log_scale_sd)
    pop_d = float(population_log_exponent_sd); pop_k = float(population_log_scale_sd)
    mu_delta = float(np.log(prior_exponent)); mu_k = 0.0

    # BBS's noise, derived from its Poisson counts. Nothing is assumed about eBird's -- it
    # falls out below, and is then compared against what TRUST_RATIO would imply.
    sx2 = np.where(n > 0, pvx, np.nan)
    rel_x = np.where(vx > _TINY, sx2 / np.maximum(vx, _TINY), np.nan)

    # A species is usable when its two products covary positively and the Poisson noise does not
    # account for its whole observed spread; otherwise it carries no information about its own
    # exponent and takes the population value.
    ok = (n >= 3) & (cxy > _TINY) & (vx > sx2 + _TINY) & (vy > _TINY)
    d_raw = np.where(ok, corrected_slope(vx, vy, cxy, np.where(ok, sx2, 0.0)), np.nan)
    ok &= np.isfinite(d_raw) & (d_raw > _TINY)
    # eBird's implied noise, and the fraction of its own spread it represents.
    w2 = np.where(ok, vx - sx2, np.nan)
    # A variance cannot be negative; sampling can push the residual slightly below zero.
    sy2 = np.where(ok, np.maximum(vy - d_raw ** 2 * w2, 0.0), np.nan)
    rel_y = np.where(ok, sy2 / np.maximum(vy, _TINY), np.nan)
    delta_raw = np.where(ok, np.log(np.maximum(d_raw, _TINY)), np.nan)

    # How precisely this species pins down its own slope. The correction divides by the signal
    # variance, so a species whose observed spread is mostly Poisson noise knows its slope far
    # less well -- that inflation is what makes such a species pool harder.
    r2 = np.where(ok, cxy ** 2 / np.maximum(vx * vy, _TINY), 0.0)
    infl = np.where(ok, (vx / np.maximum(vx - sx2, _TINY)) ** 2, np.inf)
    se2 = np.where(ok, np.maximum(1.0 - r2, 1e-6) / np.maximum(n - 2, 1) * infl, np.inf)

    delta = np.full(S, mu_delta); logk = np.full(S, mu_k)
    for _ in range(int(n_iter)):
        wo = np.where(ok, 1.0 / se2, 0.0); wp = 1.0 / tau_d ** 2
        delta = np.where(ok, (wo * np.nan_to_num(delta_raw) + wp * mu_delta) / (wo + wp),
                         mu_delta)
        d = np.exp(delta)
        own_logk = np.where(ok, my - d * (mx - log_B0), np.nan)
        se2_k = np.where(ok, vy / np.maximum(n, 1), np.inf)
        wok = np.where(ok, 1.0 / se2_k, 0.0); wpk = 1.0 / tau_k ** 2
        logk = np.where(ok, (wok * np.nan_to_num(own_logk) + wpk * mu_k) / (wok + wpk), mu_k)
        if ok.any():
            mu_delta = (wp * delta[ok].sum() + np.log(prior_exponent) / pop_d ** 2) \
                       / (wp * ok.sum() + 1.0 / pop_d ** 2)
            mu_k = (wpk * logk[ok].sum()) / (wpk * ok.sum() + 1.0 / pop_k ** 2)

    delta = np.where(ok, delta, mu_delta); logk = np.where(ok, logk, mu_k)
    d = np.exp(delta); k = np.exp(logk)
    shrink = np.where(ok, (1.0 / tau_d ** 2) / (np.where(ok, 1.0 / se2, 0.0) + 1.0 / tau_d ** 2),
                      1.0)

    out = {"k": k, "d": d, "delta": delta, "log_k": logk, "n": n, "shrinkage": shrink,
           "log_B0": log_B0, "B0": float(np.exp(log_B0)),
           "bbs_noise_var": np.where(np.isfinite(sx2), sx2, 0.0),
           "ebird_noise_var": np.where(np.isfinite(sy2), sy2, 0.0),
           "trust_ratio": float(trust_ratio),
           "mu_delta": float(mu_delta), "mu_d": float(np.exp(mu_delta)),
           "mu_logk": float(mu_k), "mu_k": float(np.exp(mu_k)),
           "tau_delta": tau_d, "tau_logk": tau_k,
           "prior": {"exponent": float(prior_exponent),
                     "log_exponent_sd": tau_d, "log_scale_sd": tau_k,
                     "population_log_exponent_sd": pop_d,
                     "population_log_scale_sd": pop_k, "trust_ratio": float(trust_ratio)},
           "n_species_with_overlap": int(ok.sum()), "n_overlap_points": int(n.sum())}
    if verbose:
        have = n > 0
        print(f"[calib] BBS -> eBird, latent abundance, partially pooled over {S} species "
              f"({int(ok.sum())} usable, {int(n.sum()):,} paired cell-years)")
        if have.any():
            print(f"[calib] BBS noise from its Poisson counts: median var "
                  f"{np.median(sx2[have]):.4f} = {np.median(rel_x[have]):.1%} of its spread")
        if ok.any():
            implied = float(np.median(rel_y[ok]))
            expect = float(trust_ratio) * float(np.median(rel_x[ok]))
            print(f"[calib] eBird noise implied by the fit: {implied:.1%} of its spread; "
                  f"a {trust_ratio:g}x-more-trustworthy product would be {expect:.1%}")
            if implied > 3 * expect:
                print(f"[calib] NOTE the implied figure is {implied/max(expect,1e-9):.0f}x that. "
                      f"The surplus is not measurement error -- it is structure one survey sees "
                      f"and the other does not (eBird resolves scales below 27 km; the two have "
                      f"different sampling emphases), which a single-latent model cannot split.")
        print(f"[calib] population: exponent d={np.exp(mu_delta):.3f} "
              f"(1 = a straight proportion), scale k={np.exp(mu_k):.4f} at a typical count of "
              f"{np.exp(log_B0):.1f} birds")
        if ok.any():
            print(f"[calib] per-species d: median {np.median(d[ok]):.3f}, "
                  f"range {d[ok].min():.3f}..{d[ok].max():.3f}")
            print(f"[calib] per-species k: median {np.median(k[ok]):.4f}, "
                  f"range {k[ok].min():.4f}..{k[ok].max():.4f}")
            print(f"[calib] shrinkage toward the population: median "
                  f"{np.median(shrink[ok]):.3f}")
        if (~ok).any():
            print(f"[calib] {int((~ok).sum())} species carry no usable signal and sit on the "
                  f"population estimate")
    return out


def apply_calibration(X_bbs_log, cal):
    """Map log1p BBS values onto the eBird log1p scale via ``E = k (B/B0)^d`` (pure).

    A zero stays a zero for any ``k`` and ``d``, so the calibrated matrix has exactly the
    occupancy of the raw one.
    """
    X = np.asarray(X_bbs_log, "float64")
    k = np.asarray(cal["k"], "float64").reshape(1, -1)
    d = np.asarray(cal["d"], "float64").reshape(1, -1)
    if X.shape[1] != k.shape[1]:
        raise ValueError(f"calibration has {k.shape[1]} species, X has {X.shape[1]}")
    B0 = float(cal.get("B0", 1.0))
    B = np.expm1(X)
    E = np.where(B > 0, k * np.power(np.maximum(B, 0.0) / B0, d), 0.0)
    return np.log1p(np.clip(E, 0.0, None)).astype("float32")


def calibration_meta(cal, species):
    """JSON-safe per-species record for ``points_meta.json``."""
    bn = np.asarray(cal["bbs_noise_var"]); en = np.asarray(cal["ebird_noise_var"])
    return {
        "direction": "bbs_to_ebird", "method": "latent_abundance_deming_hierarchical_map",
        "form": "ebird = k_s * (bbs/B0)**d_s   (raw units; zero maps to zero)",
        "B0_typical_bbs_count": cal.get("B0"),
        "trust_ratio": cal["trust_ratio"],
        "measurement_noise": {"bbs_var_median": float(np.median(bn[bn > 0])) if (bn > 0).any()
                              else None,
                              "ebird_var_median": float(np.median(en[en > 0])) if (en > 0).any()
                              else None},
        "population": {"scale_k": cal["mu_k"], "exponent_d": cal["mu_d"]},
        "prior": cal["prior"],
        "n_species_with_overlap": cal["n_species_with_overlap"],
        "n_overlap_points": cal["n_overlap_points"],
        "per_species": {str(c): {"k": float(cal["k"][i]), "d": float(cal["d"][i]),
                                 "n": int(cal["n"][i]),
                                 "shrinkage": float(cal["shrinkage"][i])}
                        for i, c in enumerate(species)},
    }
