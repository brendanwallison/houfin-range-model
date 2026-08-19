"""Put raw BBS counts on the eBird abundance scale, per species, by partial pooling.

BBS counts birds on a survey route; eBird Status & Trends gives a modelled relative
abundance. Different units. For the two to be rows of one community matrix feeding one
Ruzicka kernel they have to share a scale, because Ruzicka is ``sum(min)/sum(max)`` and is
scale-sensitive in each argument -- an uncalibrated mix would make the data SOURCE itself a
similarity signal.

**eBird is the common frame, and BBS is what gets transformed.** eBird is the richer product:
17,205 cells against BBS's ~3,900, denser per row, and modelled from far more observation
effort. So the fitted transform lands on the sparser data rather than on the majority of it.
Worth stating the consequence: BBS log-counts vary roughly 3.4x as much as eBird's, so this
compresses BBS's spread. Much of that extra spread is single-route sampling noise rather than
signal, which is the argument for compressing it -- and it is a linear rescale, so it borrows
nothing across space the way a spatial smoother would.

## Partial pooling, not thresholds

Every species gets its own slope and intercept, drawn from a population distribution:

    log1p(ebird)_si = a_s + exp(beta_s) * log1p(bbs)_si + noise
    beta_s ~ Normal(mu_beta, tau_beta)      a_s ~ Normal(mu_a, tau_a)

The slope is parameterised as ``exp(beta)`` rather than fitted directly, and that is load
bearing. A calibration factor between two abundance measures has to be POSITIVE. Fitted
directly, a species whose two products happen to disagree can land on a slope near zero --
which flattens it to a constant in every cell, destroying whatever BBS knew about it -- or
below zero, which inverts it, so more BBS birds would mean less eBird abundance. Neither is a
calibration. On the log scale both are unreachable: ``exp(beta)`` is positive for every
``beta``, and the prior pulling ``beta`` toward 0 resists collapse toward zero rather than
merely discouraging it. No threshold is involved.

A species with thousands of overlapping cell-years is fitted almost entirely by its own data.
A species with five, or with no real relationship between the two products, is pulled to the
population estimate -- automatically, and in proportion to how much its data actually says.
A species with NO overlap lands exactly on the population estimate, with no special case.

This is what replaces an earlier design with hard cutoffs (a minimum overlap count and a
minimum correlation, below which a species was dropped to a pooled fit). Those numbers were
invented, and they made the treatment of a species discontinuous in its evidence: 49 pairs
behaved completely differently from 51. Shrinkage is the same idea done continuously, and it
needs no invented constants.

## The population prior

``mu_beta`` is centred on 0 (so a slope of 1) and ``mu_a`` on 0, which says the two products
correspond up to a pure scale factor -- a proportional relationship rather than a power one.
That is the honest prior belief: they are two measurements of the same thing. Departures are
allowed, but a species has to produce evidence for one, and the population level is itself
estimated, so enough species agreeing on something else will move it.

## MAP, not a posterior

We need point estimates to apply to the data, not uncertainty about them. Coordinate ascent
alternates per-species fits, the population location and spread, and the noise scale, to
convergence. The log-slope makes the per-species step non-conjugate, so it is a small 2-D
optimisation with an analytic gradient computed from sufficient statistics rather than a
closed form -- still deterministic, still no sampler, and no JAX dependency in a CPU
preprocessing stage.
"""
import numpy as np

_MIN_SD = 1e-6


def _species_map(stats, beta0, a0, mu_beta, tau_beta, mu_a, tau_a, sigma):
    """MAP ``(beta, a)`` for one species given the population, from sufficient statistics.

    Minimises the negative log posterior

        sum_i (y_i - a - e^beta x_i)^2 / (2 sigma^2)
            + (beta - mu_beta)^2 / (2 tau_beta^2) + (a - mu_a)^2 / (2 tau_a^2)

    All the data enters through ``(n, Sx, Sy, Sxx, Sxy)``, so the cost does not grow with the
    number of paired observations. The gradient is analytic, so the optimiser is deterministic
    and needs no finite differences.
    """
    from scipy.optimize import minimize

    n, Sx, Sy, Sxx, Sxy, _Syy = stats
    s2 = sigma ** 2

    def nlp(th):
        beta, a = th
        B = np.exp(beta)
        # residual sum of squares, expanded so only the sufficient statistics appear
        rss = (_Syy - 2.0 * a * Sy - 2.0 * B * Sxy + 2.0 * a * B * Sx
               + a * a * n + B * B * Sxx)
        f = rss / (2.0 * s2) + (beta - mu_beta) ** 2 / (2.0 * tau_beta ** 2) \
            + (a - mu_a) ** 2 / (2.0 * tau_a ** 2)
        r_sum = Sy - n * a - B * Sx                 # sum of residuals
        rx_sum = Sxy - a * Sx - B * Sxx             # sum of residual * x
        g_a = -r_sum / s2 + (a - mu_a) / tau_a ** 2
        g_b = -B * rx_sum / s2 + (beta - mu_beta) / tau_beta ** 2
        return f, np.array([g_b, g_a])

    res = minimize(nlp, np.array([beta0, a0]), jac=True, method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def fit_hierarchical_calibration(pairs_by_species, n_species, prior_slope=1.0,
                                 prior_intercept=0.0, prior_log_slope_sd=0.5,
                                 prior_intercept_sd=1.0, n_iter=60, tol=1e-9,
                                 verbose=True):
    """Partially pooled per-species calibration of BBS onto the eBird scale.

    ``pairs_by_species``: ``{species_index: (x_bbs_log1p, y_ebird_log1p)}`` over overlapping
    cell-years, restricted to rows where both values are > 0.

    Returns per-species ``a``/``b`` (length ``n_species``), the population
    ``mu_beta``/``mu_a``/``tau_beta``/``tau_a``/``sigma``, per-species ``n``, and
    ``shrinkage`` -- the share of each species' estimate that came from the population rather
    than from its own data. That last one is the honest replacement for a pass/fail flag: how
    much a species' calibration rests on its own evidence is a continuous quantity, so it is
    reported as one.
    """
    S = int(n_species)
    stats = np.zeros((S, 6))                        # n, Sx, Sy, Sxx, Sxy, Syy
    for s, (x, y) in pairs_by_species.items():
        s = int(s)
        if not (0 <= s < S):
            continue
        x = np.asarray(x, "float64"); y = np.asarray(y, "float64")
        stats[s] = [x.size, x.sum(), y.sum(), float(x @ x), float(x @ y), float(y @ y)]
    n = stats[:, 0].astype("int64")
    N = int(n.sum())

    mu_beta, mu_a = float(np.log(prior_slope)), float(prior_intercept)
    tau_beta, tau_a = float(prior_log_slope_sd), float(prior_intercept_sd)
    beta = np.full(S, mu_beta); a = np.full(S, mu_a)
    sigma = 1.0
    if N > 0:
        ybar = stats[:, 2].sum() / N
        sigma = max(float(np.sqrt(max(stats[:, 5].sum() / N - ybar ** 2, 1e-12))), _MIN_SD)

    for _ in range(int(n_iter)):
        prev = np.array([mu_beta, mu_a, tau_beta, tau_a, sigma])
        for s in range(S):
            if n[s] == 0:
                beta[s], a[s] = mu_beta, mu_a       # no data: exactly the population estimate
                continue
            beta[s], a[s] = _species_map(stats[s], beta[s], a[s], mu_beta, tau_beta,
                                         mu_a, tau_a, sigma)
        # Population LOCATION is estimated; population SPREAD is not, and that is deliberate.
        # Joint MAP over a hierarchical variance is degenerate in both directions, and both
        # showed up here in testing. When species agree, the empirical spread collapses to zero
        # and the prior becomes infinitely strong, pooling every species completely. When one
        # species disagrees, its departure inflates the spread, which weakens the prior, which
        # lets it depart further -- a runaway. Neither is a property of the data; both are
        # artefacts of taking the mode of a variance. So the spread stays at the prior scale:
        # one stated belief about how much species' calibrations differ from each other, which
        # is exactly what a prior is for.
        wb = 1.0 / tau_beta ** 2
        mu_beta = (wb * beta.sum() + float(np.log(prior_slope)) / prior_log_slope_sd ** 2) / \
                  (wb * S + 1.0 / prior_log_slope_sd ** 2)
        wa = 1.0 / tau_a ** 2
        mu_a = (wa * a.sum() + float(prior_intercept) / prior_intercept_sd ** 2) / \
               (wa * S + 1.0 / prior_intercept_sd ** 2)
        if N > 0:
            rss = 0.0
            for s in range(S):
                if n[s] == 0:
                    continue
                ns, Sx, Sy, Sxx, Sxy, Syy = stats[s]
                B = np.exp(beta[s])
                rss += (Syy - 2 * a[s] * Sy - 2 * B * Sxy + 2 * a[s] * B * Sx
                        + a[s] ** 2 * ns + B ** 2 * Sxx)
            sigma = max(float(np.sqrt(max(rss, 0.0) / N)), _MIN_SD)
        if np.max(np.abs(prev - np.array([mu_beta, mu_a, tau_beta, tau_a, sigma]))) < tol:
            break

    # Final per-species pass against the converged population, so no species is left holding an
    # estimate fitted against a stale population value.
    for s in range(S):
        if n[s] == 0:
            beta[s], a[s] = mu_beta, mu_a
        else:
            beta[s], a[s] = _species_map(stats[s], beta[s], a[s], mu_beta, tau_beta,
                                         mu_a, tau_a, sigma)

    b = np.exp(beta)
    # Curvature of the likelihood in beta at the optimum, against the prior's: the share of the
    # estimate coming from the population rather than the species' own data.
    shrink = np.ones(S)
    for s in range(S):
        if n[s] == 0:
            continue
        like = (b[s] ** 2) * stats[s, 3] / sigma ** 2        # d2/dbeta2 of the fit term
        shrink[s] = float((1.0 / tau_beta ** 2) / (like + 1.0 / tau_beta ** 2))

    out = {"a": a, "b": b, "beta": beta, "n": n, "shrinkage": shrink,
           "mu_a": float(mu_a), "mu_beta": float(mu_beta), "mu_b": float(np.exp(mu_beta)),
           "tau_a": float(tau_a), "tau_beta": float(tau_beta), "sigma": float(sigma),
           "prior": {"slope": float(prior_slope), "intercept": float(prior_intercept),
                     "log_slope_sd": float(prior_log_slope_sd),
                     "intercept_sd": float(prior_intercept_sd)},
           "n_species_with_overlap": int((n > 0).sum()), "n_overlap_points": N}
    if verbose:
        fitted = n > 0
        print(f"[calib] BBS -> eBird, partially pooled over {S} species "
              f"({int(fitted.sum())} with overlap, {N:,} paired cell-years)")
        print(f"[calib] population slope {np.exp(mu_beta):.3f} "
              f"(log-slope {mu_beta:+.3f} +/- {tau_beta:.3f}), "
              f"intercept {mu_a:+.3f} +/- {tau_a:.3f}, residual sd {sigma:.3f}")
        if fitted.any():
            print(f"[calib] per-species slope: median {np.median(b[fitted]):.3f}, "
                  f"range {b[fitted].min():.3f}..{b[fitted].max():.3f} "
                  f"(positive by construction)")
            print(f"[calib] shrinkage toward the population: median "
                  f"{np.median(shrink[fitted]):.3f}, "
                  f"{int((shrink > 0.5).sum())} species more population than own data")
        if (~fitted).any():
            print(f"[calib] {int((~fitted).sum())} species have no overlap and sit exactly on "
                  f"the population estimate")
    return out


def apply_calibration(X_bbs_log, cal):
    """Map log1p BBS values onto the eBird log1p scale: ``a[s] + b[s]*x`` (pure).

    Clipped at 0 because the output feeds log1p-space Ruzicka, where a negative value is not a
    smaller abundance -- it is outside the domain, and ``sum(min)/sum(max)`` would silently
    produce a similarity above 1 or flip the denominator's sign.
    """
    X = np.asarray(X_bbs_log, "float64")
    a = np.asarray(cal["a"], "float64").reshape(1, -1)
    b = np.asarray(cal["b"], "float64").reshape(1, -1)
    if X.shape[1] != a.shape[1]:
        raise ValueError(f"calibration has {a.shape[1]} species, X has {X.shape[1]}")
    return np.clip(a + b * X, 0.0, None).astype("float32")


def calibration_meta(cal, species):
    """JSON-safe per-species record for ``points_meta.json``.

    ``shrinkage`` is the field to read: near 1 means that species' calibration is essentially
    the population relationship because its own data said little, near 0 means its own data
    determined it. There is no pass/fail flag to read instead, deliberately -- how much a
    species' estimate rests on its own evidence is a continuous quantity.
    """
    return {
        "direction": "bbs_to_ebird", "method": "hierarchical_map",
        "population": {"slope": cal["mu_b"], "log_slope": cal["mu_beta"],
                       "log_slope_sd": cal["tau_beta"],
                       "intercept": cal["mu_a"], "intercept_sd": cal["tau_a"],
                       "residual_sd": cal["sigma"]},
        "prior": cal["prior"],
        "n_species_with_overlap": cal["n_species_with_overlap"],
        "n_overlap_points": cal["n_overlap_points"],
        "per_species": {str(c): {"a": float(cal["a"][i]), "b": float(cal["b"][i]),
                                 "n": int(cal["n"][i]),
                                 "shrinkage": float(cal["shrinkage"][i])}
                        for i, c in enumerate(species)},
    }
