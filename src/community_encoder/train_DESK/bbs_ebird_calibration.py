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


def _spread_ratio_estimate(stats):
    """One species' own slope and intercept from matching SPREADS, not least squares.

    ``b = sd(y)/sd(x)`` and ``a = mean(y) - b*mean(x)``. Returns ``(log b, a, se)`` where ``se``
    is the approximate standard error of ``log b``, which is what tells the pooling step how
    much this species' own data should count.

    Why spreads rather than a regression fit -- this is the crux, and it is measured rather
    than assumed. A least-squares fit gives the best estimate of each INDIVIDUAL value: it
    knows part of a high BBS count is survey luck and pulls it back toward typical. But we
    never use individual values. The target is consumed only through a Ruzicka similarity
    table, and pulling every cell toward the mean makes all cells more alike -- so similarities
    come out inflated and, worse, the SPREAD of similarities gets compressed, which is exactly
    the structure the kernel needs to tell cells apart.

    Measured on simulated data with known truth, across BBS noise sd 0.2 to 1.2: least squares
    wins on individual values at every noise level, and loses on the similarity table at every
    noise level, by up to 3.4x. Median similarity under least squares drifts from a true 0.708
    to 0.828 as noise grows; matching spreads holds it at 0.708 throughout.

    That drift is also the reason this matters beyond accuracy. It grows with noise, so noisy
    BBS rows would end up systematically more self-similar than clean eBird rows, and the
    kernel could pick up which survey a row came from as if it were ecology.

    ``se(log(sd_y/sd_x)) ~= 1/sqrt(n-1)`` for moderate n, which is all the pooling step needs.
    """
    n, Sx, Sy, Sxx, Sxy, Syy = stats
    if n < 3:
        return None
    vx = max(Sxx / n - (Sx / n) ** 2, 0.0)
    vy = max(Syy / n - (Sy / n) ** 2, 0.0)
    if vx <= _MIN_SD ** 2 or vy <= _MIN_SD ** 2:
        return None
    b = np.sqrt(vy / vx)
    a = Sy / n - b * Sx / n
    return float(np.log(b)), float(a), float(1.0 / np.sqrt(max(n - 1.0, 1.0)))


def fit_hierarchical_calibration(pairs_by_species, n_species, prior_slope=1.0,
                                 prior_intercept=0.0, prior_log_slope_sd=0.5,
                                 prior_intercept_sd=1.0, n_iter=200, tol=1e-12,
                                 verbose=True):
    """Partially pooled per-species calibration of BBS onto the eBird scale.

    ``pairs_by_species``: ``{species_index: (x_bbs_log1p, y_ebird_log1p)}`` over overlapping
    cell-years, restricted to rows where both values are > 0.

    Each species proposes its own slope by matching spreads (see ``_spread_ratio_estimate``),
    and that proposal is shrunk toward a population value in proportion to how well its own
    data determines it. Plenty of overlap means its own estimate; little overlap means the
    population's; none at all means exactly the population's, with no special case.

    Returns per-species ``a``/``b``, the population ``mu_beta``/``mu_a``, per-species ``n``, and
    ``shrinkage`` -- the share of each estimate that came from the population rather than the
    species' own data. That is the honest replacement for a pass/fail flag: how much a species
    rests on its own evidence is a continuous quantity, so it is reported as one.
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

    # Each species' own proposal, and how precisely its data pins it down.
    own_beta = np.full(S, np.nan); own_a = np.full(S, np.nan); own_se = np.full(S, np.inf)
    for s in range(S):
        est = _spread_ratio_estimate(stats[s])
        if est is not None:
            own_beta[s], own_a[s], own_se[s] = est
    usable = np.isfinite(own_beta)

    mu_beta, mu_a = float(np.log(prior_slope)), float(prior_intercept)
    tau_beta, tau_a = float(prior_log_slope_sd), float(prior_intercept_sd)
    beta = np.full(S, mu_beta); a = np.full(S, mu_a)

    for _ in range(int(n_iter)):
        prev = (mu_beta, mu_a)
        # Shrink each species' own proposal toward the population, weighted by precision.
        w_own = np.where(usable, 1.0 / np.maximum(own_se, 1e-12) ** 2, 0.0)
        w_pop = 1.0 / tau_beta ** 2
        beta = np.where(usable, (w_own * np.nan_to_num(own_beta) + w_pop * mu_beta)
                        / (w_own + w_pop), mu_beta)
        # The intercept follows from the slope, so it is recomputed at the shrunk slope rather
        # than shrunk independently -- otherwise the line would not pass through the species'
        # own means and the calibration would carry an offset nobody asked for.
        a = np.where(usable & (n > 0),
                     np.divide(stats[:, 2], np.maximum(n, 1)) -
                     np.exp(beta) * np.divide(stats[:, 1], np.maximum(n, 1)), mu_a)
        # Population location, with its own prior so a few species cannot drag it far.
        if usable.any():
            mu_beta = (w_pop * beta[usable].sum()
                       + float(np.log(prior_slope)) / prior_log_slope_sd ** 2) / \
                      (w_pop * usable.sum() + 1.0 / prior_log_slope_sd ** 2)
            wa = 1.0 / tau_a ** 2
            mu_a = (wa * a[usable].sum() + float(prior_intercept) / prior_intercept_sd ** 2) / \
                   (wa * usable.sum() + 1.0 / prior_intercept_sd ** 2)
        if max(abs(prev[0] - mu_beta), abs(prev[1] - mu_a)) < tol:
            break

    beta = np.where(usable, beta, mu_beta)
    a = np.where(usable, a, mu_a)
    b = np.exp(beta)
    shrink = np.where(usable, (1.0 / tau_beta ** 2) /
                      (np.where(usable, 1.0 / np.maximum(own_se, 1e-12) ** 2, 0.0)
                       + 1.0 / tau_beta ** 2), 1.0)

    out = {"a": a, "b": b, "beta": beta, "n": n, "shrinkage": shrink,
           "mu_a": float(mu_a), "mu_beta": float(mu_beta), "mu_b": float(np.exp(mu_beta)),
           "tau_a": float(tau_a), "tau_beta": float(tau_beta), "sigma": float("nan"),
           "prior": {"slope": float(prior_slope), "intercept": float(prior_intercept),
                     "log_slope_sd": float(prior_log_slope_sd),
                     "intercept_sd": float(prior_intercept_sd)},
           "n_species_with_overlap": int(usable.sum()), "n_overlap_points": int(n.sum())}
    if verbose:
        print(f"[calib] BBS -> eBird by matching spreads, partially pooled over {S} species "
              f"({int(usable.sum())} with usable overlap, {int(n.sum()):,} paired cell-years)")
        print(f"[calib] population slope {np.exp(mu_beta):.3f} "
              f"(log-slope {mu_beta:+.3f}), intercept {mu_a:+.3f}")
        if usable.any():
            print(f"[calib] per-species slope: median {np.median(b[usable]):.3f}, "
                  f"range {b[usable].min():.3f}..{b[usable].max():.3f} "
                  f"(positive by construction)")
            print(f"[calib] shrinkage toward the population: median "
                  f"{np.median(shrink[usable]):.3f}, "
                  f"{int((shrink > 0.5).sum())} species more population than own data")
        if (~usable).any():
            print(f"[calib] {int((~usable).sum())} species lack usable overlap and sit exactly "
                  f"on the population estimate")
    return out


def apply_calibration(X_bbs_log, cal):
    """Map log1p BBS values onto the eBird log1p scale: ``clip(a[s] + b[s]*x, 0, inf)`` (pure).

    Clipped at 0 because a negative value is outside log1p-Ruzicka's domain -- not a smaller
    abundance -- and would let ``sum(min)/sum(max)`` exceed 1 or flip the denominator's sign.

    WHAT HAPPENS AT A TRUE ZERO, and why this is left alone rather than patched.

    An entry of 0 is a surveyed cell-year where the species was not recorded. The affine map
    sends it to ``a``, so if ``a > 0`` an absence becomes a small presence in every such cell.
    Absences are most of the matrix -- 83.3% of the real BBS community matrix -- so a positive
    intercept would give every cell-year a floor in every species, and Ruzicka cannot see past a
    shared floor: it enters both numerator and denominator, inflating similarity toward 1 and
    compressing its SPREAD, which is the structure the kernel discriminates on.

    Three reasons it is nevertheless not masked here:

    1. ``clip(..., 0)`` already handles the case where ``a <= 0``, and whether the real fitted
       intercepts are positive is UNKNOWN -- it needs the eBird trend grid, which is not
       available off-cluster. In a simulation with realistic BBS detection only 8 of 30 species
       came out with a positive intercept.
    2. Masking zeros trades the problem rather than solving it. A rare species whose count
       flickers 0, 1, 0, 1 across years would jump between 0 and ``a + b*log1p(1)`` on sampling
       luck alone, and this encoder exists to measure temporal change. Measured on a Poisson
       observation model, masking changed the fabricated year-to-year variation by under 1%
       (0.6788 -> 0.6836) -- the flicker is dominated by counting noise, not by the transform --
       while making the similarity structure slightly WORSE (0.1355 -> 0.1539).
    3. Distinguishing "present but undetected" from "genuinely absent" is a presence model.
       Neither the affine map nor a mask can do it; both collapse the two states into one. A
       patch that pretends otherwise is worse than the honest version.

    ``report_zero_effect`` is the instrument for settling this on real data. Run it on the real
    eBird grid and look at the intercept signs and the occupancy change before deciding whether
    anything is needed here.
    """
    X = np.asarray(X_bbs_log, "float64")
    a = np.asarray(cal["a"], "float64").reshape(1, -1)
    b = np.asarray(cal["b"], "float64").reshape(1, -1)
    if X.shape[1] != a.shape[1]:
        raise ValueError(f"calibration has {a.shape[1]} species, X has {X.shape[1]}")
    return np.clip(a + b * X, 0.0, None).astype("float32")


def report_zero_effect(X_bbs_log, cal, verbose=True):
    """What the calibration does to ABSENCES, on whatever data you have. Returns a dict.

    This is the measurement that decides whether ``apply_calibration`` needs anything doing
    about zeros, and it cannot be made off-cluster because it needs the real eBird grid. The
    numbers to look at:

    - ``n_positive_intercepts``. If 0, there is nothing to discuss: ``clip`` already sends every
      absence to 0.
    - ``occupancy_before`` vs ``occupancy_after``. A jump from ~17% to ~100% means every
      cell-year now shares a floor in every species.
    - ``floor_birds``. The invented abundance at an unrecorded species, in raw units.

    If those say there IS a problem, the fix is a presence model, not a mask -- see the note in
    ``apply_calibration`` for why masking trades the artifact rather than removing it.
    """
    X = np.asarray(X_bbs_log)
    a = np.asarray(cal["a"], "float64")
    C = apply_calibration(X, cal)
    pos = a > 0
    out = {"n_species": int(a.size), "n_positive_intercepts": int(pos.sum()),
           "intercept_median": float(np.median(a)),
           "intercept_min": float(a.min()), "intercept_max": float(a.max()),
           "occupancy_before": float((X > 0).mean()),
           "occupancy_after": float((C > 0).mean()),
           "floor_birds_median": float(np.expm1(np.median(np.clip(a, 0, None)))),
           "floor_birds_max": float(np.expm1(max(a.max(), 0.0)))}
    if verbose:
        print(f"[calib] intercepts: {out['n_positive_intercepts']}/{out['n_species']} positive, "
              f"median {out['intercept_median']:+.3f} "
              f"(range {out['intercept_min']:+.3f}..{out['intercept_max']:+.3f})")
        print(f"[calib] occupancy of the BBS matrix: {out['occupancy_before']:.1%} before "
              f"calibration -> {out['occupancy_after']:.1%} after")
        if out["occupancy_after"] > out["occupancy_before"] + 0.05:
            print(f"[calib] NOTE a positive intercept is giving unrecorded species a floor of "
                  f"up to {out['floor_birds_max']:.3f} birds. Every cell-year then shares that "
                  f"floor in every species, which inflates Ruzicka similarity and compresses "
                  f"its spread. If this is large, the answer is a presence model -- masking the "
                  f"zeros only converts it into fabricated year-to-year flicker.")
    return out


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
