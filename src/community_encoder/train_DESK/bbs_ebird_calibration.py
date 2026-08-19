"""Put raw BBS counts on the eBird abundance scale, per species, by partial pooling.

BBS counts birds on a survey route; eBird Status & Trends gives a modelled relative abundance.
Different units. For the two to be rows of one community matrix feeding one Ruzicka kernel they
have to share a scale, because Ruzicka is ``sum(min)/sum(max)`` and is scale-sensitive in each
argument -- an uncalibrated mix would make the data SOURCE itself a similarity signal.

**eBird is the common frame, and BBS is what gets transformed.** eBird is the richer product:
17,205 cells against BBS's ~3,900, denser per row, modelled from far more observation effort.
So the fitted transform lands on the sparser data rather than on the majority of it.

## The functional form

    E = k_s * (B / B0)^(d_s)       B = BBS mean count, E = eBird abundance, both RAW

fitted as ``log E = log k_s + d_s * (log B - log B0)`` over the cell-years where both products
recorded the species. ``B0`` is the typical BBS count, pooled across all species, and it is
what makes ``k_s`` mean something: it is eBird's value at a TYPICAL BBS count -- the actual
exchange rate between the two surveys.

Anchoring at ``B0`` rather than at ``B = 1`` is not cosmetic. Uncentred, ``log k`` is the value
where BBS counts exactly one bird, which is the far edge of the data, and it is strongly
coupled to the exponent: a species with a high ``d`` must have a tiny ``k`` to compensate. On
the real data that coupling alone dragged the reported ``k`` across ~27x before any genuine
difference in detectability, so a prior on its spread was being applied to a quantity that was
mostly an artefact of where the fit was anchored. Centred, ``k_s`` and ``d_s`` are close to
independent and the prior on ``k`` means what it says.

Two properties make this the right family, and an earlier version that worked in ``log1p``
space had neither:

**A zero maps to a zero.** ``B = 0`` gives ``E = 0`` for any ``k`` and ``d``. A surveyed
cell-year where the species was not recorded cannot become a small presence. That matters
because absences are 83.3% of the real BBS matrix, and a shared floor across every cell is
exactly what Ruzicka cannot see past -- it enters both numerator and denominator, inflating
similarity toward 1 and compressing its SPREAD, which is the structure the kernel
discriminates on. Note this holds DESPITE there being a free scale parameter: ``k`` is
multiplicative, so it cannot create a floor the way an additive intercept does.

**It can express a unit conversion.** ``d = 1`` with ``k`` free is exactly "the two products
correspond up to a scale factor", which is the honest prior for two measurements of the same
thing. The previous ``log1p(E) = b * log1p(B)`` could not represent that at all: if eBird were
simply 0.2x BBS, the implied ``b`` would have to slide from 0.235 at low abundance to 0.720 at
high, and the best single value (0.630) over-predicted by +1.8 at moderate abundance and
under-predicted by -24.5 at high. Worse, ``b = 1`` there meant ``1+E = 1+B``, i.e. E and B
IDENTICAL rather than proportional -- so the prior was making a far stronger claim than
intended, and the fitted exponent was contorting to approximate a scale change it could not
express.

## Partial pooling, not thresholds

    delta_s = log(d_s) ~ Normal(mu_delta, tau_delta)
    log k_s            ~ Normal(mu_logk,  tau_k)

A species with thousands of overlapping cell-years is fitted almost entirely by its own data.
A species with five, or with no real relationship between the two products, is pulled to the
population estimate -- in proportion to how much its data actually says. A species with NO
overlap lands exactly on the population estimate, with no special case.

This replaces an earlier design with hard cutoffs (a minimum overlap count and a minimum
correlation, below which a species was dropped to a pooled fit). Those numbers were invented,
and they made a species' treatment discontinuous in its evidence: 49 pairs behaved completely
differently from 51.

## The priors

``mu_delta`` is centred on 0, i.e. an exponent of 1, so the prior says the surveys differ by a
scale factor rather than a power. Both population locations are LEARNED, so enough species
disagreeing will move them.

FOUR separate beliefs, deliberately given four knobs -- an earlier version used one number for
two of them, which conflated things that are not the same question:

The split that matters is LOCATION versus SPREAD, and they pull in opposite directions:

- The population LOCATIONS are free to drift. We have no idea what the units conversion
  between a roadside route count and a modelled abundance index is, nor whether the typical
  exponent is exactly 1, so ``population_log_exponent_sd`` and ``population_log_scale_sd`` are
  both 5.0 -- effectively flat. The data determines where the population sits.
- The SPREAD around that location is tight. ``prior_log_exponent_sd`` and
  ``prior_log_scale_sd`` are both 0.10, so 95% of species lie within a 1.49x total span of the
  population value.

For the scale that says: species do not differ wildly in DETECTABILITY BETWEEN THE TWO
SURVEYS. Note that is the ratio, not absolute detectability -- a species that is hard for BBS
to detect is usually also hard for eBird observers, so the ratio between the two should stay
close to typical even where the absolute detectability does not. An earlier version left this
at 2.0, admitting a 2981x span, on the mistaken reasoning that species differ in
detectability. They do; the RATIO is the thing that should not.

For the exponent it says the SHAPE of the relationship is close to common across species. A
value above 1 mostly reflects BBS saturating at high abundance -- 50 stops, limited counting
time -- which is real but should be similar across species.

``exp(delta)`` keeps every exponent positive. A negative one would invert a species --
calibrating so that more BBS birds mean less eBird abundance -- which is never a calibration.

The two ``tau`` values are NOT estimated. Joint MAP over a hierarchical variance is degenerate
in both directions and both showed up in testing: when species agree the empirical spread
collapses to zero and the prior becomes infinitely strong, pooling everything completely; when
one species disagrees its departure inflates the spread, weakening the prior, letting it depart
further. Neither is a property of the data. So they stay stated beliefs about how much species'
calibrations differ, which is what a prior is for.

## Matching spreads rather than least squares

The per-species exponent is ``sd(log E)/sd(log B)``, not a regression fit. Least squares gives
the best estimate of each individual value -- it knows part of a high BBS count is survey luck
and pulls it back -- but we never use individual values. The target is consumed only through a
similarity table, and pulling every cell toward the mean makes all cells more alike, so
similarities inflate and their spread compresses. Measured against known truth across BBS noise
0.2 to 1.2, least squares won on individual values at every noise level and lost on the
similarity table at every noise level, by up to 3.4x, with median similarity drifting from a
true 0.708 to 0.828 while spread-matching held it at 0.708.

## MAP, not a posterior

Point estimates are all we apply. Coordinate ascent, deterministic, no sampler, and no JAX
dependency in a CPU preprocessing stage.
"""
import numpy as np

_MIN = 1e-12


def _species_estimate(x_log, y_log):
    """One species' own ``(log d, log k, se)`` from matching spreads in LOG-LOG space.

    ``x_log``/``y_log`` are ``log(BBS)`` and ``log(eBird)`` over cell-years where both recorded
    the species. ``d = sd(y)/sd(x)`` and ``log k = mean(y) - d*mean(x)``, so the fitted line
    passes through the species' own means. ``se ~= 1/sqrt(n-1)`` is the approximate standard
    error of ``log d``, which is what tells the pooling step how much this species' own data
    should count.
    """
    n = x_log.size
    if n < 3:
        return None
    sx, sy = x_log.std(), y_log.std()
    if sx < 1e-9 or sy < 1e-9:
        return None
    d = sy / sx
    # se on log d from the ratio of two sample spreads; se on log k from the mean of y, which
    # dominates its uncertainty. Both feed the pooling step, so a species with thin data is
    # pulled toward the population on BOTH parameters rather than only on the exponent.
    return (float(np.log(d)), float(1.0 / np.sqrt(max(n - 1.0, 1.0))),
            float(y_log.mean()), float(x_log.mean()), float(sy / np.sqrt(n)))


def fit_hierarchical_calibration(pairs_by_species, n_species, prior_exponent=1.0,
                                 prior_log_exponent_sd=0.10, prior_log_scale_sd=0.10,
                                 population_log_exponent_sd=5.0,
                                 population_log_scale_sd=5.0,
                                 n_iter=200, tol=1e-12, verbose=True):
    """Partially pooled per-species calibration of BBS onto the eBird scale: ``E = k * B^d``.

    ``pairs_by_species``: ``{species_index: (x_bbs_log1p, y_ebird_log1p)}`` over overlapping
    cell-years where both values are > 0. Converted internally to raw units and then to logs --
    the callers already hold log1p values, so this keeps the interface unchanged.

    Returns per-species ``k`` and ``d``, the learned population ``mu_delta``/``mu_logk``,
    per-species ``n``, and ``shrinkage`` -- the share of each exponent that came from the
    population rather than the species' own data. That is the honest replacement for a pass/fail
    flag: how much a species rests on its own evidence is continuous, so it is reported as one.
    """
    S = int(n_species)
    own_delta = np.full(S, np.nan); own_se = np.full(S, np.inf)
    mean_x = np.zeros(S); mean_y = np.zeros(S); se_k = np.full(S, np.inf)
    n = np.zeros(S, dtype="int64")
    xs_all = []
    for s, (x1p, y1p) in pairs_by_species.items():
        s = int(s)
        if not (0 <= s < S):
            continue
        # log1p -> raw -> log. Only positive pairs reach here, so log is safe.
        x = np.log(np.maximum(np.expm1(np.asarray(x1p, "float64")), 1e-9))
        y = np.log(np.maximum(np.expm1(np.asarray(y1p, "float64")), 1e-9))
        n[s] = x.size
        xs_all.append(x)
        est = _species_estimate(x, y)
        if est is not None:
            own_delta[s], own_se[s], mean_y[s], mean_x[s], se_k[s] = est
    usable = np.isfinite(own_delta)
    # ONE anchor for every species, so the k values are comparable to each other and a prior on
    # their spread is a statement about species rather than about where each fit was centred.
    log_B0 = float(np.concatenate(xs_all).mean()) if xs_all else 0.0

    mu_delta = float(np.log(prior_exponent)); mu_logk = 0.0
    tau_d = float(prior_log_exponent_sd); tau_k = float(prior_log_scale_sd)
    pop_sd_d = float(population_log_exponent_sd); pop_sd_k = float(population_log_scale_sd)
    delta = np.full(S, mu_delta); logk = np.full(S, mu_logk)

    for _ in range(int(n_iter)):
        prev = (mu_delta, mu_logk)
        w_own = np.where(usable, 1.0 / np.maximum(own_se, 1e-12) ** 2, 0.0)
        w_pop = 1.0 / tau_d ** 2
        delta = np.where(usable, (w_own * np.nan_to_num(own_delta) + w_pop * mu_delta)
                         / (w_own + w_pop), mu_delta)
        # The scale follows from the shrunk exponent, so the line still passes through the
        # species' own means. Shrinking the two independently would leave an offset nobody
        # asked for.
        # Shrunk toward the population like the exponent, weighted by how precisely this
        # species' data pins it down. An earlier version computed it directly from the species'
        # own means and never shrank it at all, so the prior on its spread affected only the
        # population value -- the per-species k values were completely unconstrained, and on
        # real data they spanned 612x against a prior asserting 1.2x.
        own_logk = mean_y - np.exp(delta) * (mean_x - log_B0)
        w_own_k = np.where(usable, 1.0 / np.maximum(se_k, 1e-12) ** 2, 0.0)
        w_pop_k = 1.0 / tau_k ** 2
        logk = np.where(usable, (w_own_k * own_logk + w_pop_k * mu_logk)
                        / (w_own_k + w_pop_k), mu_logk)
        if usable.any():
            mu_delta = (w_pop * delta[usable].sum()
                        + np.log(prior_exponent) / pop_sd_d ** 2) \
                       / (w_pop * usable.sum() + 1.0 / pop_sd_d ** 2)
            wk = 1.0 / tau_k ** 2
            mu_logk = (wk * logk[usable].sum()) / (wk * usable.sum() + 1.0 / pop_sd_k ** 2)
        if max(abs(prev[0] - mu_delta), abs(prev[1] - mu_logk)) < tol:
            break

    delta = np.where(usable, delta, mu_delta)
    logk = np.where(usable, logk, mu_logk)
    d = np.exp(delta); k = np.exp(logk)
    shrink = np.where(usable, (1.0 / tau_d ** 2) /
                      (np.where(usable, 1.0 / np.maximum(own_se, 1e-12) ** 2, 0.0)
                       + 1.0 / tau_d ** 2), 1.0)
    shrink_k = np.where(usable, (1.0 / tau_k ** 2) /
                        (np.where(usable, 1.0 / np.maximum(se_k, 1e-12) ** 2, 0.0)
                         + 1.0 / tau_k ** 2), 1.0)

    out = {"k": k, "d": d, "delta": delta, "log_k": logk, "n": n, "shrinkage": shrink,
           "shrinkage_k": shrink_k, "log_B0": log_B0, "B0": float(np.exp(log_B0)),
           "mu_delta": float(mu_delta), "mu_d": float(np.exp(mu_delta)),
           "mu_logk": float(mu_logk), "mu_k": float(np.exp(mu_logk)),
           "tau_delta": float(tau_d), "tau_logk": float(tau_k),
           "prior": {"exponent": float(prior_exponent),
                     "log_exponent_sd": float(prior_log_exponent_sd),
                     "log_scale_sd": float(prior_log_scale_sd),
                     "population_log_exponent_sd": float(population_log_exponent_sd),
                     "population_log_scale_sd": float(population_log_scale_sd)},
           "n_species_with_overlap": int(usable.sum()), "n_overlap_points": int(n.sum())}
    if verbose:
        print(f"[calib] BBS -> eBird as E = k*B^d, partially pooled over {S} species "
              f"({int(usable.sum())} with usable overlap, {int(n.sum()):,} paired cell-years)")
        print(f"[calib] population: scale k={np.exp(mu_logk):.4f}, exponent d="
              f"{np.exp(mu_delta):.3f}  (d=1 would mean a pure unit conversion)")
        print(f"[calib] priors at 2sd: exponents within {np.exp(2*tau_d):.2f}x of the "
              f"population, detectability ratios within {np.exp(2*tau_k):.1f}x")
        if usable.any():
            print(f"[calib] per-species exponent d: median {np.median(d[usable]):.3f}, "
                  f"range {d[usable].min():.3f}..{d[usable].max():.3f} (positive by construction)")
            print(f"[calib] per-species scale k (eBird at a typical BBS count of "
                  f"{np.exp(log_B0):.2f} birds): median {np.median(k[usable]):.4f}, "
                  f"range {k[usable].min():.4f}..{k[usable].max():.4f} "
                  f"({k[usable].max()/max(k[usable].min(),1e-12):.1f}x span)")
            print(f"[calib] shrinkage toward the population -- exponent: median "
                  f"{np.median(shrink[usable]):.3f}; scale: median "
                  f"{np.median(shrink_k[usable]):.3f}")
        if (~usable).any():
            print(f"[calib] {int((~usable).sum())} species lack usable overlap and sit exactly "
                  f"on the population estimate")
    return out


def apply_calibration(X_bbs_log, cal):
    """Map log1p BBS values onto the eBird log1p scale via ``E = k * B^d`` (pure).

    A zero stays a zero for any ``k`` and ``d``, so the occupancy pattern of the calibrated
    matrix is identical to the raw one -- nothing needs masking and no measured absence can
    become a fabricated presence. That holds despite ``k`` being a free scale, because ``k`` is
    multiplicative: unlike an additive intercept it cannot create a floor.
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
    """JSON-safe per-species record for ``points_meta.json``.

    ``shrinkage`` is the field to read: near 1 means that species' exponent is essentially the
    population value because its own data said little, near 0 means its own data determined it.
    There is no pass/fail flag to read instead, deliberately.
    """
    return {
        "direction": "bbs_to_ebird", "method": "hierarchical_map_power_law",
        "form": "ebird = k_s * (bbs/B0)**d_s   (raw units; zero maps to zero)",
        "B0_typical_bbs_count": cal.get("B0"),
        "population": {"scale_k": cal["mu_k"], "exponent_d": cal["mu_d"],
                       "log_exponent_sd_prior": cal["tau_delta"],
                       "log_scale_sd_prior": cal["tau_logk"]},
        "prior": cal["prior"],
        "n_species_with_overlap": cal["n_species_with_overlap"],
        "n_overlap_points": cal["n_overlap_points"],
        "per_species": {str(c): {"k": float(cal["k"][i]), "d": float(cal["d"][i]),
                                 "n": int(cal["n"][i]),
                                 "shrinkage": float(cal["shrinkage"][i]),
                                 "shrinkage_k": float(cal["shrinkage_k"][i])}
                        for i, c in enumerate(species)},
    }
