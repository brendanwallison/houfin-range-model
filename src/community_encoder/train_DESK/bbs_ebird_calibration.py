"""Put raw BBS counts on the eBird abundance scale, per species, by partial pooling.

BBS counts birds on a survey route; eBird Status & Trends gives a modelled relative abundance.
Different units. For the two to be rows of one community matrix feeding one Ruzicka kernel they
have to share a scale, because Ruzicka is ``sum(min)/sum(max)`` and is scale-sensitive in each
argument -- an uncalibrated mix would make the data SOURCE itself a similarity signal.

**eBird is the common frame, and BBS is what gets transformed.** eBird is the richer product:
17,205 cells against BBS's ~3,900, denser per row, modelled from far more observation effort.
So the fitted transform lands on the sparser data rather than on the majority of it.

## The functional form, and why there is no intercept

    log1p(ebird)_si  =  exp(beta_s) * log1p(bbs)_si  +  noise

Undoing the logs, that is ``(1 + E) = (1 + B)^b`` -- a power law through the origin.

The intercept is absent on purpose, and it is the most important property here. With one,
``B = 0`` maps to ``a``, so a surveyed cell-year where the species was NOT RECORDED becomes a
small presence. Absences are most of the matrix (83.3% of the real BBS community matrix), so a
positive intercept would give every cell-year a floor in every species -- and Ruzicka cannot
see past a shared floor: it enters both numerator and denominator, inflating similarity toward
1 and compressing its SPREAD, which is the structure the kernel discriminates on.

Through the origin, that is unreachable. A zero maps to a zero, the occupancy pattern of the
calibrated matrix is identical to the raw one, and nothing has to be masked, flagged, or
diagnosed. It is a guarantee rather than a measurement, which matters because the alternative's
behaviour depends on the sign of the fitted intercept, and that sign depends on BBS detection
rates that are not known in advance.

What is given up: an intercept can partially stand in for BBS under-detection, and that is
real -- in simulation with 15% detection, about a third of BBS zeros were undetected presences.
But it stands in for it crudely, applying one constant floor to genuinely-absent regions and
undetected-presence ones alike. Doing that properly needs a detection or presence model, not a
constant, and a constant that is right by accident is worse than not attempting it.

## Partial pooling, not thresholds

    beta_s ~ Normal(mu_beta, tau_beta)

A species with thousands of overlapping cell-years is fitted almost entirely by its own data. A
species with five, or with no real relationship between the two products, is pulled to the
population estimate -- in proportion to how much its data actually says. A species with NO
overlap lands exactly on the population estimate, with no special case.

This replaces an earlier design with hard cutoffs (a minimum overlap count and a minimum
correlation, below which a species was dropped to a pooled fit). Those numbers were invented,
and they made a species' treatment discontinuous in its evidence: 49 pairs behaved completely
differently from 51.

## The prior

``mu_beta`` is centred on 0, i.e. a slope of 1, which says the two products correspond up to a
pure scale factor rather than a power. That is the honest prior for two measurements of the
same thing. The population location is estimated, so enough species disagreeing will move it.

``exp(beta)`` also keeps every slope positive. A negative slope would invert a species --
calibrating so that more BBS birds mean less eBird abundance -- which is never a calibration.

``tau_beta`` is NOT estimated. Joint MAP over a hierarchical variance is degenerate in both
directions and both showed up in testing: when species agree the empirical spread collapses to
zero and the prior becomes infinitely strong, pooling everything completely; when one species
disagrees its departure inflates the spread, weakening the prior, letting it depart further.
Neither is a property of the data. So it stays a stated belief about how much species'
calibrations differ, which is what a prior is for.

## MAP, not a posterior

We need point estimates to apply, not uncertainty about them. With no intercept the per-species
step is a one-parameter fit with an analytic gradient from sufficient statistics, so coordinate
ascent is deterministic, needs no sampler, and adds no JAX dependency to a CPU stage.
"""
import numpy as np

_MIN = 1e-12


def _species_beta(Sxx, Sxy, mu_beta, tau_beta, sigma, beta0):
    """MAP ``beta`` for one species, no intercept, from sufficient statistics.

    Minimises  ``(Syy - 2 e^beta Sxy + e^2beta Sxx)/(2 sigma^2) + (beta-mu)^2/(2 tau^2)``.
    ``Syy`` is constant in ``beta`` so it drops out. The gradient is

        e^beta (e^beta Sxx - Sxy)/sigma^2  +  (beta - mu)/tau^2

    which is monotone enough that a short Newton iteration converges from the population value.
    """
    beta = float(beta0)
    for _ in range(60):
        B = np.exp(beta)
        g = B * (B * Sxx - Sxy) / sigma ** 2 + (beta - mu_beta) / tau_beta ** 2
        h = (2.0 * B * B * Sxx - B * Sxy) / sigma ** 2 + 1.0 / tau_beta ** 2
        step = g / max(h, _MIN)
        beta -= max(min(step, 1.0), -1.0)              # damped, so a bad curvature cannot bolt
        if abs(step) < 1e-12:
            break
    return beta


def fit_hierarchical_calibration(pairs_by_species, n_species, prior_slope=1.0,
                                 prior_log_slope_sd=0.5, n_iter=200, tol=1e-12, verbose=True):
    """Partially pooled per-species calibration of BBS onto the eBird scale, through the origin.

    ``pairs_by_species``: ``{species_index: (x_bbs_log1p, y_ebird_log1p)}`` over overlapping
    cell-years, restricted to rows where both values are > 0.

    Returns per-species ``b`` (length ``n_species``), the population ``mu_beta``/``mu_b``,
    per-species ``n``, and ``shrinkage`` -- the share of each estimate that came from the
    population rather than the species' own data. That is the honest replacement for a pass/fail
    flag: how much a species rests on its own evidence is continuous, so it is reported as one.
    """
    S = int(n_species)
    Sxx = np.zeros(S); Sxy = np.zeros(S); Syy = np.zeros(S); n = np.zeros(S, dtype="int64")
    for s, (x, y) in pairs_by_species.items():
        s = int(s)
        if not (0 <= s < S):
            continue
        x = np.asarray(x, "float64"); y = np.asarray(y, "float64")
        Sxx[s] = float(x @ x); Sxy[s] = float(x @ y); Syy[s] = float(y @ y); n[s] = x.size
    usable = (n > 0) & (Sxx > _MIN)
    N = int(n.sum())

    mu_beta = float(np.log(prior_slope))
    tau_beta = float(prior_log_slope_sd)
    beta = np.full(S, mu_beta)
    sigma = max(float(np.sqrt(Syy.sum() / N)) if N else 1.0, 1e-6)

    for _ in range(int(n_iter)):
        prev = (mu_beta, sigma)
        for s in range(S):
            beta[s] = _species_beta(Sxx[s], Sxy[s], mu_beta, tau_beta, sigma, beta[s]) \
                if usable[s] else mu_beta
        w = 1.0 / tau_beta ** 2
        if usable.any():
            mu_beta = (w * beta[usable].sum() + np.log(prior_slope) / prior_log_slope_sd ** 2) \
                      / (w * usable.sum() + 1.0 / prior_log_slope_sd ** 2)
        if N:
            B = np.exp(beta)
            rss = float((Syy - 2.0 * B * Sxy + B * B * Sxx)[usable].sum())
            sigma = max(float(np.sqrt(max(rss, 0.0) / N)), 1e-6)
        if max(abs(prev[0] - mu_beta), abs(prev[1] - sigma)) < tol:
            break

    beta = np.where(usable, beta, mu_beta)
    b = np.exp(beta)
    like = np.where(usable, b ** 2 * Sxx / sigma ** 2, 0.0)      # curvature from the data
    shrink = (1.0 / tau_beta ** 2) / (like + 1.0 / tau_beta ** 2)

    out = {"b": b, "beta": beta, "n": n, "shrinkage": shrink, "sigma": float(sigma),
           "mu_beta": float(mu_beta), "mu_b": float(np.exp(mu_beta)),
           "tau_beta": float(tau_beta),
           "prior": {"slope": float(prior_slope),
                     "log_slope_sd": float(prior_log_slope_sd)},
           "n_species_with_overlap": int(usable.sum()), "n_overlap_points": N}
    if verbose:
        print(f"[calib] BBS -> eBird through the origin, partially pooled over {S} species "
              f"({int(usable.sum())} with overlap, {N:,} paired cell-years)")
        print(f"[calib] population slope {np.exp(mu_beta):.3f} "
              f"(log-slope {mu_beta:+.3f}, prior sd {tau_beta:.2f}), residual sd {sigma:.3f}")
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
    """Map log1p BBS values onto the eBird log1p scale: ``b[s] * x`` (pure).

    Through the origin, so a zero stays a zero and the occupancy pattern of the calibrated
    matrix is identical to the raw one. That is the property the whole form exists for -- see the
    module docstring. Nothing needs masking and no absence can become a presence.
    """
    X = np.asarray(X_bbs_log, "float64")
    b = np.asarray(cal["b"], "float64").reshape(1, -1)
    if X.shape[1] != b.shape[1]:
        raise ValueError(f"calibration has {b.shape[1]} species, X has {X.shape[1]}")
    return np.clip(b * X, 0.0, None).astype("float32")


def calibration_meta(cal, species):
    """JSON-safe per-species record for ``points_meta.json``.

    ``shrinkage`` is the field to read: near 1 means that species' calibration is essentially
    the population relationship because its own data said little, near 0 means its own data
    determined it. There is no pass/fail flag to read instead, deliberately.
    """
    return {
        "direction": "bbs_to_ebird", "method": "hierarchical_map_through_origin",
        "form": "log1p(ebird) = b_s * log1p(bbs)",
        "population": {"slope": cal["mu_b"], "log_slope": cal["mu_beta"],
                       "log_slope_sd_prior": cal["tau_beta"], "residual_sd": cal["sigma"]},
        "prior": cal["prior"],
        "n_species_with_overlap": cal["n_species_with_overlap"],
        "n_overlap_points": cal["n_overlap_points"],
        "per_species": {str(c): {"b": float(cal["b"][i]), "n": int(cal["n"][i]),
                                 "shrinkage": float(cal["shrinkage"][i])}
                        for i, c in enumerate(species)},
    }
