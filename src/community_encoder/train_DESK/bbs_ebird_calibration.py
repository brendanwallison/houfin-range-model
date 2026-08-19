"""Put eBird relative abundance on the raw-BBS count scale, per species.

Both products measure the same latent community but in different units: BBS is birds per
route on a 50-stop roadside survey, eBird Status & Trends is a modelled relative abundance.
For the two to be rows of ONE community matrix feeding ONE Ruzicka kernel they have to share
a scale -- Ruzicka is ``sum(min)/sum(max)``, which is scale-sensitive in each argument, so
an uncalibrated mix would make source membership itself a similarity signal.

This replaces the single scalar ``k[s] = median(E/B)`` in ``trend_community._species_scale``,
which was scale-only and fit against the *published* BBS abundance raster rather than the
raw counts.

## Two choices that are easy to get backwards

**Direction: fit BBS from eBird, not the reverse.** We apply the fit to eBird values to
produce BBS-scale values, so the regression has to run in that direction. Fitting
``ebird ~ bbs`` and inverting is a different estimator and biases the result.

**Estimator: reduced major axis, not OLS.** OLS minimises residuals in y, which shrinks
predictions toward the mean -- so OLS-calibrated eBird rows would carry systematically LESS
variance than real BBS rows. In a similarity kernel that is not a small bias: eBird-derived
cell-years would look more like each other than BBS-derived ones do, and the kernel would
learn source membership. RMA (geometric-mean regression) sets

    b = sign(r) * sd(y)/sd(x)

which preserves variance by construction and is the standard estimator when the goal is
putting two measurements on a common scale rather than predicting one from the other.
``form="ols"`` remains available for comparison.

## Guards

A per-species fit is REFUSED (falling back a rung) when the correlation is below ``min_r``.
Negative is the obvious case: an inverting slope would calibrate a species so that more eBird
detections mean fewer BBS birds, which is never a calibration -- it says the two products
disagree about that species. Near-zero is the subtler one, and it matters specifically because
of RMA: the slope is ``sign(r)*sd(y)/sd(x)``, which stays large even at ``r ~ 0``, so a species
with no real agreement would still receive a confident-looking variance-matching slope with an
essentially random sign. The pooled relationship is the more honest estimate there.

The per-species ``r`` values are therefore the diagnostic to read before trusting a run: a low
median means the calibration is fitting noise and the eBird rows are not measuring the same
thing the BBS rows are.

The fallback ladder, every rung logged:

1. ``>= min_overlap_points`` overlapping cell-years with both values > 0, and ``r >= min_r``
   -> per-species ``(a_s, b_s)``
2. else -> the pooled fit across all species
3. else -> scale-only (``b = 1``, ``a = median(y - x)``), i.e. what ``k[s]`` did
"""
import numpy as np

_MIN_SD = 1e-9


def rma_fit(x, y):
    """Reduced major axis fit ``y ~ a + b*x``: ``(a, b, pearson_r)``.

    ``b = sign(r)*sd(y)/sd(x)`` preserves variance, so calibrated values keep the spread of
    the scale being mapped onto. Returns ``b = nan`` when either side is constant.
    """
    x = np.asarray(x, "float64"); y = np.asarray(y, "float64")
    sx, sy = x.std(), y.std()
    if sx < _MIN_SD or sy < _MIN_SD:
        return float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    b = float(np.sign(r) * sy / sx)
    return float(y.mean() - b * x.mean()), b, r


def ols_fit(x, y):
    """Ordinary least squares ``y ~ a + b*x``: ``(a, b, pearson_r)``. See the module note on
    why this is not the default -- it compresses the variance of what it predicts."""
    x = np.asarray(x, "float64"); y = np.asarray(y, "float64")
    if x.std() < _MIN_SD:
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(a), float(b), r


def scale_only_fit(x, y):
    """``b = 1``, ``a = median(y - x)``: a pure offset in log space, i.e. a multiplicative
    scale in raw units. The last rung, and exactly what the retired ``k[s]`` did."""
    x = np.asarray(x, "float64"); y = np.asarray(y, "float64")
    if x.size == 0:
        return 0.0, 1.0, float("nan")
    return float(np.median(y - x)), 1.0, float("nan")


_FITTERS = {"rma": rma_fit, "ols": ols_fit}


def fit_calibration(pairs_by_species, n_species, min_overlap_points=50, form="rma",
                    min_r=0.2, verbose=True):
    """Per-species calibration of eBird onto the BBS scale, with the fallback ladder.

    ``pairs_by_species``: ``{species_index: (x_ebird_log1p, y_bbs_log1p)}`` over the
    overlapping cell-years, already restricted to rows where BOTH values are > 0.

    Returns ``{"a": (S,), "b": (S,), "rung": (S,) of str, "r": (S,), "n": (S,),
    "pooled": {...}, "form": form}``. Every species gets a usable ``(a, b)``; ``rung``
    records which rung produced it so a run can be audited rather than trusted.
    """
    if form not in _FITTERS:
        raise ValueError(f"unknown calibration form {form!r}; expected one of {sorted(_FITTERS)}")
    fitter = _FITTERS[form]
    S = int(n_species)

    # Rung 2 needs the pooled relationship, so fit it first over every overlapping pair.
    xs = [np.asarray(p[0], "float64") for p in pairs_by_species.values()]
    ys = [np.asarray(p[1], "float64") for p in pairs_by_species.values()]
    x_all = np.concatenate(xs) if xs else np.zeros(0)
    y_all = np.concatenate(ys) if ys else np.zeros(0)
    pa, pb, pr = fitter(x_all, y_all) if x_all.size >= 2 else (float("nan"),) * 3
    if not np.isfinite(pb):
        pa, pb, pr = scale_only_fit(x_all, y_all)
        pooled_rung = "pooled_scale_only"
    else:
        pooled_rung = f"pooled_{form}"

    a = np.full(S, pa, dtype="float64")
    b = np.full(S, pb, dtype="float64")
    r = np.full(S, pr, dtype="float64")
    n = np.zeros(S, dtype="int64")
    rung = np.array([pooled_rung] * S, dtype=object)

    for s, (x, y) in pairs_by_species.items():
        if not (0 <= int(s) < S):
            continue
        x = np.asarray(x, "float64"); y = np.asarray(y, "float64")
        n[s] = x.size
        if x.size < int(min_overlap_points):
            continue
        aa, bb, rr = fitter(x, y)
        if not np.isfinite(bb):
            continue
        if not (np.isfinite(rr) and rr >= float(min_r)):
            # Two rejections in one test. A NEGATIVE correlation would invert this species'
            # abundance -- calibrating so more eBird detections mean fewer BBS birds. A
            # near-ZERO one is worse than it looks under RMA: the slope is sign(r)*sd(y)/sd(x),
            # so at r ~ 0 it is a pure variance match with an essentially random sign, fitted
            # to noise. Below min_r the pooled relationship is the more honest estimate.
            continue
        a[s], b[s], r[s], rung[s] = aa, bb, rr, form

    out = {"a": a, "b": b, "r": r, "n": n, "rung": rung, "form": form,
           "pooled": {"a": float(pa), "b": float(pb), "r": float(pr),
                      "n": int(x_all.size), "rung": pooled_rung},
           "min_overlap_points": int(min_overlap_points), "min_r": float(min_r)}
    if verbose:
        counts = {}
        for k in rung:
            counts[k] = counts.get(k, 0) + 1
        print(f"[calib] form={form} pooled a={pa:.3f} b={pb:.3f} r={pr:.3f} "
              f"(n={x_all.size:,})")
        print(f"[calib] rungs: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
              + f"  (min_r={min_r}, min_overlap={min_overlap_points})")
        fitted = np.array([k == form for k in rung])
        if fitted.any():
            print(f"[calib] per-species fits: median r={np.nanmedian(r[fitted]):.3f}, "
                  f"median slope={np.nanmedian(b[fitted]):.3f}, "
                  f"median n={int(np.median(n[fitted]))}")
    return out


def apply_calibration(X_ebird_log, cal):
    """Map log1p eBird values onto the BBS log1p scale: ``a[s] + b[s]*x`` (pure).

    Clipped at 0 because the output feeds ``log1p``-space Ruzicka, where a negative value is
    not a smaller abundance -- it is outside the domain, and ``sum(min)/sum(max)`` would
    silently produce a similarity greater than 1 or a negative denominator.
    """
    X = np.asarray(X_ebird_log, "float64")
    a = np.asarray(cal["a"], "float64").reshape(1, -1)
    b = np.asarray(cal["b"], "float64").reshape(1, -1)
    if X.shape[1] != a.shape[1]:
        raise ValueError(f"calibration has {a.shape[1]} species, X has {X.shape[1]}")
    return np.clip(a + b * X, 0.0, None).astype("float32")


def calibration_meta(cal, species):
    """JSON-safe per-species calibration record for ``points_meta.json``.

    These diagnostics are the artifact that reveals whether the two products agree at all,
    and are worth reading BEFORE trusting a multi-task run: a low median ``r`` means the
    calibration is fitting noise and the eBird rows are not measuring the same thing.
    """
    return {
        "form": cal["form"], "pooled": cal["pooled"],
        "min_overlap_points": cal["min_overlap_points"],
        "per_species": {str(c): {"a": float(cal["a"][i]), "b": float(cal["b"][i]),
                                 "r": (None if not np.isfinite(cal["r"][i])
                                       else float(cal["r"][i])),
                                 "n": int(cal["n"][i]), "rung": str(cal["rung"][i])}
                        for i, c in enumerate(species)},
    }
