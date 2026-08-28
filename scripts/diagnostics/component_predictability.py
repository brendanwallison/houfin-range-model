"""Which ESK components are predictable from the covariates AT ALL. No encoder, no GPU.

DESK peaks at rank 8-16 while the ESK basis's own rank curve keeps improving out to r64
(``basis_domain_gap.py`` section 3). Two readings of that gap imply opposite actions:

  (a) components 9-64 encode variation the ENVIRONMENT does not carry -- species interactions,
      dispersal history, priority effects, plus BBS observation noise at 27 km. Then DESK is near
      its ceiling, more covariates of the same kind cannot help, and the effort belongs downstream.
  (b) components 9-64 ARE environmentally determined and DESK is simply not fitting them. Then
      covariate acquisition is premature and the encoder's capacity, loss or training is the lever.

Nothing measured so far separates them, and the difference is a covariate-acquisition programme.
This script separates them by cutting the encoder out entirely: regress each ESK component on the
covariates directly and report held-out R^2 against component index. A closed-form regressor has no
optimizer, no schedule, no augmentation and no early stopping, so its R^2 is a statement about the
DATA rather than about a training run.

**Read the R^2 curve like this.**
  - R^2 collapses after ~component 8 AND does not respond to capacity or context (both laddered
    below) -> reading (a). The information is not in these covariates. Stop proposing more of them.
  - R^2 stays moderate out to 24-32 while DESK does not achieve it -> reading (b). Covariate work
    is premature; fix the encoder.
  - moderate for some components and near-zero for others -> target new covariates at the
    predictable-but-unlearned ones specifically, and name them.

**Two ladders, because a single weak regressor cannot support reading (a).** "The information is
not in the environment" is the expensive conclusion, and a low R^2 from one model is also
consistent with the model being too weak or reading too little context. So:

  1. CAPACITY: linear ridge, then piecewise-linear ADDITIVE hinges (``relu(x - knot)`` at five
     per-column quantile knots), then those plus random PAIRWISE INTERACTIONS, then RFF on a
     PCA-reduced projection. Every rung stays one closed-form multi-target solve, so capacity is a
     knob with no training dynamics to confound it.

     The ladder is deliberately NOT a plain RBF width sweep. The widest feature set here is ~7C
     (~560 columns), and an isotropic RBF in 560 dimensions is defeated by the curse of
     dimensionality: it has to resolve short length scales in the few informative directions while
     staying flat in the hundreds of uninformative ones, which no sample size on offer supports. A
     measured case is in the tests -- an isotropic RFF reached R^2 0.39 on a two-dimensional signal
     embedded in eight dimensions that an additive+interaction ridge recovers at >0.9. A flat RBF
     curve therefore licenses NO conclusion about the covariates, while a flat additive+interaction
     curve does: saturating single-covariate responses and two-way interactions are the functional
     forms species-environment relationships actually take, so exhausting them is the evidence
     reading (a) needs.
  2. CONTEXT: DESK does not see a bare per-cell covariate vector -- it has a spatial conv (kernel 3)
     and a learned output EMA over years (demographic lag, half-life ~10 yr). A point-only
     regression is therefore a WEAKER feature set than DESK's, and a low R^2 from it would not be
     evidence about the covariates. The rungs add masked 3x3 and 9x9 neighbourhood means, then the
     same channels lagged 5 and 15 years. If R^2 jumps on a context rung, that is itself a finding
     about what the architecture should carry.

**Sections 5-9 exist because a pooled R^2 cannot decide a covariate question.** They grade the
winning rung on the axes the validation suite already separates, by calling that suite's functions
UNCHANGED -- everything in ``validate_baselines`` is pure (plain arrays, no config, no file I/O, no
GPU, no checkpoint), so a diagnostic can borrow it without inheriting the validation driver. Each
axis distinguishes two situations that a single number conflates and that imply opposite purchases:

  5. NOISE CEILING (``per_dimension_signal_noise``, per component). Part of a component IS survey
     noise, and no covariate can predict noise. R^2 is restated against the achievable signal.
     This is the one most likely to revise a raw-R^2 reading of section 3: the function's own
     docstring predicts noise concentrates in the trailing directions, which is exactly where this
     script measures its lowest R^2.
  6. THE REAL BARS (``zspace_idw_baseline``, ``spacetime_idw_z``, ``baseline_panel``). R^2 against a
     component's own mean is the weakest null available. ``validate_baselines`` exists because the
     direction diagnostic beat its permutation null 0.48 to 0.22 and still LOST to inverse-distance
     interpolation at 0.51 -- "a null that a plain interpolator clears by a wide margin is not a
     bar". A component predictable only because its neighbours resemble it is geometry, not
     environment, and no new covariate changes that.
  7. LEVEL vs CHANGE. Sections 1-6 grade a level; DESK has to predict movement. Anything static
     explains level and nothing of change.
  8. DIRECTION and MAGNITUDE (``error_decomposition``, ``epoch_direction_panel``). Only the angular
     half of the error is a covariate problem; the magnitude half is calibration, and shrinkage is
     the MSE-optimal response to a poor angle, so a ridge will shrink. Reported per BAND, never per
     component -- in one dimension the angular term is zero whenever prediction and truth share a
     sign, so a "per-component angle" is sign agreement wearing the name of a direction.
  9. ERA, COVERAGE, GEOGRAPHY. Covariate quality decays backwards (HYDE decadal before 1951, BUI
     5-yearly and CONUS-only), so a pooled number averages a well-covered era with a poor one. If
     the middle components are predictable after 1990 and not before, the purchase is temporal
     resolution, not new variables.

**The payoff number is the rank curve, not R^2.** DESK's objective is dot(z_i, z_j) ~= Ružička, not
per-component variance, so a per-component R^2 does not by itself say which rank would win. The
last section pushes the regressor's held-out predictions through the SAME estimand
``basis_domain_gap.rank_curve`` uses, next to ESK's own projection of the identical points. That
makes three curves comparable on one point set: the ESK oracle (the ceiling), this regressor (what
the covariates support with no encoder), and DESK's swept bestR (from the sweep, not computed here).

Grading is on the SAME spatially blocked holdout DESK is graded on -- the saved
``holdout_cells.npy``/``buffer_cells.npy`` when a trained run is present, otherwise redrawn from
the identical config knobs and seed. Buffer cells are in neither set. Reusing the split is the
whole point: an R^2 measured on a different split is not comparable to the DESK number it exists
to be compared against.

Run on TACC as a batch job, on the lightweight ``vm-small`` queue. Not a login node -- the peak is
~2 GB and the projection runs for minutes, which is what a shared login node's process reaper
kills, and the bare invocation this file first carried was copied from ``basis_domain_gap.py``,
whose 0.04 GiB peak is what justifies its. But not a compute node either: at a 133x224 grid the
biggest BLAS step is ~10 s on 16 cores, so nothing here can use a 128-core node. The dominant cost
is the ESK projection, not the ridge -- ``project_into_z`` caps its CPU batch at 5,000 rows and
holds ~5 live ``(batch, M)`` tensors, so at M=16,000 landmarks that is ~1.6 GB and several minutes.

    bash scripts/tacc/submit_predictability.sh          # -> scripts/tacc/23_predictability.slurm

The wrapper checks the states dir, the schema sidecar and the three basis files before queueing,
and says whether the trained run's holdout masks were found (they are what makes the R^2
comparable to DESK's own val numbers). Rows are subsampled BEFORE either the ESK projection or the
feature assembly, so the other ~2 GB claimant is the feature matrix, ``max_rows x 7C`` float32
(~0.7 GB at 300k rows and C=80) plus one standardized copy. The interaction rung's Gram is
``n * d^2`` with ``d = 6C + pairs``, accumulated in chunks sized to a 256 MiB budget.
``--max-rows``, ``--pairs`` and ``--rff-width`` bound all of it. Writes
``component_predictability.json``.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.community_encoder.train_DESK import covariate_io as cio                 # noqa: E402
from src.community_encoder.train_DESK.config_utils import load_config           # noqa: E402
from src.community_encoder.train_DESK.eigenbasis_diag import ruzicka_gram       # noqa: E402
from src.community_encoder.train_DESK.esk_kernel import (                       # noqa: E402
    coarse_spatial, project_points_to_z)
# Everything below is reused UNCHANGED from the validation suite. All of it is pure -- plain
# arrays in, no config, no file I/O, no GPU, no checkpoint -- which is why a diagnostic can call
# it without inheriting the validation driver's dependencies. `_era_of` is imported despite the
# underscore rather than open-coding the decade convention a fourth time (it is already
# duplicated in validate_spacetime's period loop and in per_era_attenuation).
from src.community_encoder.train_DESK.validate_baselines import (               # noqa: E402
    ATTEN_GAP, ATTEN_GAP_TOL, DEFAULT_EPOCHS, _era_of, baseline_panel,
    epoch_direction_panel, error_decomposition, per_dimension_signal_noise,
    per_era_attenuation, spacetime_idw_baseline, spacetime_idw_z, zspace_idw_baseline)

# (year_offset, neighbourhood_width) blocks the feature matrix is built from. The rungs below are
# cumulative subsets of these, so the whole matrix is gathered once and each rung is a column slice.
BLOCKS = ((0, 1), (0, 3), (0, 9), (5, 1), (5, 9), (15, 1), (15, 9))
RUNGS = {
    "point": ((0, 1),),
    "+spatial": ((0, 1), (0, 3), (0, 9)),
    "+temporal": BLOCKS,
}
RANKS = (8, 16, 24, 32, 48, 64)
# Multipliers on the median-heuristic RBF gamma, scanned on the inner split. Wide
# enough to span a 100x mis-set scale in either direction, because the heuristic
# missing by ~64x is exactly what a test caught.
BANDWIDTH_MULTS = (0.01, 0.1, 1.0, 10.0, 100.0)
#: Ridge penalties offered to the inner split. The top of this grid used to be 1.0, which was far
#: too small for the widest rungs: on the real run the additive (12,684 features) and interaction
#: (14,684 features) rungs scored WORSE than plain linear (0.139 and 0.119 against 0.175) on 80,415
#: training rows. That is under-regularisation, not absent structure, and it made those two rungs
#: uninformative -- they could not support the "capacity is exhausted" reading they exist to
#: support. The grid now reaches 1e4 so a 14k-column basis can actually be shrunk.
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)
#: Component bands. Direction and magnitude are reported over these rather than per component
#: because an angle needs at least two dimensions -- see `banded_direction`.
BANDS = ((0, 8), (8, 16), (16, 32), (32, 64))


# --- ridge, multi-target, one solve for every component ---------------------------------------

def _chunk_for(d, budget_mib=256):
    """Rows per chunk so the mapped float64 block stays inside ``budget_mib``.

    A fixed chunk is a memory bug waiting for the widest rung: 20,000 rows of a 5,400-column
    interaction map is 860 MB of float64 per chunk, for a Gram of 230 MB.
    """
    return max(500, int(budget_mib * 2 ** 20 // max(8 * d, 1)))


def _accum(F, Y, rows, chunk=None, fmap=None):
    """Accumulate ``(n, sum_x, sum_y, XtX, XtY)`` over ``rows`` of ``F``, optionally mapped.

    Chunked because ``fmap`` can widen 500 columns to 4096: materializing that for 300k rows is
    5 GB, while the Gram it feeds is 134 MB. The sums are accumulated alongside so the centering
    below needs no second pass -- and centering is what removes the intercept, which otherwise
    has to be excluded from the ridge penalty by hand.
    """
    d = (fmap(F[rows[:1]]) if fmap is not None else F[rows[:1]]).shape[1]
    chunk = chunk or _chunk_for(d)
    L = Y.shape[1]
    XtX = np.zeros((d, d), dtype="float64")
    XtY = np.zeros((d, L), dtype="float64")
    sx = np.zeros(d, dtype="float64")
    sy = np.zeros(L, dtype="float64")
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        xb = np.asarray(F[r], dtype="float64")
        if fmap is not None:
            xb = fmap(xb)
        yb = np.asarray(Y[r], dtype="float64")
        XtX += xb.T @ xb
        XtY += xb.T @ yb
        sx += xb.sum(0)
        sy += yb.sum(0)
    return len(rows), sx, sy, XtX, XtY


def _solve(acc, alpha):
    """Ridge coefficients and target means from an accumulation, centered analytically.

    ``alpha`` is scaled by ``n`` so the same value means the same thing at any sample size --
    otherwise every rung of a ladder that changes the row count would need its own grid.
    """
    n, sx, sy, XtX, XtY = acc
    mx, my = sx / n, sy / n
    A = XtX - n * np.outer(mx, mx)
    B = XtY - n * np.outer(mx, my)
    A = A + alpha * n * np.eye(A.shape[0])
    try:
        c = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, B, rcond=None)[0]
    return c, mx, my


def _predict(F, rows, coef, mx, my, chunk=None, fmap=None):
    chunk = chunk or _chunk_for(coef.shape[0])
    out = np.empty((len(rows), coef.shape[1]), dtype="float64")
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        xb = np.asarray(F[r], dtype="float64")
        if fmap is not None:
            xb = fmap(xb)
        out[s:s + chunk] = (xb - mx) @ coef + my
    return out


def held_out_r2(y_true, y_pred, train_mean):
    """Per-component held-out R^2, plus the constant-predictor check that keeps it honest.

    R^2 is against the HELD-OUT variance -- the "is this component predictable" quantity. That
    alone can read positive for a model that is worse than predicting the training mean, whenever
    the val mean has shifted, so ``beats_const`` is reported next to it and any component failing
    it is not predictable no matter what its R^2 says.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    sse = ((y_true - y_pred) ** 2).sum(0)
    sst = ((y_true - y_true.mean(0)) ** 2).sum(0)
    sse_c = ((y_true - np.asarray(train_mean, dtype="float64")) ** 2).sum(0)
    r2 = 1.0 - sse / np.maximum(sst, 1e-300)
    return r2, sse < sse_c


def _rff(d_in, d_out, gamma, seed, proj=None):
    """Random Fourier features for the RBF kernel: ``sqrt(2/D) cos(x W + b)``.

    ``proj`` pre-projects the input (a PCA basis), which is what makes an isotropic kernel usable
    at all here -- see the module docstring on the curse of dimensionality. Deterministic in
    ``seed`` so train and val get the identical map; the failure mode if they do not is a near-zero
    R^2 that looks exactly like an unpredictable component.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=np.sqrt(2.0 * gamma), size=(d_in, d_out))
    b = rng.uniform(0.0, 2.0 * np.pi, size=d_out)
    scale = np.sqrt(2.0 / d_out)
    if proj is None:
        return lambda x: scale * np.cos(x @ W + b)
    return lambda x: scale * np.cos((x @ proj) @ W + b)


def _median_bandwidth(F, rows, rng, n=2000, proj=None):
    """RBF length scale from the median pairwise distance of a training subsample."""
    take = rows[rng.permutation(len(rows))[:min(n, len(rows))]]
    S = np.asarray(F[take], dtype="float64")
    if proj is not None:
        S = S @ proj
    d2 = np.maximum((S ** 2).sum(1)[:, None] + (S ** 2).sum(1)[None, :] - 2 * S @ S.T, 0.0)
    iu = np.triu_indices(len(S), k=1)
    med = np.median(np.sqrt(d2[iu]))
    return 1.0 / (2.0 * max(med, 1e-6) ** 2)


def _hinges(F, rows, n_knots=5):
    """``x -> [x, relu(x - q_1), ..., relu(x - q_k)]`` at per-column TRAIN quantile knots.

    A piecewise-linear additive expansion, which is the cheapest honest escape from linearity in
    high dimensions: it costs ``(1 + k)`` columns per input rather than paying for a joint density,
    and the shape it buys -- a response that saturates or switches on past a threshold -- is the
    one single-covariate species-environment relationships actually have.

    Knots come from the training rows only. Fitting them on all rows would leak the held-out
    covariate distribution into the basis, which is a smaller leak than fitting coefficients there
    but is still one, and it is free to avoid.
    """
    q = np.quantile(np.asarray(F[rows], dtype="float64"),
                    np.linspace(1.0 / (n_knots + 1), n_knots / (n_knots + 1), n_knots),
                    axis=0).astype("float32")                      # (k, d)

    def _f(x):
        x = np.asarray(x, dtype="float64")
        h = np.maximum(x[:, None, :] - q[None, :, :], 0.0)          # (n, k, d)
        return np.concatenate([x, h.reshape(len(x), -1)], axis=1)
    return _f, q.shape[0]


def _interactions(base_fn, d_raw, n_pairs, seed):
    """``base_fn(x)`` with ``n_pairs`` random products ``x_i * x_j`` appended.

    Random pairs rather than the full ``d(d-1)/2``: at d=560 the complete second-order basis is
    157k columns, which is not a ridge solve any more. A random subset still answers the question
    the rung exists for -- does ANY two-way interaction structure raise R^2 -- because if a large
    random sample of pairs moves nothing, the specific pairs are unlikely to.
    """
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, d_raw, size=n_pairs)
    jj = rng.integers(0, d_raw, size=n_pairs)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]

    def _f(x):
        x = np.asarray(x, dtype="float64")
        return np.concatenate([base_fn(x), x[:, ii] * x[:, jj]], axis=1)
    return _f


def _pca_basis(F, rows, q, seed=0):
    """Top-``q`` PCA directions of the TRAIN rows, as a ``(d, q)`` projection."""
    S = np.asarray(F[rows[np.random.default_rng(seed).permutation(len(rows))[:20000]]],
                   dtype="float64")
    S = S - S.mean(0)
    w, V = np.linalg.eigh(S.T @ S / max(len(S) - 1, 1))
    return V[:, np.argsort(w)[::-1][:q]].astype("float32")


def capacity_ladder(F, tr, n_pairs=2000, pca_dim=48, rff_width=2048, rng=None):
    """The closed-form capacity rungs, as ``[(name, fmap)]`` with ``fmap=None`` meaning linear.

    Ordered by capacity so a monotone-flat curve across all four is the evidence reading (a)
    requires: linear, additive, additive + two-way, and smooth-high-order-in-the-dominant-
    directions. Each rung strictly contains the previous one's function class except the last,
    which trades coverage of all inputs for unbounded order within a subspace.
    """
    rng = rng or np.random.default_rng(0)
    d = F.shape[1]
    hin, k = _hinges(F, tr)
    q = min(int(pca_dim), d)
    P = _pca_basis(F, tr, q)
    g = _median_bandwidth(F, tr, rng, proj=P)
    # The RBF rung is offered at every BANDWIDTH_MULTS scale as ONE rung with several feature
    # maps, so fit_and_score picks the scale per component on the inner split. A single median-
    # heuristic gamma was measured 64x too wide on a test target (see fit_and_score), and a rung
    # that fails only because its scale was assumed contributes nothing to the capacity argument.
    return [
        ("linear", (None,), d),
        (f"additive(k={k})", (hin,), d * (1 + k)),
        (f"additive+{n_pairs}pairs", (_interactions(hin, d, n_pairs, seed=11),),
         d * (1 + k) + n_pairs),
        (f"rff{rff_width}@pca{q}",
         tuple(_rff(q, int(rff_width), g * m, seed=12, proj=P) for m in BANDWIDTH_MULTS),
         int(rff_width)),
    ]


def fit_and_score(F, Y, tr, va, alphas=ALPHAS, fmaps=(None,), inner_frac=0.25, rng=None,
                  predict_rows=None):
    """Ridge on ``tr``, ``(feature map, alpha)`` chosen on an inner split of ``tr``, scored on ``va``.

    Neither knob is chosen on ``va``: that is the held-out set every number here is compared to
    DESK's on, and selecting on it would make this diagnostic optimistic by exactly the amount that
    matters. The inner split is a slice of a permutation of the training rows -- the spatial
    separation that matters is already in the outer blocked split.

    ``fmaps`` is a sequence because the RBF length scale cannot be fixed by the median heuristic
    alone. A single heuristic bandwidth was 64x too wide for a sin/cos target in
    ``tests/test_component_predictability.py``, and the capacity ladder scored 0.036 on a signal it
    should have recovered at 0.9 -- which would have read as "the information is not in the
    environment", the script's most expensive conclusion, produced entirely by a mis-set scale. So
    the scale is a selected knob like alpha, chosen per component on the inner split.

    ``predict_rows`` widens WHERE the fitted model is evaluated without changing what it is scored
    on. The validation machinery this feeds needs a prediction at rows other than ``va``:
    ``baseline_panel`` computes ``norm(z_desk - z_obs)`` over every row before masking, and
    ``epoch_direction_panel`` reads arbitrary ``(cell, year)`` pairs. Scoring stays on ``va``
    exactly as before -- the returned ``r2`` is unaffected -- so this costs one extra predict pass
    and buys no optimism.
    """
    rng = rng or np.random.default_rng(0)
    perm = rng.permutation(len(tr))
    cut = int(len(tr) * (1.0 - inner_frac))
    itr, iva = tr[perm[:cut]], tr[perm[cut:]]
    L = Y.shape[1]
    best = np.full(L, -np.inf)
    pick_f, pick_a = np.zeros(L, int), np.zeros(L, int)
    for fi, f in enumerate(fmaps):
        acc_i = _accum(F, Y, itr, fmap=f)
        for ai, a in enumerate(alphas):
            c, mx, my = _solve(acc_i, a)
            r2, _ = held_out_r2(Y[iva], _predict(F, iva, c, mx, my, fmap=f), my)
            upd = r2 > best
            best = np.where(upd, r2, best)
            pick_f, pick_a = np.where(upd, fi, pick_f), np.where(upd, ai, pick_a)
        del acc_i
    rows = va if predict_rows is None else np.asarray(predict_rows)
    pred = np.empty((len(rows), L), dtype="float64")
    tmean = np.zeros(L)
    for fi in sorted(set(int(v) for v in pick_f)):
        acc = _accum(F, Y, tr, fmap=fmaps[fi])
        for ai in sorted(set(int(a) for a in pick_a[pick_f == fi])):
            c, mx, my = _solve(acc, alphas[ai])
            p = _predict(F, rows, c, mx, my, fmap=fmaps[fi])
            cols = np.where((pick_f == fi) & (pick_a == ai))[0]
            pred[:, cols] = p[:, cols]
            tmean[cols] = my[cols]
        del acc
    # Scored on `va` whatever `predict_rows` asked for. Locating va inside rows rather than
    # re-predicting keeps the two exactly consistent -- a second predict pass with a different
    # chunking could differ in the last bits and make the reported r2 not quite the r2 of the
    # array handed downstream.
    if predict_rows is None:
        pred_va = pred
    else:
        pos = {int(v): i for i, v in enumerate(rows)}
        pred_va = pred[np.array([pos[int(v)] for v in va])]
    r2, beats = held_out_r2(Y[va], pred_va, tmean)
    return dict(r2=r2, beats_const=beats, alpha=[alphas[int(j)] for j in pick_a],
                fmap=pick_f.tolist(), inner_r2=best), pred


# --- feature assembly --------------------------------------------------------------------------

def _masked_mean(grid, mask, k):
    """Neighbourhood mean of ``(H,W,C)`` over ``k x k``, counting only valid cells.

    ``norm_grid`` zero-fills invalid cells and 0 is the post-norm channel mean, so a plain filter
    would silently pull the mean toward 0 near every coast and border -- a real gradient in the
    feature that is an artifact of the fill. Dividing by the filtered mask removes it.
    """
    from scipy.ndimage import uniform_filter
    if k <= 1:
        return grid
    w = mask.astype("float32")
    num = uniform_filter(grid * w[..., None], size=(k, k, 1), mode="nearest")
    den = uniform_filter(w, size=(k, k), mode="nearest")[..., None]
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0).astype("float32")


def build_features(pidx, states_dir, schema, mu, sd, blocks=BLOCKS, verbose=True):
    """Gather ``(N, len(blocks)*C)`` covariate features for every point, one state load per year.

    Each block is a ``(year_offset, width)`` pair. A lagged year outside the available range is
    CLAMPED rather than dropped, matching what the streams already do between snapshots ("hold
    constant rather than extrapolating") -- dropping instead would delete the early-era points,
    which are the ones the historical reconstruction exists for.

    Returns ``(F, valid, cols)`` where ``valid`` marks rows whose whole feature vector is finite
    and ``cols`` maps each block to its column slice.
    """
    avail = sorted(int(f[6:10]) for f in os.listdir(states_dir)
                   if f.startswith("state_") and f.endswith(".npz"))
    if not avail:
        raise SystemExit(f"no state_*.npz in {states_dir}")
    lo, hi = avail[0], avail[-1]
    py = pidx[:, 2].astype(int)
    pr, pc = pidx[:, 0].astype(int), pidx[:, 1].astype(int)
    C = int(schema["streams"][-1]["end"])
    F = np.full((len(pidx), C * len(blocks)), np.nan, dtype="float32")
    cols = {b: slice(i * C, (i + 1) * C) for i, b in enumerate(blocks)}
    want = {}                                     # state year -> [(block, rows)]
    for b in blocks:
        off, _k = b
        need = np.clip(py - off, lo, hi)
        for ys in np.unique(need):
            want.setdefault(int(ys), []).append((b, np.where(need == ys)[0]))
    for i, ys in enumerate(sorted(want)):
        base, m = cio.norm_grid(cio.load_state_stack(ys, states_dir, schema), mu, sd)
        for k in sorted({b[1] for b, _ in want[ys]}):
            g = _masked_mean(base, m, k)
            for b, rows in want[ys]:
                if b[1] != k:
                    continue
                vals = g[pr[rows], pc[rows]]
                vals[~m[pr[rows], pc[rows]]] = np.nan
                F[rows, cols[b]] = vals
            del g
        del base, m
        if verbose and (i % 20 == 0 or i == len(want) - 1):
            print(f"  states {i + 1}/{len(want)} (year {ys})", flush=True)
    return F, np.isfinite(F).all(1), cols


# --- the rank curve, on predictions instead of on a basis --------------------------------------

def rank_curve(X_va, z_va, X_tr=None, z_tr=None, ranks=RANKS):
    """Kernel agreement of ``dot(z_i[:r], z_j[:r])`` with EXACT Ružička, per rank.

    Same estimand as ``basis_domain_gap.rank_curve`` and the trainer's kernel metric -- but a
    RIDGE PREDICTION cannot be fed to it raw. Ridge shrinks toward the mean, per component and by
    a different amount for each, so predicted dot products are systematically small. Raw MSE then
    REWARDS shrinkage whenever the truth is closer to zero than the oracle's dot products are: a
    synthetic run had the weakest rung scoring 41.9 against the oracle's 272.9, which reads as
    "the regressor beats ESK" and means the opposite. Two fixes, both reported:

    ``mse`` calibrates ``a * pred + b`` per rank on TRAIN pairs and applies it to the val pairs.
    Two nuisance parameters, fitted off the held-out set, which removes the shrinkage bias without
    granting the regressor any of the fit that matters. The oracle's ``scale`` should land near 1;
    a regressor's ``scale`` well above 1 is the shrinkage, quantified.

    ``corr`` is Pearson r between predicted dot products and truth, which no affine map can move.
    It is the number to trust when the two disagree, and it is what "which rank wins" means.

    ``mse_raw`` is the uncalibrated value, kept only so the oracle row stays numerically
    comparable to ``basis_domain_gap``'s curve on the same estimand.
    """
    Rva = ruzicka_gram(np.asarray(X_va, dtype="float64"))
    iu = np.triu_indices(len(X_va), k=1)
    t_va = Rva[iu]
    zv = np.asarray(z_va, dtype="float64")
    zt = None
    if X_tr is not None and z_tr is not None and len(X_tr) > 2:
        Rtr = ruzicka_gram(np.asarray(X_tr, dtype="float64"))
        iut = np.triu_indices(len(X_tr), k=1)
        t_tr = Rtr[iut]
        zt = np.asarray(z_tr, dtype="float64")
    out = {}
    for r in ranks:
        rr = min(int(r), zv.shape[1])
        p_va = (zv[:, :rr] @ zv[:, :rr].T)[iu]
        a, b = 1.0, 0.0
        if zt is not None:
            p_tr = (zt[:, :rr] @ zt[:, :rr].T)[iut]
            a, b = np.linalg.lstsq(np.stack([p_tr, np.ones_like(p_tr)], 1), t_tr, rcond=None)[0]
        sd = p_va.std()
        out[int(r)] = {
            "mse": float(np.mean((a * p_va + b - t_va) ** 2)),
            "mse_raw": float(np.mean((p_va - t_va) ** 2)),
            "corr": float(np.corrcoef(p_va, t_va)[0, 1]) if sd > 1e-12 else 0.0,
            "scale": float(a),
        }
    return out


def best_rank(curve, key="corr", min_gain=0.01):
    """The rank that wins on ``key``, or ``None`` when the curve is FLAT.

    argmax alone invents a winner out of a tie: a synthetic run printed ``bestR=48`` for a curve
    reading +0.183 at every rank. "Which rank wins" is the question this whole script feeds, so a
    tie has to be reported as a tie -- a spurious bestR of 48 and a real one mean opposite things
    for latent_dim. ``min_gain`` is on the same scale as the key (correlation points, or relative
    MSE reduction for ``mse``).
    """
    ranks = sorted(curve)
    if key == "corr":
        vals = [curve[r]["corr"] for r in ranks]
        b = int(np.argmax(vals))
        gain = vals[b] - vals[0]
    else:
        vals = [curve[r][key] for r in ranks]
        b = int(np.argmin(vals))
        gain = (vals[0] - vals[b]) / max(abs(vals[0]), 1e-300)
    return (ranks[b], gain) if gain > min_gain else (None, gain)


def band_means(r2, beats, bands=((0, 8), (8, 16), (16, 32), (32, 64))):
    """Mean R^2 per component band, counting a component that loses to the training mean as 0."""
    r = np.where(beats, np.asarray(r2), 0.0)
    return {f"{a + 1}-{b}": float(np.mean(r[a:b])) for a, b in bands if a < len(r)}


def verdict(rungs, caps, var_share):
    """Print the three-way reading the script exists to decide between, with its own numbers.

    The thresholds are judgment calls, so every quantity they are applied to is printed first:
    a reader who disagrees with 0.10 or 0.30 can re-read the same table against their own line.
    """
    best_name, best = max(list(rungs.items()) + list(caps.items()),
                          key=lambda kv: np.mean(np.where(kv[1]["beats_const"], kv[1]["r2"], 0.0)))
    b = band_means(best["r2"], best["beats_const"])
    # Components 9-32 straight off the array rather than by averaging band labels: at a latent_dim
    # below 17 those labels do not exist, the average of an empty list is nan, and every threshold
    # below silently compares False -- printing "nan" inside a verdict rather than refusing.
    r2b = np.where(best["beats_const"], np.asarray(best["r2"], dtype="float64"), 0.0)
    if len(r2b) <= 8:
        print("\n=== verdict ===")
        print(f"  latent_dim is {len(r2b)}: there are no components past 8 to ask the question "
              f"about, so no reading is available.")
        return best_name, b
    mid = float(np.mean(r2b[8:min(32, len(r2b))]))
    ctx = max(np.mean(np.where(v["beats_const"], v["r2"], 0.0)) for v in rungs.values())
    ctx0 = np.mean(np.where(rungs["point"]["beats_const"], rungs["point"]["r2"], 0.0))
    cap = max(np.mean(np.where(v["beats_const"], v["r2"], 0.0)) for v in caps.values()) if caps else ctx
    print("\n=== verdict ===")
    print(f"  best model: {best_name}")
    print("  mean held-out R^2 by component band: "
          + "  ".join(f"{k}={v:.3f}" for k, v in b.items()))
    print(f"  components 9-32 mean R^2 = {mid:.3f}")
    print(f"  context ladder moved mean R^2 {ctx0:.3f} -> {ctx:.3f}; capacity ladder reached {cap:.3f}")
    responds = (ctx - ctx0) > 0.02 or (cap - ctx) > 0.02
    r2 = r2b
    if mid < 0.10 and not responds:
        print("  -> (a) NOT IN THE ENVIRONMENT. Components 9-32 are near-unpredictable and neither\n"
              "     more capacity nor more context moves them, so DESK is close to its ceiling and\n"
              "     more covariates of the same kind will not buy rank. Spend the effort downstream.")
    elif mid > 0.30:
        print("  -> (b) THE ENCODER IS THE LIMIT. Components 9-32 are predictable from covariates\n"
              "     DESK ALREADY HAS, by a regressor with no training loop. Covariate acquisition is\n"
              "     premature; the capacity, loss or training schedule is the lever.")
    else:
        good = [int(i) + 1 for i in np.where(r2[:32] > 0.30)[0] if i >= 8]
        weak = [int(i) + 1 for i in np.where(r2[:32] < 0.10)[0] if i >= 8]
        # (c) exists for the case where SOME components are predictable and others are not, so
        # that new covariates can be aimed at the second group. With no near-zero component its
        # advice has no referent. The real run printed exactly that -- "(c) MIXED ... Near-zero:
        # none" -- because the 9-32 mean of 0.295 fell under the 0.30 cutoff for (b) by 0.005,
        # and then contradicted itself. A branch that cannot name its target is (b).
        if not weak:
            print(f"  -> (b) THE ENCODER IS THE LIMIT (9-32 mean {mid:.3f}, just under the 0.30\n"
                  f"     line, but NO component in 9-32 is near zero -- there is no "
                  f"predictable-but-unlearned\n     split to exploit). Components 9-32 are "
                  f"predictable from covariates DESK ALREADY HAS.\n     Covariate acquisition is "
                  f"premature; capacity, loss or training is the lever.")
        else:
            print(f"  -> (c) MIXED. Predictable past 8: components {good if good else 'none'}. "
                  f"Near-zero: {weak}.")
            print("     Target new covariates at the near-zero components specifically, and check "
                  "what\n     the predictable ones share before assuming DESK cannot reach them.")
    if responds:
        print(f"  NOTE the ladders DID move R^2 (context +{ctx - ctx0:.3f}, capacity "
              f"+{cap - ctx:.3f}). A point-only linear number would have understated the\n"
              f"  covariates; whatever rung won here is the feature set the encoder should carry.")
    cum = np.cumsum(var_share * r2) / np.maximum(np.cumsum(var_share), 1e-300)
    print("  predictable share of target variance out to rank r: "
          + "  ".join(f"r{r}={cum[min(r, len(cum)) - 1]:.3f}" for r in RANKS if r <= len(cum)))
    return best_name, b


# --- the decompositions, all borrowed from the validation suite --------------------------------

#: Below this signal share the achievable-R^2 rescale is REFUSED rather than reported. Dividing by
#: a near-zero share amplifies without bound: a synthetic run printed "R^2 vs ACHIEVABLE -215" from
#: a raw R^2 of -0.0005 over a share of 0.006. A component that is 97% noise has no meaningful
#: achievable R^2 -- that IS the finding, and it belongs in the signal-share column.
MIN_SIGNAL_SHARE = 0.05


def achievable_r2(r2, sn, min_share=MIN_SIGNAL_SHARE):
    """Rescale each component's R^2 by the share of its variance that is not survey noise.

    R^2 against total variance answers "how much of this component can be predicted", which is the
    wrong question when part of the component IS measurement noise. ``per_dimension_signal_noise``
    measures that share directly: adjacent-year within-cell differences are essentially ``2 sigma^2``
    (real community change over one year is small), and fixed-gap differences carry the same noise
    plus real change. A component that is 90% noise cannot be predicted from ANY covariate, and
    reporting its low R^2 as a covariate gap is a category error -- which is the specific mistake
    this rescale prevents, since the module's own docstring predicts noise concentrates in the
    trailing directions, exactly where this script measured its lowest R^2.

    Bounded on BOTH sides, and refused where it cannot mean anything:

    * above 1.0, because a ratio over 1 says the noise estimate is the imprecise quantity, not that
      the model beat the ceiling;
    * below 0.0, because "captured a negative fraction of the achievable signal" is not a reading --
      failing to beat the mean is already what the raw R^2 says;
    * NaN where the signal share is under ``min_share``. This is the case the first version got
      badly wrong: it divided regardless, and a raw R^2 of -0.0005 over a share of 0.006 printed as
      -215. Where the share is that low, the share itself is the answer.

    Returns ``(achievable, signal_share)``, both per component, or ``(None, None)`` if
    ``per_dimension_signal_noise`` declined for want of pairs.
    """
    if "signal_var" not in sn:
        return None, None
    sig = np.asarray(sn["signal_var"], dtype="float64")
    tot = np.asarray(sn["total_var"], dtype="float64")
    share = np.where(tot > 1e-300, sig / np.maximum(tot, 1e-300), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ach = np.clip(np.asarray(r2, dtype="float64") / share, 0.0, 1.0)
    return np.where(share >= float(min_share), ach, np.nan), share


def per_component_r2_of(pred, truth):
    """Per-component R^2 of an arbitrary prediction array against ``truth``, NaN rows skipped.

    Used for the IDW bars, which come back with NaN on rows the interpolation could not reach (a
    year with fewer than ``k`` training cells). Dropping those rows PER COMPONENT rather than
    globally keeps every component scored on as many rows as it has, and the count is returned so a
    component scored on a handful of rows is visible rather than silently comparable.
    """
    pred = np.asarray(pred, dtype="float64")
    truth = np.asarray(truth, dtype="float64")
    out = np.full(pred.shape[1], np.nan)
    n = np.zeros(pred.shape[1], dtype=int)
    for l in range(pred.shape[1]):
        m = np.isfinite(pred[:, l]) & np.isfinite(truth[:, l])
        n[l] = int(m.sum())
        if n[l] < 30:
            continue
        t = truth[m, l]
        sst = ((t - t.mean()) ** 2).sum()
        if sst <= 1e-300:
            continue
        out[l] = 1.0 - ((t - pred[m, l]) ** 2).sum() / sst
    return out, n


def banded_direction(pred, truth, bands=BANDS):
    """Magnitude/angular error split per component BAND, via ``error_decomposition`` unchanged.

    Per component is deliberately refused. ``error_decomposition`` splits ``||a-b||^2`` into
    ``(||a||-||b||)^2`` plus an angular remainder; in one dimension that remainder is zero whenever
    prediction and truth share a sign, so a "per-component angle" is sign agreement wearing the
    name of a direction. An angle needs at least two dimensions, so the finest honest granularity
    is a band.

    Reports the ANGULAR SHARE of the error alongside the magnitude share because the two trade off:
    shrinkage is the MSE-optimal response to a poor angle, so a ridge will shrink, and only the
    angular half is a covariate problem at all -- the magnitude half is calibration.
    """
    pred = np.asarray(pred, dtype="float64")
    truth = np.asarray(truth, dtype="float64")
    out = {}
    for a, b in bands:
        b = min(b, pred.shape[1])
        if a >= b:
            continue
        tot, mag, ang, cos = error_decomposition(pred[:, a:b], truth[:, a:b])
        mt = np.mean(tot)
        out[f"{a + 1}-{b}"] = {
            "n_dims": int(b - a),
            "mag_share": float(np.mean(mag) / max(mt, 1e-300)),
            "ang_share": float(np.mean(ang) / max(mt, 1e-300)),
            "median_cos": float(np.nanmedian(cos)),
            "norm_ratio": float(np.median(
                np.linalg.norm(pred[:, a:b], axis=1)
                / np.maximum(np.linalg.norm(truth[:, a:b], axis=1), 1e-12))),
        }
    return out


def gap_pairs(pidx, gap=ATTEN_GAP, tol=ATTEN_GAP_TOL):
    """Within-cell row-index pairs ``(i_early, i_late)`` separated by ``gap +/- tol`` years.

    The gap is FIXED rather than each cell's own span, for the reason recorded on
    ``per_era_attenuation``: a record starting in 1966 spans ~59 years and one starting in 2010
    spans <=15, so a per-cell span ranks cells by RECORD LENGTH instead of by the thing being
    asked. Using ``ATTEN_GAP`` specifically means the signal/noise numbers from
    ``per_dimension_signal_noise`` describe the same quantity this differencing produces.

    One pair per (cell, earlier year), matching that function's own ``break``.
    """
    by_cell = {}
    for i, (r, c, y) in enumerate(np.asarray(pidx)):
        by_cell.setdefault((int(r), int(c)), []).append((int(y), i))
    lo, hi = int(gap) - int(tol), int(gap) + int(tol)
    ea, la = [], []
    for rows in by_cell.values():
        rows.sort()
        for a, (y0, i0) in enumerate(rows):
            for (y1, i1) in rows[a + 1:]:
                d = y1 - y0
                if lo <= d <= hi:
                    ea.append(i0); la.append(i1)
                    break
                if d > hi:
                    break
    return np.array(ea, dtype=int), np.array(la, dtype=int)


def nochange_rows(pidx, recent_year):
    """``(to_rec, has_rec)``: for each row, the index of its own cell's ``recent_year`` row.

    The no-change null has no standalone helper in the validation suite -- it is inlined in
    ``validate_spacetime.run_validate``, again inside ``zspace_reconstruction``, and a third time as
    a per-row loop in ``baseline_panel``. This is that block, kept identical including the ``-1``
    sentinel and the ``has_rec`` gate: a cell with no recent-year row must come back NaN rather
    than silently borrowing row 0.
    """
    pidx = np.asarray(pidx)
    rec = {}
    for k in np.flatnonzero(pidx[:, 2] == int(recent_year)):
        rec[(int(pidx[k, 0]), int(pidx[k, 1]))] = int(k)
    to_rec = np.array([rec.get((int(r), int(c)), -1) for r, c, _y in pidx], dtype=int)
    return to_rec, to_rec >= 0


def bui_avail_grid(states_dir, schema, year):
    """Per-cell BUI coverage fraction ``(H, W)``, located through the loader, never by position.

    ``indicator_channels`` is the authority. The channel's offset within its stream is NOT a fixed
    constant and the repo disagrees with itself about it: ``preprocess/bui.py`` writes
    ``sorted(variables)`` into the manifest, which puts ``bui_avail`` FIRST, while
    ``tests/test_desk_training.py`` asserts it is last on a hand-built schema. Only the loader
    resolves it correctly.

    Filtered by stream name rather than taking ``indicator_channels(...)[0]``: today bui is the only
    stream declaring an indicator, so the first entry happens to be right, and it would silently
    become another stream's channel the day a second one declares one.

    Availability is computed once from the first BUI snapshot and reused for every year, so this is
    a time-invariant geographic mask and the choice of ``year`` does not matter. It also arrives
    un-standardised (``fit_norm`` pins mu=0/sd=1 on indicator channels), so the value is the raw
    fraction: 0 absent, 1 fully covered.
    """
    bui = next((st for st in schema["streams"] if st.get("indicator_variable")
                and st["name"] == "bui"), None)
    if bui is None:
        return None
    variables = [str(v) for v in (bui.get("variables") or [])]
    ch = int(bui["start"]) + variables.index(str(bui["indicator_variable"]))
    if ch not in cio.indicator_channels(schema):
        raise SystemExit(f"resolved BUI availability to channel {ch}, which covariate_io does not "
                         f"list as an indicator channel {cio.indicator_channels(schema)}")
    return cio.load_state_stack(year, states_dir, schema)[..., ch]


def r2_by_group(pred, truth, groups):
    """``{label: (r2 per component, n)}`` for a dict of boolean row masks."""
    return {k: per_component_r2_of(pred[m], truth[m]) for k, m in groups.items()
            if int(np.asarray(m).sum()) >= 30}


def _jsonable(o):
    """JSON default for the validation suite's returns, which nest numpy arrays and scalars.

    ``epoch_direction_panel`` carries per-site ``rows``/``cols`` arrays and ``per_era_attenuation``
    numpy scalars, so a plain ``json.dump`` raises. Converting here rather than sanitising each
    return keeps those dicts stored exactly as the functions produced them.
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-rows", type=int, default=300000,
                    help="cap on train+val rows (subsampled within each side)")
    ap.add_argument("--pairs", type=int, default=2000,
                    help="random pairwise-interaction columns on the interaction rung")
    ap.add_argument("--pca-dim", type=int, default=48,
                    help="PCA dimension the RBF rung's isotropic kernel lives in")
    ap.add_argument("--rff-width", type=int, default=2048,
                    help="random-Fourier width on the PCA rung")
    ap.add_argument("--curve-points", type=int, default=700,
                    help="held-out points to SCORE the rank curve on; an equal disjoint sample "
                         "is used to fit its two calibration scalars, so 2x this many are drawn")
    ap.add_argument("--out", default="component_predictability.json")
    args = ap.parse_args()

    cfg = load_config(os.environ.get("ESK_DESK_CONFIG") or None)
    paths, desk_cfg = cfg["paths"], cfg["desk"]
    states_dir = os.path.join(paths["hist_dir"], "yearly_states")
    schema = cio.load_schema(states_dir)
    z_dir = desk_cfg["z_dir"]
    rng = np.random.default_rng(0)

    from src.community_encoder.train_DESK.desk_training import load_point_set, supervised_cells
    from src.config_utils import target_points_dir
    points_dir = target_points_dir(cfg)
    Xp, pidx, _w, sup = load_point_set(points_dir)
    latent_dim = int(desk_cfg.get("latent_dim") or json.load(
        open(os.path.join(z_dir, "meta.json"), encoding="utf-8"))["latent_dim"])
    # The projection is deferred until after the subsample -- see the note at `sel` below. It is
    # the single most expensive step in the script on CPU (every point against ~16k landmarks over
    # all species), and only --max-rows of them are ever used.
    print(f"points: {points_dir}  {Xp.shape[0]:,} rows ({int(sup.sum()):,} supervised), "
          f"years {int(pidx[:, 2].min())}-{int(pidx[:, 2].max())}, latent_dim {latent_dim}")

    # The DESK split, reused rather than redrawn wherever a trained run left it on disk. An R^2 on
    # a different split is not comparable to the DESK numbers this exists to sit beside.
    H, W = cio.load_state_stack(int(desk_cfg.get("label_year", 2025)), states_dir, schema).shape[:2]
    dd = paths["desk_output_dir"]
    hp, bp = os.path.join(dd, "holdout_cells.npy"), os.path.join(dd, "buffer_cells.npy")
    if os.path.exists(hp) and os.path.exists(bp):
        holdout, buffer_cells = np.load(hp).astype(bool), np.load(bp).astype(bool)
        print(f"split: reusing the trained run's {int(holdout.sum())} held-out / "
              f"{int(buffer_cells.sum())} buffer cells from {dd}")
        if holdout.shape != (H, W):
            raise SystemExit(f"saved holdout is {holdout.shape} but states are {(H, W)}")
    else:
        from src.community_encoder.train_DESK.augment import blocked_holdout
        t = desk_cfg.get("trend", {})
        k = int(desk_cfg.get("spatial_conv", {}).get("kernel", 3)) \
            if desk_cfg.get("spatial_conv", {}).get("enabled", True) else 0
        buf = max(k // 2, int(t.get("buffer_floor") or 0))
        holdout, buffer_cells = blocked_holdout(
            supervised_cells(pidx, sup, (H, W)), block_cells=int(t.get("block_cells", 6)),
            holdout_frac=float(t.get("holdout_frac", 0.15)), buffer_cells=buf,
            seed=int(t.get("seed", 0)))
        print(f"split: redrawn from config (block={int(t.get('block_cells', 6))}, buffer={buf}, "
              f"seed={int(t.get('seed', 0))}) -> {int(holdout.sum())} val cells")

    # mu/sd from the trained run when present, so the features are standardized exactly as DESK's
    # inputs were. Refitting here on the same training cells is equivalent in expectation but not
    # identical, and an unexplained difference between two diagnostics is worse than a dependency.
    mp = os.path.join(dd, "desk_meta.npz")
    if os.path.exists(mp):
        dm = np.load(mp)
        cio.assert_schema_compatible(
            json.loads(str(dm["state_schema"])) if "state_schema" in dm else schema, schema,
            "desk_meta vs states")
        mu, sd = dm["mu"], dm["sd"]
        print(f"norm: mu/sd from {mp}")
    else:
        ly = int(desk_cfg.get("label_year", 2025))
        cov0 = cio.load_state_stack(ly, states_dir, schema)
        fit = np.isfinite(cov0).all(-1) & supervised_cells(pidx, sup, (H, W)) \
            & (~holdout) & (~buffer_cells)
        mu, sd = cio.fit_norm(cov0[fit].astype("float32"), schema)
        print(f"norm: fitted on {int(fit.sum())} training cells at {ly} (no desk_meta.npz)")

    # SUBSAMPLE FIRST, then project and build features. Both of those are per-point costs and
    # only --max-rows rows are ever read, so doing either first is pure waste -- the same mistake
    # commit d0bdd34 fixed in the ESK rank curve. Assembling 7C float32 for every point is a 2 GB
    # matrix plus a 2 GB standardized copy at the BBS+eBird target (~12,400 cells x ~60 years),
    # and projecting every point into the basis is ~744k rows against ~16k landmarks over all
    # species, which is the single heaviest step in the script on CPU.
    #
    # The split, the buffer and the supervise flag all live in point space and need neither, so
    # they gate the sample first. The two filters that CANNOT be applied first are the non-finite
    # covariate drop and the non-finite z drop, which is why the final counts land slightly under
    # the cap rather than exactly on it.
    in_val = holdout[pidx[:, 0], pidx[:, 1]]
    in_buf = buffer_cells[pidx[:, 0], pidx[:, 1]]
    base = sup & (~in_buf)
    tr0 = np.where(base & (~in_val))[0]
    va0 = np.where(base & in_val)[0]
    print(f"\ncandidate rows: {len(tr0):,} train, {len(va0):,} val   "
          f"({int(in_buf.sum()):,} rows dropped for sitting in a buffer cell, "
          f"{int((~sup).sum()):,} unsupervised)")
    cap = max(1000, args.max_rows // 2)
    if len(tr0) > cap:
        tr0 = tr0[rng.permutation(len(tr0))[:cap]]
        print(f"  train subsampled to {len(tr0):,} (--max-rows)")
    if len(va0) > cap:
        va0 = va0[rng.permutation(len(va0))[:cap]]
        print(f"  val subsampled to {len(va0):,} (--max-rows)")
    sel = np.concatenate([tr0, va0])

    print(f"projecting {len(sel):,} communities into the pinned ESK basis", flush=True)
    z_sel = project_points_to_z(Xp[sel], z_dir, latent_dim)
    if z_sel is None:
        raise SystemExit(f"needs the saved ESK projection in {z_dir}")

    print(f"assembling {len(BLOCKS)} x {int(schema['streams'][-1]['end'])} features "
          f"for {len(sel):,} points", flush=True)
    F, ok_cov, cols = build_features(pidx[sel], states_dir, schema, mu, sd)
    # Row indices are now positions within `sel`, and everything downstream -- Y, the community
    # vectors for the rank curve, and the per-model prediction arrays -- is indexed the same way.
    ok = ok_cov & np.isfinite(z_sel).all(1)
    Xs = Xp[sel]
    Y = np.asarray(z_sel, dtype="float64")
    tr = np.where(ok & (np.arange(len(sel)) < len(tr0)))[0]
    va = np.where(ok & (np.arange(len(sel)) >= len(tr0)))[0]
    print(f"rows: {len(tr):,} train, {len(va):,} val ({int((~ok_cov).sum()):,} dropped for "
          f"non-finite covariates, {int((ok_cov & ~ok).sum()):,} for non-finite z)")
    if len(va) < 500:
        raise SystemExit(f"only {len(va)} held-out rows; every number below would be noise")

    var_share = Y[tr].var(0)
    var_share = var_share / var_share.sum()

    print("\n=== 1. CONTEXT LADDER (linear ridge) ===")
    # preds are kept over EVERY row of `sel`, not just `va`. baseline_panel differences z_desk
    # against z_obs across all rows before masking, and epoch_direction_panel reads arbitrary
    # (cell, year) pairs -- including training cells, which is how its IDW bar gets its sources.
    # Scoring is still on `va` alone; see fit_and_score's predict_rows note.
    all_rows = np.arange(len(sel))
    rungs, preds = {}, {}
    for name, blocks in RUNGS.items():
        sub = np.concatenate([np.arange(cols[b].start, cols[b].stop) for b in blocks])
        res, p = fit_and_score(F[:, sub], Y, tr, va, rng=np.random.default_rng(1),
                               predict_rows=all_rows)
        rungs[name], preds[name] = res, p
        b = band_means(res["r2"], res["beats_const"])
        print(f"  {name:<10} {len(sub):>4} features   "
              + "  ".join(f"{k}={v:.3f}" for k, v in b.items())
              + f"   | mean {np.mean(np.where(res['beats_const'], res['r2'], 0.0)):.3f}")

    print("\n=== 2. CAPACITY LADDER (closed-form rungs on the widest context rung) ===")
    caps = {}
    sub = np.concatenate([np.arange(cols[b].start, cols[b].stop) for b in BLOCKS])
    Fw = np.ascontiguousarray(F[:, sub])
    # Standardized once, on TRAIN rows only: the hinge knots are quantiles and the RBF length
    # scale is a distance, so both are meaningless on raw channels that differ by 10^6.
    Fw = ((Fw - Fw[tr].mean(0)) / np.maximum(Fw[tr].std(0), 1e-6)).astype("float32")
    for name, fmaps, width in capacity_ladder(Fw, tr, n_pairs=args.pairs, pca_dim=args.pca_dim,
                                              rff_width=args.rff_width,
                                              rng=np.random.default_rng(2)):
        res, p = fit_and_score(Fw, Y, tr, va, fmaps=fmaps, rng=np.random.default_rng(4),
                               predict_rows=all_rows)
        caps[name], preds[name] = res, p
        b = band_means(res["r2"], res["beats_const"])
        print(f"  {name:<22} {width:>6} features   "
              + "  ".join(f"{k}={v:.3f}" for k, v in b.items())
              + f"   | mean {np.mean(np.where(res['beats_const'], res['r2'], 0.0)):.3f}",
              flush=True)

    print("\n=== 3. PER-COMPONENT R^2 (best model) and TARGET VARIANCE SHARE ===")
    best_name, _ = max(list(rungs.items()) + list(caps.items()),
                       key=lambda kv: np.mean(np.where(kv[1]["beats_const"], kv[1]["r2"], 0.0)))
    br2 = np.where(rungs.get(best_name, caps.get(best_name))["beats_const"],
                   rungs.get(best_name, caps.get(best_name))["r2"], 0.0)
    for a in range(0, len(br2), 8):
        print(f"  l{a + 1:>3}-{min(a + 8, len(br2)):<3} R^2 "
              + " ".join(f"{v:6.3f}" for v in br2[a:a + 8]))
        print("          var% " + " ".join(f"{100 * v:6.2f}" for v in var_share[a:a + 8]))

    print("\n=== 4. RANK CURVE: what the covariates support, against the ESK oracle ===")
    # Two DISJOINT held-out samples: one to fit the two calibration scalars, one to score on.
    # Both come from `va` because the models only ever predicted there, and a and b are two
    # nuisance parameters fitted on different pairs than they are evaluated on -- so this
    # removes the shrinkage bias without handing any model a fit on its own score. Every row
    # including the oracle gets the identical treatment, which is what keeps them comparable.
    cr = np.random.default_rng(5)
    pool = cr.permutation(len(va))
    n_c = min(args.curve_points, len(pool) // 2)
    ev, ca = pool[:n_c], pool[n_c:2 * n_c]
    if n_c < 50:
        print(f"  only {len(va)} held-out rows: too few for a two-sample curve, SKIPPED")
        curves = {}
    else:
        curves = {"esk_oracle": rank_curve(Xs[va[ev]], Y[va[ev]], Xs[va[ca]], Y[va[ca]])}
        for name in list(rungs) + list(caps):
            curves[name] = rank_curve(Xs[va[ev]], preds[name][va[ev]],
                                      Xs[va[ca]], preds[name][va[ca]])
        print(f"  {n_c} scoring points ({n_c * (n_c - 1) // 2:,} pairs), {n_c} disjoint "
              f"calibration points. corr is the shrinkage-proof column -- read it first.")
        for name, c in curves.items():
            bc, gc = best_rank(c, "corr")
            bm, gm = best_rank(c, "mse")
            print(f"    {name:<20} " + " ".join(f"r{k}:{c[k]['corr']:+.3f}" for k in sorted(c))
                  + f"   corr {'FLAT' if bc is None else f'bestR={bc}'} (+{gc:.3f})"
                  + f"  mse {'FLAT' if bm is None else f'bestR={bm}'} ({100 * gm:+.0f}%)"
                  + f"  scale@{min(32, max(c))}={c[min(32, max(c))]['scale']:.2f}")
        # The oracle's UNCALIBRATED curve, because that is the one basis_domain_gap prints and the
        # one the 11x r8->r64 claim rests on. If it improves sharply here while its calibrated and
        # corr columns are flat, the rank gain is largely MAGNITUDE -- truncation shrinks ||z||^2,
        # and a per-rank affine fit absorbs exactly that. Then this section says nothing about the
        # tail and basis_domain_gap's raw curve is the number to use.
        o = curves["esk_oracle"]
        print("    esk_oracle mse_raw  "
              + " ".join(f"r{k}={o[k]['mse_raw']:.5f}" for k in sorted(o)))
        bo_raw, go_raw = best_rank(o, "mse_raw")
        bo_cal = best_rank(o, "mse")[0]
        if bo_raw is not None and bo_cal is None:
            print(f"    NOTE the oracle's RAW curve improves {100 * go_raw:.0f}% to r{bo_raw} while "
                  f"its calibrated curve is flat.\n    That gain is scale, not tail structure. "
                  f"Read basis_domain_gap's raw curve for the tail claim.")
        ob = best_rank(curves["esk_oracle"], "corr")[0]
        rb = best_rank(curves[best_name], "corr")[0]
        print(f"\n  oracle bestR(corr)={ob or 'FLAT'}, {best_name} bestR(corr)={rb or 'FLAT'}. "
              f"A regressor whose corr "
              f"stops\n  improving well before the oracle's is the COVARIATE ceiling on rank; "
              f"DESK's swept bestR\n  below THIS one is encoder headroom. Compare shapes against "
              f"the sweep's k@24, never these\n  absolute values. A scale far above 1 is ridge "
              f"shrinkage, not a basis property.")

    # ---------------------------------------------------------------------------------------
    # Everything below grades the WINNING rung on the axes the validation suite already
    # separates, by calling that suite's functions unchanged. A single pooled R^2 conflates a
    # covariate signal with spatial smoothness, a level with a change, a well-covered era with a
    # poorly covered one, and a real gap with survey noise. Each of those implies a different
    # answer to "which covariate should we acquire", so the pooled number cannot decide it.
    # ---------------------------------------------------------------------------------------
    best_all = dict(rungs); best_all.update(caps)
    best_name = max(best_all, key=lambda k: np.mean(
        np.where(best_all[k]["beats_const"], best_all[k]["r2"], 0.0)))
    P = preds[best_name]                                  # (len(sel), L), all rows
    pidx_sel = pidx[sel]
    r2_best = np.asarray(best_all[best_name]["r2"], dtype="float64")
    extra = {"best_model": best_name}
    print(f"\n  (sections 5-9 grade the winning rung, {best_name})")

    print("\n=== 5. NOISE CEILING: how much of each component is even measurable ===")
    sn = per_dimension_signal_noise(pidx_sel, Y)
    ach, share = achievable_r2(r2_best, sn)
    M_MIN_SHARE = MIN_SIGNAL_SHARE
    extra["signal_noise"] = sn
    if ach is None:
        print(f"  unavailable: {sn.get('note')}")
    else:
        print(f"  {sn['n_adjacent_pairs']:,} adjacent-year pairs, {sn['n_gap_pairs']:,} at "
              f"{sn['gap_years']}+/-{sn['gap_tol']} yr. snr slope {sn['snr_slope']:+.4f}, "
              f"leading-8 median {sn['snr_leading_8']:.3f} vs trailing-8 "
              f"{sn['snr_trailing_8']:.3f}")
        extra["achievable_r2"] = ach.tolist()
        extra["signal_share"] = share.tolist()
        _nref = int(np.sum(~np.isfinite(ach)))
        if _nref:
            print(f"  {_nref} of {len(ach)} components carry under "
                  f"{100 * M_MIN_SHARE:.0f}% real signal, so an achievable-R^2 is REFUSED for "
                  f"them\n  (n/a below) -- at that share the rescale amplifies noise without "
                  f"bound and the share IS the finding")
        for a, b in BANDS:
            b = min(b, len(ach))
            if a >= b:
                continue
            _a = np.nanmean(ach[a:b]) if np.isfinite(ach[a:b]).any() else np.nan
            _as = "    n/a" if not np.isfinite(_a) else f"{_a:7.3f}"
            print(f"  l{a + 1:>3}-{b:<3} signal share {np.nanmean(share[a:b]):.3f}   "
                  f"R^2 vs total {np.nanmean(r2_best[a:b]):7.3f}   "
                  f"R^2 vs ACHIEVABLE {_as}"
                  f"   ({int(np.isfinite(ach[a:b]).sum())}/{b - a} defined)")
        # The reading that would overturn section 3: if the trailing components are mostly noise,
        # their low raw R^2 was never a covariate gap and no covariate can close it.
        if np.nanmean(share[32:]) < 0.5 and np.nanmean(share[:8]) > 0.7:
            print("  -> the TAIL IS MOSTLY NOISE. Components 33-64 carry more survey noise than\n"
                  "     temporal signal, so their low R^2 is not a covariate gap and no covariate\n"
                  "     can close it. The ESK oracle can 'predict' that noise only because it is\n"
                  "     projected FROM the same communities; a covariate model never can.")
    atten = per_era_attenuation(pidx_sel, Y)
    extra["per_era_attenuation"] = atten
    if atten:
        print("  per-era noise share of the gap difference, and the dir-cos attenuation it causes:")
        for era in sorted(atten):
            v = atten[era]
            print(f"    {era:<7} noise_share={v['noise_share_of_long_gap']:.3f}  "
                  f"dir_cos_attenuation={v['dir_cos_attenuation']:.3f}  "
                  f"n_adj={v['n_adjacent_pairs']:,}")

    print("\n=== 6. AGAINST THE REAL BARS, not the component mean ===")
    # An R^2 against a component's own mean is the weakest null available. validate_baselines
    # exists because the direction diagnostic beat its permutation null 0.48 to 0.22 and still
    # LOST to inverse-distance interpolation at 0.51 -- "a null that a plain interpolator clears
    # by a wide margin is not a bar". These are the bars.
    #
    # ONE call each, on the full 64-vector, because interpolation is linear and acts on each
    # component independently: the (n, 64) result gives every component's bar at once. Calling a
    # panel per component would re-sweep the 9 SPACETIME_RATIOS and rebuild the per-year KD-trees
    # 64 times for arithmetic that is already columnwise.
    va_mask = np.zeros(len(sel), bool); va_mask[va] = True
    ho_years = [int(y) for y in (desk_cfg.get("trend", {}).get("holdout_years") or [])]
    # verbose=True on purpose: this prints a WARNING when the fitted anisotropy lands on an
    # endpoint of SPACETIME_RATIOS, which means censored rather than measured. Swallowing that
    # would hide a bar that was never actually fitted.
    _st_err, st_ratio = spacetime_idw_baseline(pidx_sel, Y, holdout, va_mask,
                                              exclude_years=ho_years, verbose=True)
    Z_st = spacetime_idw_z(pidx_sel, Y, holdout, st_ratio, exclude_years=ho_years)
    _sp_err, Z_sp = zspace_idw_baseline(pidx_sel, Y, holdout, va_mask, return_z=True)
    r2_st, _n_st = per_component_r2_of(Z_st[va], Y[va])
    r2_sp, n_sp = per_component_r2_of(Z_sp, Y[va])              # zi is already va-aligned
    extra["spacetime_ratio_cells_per_year"] = st_ratio
    extra["r2_spacetime_idw"] = r2_st.tolist()
    extra["r2_spatial_idw"] = r2_sp.tolist()
    bar = np.fmax(np.nan_to_num(r2_st, nan=-np.inf), np.nan_to_num(r2_sp, nan=-np.inf))
    bar = np.where(np.isfinite(bar), bar, np.nan)
    gain = r2_best - bar
    extra["r2_gain_over_best_bar"] = gain.tolist()
    print(f"  space-time IDW anisotropy {st_ratio:g} cells/yr; spatial-IDW rows scored per "
          f"component: median {int(np.median(n_sp)):,}")
    print(f"  {'band':<9} {'covariates':>10} {'spatial IDW':>12} {'st IDW':>8} {'GAIN':>8}")
    for a, b in BANDS:
        b = min(b, len(r2_best))
        if a >= b:
            continue
        print(f"  l{a + 1:>3}-{b:<4} {np.nanmean(r2_best[a:b]):10.3f} "
              f"{np.nanmean(r2_sp[a:b]):12.3f} {np.nanmean(r2_st[a:b]):8.3f} "
              f"{np.nanmean(gain[a:b]):+8.3f}")
    _mid = np.nanmean(gain[8:min(32, len(gain))])
    if _mid <= 0.02:
        print(f"  -> components 9-32 gain only {_mid:+.3f} over a plain interpolator. Their "
              f"predictability is\n     SPATIAL SMOOTHNESS, not environmental signal, and new "
              f"covariates of any kind inherit\n     that ceiling. This overturns a raw-R^2 "
              f"reading of section 3.")
    else:
        print(f"  -> components 9-32 beat the best interpolator by {_mid:+.3f}, so their "
              f"predictability is\n     genuinely environmental rather than geometric.")
    # The full 6-rung ladder, pooled, as the tie to the existing validation report. Two rungs are
    # structurally n/a here: under a SPATIAL holdout a held-out cell is held out in every year, so
    # it has no training years of its own and cell_nearest_year/cell_trend cannot run. That is
    # documented in baseline_panel's docstring, not a defect.
    recent_year = int(json.load(open(os.path.join(points_dir, "points_meta.json"),
                                    encoding="utf-8"))["recent_year"])
    extra["baseline_panel"] = baseline_panel(pidx_sel, Y, P, holdout, recent_year,
                                            buffer_mask=buffer_cells, heldout_only=True,
                                            verbose=True, exclude_years=ho_years)

    print("\n=== 7. LEVEL vs CHANGE: the quantity DESK actually has to predict ===")
    # Sections 1-6 all grade a LEVEL. Anything static explains level and not change, and this
    # project's own finding is that direction is where the model fails (dcos 0.21 against a 0.19
    # no-covariate baseline, over-moving 2.2x). Same gap as section 5's signal/noise, so those
    # numbers describe this exact quantity.
    ea, la = gap_pairs(pidx_sel, ATTEN_GAP, ATTEN_GAP_TOL)
    extra["change"] = {"n_pairs": int(len(ea)), "gap_years": ATTEN_GAP}
    if len(ea) < 200:
        print(f"  only {len(ea)} within-cell pairs at {ATTEN_GAP}+/-{ATTEN_GAP_TOL} yr; SKIPPED")
    else:
        d_tr = np.isin(ea, tr) & np.isin(la, tr)
        d_va = np.isin(ea, va) & np.isin(la, va)
        print(f"  {len(ea):,} within-cell pairs at {ATTEN_GAP}+/-{ATTEN_GAP_TOL} yr "
              f"({int(d_tr.sum()):,} train, {int(d_va.sum()):,} val)")
        if int(d_tr.sum()) < 100 or int(d_va.sum()) < 100:
            print("  too few on one side of the split; SKIPPED")
        else:
            dF = np.ascontiguousarray(Fw[la] - Fw[ea])
            dY = Y[la] - Y[ea]
            res_d, _ = fit_and_score(dF, dY, np.where(d_tr)[0], np.where(d_va)[0],
                                     rng=np.random.default_rng(21))
            r2_ch = np.asarray(res_d["r2"], dtype="float64")
            extra["change"]["r2"] = r2_ch.tolist()
            # Compared against the LINEAR level rung, not the best (nonlinear) one. The change fit
            # is linear, so grading it against a capacity-laddered level number would measure the
            # model class as well as the quantity, and the level/change gap is the only thing this
            # section is for. `linear` is the same feature set (widest context), same solver.
            r2_lvl = np.asarray(caps["linear"]["r2"], dtype="float64")
            extra["change"]["r2_level_linear"] = r2_lvl.tolist()
            print("  both fits are LINEAR on the same features, so the gap is the quantity")
            print(f"  {'band':<9} {'R^2 level':>10} {'R^2 CHANGE':>11}")
            for a, b in BANDS:
                b = min(b, len(r2_ch))
                if a >= b:
                    continue
                print(f"  l{a + 1:>3}-{b:<4} {np.nanmean(r2_lvl[a:b]):10.3f} "
                      f"{np.nanmean(r2_ch[a:b]):11.3f}")
            _cl = np.nanmean(r2_ch[:8]); _ll = np.nanmean(r2_lvl[:8])
            if _cl < 0.25 * max(_ll, 1e-9):
                print(f"  -> the covariates explain LEVEL ({_ll:.3f}) far better than CHANGE "
                      f"({_cl:.3f}).\n     Every R^2 in sections 1-6 is answering the easier "
                      f"question. A covariate that varies\n     mostly in space cannot help "
                      f"here however high its level R^2.")

    print("\n=== 8. DIRECTION and MAGNITUDE (bands; an angle needs >=2 dimensions) ===")
    print(f"  {'band':<9} {'dims':>5} {'ang share':>10} {'mag share':>10} {'median cos':>11} "
          f"{'|pred|/|true|':>13}")
    bd = banded_direction(P[va], Y[va])
    extra["banded_error_split"] = bd
    for k, v in bd.items():
        print(f"  l{k:<8} {v['n_dims']:>5} {v['ang_share']:10.3f} {v['mag_share']:10.3f} "
              f"{v['median_cos']:11.3f} {v['norm_ratio']:13.3f}")
    print("  (only the ANGULAR share is a covariate problem; the magnitude share is calibration,\n"
          "   and ridge shrinkage is the MSE-optimal response to a poor angle)")
    # The epoch panel, unchanged, on the full vector. It wants z_model as a DICT keyed by
    # (row, col, year) rather than an array -- the one shape adaptation in this whole section.
    zmodel = {(int(r), int(c), int(y)): P[i]
              for i, (r, c, y) in enumerate(pidx_sel)}
    extra["epoch_direction_panel"] = epoch_direction_panel(
        pidx_sel, None, Y, zmodel, holdout, buffer_cells, epochs=DEFAULT_EPOCHS,
        exclude_years=ho_years, z_spacetime=Z_st, verbose=True)

    print("\n=== 9. WHERE the predictability lives: era, coverage, geography ===")
    yv = pidx_sel[va, 2]
    groups = {f"era {e}": (_era_of(yv) == e) for e in sorted(set(_era_of(yv)))}
    avail = bui_avail_grid(states_dir, schema, int(desk_cfg.get("label_year", 2025)))
    if avail is not None:
        av = avail[pidx_sel[va, 0], pidx_sel[va, 1]]
        groups["BUI covered"] = av > 0.5
        groups["BUI absent"] = av <= 0.5
        print(f"  BUI coverage on val rows: {float(np.mean(av > 0.5)):.1%} covered "
              f"(CONUS-only stream; outside it the encoder has HYDE alone)")
    quad = coarse_spatial(pidx_sel[va], regions=2)
    for q in sorted(set(int(v) for v in quad)):
        groups[f"quadrant {q}"] = quad == q
    by = r2_by_group(P[va], Y[va], groups)
    extra["r2_by_group"] = {k: {"r2": v[0].tolist(), "n": v[1].tolist()} for k, v in by.items()}
    print(f"  {'group':<14} {'n rows':>8} "
          + " ".join(f"{f'l{a + 1}-{min(b, latent_dim)}':>9}" for a, b in BANDS
                     if a < latent_dim))
    for k, (r2g, ng) in by.items():
        print(f"  {k:<14} {int(np.max(ng)):>8,} "
              + " ".join(f"{np.nanmean(r2g[a:min(b, len(r2g))]):9.3f}" for a, b in BANDS
                         if a < len(r2g)))
    _eras = {k: v for k, v in by.items() if k.startswith("era ")}
    if len(_eras) >= 2:
        _mids = {k: np.nanmean(v[0][8:32]) for k, v in _eras.items()}
        _lo, _hi = min(_mids, key=_mids.get), max(_mids, key=_mids.get)
        if _mids[_hi] - _mids[_lo] > 0.10:
            print(f"  -> ERA GAP: components 9-32 reach {_mids[_hi]:.3f} in {_hi} but only "
                  f"{_mids[_lo]:.3f} in {_lo}.\n     Covariate resolution in the weak era is the "
                  f"purchase, not new variables -- HYDE is decadal\n     before 1951 and BUI is "
                  f"5-yearly.")

    bn, bands = verdict(rungs, caps, var_share)
    json.dump({
        "points_dir": points_dir, "n_train": len(tr), "n_val": len(va),
        "latent_dim": latent_dim, "var_share": var_share.tolist(),
        "context_ladder": {k: {"r2": v["r2"].tolist(), "beats_const": v["beats_const"].tolist(),
                               "bands": band_means(v["r2"], v["beats_const"])}
                           for k, v in rungs.items()},
        "capacity_ladder": {k: {"r2": v["r2"].tolist(), "beats_const": v["beats_const"].tolist(),
                                "bands": band_means(v["r2"], v["beats_const"])}
                            for k, v in caps.items()},
        "rank_curves": curves, "best_model": bn, "best_bands": bands,
        "decompositions": extra,
    }, open(args.out, "w"), indent=2, default=_jsonable)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
