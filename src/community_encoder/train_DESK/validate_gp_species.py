"""GP species validation: does DESK, used as a GP kernel, predict single species in held-out blocks?

Companion to ``validate_bbs_routes``, which asks whether ``z(x).z(x')`` reproduces observed
COMMUNITY similarity. This asks the question the kernel is deployed for: put ``s^2 z.z'`` under a
GP, condition on training blocks, predict one species' abundance in held-out blocks. Design:
``docs/methods/gp_species_validation.md``.

RAW vs EMA. DESK is trained through a learned causal output EMA (demographic lag) and
supervised on z_ema, but the cube exports RAW z because the population model supplies lag
itself. A static GP has no dynamics, so neither is exactly the deployed object; both are graded.

WHICH SPECIES. Every BBS species that crosswalks to the eBird taxonomy, minus the reference
community and minus House Finch. The community is what DESK was trained to reproduce, so grading
on it would be grading on the training target; House Finch is the deployment species, so grading
on it would be double-dipping. No species is dropped for being rare: a species leaves a METRIC
only when that metric is undefined for it, and the report counts those per metric.

WHICH RUN. Point ``paths.desk_output_dir`` at a run trained WITH a holdout -- the production run
has ``holdout_frac=0`` and nothing to grade. ``config/overlays/gp_species_base.json`` points at
production's holdout predecessor. Results must never select that run's epoch or configuration.

HOLDOUTS, as in the existing suite. Spatial blocks always; and on a ``desk_tempho_*`` checkpoint
the withheld early decades too, excluded from EVERY fit in every cell. Scored in three groups --
space (held-out blocks, trained years), time (training cells, withheld years), space_time (both)
-- because the spacetime GP's advantage depends on neighbours observed in the same years, which
only the space group has. Run one overlay per checkpoint, e.g.
``ESK_DESK_CONFIG=config/overlays/desk_tempho_1985.json``.

THE PREDICTORS (all share the GP machinery in ``gp_kernels``):

    desk            s^2 z_ema(x).z_ema(x')                        what DESK was trained on
    desk_raw        s^2 z_raw(x).z_raw(x')                        what the downstream model gets
    no_change       desk's fit, with each held-out cell's modern-epoch z used for every year
                    (no_change_raw: the same for desk_raw)
    spacetime       Matern(space) x exponential(time)            shared shape, per-species scale
    covariate       ARD-RBF on DESK's covariates + output EMA     matched to z_ema
                    (covariate_raw: input-side EMA only, matched to z_raw)
    intercept       the training mean; the floor for LEVEL (it knows nothing about place)
    esk_oracle_independent
                    the observed community from a DISJOINT half of the cell-epoch's years,
                    projected into the ESK basis: the ceiling (see independent_oracle_z)

``no_change`` predicts the SAME value for a cell in every year, so its predicted change is exactly
zero and the change-skill denominator is just the observed change. That is the honest null for
change; it abstains on direction, and the report says so rather than scoring it as wrong.

PRIMARY METRIC, fixed before any result: per-species RMSE skill on held-out same-cell change
(early epoch -> modern epoch), ``1 - rmse(desk) / rmse(no_change)``, pooled as the median across
species and the share of species above zero, with a bootstrap over held-out blocks AND species.
"""
import argparse
import json
import os
import time

import numpy as np

from src.config_utils import load_config

from . import gp_kernels as gpk
from .validation_core import GROUPS, load_holdout_masks, row_splits
from .validate_bbs_routes import (EARLY_WINDOW, EPOCH_EARLY, EPOCH_MODERN, MIN_EPOCH_YEARS,
                                  MODERN_WINDOW, epoch_gate)

HOUSE_FINCH_CODE = "houfin"


# ----------------------------- species layout -----------------------------

def species_layout(community_codes, matched_codes, focal_codes=(HOUSE_FINCH_CODE,)):
    """Column layout: community block then evaluation block. ``(community, evaluation)``. Pure.

    The community block is ``community_codes`` IN ITS GIVEN ORDER -- it must be
    ``species_order(community_csv)``, the layout the ESK basis was fitted with, because the oracle
    projects it. A community species BBS cannot survey keeps its (all-zero) column: compacting it
    out would shift every later column, which is the 94-of-96 misalignment this pipeline has had.

    The evaluation block is every matched code outside the community and the focal set, sorted, so
    it is deterministic across runs. Focal species are removed from BOTH blocks; a focal species
    in the community list is an error upstream and raises here.
    """
    comm = [str(c).lower() for c in community_codes]
    focal = {str(f).lower() for f in focal_codes}
    leaked = focal & set(comm)
    if leaked:
        raise ValueError(f"focal species {sorted(leaked)} is in the community list; DESK would "
                         "have been trained on the deployment species")
    if len(set(comm)) != len(comm):
        raise ValueError("community list has duplicate species codes")
    cset = set(comm)
    ev = sorted({str(c).lower() for c in matched_codes} - cset - focal)
    return comm, ev


def crosswalk_all_species(bbs_species_path, ebird_taxonomy_path):
    """Every BBS AOU that maps to exactly one eBird species code. ``(df[aou, species_code], diag)``.

    Joins on normalized scientific name, as ``bbs_crosswalk`` does, but over the WHOLE taxonomy
    instead of a community -- ``build_crosswalk`` would print every one of ~11k unmatched eBird
    codes. Unidentified, hybrid, slash and form rows drop out at the join because the taxonomy
    holds species-category rows only. An AOU resolving to more than one code is ambiguous and is
    dropped and reported, not guessed. Several AOUs on one code (a lump) are kept and summed
    downstream, matching the community path.
    """
    from src.data.identify.bbs_crosswalk import (_read_species_table, load_ebird_taxonomy,
                                                 normalize_bbs_species)
    bbs_norm = normalize_bbs_species(_read_species_table(bbs_species_path))
    tax = load_ebird_taxonomy(ebird_taxonomy_path)
    m = bbs_norm.merge(tax, on="sci_norm", how="inner")[["aou", "species_code"]].drop_duplicates()
    per_aou = m.groupby("aou")["species_code"].nunique()
    split = sorted(int(a) for a in per_aou[per_aou > 1].index)
    m = m[~m["aou"].isin(split)].reset_index(drop=True)
    diag = {"n_bbs_aou": int(bbs_norm["aou"].nunique()), "n_aou_matched": int(m["aou"].nunique()),
            "n_codes_matched": int(m["species_code"].nunique()), "split_aous_dropped": split}
    return m, diag


def load_all_species(config):
    """Raw BBS, all species, same aggregation as the community target. ``(X_raw, keys, layout)``.

    The aggregation is ``validate_bbs_routes.observed_counts`` -- the SAME function the community
    target uses -- with only the species list changed.
    """
    import pandas as pd

    from src.config_utils import load_data_config, target_points_dir
    from src.data.preprocess import bbs

    from .bbs_community_points import species_order
    from .validate_bbs_routes import assert_same_layout

    dcfg = load_data_config()
    community_csv = (config.get("trend", {}) or {}).get("community_trend_list") \
        or dcfg["community_trend_list"]
    focal = str(dcfg.get("focal_species_code") or HOUSE_FINCH_CODE).lower()
    bbs_species = config.get("bbs", {}).get("species_list") or \
        os.path.join(bbs.BBS_PARENT_DIR, "SpeciesList.csv")
    xw, xdiag = crosswalk_all_species(
        bbs_species, os.path.join(dcfg["datasets_root"], "avonet", "eBird_taxonomy.csv"))
    # House Finch by AOU as well as by code: a lumped or renamed AOU must not carry it in.
    xw = xw[(xw["aou"] != bbs.HOUSE_FINCH_AOU) & (xw["species_code"].str.lower() != focal)]

    comm, ev = species_layout(species_order(community_csv), xw["species_code"],
                              focal_codes=(focal, HOUSE_FINCH_CODE))
    # The oracle projects the community block through the basis, so its layout must be the one
    # the basis was fitted with -- checked position by position, as load_observed does.
    pm_path = os.path.join(target_points_dir(config) or "", "points_meta.json")
    if not os.path.exists(pm_path):
        raise FileNotFoundError(f"no points_meta.json at {pm_path}; cannot verify the community "
                                "layout the ESK oracle projects")
    with open(pm_path, encoding="utf-8") as fh:
        trained = [str(s) for s in (json.load(fh).get("species") or [])]
    if trained:
        assert_same_layout(trained, comm, pm_path)

    from .validate_bbs_routes import observed_counts
    X_raw, keys, dropped = observed_counts(comm + ev, xw)
    layout = {"community": comm, "evaluation": ev, "n_community": len(comm),
              "n_evaluation": len(ev), "focal_excluded": focal, "crosswalk": xdiag,
              "presence_triples_outside_coverage": int(dropped), "community_csv": community_csv}
    print(f"[gp-species] {len(ev)} evaluation species (+{len(comm)} community columns for the "
          f"oracle) over {len(keys):,} surveyed cell-years; {focal} excluded")
    return X_raw, keys, layout


# ----------------------------- splits -----------------------------

def change_sets(tkeys, row_group, withheld):
    """Which held-out cells give a same-cell change, and from which rows. ``{name: rows}``. Pure.

    ``row_group`` labels the predicted rows (0 = an in-sample modern row of a training cell). A
    change needs an early and a modern epoch from the same cell:

    * held-out block cells: modern rows are ``space`` rows; early rows are ``space_time`` rows when
      years were withheld (the set is then ``space_time``), else ``space`` rows (set ``space``).
    * training cells with withheld early years (set ``time``): early rows are ``time`` rows, modern
      rows are that cell's own TRAINING rows, predicted in-sample. That is the honest question for
      temporal extrapolation: given a cell's observed present, how well is its unobserved past
      predicted.

    Returns indices into the predicted rows; ``epoch_gate`` is applied per set by the caller.
    """
    yr = np.asarray(tkeys)[:, 2]
    early = (yr >= EPOCH_EARLY[0]) & (yr <= EPOCH_EARLY[1])
    modern = (yr >= EPOCH_MODERN[0]) & (yr <= EPOCH_MODERN[1])
    g = np.asarray(row_group)
    out = {}
    if len(withheld):
        out["space_time"] = np.where(((g == 1) & modern) | ((g == 3) & early))[0]
        out["time"] = np.where(((g == 0) & modern) | ((g == 2) & early))[0]
    else:
        out["space"] = np.where((g == 1) & (modern | early))[0]
    return out


def modern_reference_keys(keys, modern=EPOCH_MODERN):
    """Every ``(cell, year)`` in the modern epoch for each cell in ``keys``. ``(M, 3)``. Pure.

    DESK has a z for any cell-year, surveyed or not, so the no-change reference is the cell's
    mean z over the whole modern epoch rather than over whichever years happened to be surveyed.
    """
    cells = np.unique(np.asarray(keys)[:, :2], axis=0)
    years = np.arange(int(modern[0]), int(modern[1]) + 1)
    out = np.empty((len(cells) * len(years), 3), "int32")
    out[:, :2] = np.repeat(cells, len(years), axis=0)
    out[:, 2] = np.tile(years, len(cells))
    return out


def no_change_z(keys, ref_keys, Z_ref):
    """Each row's cell-mean z over the modern epoch, NaN-aware. ``(N, L)``. Pure."""
    acc = {}
    for k, z in zip(ref_keys, Z_ref):
        if np.all(np.isfinite(z)):
            acc.setdefault((int(k[0]), int(k[1])), []).append(z)
    mean = {c: np.mean(v, 0) for c, v in acc.items()}
    L = Z_ref.shape[1]
    out = np.full((len(keys), L), np.nan, "float32")
    for i, k in enumerate(keys):
        v = mean.get((int(k[0]), int(k[1])))
        if v is not None:
            out[i] = v
    return out


# ----------------------------- metrics -----------------------------

def gaussian_lpd(y, mean, var):
    """Log predictive density of each observation. Pure."""
    return -0.5 * (np.log(2 * np.pi * var) + (y - mean) ** 2 / var)


def gaussian_crps(y, mean, var):
    """Closed-form CRPS of a Gaussian predictive (lower is better). Pure."""
    from scipy.stats import norm
    sd = np.sqrt(var)
    z = (y - mean) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def interval_coverage(y, mean, var, level):
    """Share of observations inside the central ``level`` predictive interval. Pure."""
    from scipy.stats import norm
    h = norm.ppf(0.5 + level / 2) * np.sqrt(var)
    return (np.abs(y - mean) <= h).astype("float64")


def block_sums(values, block_id):
    """Sum each column of ``values (n, S)`` within blocks. ``(blocks, (n_blocks, S))``. Pure."""
    blocks, inv = np.unique(block_id, return_inverse=True)
    out = np.zeros((len(blocks), values.shape[1]))
    np.add.at(out, inv, values)
    return blocks, out


def skill_from_sse(sse_model, sse_base):
    """``1 - sqrt(SSE_model / SSE_base)`` per species; NaN where the base has zero error. Pure.

    Zero base error means there was nothing to predict -- e.g. a species absent from every
    held-out row, where no_change and the truth are both zero. That is an undefined metric, not
    a perfect or a failed one.
    """
    sm, sb = np.asarray(sse_model, "float64"), np.asarray(sse_base, "float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sb > 0, 1.0 - np.sqrt(sm / np.where(sb > 0, sb, 1.0)), np.nan)


def pooled_skill(block_sse_model, block_sse_base, n_boot=1000, seed=0, alpha=0.05,
                 block_signal=None):
    """Median species skill and share of species above zero, with a two-way bootstrap. Pure.

    Inputs are ``(n_blocks, n_species)`` SSE tables. Each draw resamples held-out BLOCKS (the unit
    the split drew, so spatial correlation inside a block stays inside the draw) and, separately,
    SPECIES (the population the claim is about). Skill is recomputed from the resampled block sums,
    not averaged from per-block skills, so a block with little data weighs little.

    ``block_signal`` (``(n_blocks, n_species)``, units where the truth carries signal) applies the
    undefined-metric rule INSIDE every draw: a draw that misses all of a species' signal-bearing
    blocks leaves it with a near-zero reference error, and its skill there is noise of any size.
    The rule used to be applied once, on the full sample, and the draws reintroduced exactly the
    denominators it had removed.
    """
    sm, sb = np.asarray(block_sse_model), np.asarray(block_sse_base)
    sk = skill_from_sse(sm.sum(0), sb.sum(0))
    defined = np.isfinite(sk)
    out = {"n_species_defined": int(defined.sum()),
           "n_species_undefined": int((~defined).sum())}
    if defined.sum() < 3:
        out["note"] = "fewer than 3 species with a defined skill; nothing to pool"
        return out, sk
    out["median"] = float(np.median(sk[defined]))
    out["share_above_zero"] = float((sk[defined] > 0).mean())
    rng = np.random.default_rng(seed)
    nb, ns = sm.shape
    sig = None if block_signal is None else np.asarray(block_signal, "float64")
    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot).astype("float64")
    med, share = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    for i in range(n_boot):
        s = skill_from_sse(W[i] @ sm, W[i] @ sb)
        if sig is not None:
            s = np.where(W[i] @ sig > 0, s, np.nan)
        pick = rng.integers(0, ns, ns)
        v = s[pick]
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            med[i], share[i] = np.median(v), (v > 0).mean()
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out["median_ci"] = [float(np.nanpercentile(med, lo)), float(np.nanpercentile(med, hi))]
    out["share_above_zero_ci"] = [float(np.nanpercentile(share, lo)),
                                  float(np.nanpercentile(share, hi))]
    out["n_boot"] = int(n_boot)
    return out, sk


def predicted_raw(mean, var):
    """Expected count of an observation under the Gaussian-on-log1p model. Pure.

    ``E[x] = E[exp(y)] - 1`` with ``y ~ N(mean, var)``: ``exp(mean + var/2) - 1``. Callers pass
    the predictor's per-species OBSERVATION noise as ``var``, not its full predictive variance:
    the observed log1p includes that noise, so the truth's epoch mean carries its Jensen inflation
    and the prediction must too -- but the latent function's own uncertainty is not part of a
    point prediction. Including it (as the first runs did) made a local GP, sure of a cell's
    present and unsure of its past, predict a decline of ~(v_early - v_modern)/2 that it does not
    believe in, on every cell, from the back-transform alone.
    """
    return np.expm1(np.asarray(mean, "float64") + 0.5 * np.asarray(var, "float64"))


def direction_by_species(obs_e, obs_m, pred_e, pred_m, noise_sd=None):
    """``species_change_agreement`` per species, ACROSS CELLS. ``[dict]``. Pure.

    The existing function scores species within a row; given one species' column it scores cells
    within a species, with the same majority-direction correction and the same abstain-on-zero
    rule. ``noise_sd`` (one per species, the split-half noise sd of observed change) is passed as
    ``noise_floor_abs``: cells whose observed change is within survey noise are coin flips, and
    scoring them drags every predictor toward the null. Cells with no prediction are skipped.
    """
    from .validate_baselines import species_change_agreement
    # Snap predicted change below ``tol`` to exactly zero. A predictor that is constant over
    # years (no_change, intercept) gives epoch means that differ only by float rounding, and
    # species_change_agreement read those 1e-16 differences as committed directions -- scoring
    # an abstaining null as a confident wrong guess (-0.54 in the first run).
    tol = 1e-9
    d = np.asarray(pred_m, "float64") - np.asarray(pred_e, "float64")
    pred_m = np.where(np.abs(d) < tol, pred_e, pred_m)
    ok = np.isfinite(pred_e).all(1) & np.isfinite(pred_m).all(1)
    out = []
    for s in range(obs_e.shape[1]):
        nf = 0.0 if noise_sd is None or not np.isfinite(noise_sd[s]) else float(noise_sd[s])
        out.append(species_change_agreement(obs_e[ok, s:s + 1], obs_m[ok, s:s + 1],
                                            pred_e[ok, s:s + 1], pred_m[ok, s:s + 1],
                                            noise_floor_abs=nf))
    return out


def independent_oracle_z(keys, X_comm_raw, z_dir, latent, norm_tol=0.5, withheld=()):
    """The INDEPENDENT ESK ceiling: each row's community position from a DISJOINT half of its
    cell-epoch's years. ``(Z (N, L) with NaN where unavailable, info)``.

    The first version projected the community observed in the SAME cell-year as the species being
    predicted, so it shared that survey's route, observer and weather noise -- the mistake the
    route suite fixed by renaming its same-rows projection "truncation fidelity, not a ceiling".
    Here every cell-epoch's years are split ABBA (``validation_core.split_half_groups``); rows in
    half A get the projection of half B's mean community, and vice versa. Rows outside the two
    epochs, or in a cell-epoch too short to split, have no independent observation (NaN).

    REPRESENTABILITY GATE, as in the route suite (``bbs_routes.oracle_norm_tol``): a window-mean
    community can project off the basis's span, measured once at ||z||^2 0.15 against 0.672 for
    annual communities. If the half-window projections keep less than ``norm_tol`` of the annual
    squared norm, the oracle is refused -- a number computed off-span measures that mismatch, not
    a ceiling.

    WITHHELD YEARS form their own groups. A cell-epoch spanning trained and withheld years (the
    early epoch 1966-1986 against a 1966-1975 holdout) used to be split as one group, so a TRAINED
    row's oracle feature could average in withheld-decade communities -- the oracle was fitted on
    the decade it is then scored on. ``info['half_reliability']`` is the split-half correlation of
    the two halves' projections: the oracle's features are themselves a noisy half-window
    measurement, and a low reliability attenuates its skill (errors in variables), so a weak
    oracle is not by itself evidence that the community cannot predict the species.
    """
    from .esk_kernel import project_points_to_z
    from .validate_bbs_routes import epoch_mean_observed
    from .validation_core import split_half_groups

    keys = np.asarray(keys)
    yr = keys[:, 2]
    wh = set(int(y) for y in withheld)
    groups = {}
    for i, (r, c, y) in enumerate(keys):
        for name, (lo, hi) in (("early", EPOCH_EARLY), ("modern", EPOCH_MODERN)):
            if lo <= int(y) <= hi:
                groups.setdefault((int(r), int(c), name, int(y) in wh), []).append(i)
    glist = list(groups.values())
    A, B, ok = split_half_groups(glist, years=yr)
    A = [a for a, k in zip(A, ok) if k]
    B = [b for b, k in zip(B, ok) if k]
    if not A:
        return None, {"reason": "no cell-epoch has two years to split"}
    zA = project_points_to_z(epoch_mean_observed(X_comm_raw, A).astype("float32"), z_dir, latent)
    zB = project_points_to_z(epoch_mean_observed(X_comm_raw, B).astype("float32"), z_dir, latent)
    if zA is None:
        return None, {"reason": f"no ESK projection saved in {z_dir}"}
    rng = np.random.default_rng(0)
    sample = rng.choice(len(keys), min(20000, len(keys)), replace=False)
    from src.data.preprocess.bbs_community import log1p_community
    z_ann = project_points_to_z(log1p_community(X_comm_raw[sample]), z_dir, latent)
    n_ann = float(np.mean((z_ann ** 2).sum(1)))
    n_half = float(np.mean(np.concatenate([(zA ** 2).sum(1), (zB ** 2).sum(1)])))
    # Split-half reliability of the projected features: per dimension, the correlation of the two
    # halves across cell-epochs, averaged with variance weights.
    za, zb = np.asarray(zA, "float64"), np.asarray(zB, "float64")
    va = 0.5 * (za.var(0) + zb.var(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.array([np.corrcoef(za[:, d], zb[:, d])[0, 1] if va[d] > 0 else np.nan
                       for d in range(za.shape[1])])
    fin = np.isfinite(cr)
    rel = float(np.sum(cr[fin] * va[fin]) / max(va[fin].sum(), 1e-300)) if fin.any() else None
    info = {"half_window_norm2": n_half, "annual_norm2": n_ann,
            "norm_ratio": n_half / max(n_ann, 1e-12), "norm_tol": float(norm_tol),
            "n_cell_epochs_split": len(A), "n_cell_epochs_unsplittable": int((~ok).sum()),
            "half_reliability": rel, "withheld_years_grouped_separately": bool(wh)}
    if info["norm_ratio"] < norm_tol:
        info["reason"] = (f"refused its representability gate: half-window projections keep "
                          f"{info['norm_ratio']:.2f} of the annual ||z||^2 (< {norm_tol})")
        return None, info
    Z = np.full((len(keys), zA.shape[1]), np.nan, "float32")
    for a, b, za, zb in zip(A, B, zA, zB):
        Z[list(a)] = zb                       # rows in half A get the community of half B
        Z[list(b)] = za
    info["n_rows_with_oracle"] = int(np.isfinite(Z).all(1).sum())
    return Z, info


# ----------------------------- inputs for the baselines -----------------------------

def rows_from_stack(stack, years, cells, keys):
    """Gather ``stack (T, n_cells, C)`` at each ``(row, col, year)`` of ``keys``. Pure.

    Rows whose cell or year is not in the stack come back NaN rather than raising, so a key
    outside the encoded span is visible as missing instead of silently taking a neighbour.
    """
    cell_ix = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    year_ix = {int(y): t for t, y in enumerate(years)}
    out = np.full((len(keys), stack.shape[-1]), np.nan, "float32")
    for i, (r, c, y) in enumerate(np.asarray(keys)):
        ci, ti = cell_ix.get((int(r), int(c))), year_ix.get(int(y))
        if ci is not None and ti is not None:
            out[i] = stack[ti, ci]
    return out


def covariates_for_keys(config, keys, both=False):
    """DESK's own normalized covariates at each key, EMA'd at DESK's learned half-life. ``(N, C)``.

    With ``both=True`` returns ``(ema, raw)``: ``raw`` carries only the light input-side EMA
    (``ema_tau`` at state-build time), matching DESK's RAW z, which is what the downstream model
    consumes; ``ema`` adds DESK's learned output EMA, matching z_ema, which is what DESK was
    trained on.

    The covariate GP's inputs. Same states, same channel transforms, same training-pixel mu/sd as
    the checkpoint, and the same causal EMA DESK applies to its output -- so the covariate GP sees
    the temporal smoothing DESK sees rather than losing on change for want of it. What it does NOT
    get is DESK's 5x5 spatial convolution: it sees each cell's own covariates only. That
    asymmetry is recorded in the report rather than corrected.
    """
    from . import covariate_io as cio
    from .desk_training import apply_output_ema

    run_dir = config["paths"]["desk_output_dir"]
    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    schema = json.loads(str(dm["schema"]))
    states_dir = os.path.join(config["paths"]["hist_dir"], "yearly_states")
    cio.assert_schema_compatible(schema, cio.load_schema(states_dir), context="gp-species")
    mu, sd = dm["mu"].astype("float32"), dm["sd"].astype("float32")
    hl = float(dm["ema_half_life"])
    warm = int(dm["ema_warmup_start"]) if "ema_warmup_start" in dm.files else 1940
    keys = np.asarray(keys)
    cells = np.unique(keys[:, :2], axis=0)
    years = list(range(min(warm, int(keys[:, 2].min())), int(keys[:, 2].max()) + 1))
    stack = valid = None
    for t, y in enumerate(years):
        try:
            covn, mask = cio.norm_grid(cio.load_state_stack(y, states_dir, schema), mu, sd)
        except FileNotFoundError:
            continue
        if stack is None:
            stack = np.full((len(years), len(cells), covn.shape[-1]), np.nan, "float32")
            valid = np.zeros((len(years), len(cells)), bool)
        stack[t] = covn[cells[:, 0], cells[:, 1]]
        valid[t] = mask[cells[:, 0], cells[:, 1]]
        stack[t][~valid[t]] = np.nan
    if stack is None:
        raise FileNotFoundError(f"no yearly states under {states_dir}")
    raw = rows_from_stack(stack, years, cells, keys) if both else None
    stack = apply_output_ema(stack, hl, valid=valid)
    print(f"[gp-species] covariates: {stack.shape[-1]} channels, EMA half-life {hl:.2f} yr")
    ema = rows_from_stack(stack, years, cells, keys)
    return (ema, raw) if both else ema


def spacetime_inputs(keys, cell_km):
    """``[x_km, y_km, year]`` per key, on the model grid. Pure."""
    k = np.asarray(keys, "float64")
    return np.column_stack([k[:, 1] * cell_km, k[:, 0] * cell_km, k[:, 2]])


def cooccurrence_similarity(Y_eval, Y_comm, top=5, exclude=None):
    """How closely each evaluation species tracks its nearest community species. Pure.

    Pearson correlation of log1p abundance over TRAINING rows between each evaluation species and
    each community species. Returns ``(max, mean of top-k)`` per evaluation species, NaN for a
    species with no variance. A data-driven "how much like the community is this species", defined
    for every species, unlike the AVONET measures.
    """
    def _std(A):
        A = np.asarray(A, "float64")
        sd = A.std(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (A - A.mean(0)) / np.where(sd > 0, sd, np.nan)
    C = _std(Y_eval).T @ np.nan_to_num(_std(Y_comm)) / len(Y_eval)        # (S_eval, S_comm)
    C = np.where(np.isfinite(C), C, np.nan)
    if exclude is not None:                      # exclude[i]: column of evaluation species i itself
        C[np.arange(len(exclude)), np.asarray(exclude, int)] = np.nan
    srt = -np.sort(-np.nan_to_num(C, nan=-np.inf), axis=1)
    k = min(int(top), C.shape[1])
    mx = srt[:, 0]
    tk = srt[:, :k].mean(1)
    dead = ~np.isfinite(C).any(1)
    mx, tk = mx.astype("float64"), tk.astype("float64")
    mx[dead], tk[dead] = np.nan, np.nan
    return mx, tk


def distance_to_training_km(test_cells, train_cells, cell_km):
    """Distance from each test cell to the nearest training cell. ``(n,)`` km. Pure."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(train_cells, "float64")).query(np.asarray(test_cells, "float64"))
    return d * float(cell_km)


def mahalanobis_novelty(X_test, X_train):
    """Mahalanobis distance of each test row from the training covariate distribution. Pure.

    Pseudo-inverse covariance, because channels that never vary (or are exact transforms of one
    another) make the covariance singular. Measures how far outside the training environment a
    held-out row sits -- extrapolation in covariate space rather than in geography.
    """
    Xt = np.asarray(X_train, "float64")
    mu = Xt.mean(0)
    P = np.linalg.pinv(np.cov(Xt, rowvar=False))
    D = np.asarray(X_test, "float64") - mu
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", D, P, D), 0.0, None))


def persistence_prediction(keys, Y, tr_rows, te_rows, modern=EPOCH_MODERN):
    """Stasis in OBSERVED abundance. ``((mean, var), noise)``. Pure.

    Each predicted row gets its own cell's mean log1p abundance over the cell's TRAINING rows in
    the modern epoch: a place whose present is observed, backcast as unchanged. That is the honest
    LEVEL competitor for the ``time`` group, where the spacetime GP has the same information and
    desk (a pure z readout, like the deployed model) does not. NaN for cells with no modern
    training rows -- every held-out block cell. Its predicted change is zero by construction, so it
    is graded on level only. Variance: the pooled within-cell spread of those rows plus the cell
    mean's own sampling variance; ``noise`` is the within-cell spread.
    """
    k = np.asarray(keys)
    tr = np.asarray(tr_rows)
    yr = k[tr, 2]
    sel = tr[(yr >= modern[0]) & (yr <= modern[1])]
    Yf = np.asarray(Y, "float64")
    S = Yf.shape[1]
    cell = lambda rows: k[rows, 0].astype(np.int64) * 100000 + k[rows, 1].astype(np.int64)
    uc, inv, cnt = np.unique(cell(sel), return_inverse=True, return_counts=True)
    sums = np.zeros((len(uc), S))
    np.add.at(sums, inv, Yf[sel])
    means = sums / cnt[:, None]
    within = (((Yf[sel] - means[inv]) ** 2).sum(0) / max(len(sel) - len(uc), 1))
    within = np.maximum(within, gpk.species_floor(Yf[tr].var(0)))
    tc = cell(np.asarray(te_rows))
    pos = np.searchsorted(uc, tc)
    ok = pos < len(uc)
    ok[ok] = uc[pos[ok]] == tc[ok]
    mean = np.full((len(te_rows), S), np.nan)
    var = np.full_like(mean, np.nan)
    mean[ok] = means[pos[ok]]
    var[ok] = within[None, :] * (1.0 + 1.0 / cnt[pos[ok]][:, None])
    return (mean, var), within


def thin_rows(rows, frac, rng):
    """A random ``frac`` of training rows, shared by every species and every predictor. Pure.

    Thinning ROWS rather than each species' detections separately keeps one conditioning set for
    all species (so the baselines stay one eigendecomposition per block). Every species loses
    detections in proportion, and the data-poor curve is read on each species' REMAINING
    detections, which spans the low end because species differ in prevalence.
    """
    rows = np.asarray(rows)
    n = max(1, int(round(frac * len(rows))))
    return np.sort(rng.choice(rows, n, replace=False)) if n < len(rows) else rows


# ----------------------------- fit + predict, every predictor -----------------------------

def spatial_blocks(keys, rows, max_rows=1000):
    """Partition ``rows`` into spatially contiguous blocks of WHOLE cells. Pure.

    Cells are ordered along a Morton (Z-order) curve over (row, col), so consecutive cells are
    neighbours, and cut into consecutive runs of at most ``max_rows`` rows. A cell is never split:
    all its years stay in one block, which is what puts same-cell temporal pairs -- the pairs that
    identify a temporal lengthscale and a persistent level -- inside the fit. A random subsample
    of a continent contains almost none of them.
    """
    rows = np.asarray(rows)
    k = np.asarray(keys)[rows]
    r, c = k[:, 0].astype(np.int64), k[:, 1].astype(np.int64)
    code = np.zeros(len(rows), np.int64)
    for i in range(16):
        code |= (((r >> i) & 1) << (2 * i + 1)) | (((c >> i) & 1) << (2 * i))
    order = np.lexsort((k[:, 2], code))
    rs, cs = rows[order], code[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
    ends = np.r_[starts[1:], len(rs)]
    blocks, cur, n_cur = [], [], 0
    for s, e in zip(starts, ends):
        if n_cur and n_cur + (e - s) > max_rows:
            blocks.append(np.sort(np.concatenate(cur)))
            cur, n_cur = [], 0
        cur.append(rs[s:e])
        n_cur += e - s
    if cur:
        blocks.append(np.sort(np.concatenate(cur)))
    return blocks


def self_index(tr, te):
    """Position of each predicted row among the (sorted) training rows; -1 where it is not one."""
    tr, te = np.asarray(tr), np.asarray(te)
    pos = np.searchsorted(tr, te)
    ok = pos < len(tr)
    ok[ok] = tr[pos[ok]] == te[ok]
    return np.where(ok, pos, -1)


def fit_predict_all(D, tr, te, opts, rng, verbose=True):
    """Every predictor fitted on training rows ``tr`` and predicting held-out rows ``te``.

    ``D`` holds per-row arrays over a common row set (``Y``, ``Z``, ``Z_nc``, ``F_st``,
    ``F_cov``, ``keys``, optionally ``Z_oracle``) and ``groups`` / ``local_groups`` for the test
    rows. Returns ``(preds {name: (mean, var)}, fits, noise {name: per-species noise variance})``.
    Shared by the main run and every thinning fraction, so the two cannot differ in anything but
    the data.

    In-sample predicted rows (the ``time`` set's modern epoch is the cell's own training rows)
    get every predictor's exact LEAVE-ONE-OUT prediction, so no predictor is graded on a row it
    was fitted to: for an exact local GP that row is its own nearest neighbour and the in-sample
    "prediction" reproduced its noise, which the observed change shares.

    ``noise`` is each predictor's fitted per-species observation variance, what the abundance
    back-transform needs (``predicted_raw``): the latent function's uncertainty is deliberately
    left out of a point prediction, or a predictor less certain in the past than in the present
    predicts a decline it does not believe in.
    """
    tr = np.sort(np.asarray(tr))
    Ytr = D["Y"][tr].astype("float64")
    sidx = self_index(tr, te)
    in_tr = sidx >= 0
    y_self = None
    if in_tr.any():
        y_self = np.full((len(te), Ytr.shape[1]), np.nan)
        y_self[in_tr] = D["Y"][np.asarray(te)[in_tr]]
    preds, fits, noise = {}, {}, {}
    m = gpk.fit(D["Z"][tr], Ytr)
    fits["desk"] = m
    preds["desk"] = gpk.predict(m, D["Z"][te], y_self=y_self)
    preds["no_change"] = gpk.predict(m, D["Z_nc"])
    noise["desk"] = noise["no_change"] = m["n2"]
    if D.get("Z_raw") is not None:
        mr = gpk.fit(D["Z_raw"][tr], Ytr)
        fits["desk_raw"] = mr
        preds["desk_raw"] = gpk.predict(mr, D["Z_raw"][te], y_self=y_self)
        preds["no_change_raw"] = gpk.predict(mr, D["Z_nc_raw"])
        noise["desk_raw"] = noise["no_change_raw"] = mr["n2"]
    # Pooled arms: the same kernels with every species' amplitude under one cross-species prior
    # (gp_kernels.fit_pooled_scales). Applied to every model, so pooling cannot tilt a comparison.
    if opts.get("pooled", True):
        mp = gpk.fit(D["Z"][tr], Ytr, pool=True)
        fits["desk_pooled"] = mp
        preds["desk_pooled"] = gpk.predict(mp, D["Z"][te], y_self=y_self)
        noise["desk_pooled"] = mp["n2"]
        if D.get("Z_raw") is not None:
            mrp = gpk.fit(D["Z_raw"][tr], Ytr, pool=True)
            fits["desk_raw_pooled"] = mrp
            preds["desk_raw_pooled"] = gpk.predict(mrp, D["Z_raw"][te], y_self=y_self)
            noise["desk_raw_pooled"] = mrp["n2"]
    n = len(tr)
    vy = np.maximum(Ytr.var(0) * (1 + 1 / n), gpk.VAR_FLOOR)
    preds["intercept"] = (np.broadcast_to(Ytr.mean(0), (len(te), Ytr.shape[1])),
                          np.broadcast_to(vy, (len(te), Ytr.shape[1])))
    noise["intercept"] = vy
    Zo = D.get("Z_oracle")
    if Zo is not None:
        # Only epoch rows with an independent half have an oracle z, so the oracle GP fits on
        # those training rows and predicts those held-out rows; the rest are NaN and every
        # comparison with it is scored on the rows where both exist.
        otr = tr[np.isfinite(Zo[tr]).all(1)]
        ote = np.isfinite(Zo[te]).all(1)
        mo = gpk.fit(Zo[otr], D["Y"][otr].astype("float64"))
        fits["esk_oracle_independent"] = mo
        noise["esk_oracle_independent"] = mo["n2"]
        mu = np.full((len(te), Ytr.shape[1]), np.nan)
        va = np.full_like(mu, np.nan)
        if ote.any():
            ys = None if y_self is None else y_self[ote]
            if ys is not None:
                # the oracle's training set is otr, so only rows that are IN otr are in-sample
                ys = np.where(np.isin(np.asarray(te)[ote], otr)[:, None], ys, np.nan)
            mu[ote], va[ote] = gpk.predict(mo, Zo[te][ote], y_self=ys)
        preds["esk_oracle_independent"] = (mu, va)
    if opts.get("baselines", True):
        ybar = Ytr.mean(0)
        var_y = Ytr.var(0) * n / max(n - 1, 1)
        blocks = spatial_blocks(D["keys"], tr, int(opts.get("block_rows", 1000)))
        # The SHAPE is fitted on random whole blocks totalling ~n_fit rows: contiguous, so a cell's
        # own other years and its neighbours are in the fit. The per-species SCALES then come from
        # every training row, block by block (gp_kernels.blocked_scales).
        shape_blocks, n_sel = [], 0
        for i in rng.permutation(len(blocks)):
            shape_blocks.append(blocks[i])
            n_sel += len(blocks[i])
            if n_sel >= opts["n_fit"]:
                break
        groups = D.get("local_groups", D["groups"])
        th_cov = np.full(D["F_cov"].shape[1], np.log(np.sqrt(max(D["F_cov"].shape[1], 1))))
        arms = [("spacetime", "spacetime", "F_st", np.log([300.0, 20.0])),
                ("spacetime_sum", "spacetime_sum", "F_st",
                 np.array([np.log(300.0), np.log(150.0), np.log(20.0), 0.0])),
                ("covariate", "covariate", "F_cov", th_cov)]
        # The covariate GP matched to RAW z (input-side EMA only), so desk_raw has a like-for-like
        # rival. Main run only: it is a third shape fit, and thinning is about data, not lag.
        if opts.get("raw_baseline", True) and D.get("F_cov_raw") is not None:
            arms.append(("covariate_raw", "covariate", "F_cov_raw", th_cov))
        for label, kind, Fkey, th0 in arms:
            t0 = time.perf_counter()
            shape = gpk.fit_shared_shape(kind, [D[Fkey][b] for b in shape_blocks],
                                         [D["Y"][b] for b in shape_blocks], th0,
                                         n_iter=opts["shape_iters"], verbose=verbose,
                                         ybar=ybar, var_y=var_y)
            sc = gpk.blocked_scales(kind, shape["theta"], [D[Fkey][b] for b in blocks],
                                    [D["Y"][b] for b in blocks], ybar, var_y)
            shape.update({"s2": sc["s2"], "n2": sc["n2"], "ok": sc["ok"], "ybar": ybar,
                          "n_scale_blocks": len(blocks)})
            fits[label] = shape
            for k in opts["k_nn"]:
                name = label if k == opts["k_nn"][0] else f"{label}_k{k}"
                preds[name] = gpk.predict_local(shape, D[Fkey][tr], Ytr, D[Fkey][te], groups,
                                                k=k, k_max=opts["k_max"], self_idx=sidx)
                noise[name] = shape["n2"]
            if opts.get("pooled", True):
                fit_rows = np.sort(np.concatenate(shape_blocks))
                ps = gpk.pool_shape_scales(shape, D[Fkey][fit_rows], D["Y"][fit_rows])
                ps["ybar"] = ybar
                fits[f"{label}_pooled"] = ps
                preds[f"{label}_pooled"] = gpk.predict_local(
                    ps, D[Fkey][tr], Ytr, D[Fkey][te], groups, k=opts["k_nn"][0],
                    k_max=opts["k_max"], self_idx=sidx)
                noise[f"{label}_pooled"] = ps["n2"]
            if verbose:
                print(f"[gp-species] {label} baseline fitted and predicted in "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    return preds, fits, noise


def shape_summary(shape):
    """Lengthscales and the optimizer's own convergence verdict for a baseline's shape fit. Pure.

    An under-converged baseline is a strawman, so the report carries L-BFGS-B's verdict and which
    lengthscales sit at a bound (a lengthscale pinned at the upper bound means that input, or
    time, carries no usable structure -- a finding, not a failure).
    """
    tr = np.asarray(shape.get("trace", []), "float64")
    return {"lengthscales": np.exp(shape["theta"]).tolist(),
            "converged": bool(shape.get("converged", False)),
            "message": shape.get("message"), "n_iterations": shape.get("n_iterations"),
            "at_bound": shape.get("at_bound"),
            "sum_nll_first": float(tr[0]) if len(tr) else None,
            "sum_nll_final": float(tr[-1]) if len(tr) else None}


# ----------------------------- predictor registry -----------------------------

#: Every predictor this suite can emit and what it is FOR -- the GP counterpart of
#: ``validate_bbs_routes.PREDICTOR_ROLES``. A predictor that is expected but produces neither a
#: result nor a named reason is a gap the report lists, never a silent hole: an absent comparison
#: is how a missing ceiling once went unnoticed across several runs.
GP_PREDICTOR_ROLES = {
    "desk": "the model under test: z_ema, what DESK was trained on",
    "desk_raw": ("raw z, what the cube exports to the population model -- with no lag supplied "
                 "here, a lower bound on the deployed features"),
    "no_change": ("decomposition null: desk's own fit at each cell's modern-epoch z, so its "
                  "predicted change is exactly zero. NOT a competitor"),
    "no_change_raw": "the same null for desk_raw",
    "intercept": "the level floor: the training mean, which knows nothing about place",
    "spacetime": ("the honest bar: Matern(space) x exponential(time) on OBSERVED abundance from "
                  "training rows only; its strength is neighbours observed in the same years"),
    "spacetime_sum": ("the honest TEMPORAL rival: a persistent spatial field plus a "
                      "spatiotemporal one, so a backcast keeps what does not change instead of "
                      "reverting to the continental mean as the product kernel does"),
    "persistence": ("stasis in OBSERVED abundance: each cell's own modern-epoch training mean, "
                    "for every year. Level only (its change is zero); defined only where the cell "
                    "has modern training rows -- the time group"),
    "covariate": "ARD on DESK's own covariates with DESK's output EMA -- matched to z_ema",
    "covariate_raw": "ARD on DESK's covariates with the input-side EMA only -- matched to z_raw",
    "esk_oracle_independent": ("THE ceiling: the observed community from a DISJOINT half of the "
                               "same cell-epoch's years, projected into the ESK basis"),
}
#: Whether a predictor shares the target's noise draw. None does: the same-rows oracle that did
#: was removed (see ``independent_oracle_z``). Kept as a registry so a new row must declare it.
GP_SHARES_TARGET_NOISE = {k: False for k in GP_PREDICTOR_ROLES}

SEED_CAVEAT = ("single DESK checkpoint = single training seed. Seed-to-seed spread of DESK is "
               "6.6% sd (15.6% range) on val_kernel (desk_hp hl4 replicates); the CIs here "
               "resample blocks and species, NOT training seeds, so they understate the "
               "uncertainty about DESK itself")
ESTIMAND = ("change and epoch level are log1p of the epoch-MEAN abundance "
            "(validate_bbs_routes.epoch_mean_observed), for truth and prediction alike; a "
            "prediction's expected count is exp(mean + sigma^2/2) - 1 with sigma^2 the "
            "predictor's fitted per-species OBSERVATION noise (its latent uncertainty is not part "
            "of a point prediction); a negative expected count is clipped to 0 by log1p. Change "
            "skill is defined only for species whose change is resolvable above split-half noise. "
            "Per-row level is RMSE on log1p of the cell-year mean count")
#: Level windows, the route suite's buckets: its DESK skill sits almost entirely in the early one.
WINDOWS = {"early": EARLY_WINDOW, "mid": (EARLY_WINDOW[1] + 1, MODERN_WINDOW[0] - 1),
           "modern": MODERN_WINDOW}


def expected_predictors(opts):
    """The predictors this configuration must account for, by name. Pure."""
    names = ["desk", "desk_raw", "no_change", "no_change_raw", "intercept",
             "esk_oracle_independent", "persistence"]
    arms = (["spacetime", "spacetime_sum", "covariate", "covariate_raw"]
            if opts.get("baselines", True) else [])
    names += arms
    names += [f"{a}_k{k}" for a in arms for k in opts["k_nn"][1:]]
    if opts.get("pooled", True):
        names += ["desk_pooled", "desk_raw_pooled"] + [f"{a}_pooled" for a in arms]
    return names


def role_of(name):
    """The registry role of a predictor, including its pooled and support-sensitivity variants."""
    base, note = name, ""
    if name.endswith("_pooled"):
        base, note = name[:-len("_pooled")], " (amplitude under the cross-species prior)"
    elif "_k" in name and name.rsplit("_k", 1)[1].isdigit():
        base, k = name.rsplit("_k", 1)
        note = f" (local support k={k}: sensitivity check)"
    return GP_PREDICTOR_ROLES.get(base, "UNREGISTERED") + note


# ----------------------------- scoring -----------------------------

REFERENCES = ("no_change", "intercept")


def comparisons(names):
    """Which (model, reference) pairs the report scores. Pure.

    Every model against no_change (the change null) and intercept (the level floor); desk against
    every other model, so "desk vs spacetime" and "desk vs desk_raw" are reported directly; and
    desk_raw against its own null, no_change_raw. Nulls never appear on the model side.
    """
    models = [p for p in names if not p.startswith(("no_change", "intercept"))]
    pairs = [(p, ref) for ref in REFERENCES for p in models]
    pairs += [("desk", p) for p in models if p != "desk"]
    if "desk_raw" in names and "no_change_raw" in names:
        pairs.append(("desk_raw", "no_change_raw"))
    # The pooled DESK arms against every pooled rival, like for like.
    for d, rivals in (("desk_pooled", ("spacetime_pooled", "spacetime_sum_pooled",
                                       "covariate_pooled")),
                      ("desk_raw_pooled", ("covariate_raw_pooled",))):
        if d in names:
            pairs += [(d, p) for p in rivals if p in names]
    return pairs


def _pair_table(err, avail, block, region, cell_of, pairs, n_boot, seed, region_ok, bal_kw,
                truth, regions=True, species_ok=None):
    """Pooled skill per pair, on the rows where BOTH predictors exist, with regional balance.

    ``err[name]`` is a per-unit squared-error table (rows or cells x species), ``avail[name]`` the
    units where that predictor exists. Every pair gets the two-way-bootstrapped pooled skill and,
    unless ``regions`` is off, a per-region median and ``balanced_over_strata`` across regions --
    BBS is coast-heavy, so the population-weighted figure grades a model where the survey is dense
    and cannot see a deficit elsewhere. Both are reported; their gap IS the coverage bias.

    ``truth`` (units x species) decides where a species' skill is DEFINED: only over units where
    the truth carries any signal for it (a detection for level, a nonzero observed change for
    change). Elsewhere the reference's error is not zero but merely tiny -- a baseline predicting
    1e-6 where the species is absent -- and a skill computed against it is noise of any size: the
    first audited run reported regional skills of -7 to -1858 from exactly that. The rule is the
    suite's "undefined metric" rule, applied per region, to the pooled figure, and inside every
    bootstrap draw.

    ``species_ok`` restricts the species a skill is defined for (the change tables pass the
    split-half ``resolvable`` mask): a species whose observed change is indistinguishable from its
    own noise has nothing to predict, and any predictor that moves scores below no_change on it in
    expectation. Pooling over such species drags every model's median toward zero whatever the
    model is -- the first runs' median was over every species with ANY nonzero change.
    """
    from .validate_bbs_routes import balanced_over_strata
    out, per = {}, {}
    ok_sp = None if species_ok is None else np.asarray(species_ok, bool)
    for a, b in pairs:
        if a not in err or b not in err:
            continue
        m = avail[a] & avail[b]
        key = f"{a}_vs_{b}"
        if m.sum() == 0:
            out[key] = {"unavailable": "no units where both predictors exist"}
            continue
        _, sa = block_sums(err[a][m], block[m])
        _, sb = block_sums(err[b][m], block[m])
        _, sig = block_sums((np.abs(truth[m]) > 0).astype("float64"), block[m])
        undefined = ~(sig.sum(0) > 0)
        if ok_sp is not None:
            undefined |= ~ok_sp
        sa[:, undefined], sb[:, undefined] = 0.0, 0.0      # -> skill NaN: no signal to grade
        pooled, sk = pooled_skill(sa, sb, n_boot, seed, block_signal=sig)
        per[key] = sk
        if regions:
            strata = {}
            for rg in np.unique(region[m]):
                mm = m & (region == rg)
                ra, rb = err[a][mm].sum(0), err[b][mm].sum(0)
                skr = skill_from_sse(ra, rb)
                skr[~(np.abs(truth[mm]) > 0).any(0)] = np.nan
                if ok_sp is not None:
                    skr[~ok_sp] = np.nan
                fin = np.isfinite(skr)
                ok, why = region_ok.get(int(rg), (False, "no viability verdict"))
                strata[str(int(rg))] = {
                    "qualified": bool(ok and fin.sum() >= 3), "reason": why,
                    "n_cells": int(len(np.unique(cell_of[mm], axis=0))),
                    "median": float(np.median(skr[fin])) if fin.sum() >= 3 else float("nan")}
            bal = balanced_over_strata(strata, "median", **bal_kw)
            pooled = {**pooled, "regions": strata, "balanced_median": bal.get("balanced"),
                      "population_weighted_median": bal.get("population_weighted"),
                      "n_regions_qualified": bal.get("n_strata", 0)}
        out[key] = pooled
    return out, per


def decomposition(d_pred, d_obs, resolvable):
    """Magnitude vs direction of each species' predicted change across cells. ``(summary, per)``.

    ``validate_baselines.error_decomposition`` per species, treating the vector over cells as the
    object: ``||p - o||^2 = (||p|| - ||o||)^2 + 2||p|| ||o|| (1 - cos)``. RMSE skill and captured
    share mix the two, and they trade off -- at direction cosine rho the MSE-optimal magnitude is
    ``rho * ||o||`` -- so a negative skill can be a calibration problem (right direction, too big a
    move) rather than a direction problem. DESK was measured over-moving ~2.2x against that
    optimum on its own target. ``overmove = (||p||/||o||) / cos`` is 1.0 at MSE calibration.
    Resolvable species only; cells without a prediction are left out.
    """
    from .validate_baselines import error_decomposition
    ok = np.isfinite(d_pred).all(1)
    p, o = np.asarray(d_pred, "float64")[ok].T, np.asarray(d_obs, "float64")[ok].T
    total, mag, ang, cos = error_decomposition(p, o)
    ratio = np.linalg.norm(p, axis=1) / np.maximum(np.linalg.norm(o, axis=1), 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        over = np.where(cos > 0, ratio / cos, np.nan)
        mag_share = np.where(total > 0, mag / total, np.nan)
    res = np.asarray(resolvable, bool)
    med = lambda v: float(np.nanmedian(v[res])) if np.isfinite(v[res]).any() else None
    return ({"median_cos": med(cos), "median_norm_ratio": med(ratio), "median_overmove": med(over),
             "median_magnitude_share_of_error": med(mag_share),
             "share_cos_positive": (float(np.nanmean(cos[res] > 0)) if res.any() else None),
             "note": "resolvable species; overmove 1.0 = MSE-calibrated magnitude at this angle"},
            {"cos": cos, "norm_ratio": ratio, "overmove": over})


def _direction_report(dir_tables):
    rep, per = {}, {}
    for name, tab in dir_tables.items():
        ds = np.array([t.get("direction_skill", np.nan) for t in tab], "float64")
        n_abst = sum(1 for t in tab if "n_species_committed" in t and "direction_skill" not in t)
        rep[name] = {
            "n_species_scored": int(np.isfinite(ds).sum()),
            "n_species_abstained_or_undefined": int((~np.isfinite(ds)).sum()),
            "n_abstained": int(n_abst),
            "median_direction_skill": float(np.nanmedian(ds)) if np.isfinite(ds).any() else None}
        per[name] = ds
    return rep, per


def _prob_table(Y, preds, sel, avail):
    """Log density, CRPS and interval coverage per predictor, on its own available rows."""
    out = {}
    for name, (mu, var) in preds.items():
        m = sel[avail[name][sel]] if avail is not None else sel
        if len(m) == 0:
            continue
        mu_s, va_s, y = np.asarray(mu)[m], np.asarray(var)[m], Y[m]
        n = float(len(m))
        out[name] = {
            "n_rows": int(n),
            "median_lpd_per_row": float(np.median(gaussian_lpd(y, mu_s, va_s).sum(0) / n)),
            "median_crps": float(np.median(gaussian_crps(y, mu_s, va_s).sum(0) / n)),
            "coverage50_pooled": float(interval_coverage(y, mu_s, va_s, 0.5).mean()),
            "coverage90_pooled": float(interval_coverage(y, mu_s, va_s, 0.9).mean())}
    return out


# ----------------------------- run -----------------------------

#: Which species are graded. ``out_of_community`` is the transfer question and the default;
#: ``community`` grades the 96 species DESK's target is built from -- a CONSISTENCY run, the
#: per-species analogue of the route suite, never evidence about transfer (DESK was trained on
#: these species' similarity structure; it is out-of-sample in space and time only).
SPECIES_MODES = ("out_of_community", "community")


def run(config=None, out_dir=None, n_boot=1000, seed=0, opts=None):
    t0 = time.perf_counter()
    config = config or load_config()
    opts = {"baselines": True, "pooled": True, "n_fit": 3000, "shape_iters": 400,
            "k_nn": (32, 8), "k_max": 4000, "thin": (0.3, 0.1, 0.03),
            "species": "out_of_community", **(opts or {})}
    if opts["species"] not in SPECIES_MODES:
        raise ValueError(f"species mode {opts['species']!r}; one of {SPECIES_MODES}")
    community_mode = opts["species"] == "community"
    rng = np.random.default_rng(seed)
    run_dir = config["paths"]["desk_output_dir"]
    out_dir = out_dir or os.path.join(run_dir, "gp_species_community" if community_mode
                                      else "gp_species")
    os.makedirs(out_dir, exist_ok=True)
    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    latent = int(dm["latent_dim"])
    tr_cfg = (config.get("desk", {}) or {}).get("trend", {}) or {}
    br_cfg = config.get("bbs_routes", {}) or {}
    block_cells = int(tr_cfg.get("block_cells", 6))
    withheld = [int(y) for y in (tr_cfg.get("holdout_years") or [])]
    common = [int(y) for y in (tr_cfg.get("common_holdout_years") or [])]
    cell_km = float(opts.get("cell_km") or _cell_km())
    n_regions = int(br_cfg.get("spatial_regions", 2) or 2)
    bal_kw = {k: v for k, v in (br_cfg.get("balance") or {}).items()
              if k in ("n_min", "cap", "power")}

    from src.data.preprocess.bbs_community import log1p_community
    from .esk_kernel import coarse_spatial
    from .validate_bbs_routes import stratum_viable
    from .validation_core import captured_share, change_noise, epoch_values, split_half_change

    X_raw, keys, layout = load_all_species(config)
    nc = layout["n_community"]
    Ylog = log1p_community(X_raw)
    if community_mode:
        # Grade the community's own columns. The independent oracle still projects the WHOLE
        # community, including the graded species, but from the disjoint half of the years -- the
        # route suite's esk_oracle_independent, applied per species.
        Y_eval = Ylog[:, :nc]
        layout = {**layout, "evaluation": list(layout["community"])}
    else:
        Y_eval = Ylog[:, nc:]
    X_comm_raw = np.asarray(X_raw[:, :nc], "float64")
    # Regions over ALL surveyed rows, the route suite's definition (coarse_spatial bins the
    # occupied extent of the points it is given, so it must be given the same points).
    region_all = coarse_spatial(keys, regions=n_regions)

    ho, bf, buffer_note = load_holdout_masks(run_dir, tr_cfg.get("buffer_floor"))
    if ho is None or not ho.any():
        raise ValueError(f"{run_dir}: no held-out cells ({buffer_note}) -- holdout_frac=0 (the "
                         "production run?). Point paths.desk_output_dir at a run trained with a "
                         "holdout.")
    is_train, group, block_id = row_splits(keys, ho, bf, block_cells, withheld, common)
    yr = keys[:, 2]
    # Training cells with a withheld early epoch also need their MODERN rows predicted (in-sample)
    # so their change can be scored: the "time" change set.
    time_cells = {(int(a), int(b)) for a, b in keys[group == 2, :2]}
    in_time_cell = np.array([(int(a), int(b)) in time_cells for a, b in keys[:, :2]], bool)
    insample = is_train & in_time_cell & (yr >= EPOCH_MODERN[0]) & (yr <= EPOCH_MODERN[1])
    pred = (group > 0) | insample
    use = is_train | pred
    print(f"[gp-species] withheld years: "
          f"{(str(withheld[0]) + '-' + str(withheld[-1])) if withheld else 'none'}; scored rows: "
          + ", ".join(f"{GROUPS[g]} {int((group == g).sum()):,}" for g in GROUPS)
          + f"; buffer {buffer_note}")

    # One encode for everything, raw and EMA'd together: rows to fit and predict, plus the modern
    # epoch of every predicted cell for the no-change reference.
    from .validate_bbs_routes import desk_z_ema
    ref_keys = modern_reference_keys(keys[pred])
    want = np.concatenate([keys[use], ref_keys]).astype("int32")
    want_u, inv = np.unique(want, axis=0, return_inverse=True)
    Zu, zinfo, Zu_raw = desk_z_ema(config, want_u, return_raw=True)
    n_rows = int(use.sum())
    Z = np.full((len(keys), latent), np.nan, "float32")
    Z_raw = np.full((len(keys), latent), np.nan, "float32")
    Z[use], Z_raw[use] = Zu[inv[:n_rows]], Zu_raw[inv[:n_rows]]
    Z_ref, Z_ref_raw = Zu[inv[n_rows:]], Zu_raw[inv[n_rows:]]

    F_cov = np.full((len(keys), 0), np.nan, "float32")
    F_cov_raw = F_cov
    if opts["baselines"]:
        Fc, Fr = covariates_for_keys(config, keys[use], both=True)
        F_cov = np.full((len(keys), Fc.shape[1]), np.nan, "float32")
        F_cov_raw = np.full_like(F_cov, np.nan)
        F_cov[use], F_cov_raw[use] = Fc, Fr
    F_st = spacetime_inputs(keys, cell_km)

    # ONE common row set for every predictor: finite z (both forms), finite covariates (both
    # forms), finite no-change z. The oracle alone covers a subset (epoch rows with an
    # independent half) and is compared on the rows where both predictors exist.
    finite = np.isfinite(Z).all(1) & np.isfinite(Z_raw).all(1)
    if F_cov.shape[1]:
        finite &= np.isfinite(F_cov).all(1) & np.isfinite(F_cov_raw).all(1)
    Z_nc_all = np.full_like(Z, np.nan)
    Z_nc_all[pred] = no_change_z(keys[pred], ref_keys, Z_ref)
    Z_nc_raw = np.full_like(Z, np.nan)
    Z_nc_raw[pred] = no_change_z(keys[pred], ref_keys, Z_ref_raw)
    tr_rows = np.where(is_train & finite)[0]
    te_rows = np.where(pred & finite & np.isfinite(Z_nc_all).all(1)
                       & np.isfinite(Z_nc_raw).all(1))[0]
    n_drop = int(pred.sum() - len(te_rows))
    row_group = group[te_rows]

    # Species with ZERO training detections leave the evaluation entirely. Every predictor's
    # posterior mean for them is exactly the training mean (zero), so no metric can tell any two
    # predictors apart, and keeping them only adds exact-zero skills that drag the pooled medians
    # toward 0. In the first run these were 91 species absent from the study area in June
    # (arctic breeders, pelagics, Alaskan specialties, vagrants) plus 7 whose whole range fell in
    # held-out blocks. Not a data-poverty filter: one training detection is enough to stay in.
    det_all = (Y_eval[tr_rows] > 0).sum(0)
    keep_sp = det_all > 0
    # A FIXED species set (A7): each tempho run filters on its OWN training years, so without a
    # shared list three runs grade three different populations and a cross-run decay curve mixes
    # them. Listed species that a run cannot grade (no training detection) are reported, not
    # silently replaced.
    fixed = opts.get("species_codes")
    not_gradable = []
    if fixed is not None:
        fixed = [str(c) for c in fixed]
        listed = np.isin(np.asarray(layout["evaluation"]), fixed)
        not_gradable = [c for c, l, k in zip(layout["evaluation"], listed, keep_sp) if l and not k]
        keep_sp = keep_sp & listed
    dropped_species = [c for c, k in zip(layout["evaluation"], keep_sp) if not k]
    Y_eval = Y_eval[:, keep_sp]
    layout = {**layout, "evaluation": [c for c, k in zip(layout["evaluation"], keep_sp) if k]}
    print(f"[gp-species] dropped {len(dropped_species)} species with no training detections; "
          f"{int(keep_sp.sum())} evaluated")
    print(f"[gp-species] {len(tr_rows):,} training rows, {len(te_rows):,} predicted rows "
          f"({int((row_group == 0).sum()):,} in-sample modern rows of time-group cells; "
          f"{n_drop} dropped: outside the covariate footprint or no modern z)")

    unavailable = {}
    Z_oracle, oracle_info = independent_oracle_z(
        keys, X_comm_raw, config["desk"]["z_dir"], latent,
        norm_tol=float(br_cfg.get("oracle_norm_tol", 0.5)), withheld=withheld)
    if Z_oracle is None:
        unavailable["esk_oracle_independent"] = oracle_info.get("reason", "unavailable")
    print(f"[gp-species] independent oracle: {oracle_info}")
    if not opts["baselines"]:
        for a in ("spacetime", "spacetime_sum", "covariate", "covariate_raw"):
            unavailable[a] = "disabled by --no-baselines"
    if not opts.get("pooled", True):
        unavailable["desk_pooled"] = unavailable["desk_raw_pooled"] = "disabled by --no-pooled"

    bid = block_id[te_rows]
    # Local GPs condition each group of test rows on the union of their neighbours, capped at
    # k_max by distance to the NEAREST test row. Grouping by block alone let a block's many modern
    # rows crowd its sparse early rows' neighbours out of that cap; grouping by block x decade
    # keeps every era's support.
    local_groups = bid * 1000 + (keys[te_rows, 2] // 10)
    D = {"Y": Y_eval, "Z": Z, "Z_nc": Z_nc_all[te_rows], "Z_raw": Z_raw,
         "Z_nc_raw": Z_nc_raw[te_rows], "F_st": F_st, "F_cov": F_cov, "F_cov_raw": F_cov_raw,
         "Z_oracle": Z_oracle, "groups": bid, "local_groups": local_groups, "keys": keys}
    preds, fits, noise = fit_predict_all(D, tr_rows, te_rows, opts, rng)
    preds["persistence"], noise["persistence"] = persistence_prediction(
        keys, Y_eval, tr_rows, te_rows)
    Yte = Y_eval[te_rows].astype("float64")
    raw_te = np.expm1(Yte)
    Ytr = Y_eval[tr_rows]
    names = list(preds)
    tk = keys[te_rows]
    reg = region_all[te_rows]
    ev = layout["evaluation"]
    pairs = comparisons(names)
    avail_rows = {n_: np.isfinite(np.asarray(mu)).all(1) for n_, (mu, _v) in preds.items()}
    expected = expected_predictors(opts)
    gaps = [p for p in expected if p not in preds and p not in unavailable]

    rep = {"run_dir": run_dir, "basis": config["desk"].get("z_dir"),
           "ema": zinfo, "best_epoch_of_checkpoint": int(dm["best_epoch"])
           if "best_epoch" in dm.files else None,
           "withheld_years": withheld, "common_holdout_years": common, "buffer": buffer_note,
           "estimand": ESTIMAND, "seed_caveat": SEED_CAVEAT,
           "species_mode": opts["species"],
           "species_mode_note": ("COMMUNITY species: a consistency check against the route suite, "
                                 "in-sample in species (DESK's target is built from them), "
                                 "out-of-sample in space and time. Not evidence about transfer."
                                 if community_mode else
                                 "out-of-community species minus House Finch: the transfer "
                                 "question"),
           "selection_caveat": ("the checkpoint's epoch was selected on a held-out kernel metric "
                                "over these same blocks: a mild optimistic bias for desk"),
           "covariate_gp_caveat": ("the covariate GP sees each cell's own covariates; desk also "
                                   "sees a 5x5 neighbourhood through its convolution"),
           "regions": {"spatial_regions_per_axis": n_regions, "balance": bal_kw},
           "predictors": {p: role_of(p) for p in names},
           "shares_target_noise": {p: GP_SHARES_TARGET_NOISE.get(p, False) for p in names},
           "unavailable": unavailable, "completeness_gaps": gaps,
           "oracle": oracle_info,
           "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opts.items()},
           "layout": {k: v for k, v in layout.items() if k not in ("community", "evaluation")},
           "rows": {"train": int(len(tr_rows)), "predicted": int(len(te_rows)),
                    "dropped": n_drop, "in_sample_modern": int((row_group == 0).sum()),
                    **{GROUPS[g]: int((row_group == g).sum()) for g in GROUPS}},
           "dropped_zero_training_detections": {"n": len(dropped_species),
                                                "species": dropped_species},
           "fixed_species_list": ({"n_listed": len(fixed), "n_graded": int(keep_sp.sum()),
                                   "listed_but_not_gradable": not_gradable}
                                  if fixed is not None else None),
           "pooled_priors": {k: v["prior"] for k, v in fits.items() if "prior" in v},
           "baseline_shapes": {k: shape_summary(v) for k, v in fits.items() if "theta" in v},
           "primary": {}, "change": {}, "change_all_defined": {}, "level": {},
           "level_by_window": {}, "probabilistic": {},
           "direction": {}, "change_noise": {}, "change_viability": {}, "captured": {},
           "decomposition": {}, "epoch_gate": {}}
    per_species = {}

    # ---- level, per held-out group (never on in-sample rows) ----
    for g, gname in GROUPS.items():
        sel = np.where(row_group == g)[0]
        if len(sel) == 0:
            continue
        err = {n_: ((Yte[sel] - np.asarray(mu)[sel]) ** 2).astype("float32")
               for n_, (mu, _v) in preds.items()}
        avail = {n_: avail_rows[n_][sel] for n_ in preds}
        cells_g = tk[sel, :2]
        region_ok = {}
        for rg in np.unique(reg[sel]):
            n_c = len(np.unique(cells_g[reg[sel] == rg], axis=0))
            region_ok[int(rg)] = (n_c >= 30, "" if n_c >= 30 else f"only {n_c} cells (needs >= 30)")
        rep["level"][gname], per = _pair_table(err, avail, bid[sel], reg[sel], cells_g, pairs,
                                               n_boot, seed, region_ok, bal_kw, truth=Yte[sel])
        per_species.update({f"level_skill_{gname}_{k}": v for k, v in per.items()})
        rep["level_by_window"][gname] = {}
        for wname, (lo, hi) in WINDOWS.items():
            w = (tk[sel, 2] >= lo) & (tk[sel, 2] <= hi)
            if w.sum() == 0:
                continue
            rep["level_by_window"][gname][wname], _ = _pair_table(
                {k: v[w] for k, v in err.items()}, {k: v[w] for k, v in avail.items()},
                bid[sel][w], reg[sel][w], cells_g[w], pairs, n_boot, seed, region_ok, bal_kw,
                truth=Yte[sel][w], regions=False)
        rep["probabilistic"][gname] = _prob_table(Yte, preds, sel, avail_rows)

    # ---- change, per change set ----
    sets = change_sets(tk, row_group, withheld)
    change_store = {}
    for sname, rows in sets.items():
        cells, e_loc, m_loc, gate = epoch_gate(tk[rows], EPOCH_EARLY, EPOCH_MODERN,
                                               MIN_EPOCH_YEARS)
        rep["epoch_gate"][sname] = gate
        if len(cells) == 0:
            continue
        e_rows = [rows[np.asarray(r)] for r in e_loc]
        m_rows = [rows[np.asarray(r)] for r in m_loc]
        first = [int(r[0]) for r in e_rows]
        cell_block, cell_reg = bid[first], reg[first]
        d_full, d_a, d_b = split_half_change(raw_te, e_rows, m_rows, tk[:, 2])
        nz = change_noise(d_full, d_a, d_b, seed=seed)
        oe, om = epoch_values(raw_te, e_rows, m_rows)
        # Viability, the route suite's rule: enough cells, and real change (squared change minus
        # its split-half noise, per cell) distinguishable from zero. stratum_viable takes
        # (observed, floor) and tests floor - observed > 0; here "floor" is the squared change and
        # "observed" its noise, so floor - observed is the per-cell excess of real change.
        ok_all, why_all, st_all = stratum_viable(nz["cell_noise"].mean(1), nz["cell_sq"].mean(1),
                                                 len(cells))
        region_ok = {}
        for rg in np.unique(cell_reg):
            cm = cell_reg == rg
            ok_r, why_r, _ = stratum_viable(nz["cell_noise"][cm].mean(1),
                                            nz["cell_sq"][cm].mean(1), int(cm.sum()))
            region_ok[int(rg)] = (ok_r, why_r)
        rep["change_viability"][sname] = {"qualified": bool(ok_all), "reason": why_all,
                                          **st_all}
        dpred, err, avail, dirs, cap, dec = {}, {}, {}, {}, {}, {}
        for n_, (mu, var) in preds.items():
            if n_ == "persistence":
                continue                       # zero predicted change by construction: level only
            pe, pm = epoch_values(predicted_raw(mu, noise[n_][None, :]), e_rows, m_rows)
            dp = pm - pe
            dpred[n_] = dp
            err[n_] = (dp - d_full) ** 2
            avail[n_] = np.isfinite(dp).all(1)
            dirs[n_] = direction_by_species(oe, om, pe, pm, noise_sd=np.sqrt(nz["noise"]))
            per_c, pooled_c = captured_share(d_full, dp, nz)
            v = per_c[np.isfinite(per_c)]
            cap[n_] = {"median": float(np.median(v)) if len(v) else None,
                       "share_above_zero": float((v > 0).mean()) if len(v) else None,
                       "pooled": pooled_c}
            per_species[f"captured_{sname}_{n_}"] = per_c
            dsum, dper = decomposition(dp, d_full, nz["resolvable"])
            dec[n_] = dsum
            per_species.update({f"{k}_{sname}_{n_}": v_ for k, v_ in dper.items()})
        cpairs = [(a, b) for a, b in pairs if b != "intercept" and "persistence" not in (a, b)]
        # Skill is DEFINED only for species whose change is resolvable above its own split-half
        # noise (A1). The previous definition -- any nonzero change anywhere -- is kept beside it
        # so older reports can be compared, never as the headline.
        rep["change"][sname], per = _pair_table(err, avail, cell_block, cell_reg, cells, cpairs,
                                                n_boot, seed, region_ok, bal_kw, truth=d_full,
                                                species_ok=nz["resolvable"])
        rep["change_all_defined"][sname], _ = _pair_table(
            err, avail, cell_block, cell_reg, cells, cpairs, n_boot, seed, region_ok, bal_kw,
            truth=d_full, regions=False)
        per_species.update({f"change_skill_{sname}_{k}": v for k, v in per.items()})
        rep["direction"][sname], per = _direction_report(dirs)
        per_species.update({f"direction_skill_{sname}_{k}": v for k, v in per.items()})
        ok = nz["ms_obs"] > 0
        rep["change_noise"][sname] = {
            "n_cells": int(len(cells)), "n_species_with_change": int(ok.sum()),
            "n_resolvable": int(nz["resolvable"].sum()),
            "median_signal_share": float(np.nanmedian(nz["signal_share"][ok])) if ok.any()
            else None,
            "pooled_signal_share": float(1 - nz["noise"][ok].sum() / nz["ms_obs"][ok].sum())
            if ok.any() else None,
            "median_ceiling_skill_resolvable": float(np.median(nz["ceiling_skill"][
                nz["resolvable"]])) if nz["resolvable"].any() else None,
            "note": "split-half is ABBA over years within each epoch; noise slightly conservative"}
        rep["captured"][sname] = cap
        rep["decomposition"][sname] = dec
        per_species.update({f"resolvable_{sname}": nz["resolvable"],
                            f"signal_share_{sname}": nz["signal_share"],
                            f"noise_change_{sname}": nz["noise"],
                            f"ms_obs_change_{sname}": nz["ms_obs"]})
        change_store[sname] = {"cells": cells, "e_rows": e_rows, "m_rows": m_rows,
                               "cell_block": cell_block, "cell_region": cell_reg, "noise": nz,
                               "sse": {k: v[avail[k]].sum(0) for k, v in err.items()}}
    primary_set = "space_time" if withheld else "space"
    if "desk_vs_no_change" in rep["change"].get(primary_set, {}):
        rep["primary"] = {"metric": f"per-species RMSE skill on held-out same-cell change "
                                    f"({primary_set}), desk vs no_change, over species whose "
                                    f"change is resolvable above split-half noise",
                          "set": primary_set,
                          "n_resolvable": rep["change_noise"][primary_set]["n_resolvable"],
                          "viable": rep["change_viability"][primary_set]["qualified"],
                          **rep["change"][primary_set]["desk_vs_no_change"]}
    else:
        rep["primary"] = {"note": f"no {primary_set} cell passes the epoch gate"}

    # ---- extrapolation degree, per predicted row ----
    train_cells = np.unique(keys[tr_rows, :2], axis=0)
    dist_km = distance_to_training_km(tk[:, :2], train_cells, cell_km)
    novelty = (mahalanobis_novelty(F_cov[te_rows], F_cov[tr_rows]) if F_cov.shape[1]
               else np.full(len(te_rows), np.nan))
    yrs_before = np.clip(EPOCH_MODERN[0] - tk[:, 2], 0, None).astype("float64")
    # How far a withheld year lies BEYOND the training edge -- the independent variable of the
    # tempho design (1 yr past the edge for 1975, 11 for 1985, 21 for 1995 at year 1975).
    edge = (max(withheld) + 1) if withheld else None
    reach = (np.where(np.isin(tk[:, 2], withheld), edge - tk[:, 2], 0).astype("float64")
             if withheld else np.zeros(len(tk)))

    # ---- per-species table, the input to the regressions ----
    scored = row_group > 0
    det_tr, det_te = (Ytr > 0).sum(0), (Yte[scored] > 0).sum(0)
    if community_mode:
        # A community species' nearest community species must not be itself.
        cmax, ctop = cooccurrence_similarity(Ytr, Ylog[tr_rows][:, :nc],
                                             exclude=[layout["community"].index(c) for c in ev])
    else:
        cmax, ctop = cooccurrence_similarity(Ytr, Ylog[tr_rows][:, :nc])
    table = {"species_code": np.array(ev), "n_train_detections": det_tr,
             "n_heldout_detections": det_te, "prevalence_train": det_tr / max(len(tr_rows), 1),
             "cooc_max": cmax, "cooc_top5": ctop,
             "s2_desk": fits["desk"]["s2"], "n2_desk": fits["desk"]["n2"],
             "fit_ok_desk": fits["desk"]["ok"], **per_species}
    import pandas as pd
    pd.DataFrame(table).to_csv(os.path.join(out_dir, "per_species.csv"), index=False)
    with open(os.path.join(out_dir, "layout.json"), "w", encoding="utf-8") as fh:
        json.dump({"community": layout["community"], "evaluation": ev}, fh, indent=1)

    change_arrays = {}
    for sname, cs in change_store.items():
        e_rows, m_rows = cs["e_rows"], cs["m_rows"]
        change_arrays.update({
            f"change_cells_{sname}": cs["cells"], f"change_cell_block_{sname}": cs["cell_block"],
            f"change_cell_region_{sname}": cs["cell_region"],
            f"change_cell_dist_km_{sname}": np.array([dist_km[r[0]] for r in e_rows]),
            f"change_cell_novelty_{sname}": np.array([np.nanmean(novelty[r]) for r in e_rows]),
            f"change_cell_reach_{sname}": np.array([np.mean(reach[r]) for r in e_rows]),
            f"change_early_rows_{sname}": np.array([np.asarray(r) for r in e_rows], dtype=object),
            f"change_modern_rows_{sname}": np.array([np.asarray(r) for r in m_rows],
                                                    dtype=object),
            f"change_resolvable_{sname}": cs["noise"]["resolvable"],
            **{f"change_sse_{sname}_{k}": v for k, v in cs["sse"].items()}})
    np.savez_compressed(
        os.path.join(out_dir, "heldout_predictions.npz"),
        keys=tk, block_id=bid, region=reg, y=Yte.astype("float32"), species=np.array(ev),
        row_group=row_group, change_set_names=np.array(list(change_store)),
        dist_to_train_km=dist_km, cov_novelty=novelty, years_before_modern=yrs_before,
        years_beyond_training_edge=reach,
        **{f"mean_{k}": np.asarray(v[0], "float32") for k, v in preds.items()},
        **{f"var_{k}": np.asarray(v[1], "float32") for k, v in preds.items()},
        **change_arrays)

    # ---- thinning arm: the data-poor claim, test set fixed ----
    if opts["thin"]:
        cs = change_store.get(primary_set)
        rep["thinning"] = run_thinning(
            D, tr_rows, te_rows, Yte, np.where(scored)[0],
            cs["e_rows"] if cs else [], cs["m_rows"] if cs else [], opts, seed, out_dir,
            primary_set, cs["noise"] if cs else None)

    rep["elapsed_s"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=_json_default)
    _print_summary(rep)
    return rep


def run_thinning(D, tr_rows, te_rows, Yte, scored, e_rows, m_rows, opts, seed, out_dir,
                 change_set, noise):
    """Refit every predictor on a thinned training set; score on the SAME held-out rows.

    Level on ``scored`` (all held-out groups, no in-sample rows); change on the primary change
    set, on the abundance estimand, as the share of AVAILABLE change captured (resolvable species,
    ``noise`` from the full-data split-half: the truth does not change with the fraction). Saves
    per-species SSE per predictor per fraction, with each species' remaining training detections
    -- the x-axis of the data-poor curve.
    """
    from .validation_core import captured_share, epoch_values
    fracs = [1.0] + [float(f) for f in opts["thin"]]
    out = {"n_train_rows": [], "level_sse": {}, "change_sse": {}, "n_train_detections": [],
           "captured_pooled": {}}
    have_change = len(e_rows) > 0
    raw_te = np.expm1(Yte)
    if have_change:
        oe, om = epoch_values(raw_te, e_rows, m_rows)
        d_obs = om - oe
    for i, f in enumerate(fracs):
        rng = np.random.default_rng(seed + 1000 * (i + 1))
        tr = thin_rows(tr_rows, f, rng)
        print(f"[gp-species] thinning {f:g}: {len(tr):,} training rows", flush=True)
        preds, _, pnoise = fit_predict_all(
            D, tr, te_rows, {**opts, "k_nn": opts["k_nn"][:1], "raw_baseline": False}, rng,
            verbose=False)
        out["n_train_rows"].append(int(len(tr)))
        out["n_train_detections"].append((D["Y"][tr] > 0).sum(0))
        for name, (mu, var) in preds.items():
            mu = np.asarray(mu)
            ok = np.isfinite(mu[scored]).all(1)
            out["level_sse"].setdefault(name, []).append(
                ((Yte[scored][ok] - mu[scored][ok]) ** 2).sum(0))
            if have_change:
                pe, pm = epoch_values(predicted_raw(mu, pnoise[name][None, :]), e_rows, m_rows)
                dp = pm - pe
                okc = np.isfinite(dp).all(1)
                out["change_sse"].setdefault(name, []).append(
                    ((dp[okc] - d_obs[okc]) ** 2).sum(0))
                if noise is not None:
                    out["captured_pooled"].setdefault(name, []).append(
                        captured_share(d_obs, dp, noise)[1])
    arrays = {"fractions": np.array(fracs), "n_train_rows": np.array(out["n_train_rows"]),
              "n_train_detections": np.stack(out["n_train_detections"]),
              "change_set": np.array(change_set)}
    for kind in ("level_sse", "change_sse"):
        for name, v in out[kind].items():
            arrays[f"{kind}_{name}"] = np.stack(v)
    if have_change:
        arrays["change_sse_zero"] = (d_obs ** 2).sum(0)
    np.savez_compressed(os.path.join(out_dir, "thinning.npz"), **arrays)
    return {"fractions": fracs, "n_train_rows": out["n_train_rows"], "change_set": change_set,
            "captured_pooled_by_fraction": out["captured_pooled"],
            "note": "per-species curves (skill vs remaining detections) are in thinning.npz"}


def _cell_km():
    from src.config_utils import load_data_config
    return load_data_config()["grid"]["target_res_m"] / 1000.0


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _print_summary(rep):
    p = rep["primary"]
    print("\n[gp-species] PRIMARY:", p.get("metric", p.get("note")))
    if "median" in p:
        print(f"  median skill {p['median']:+.3f}  CI {p['median_ci'][0]:+.3f}..."
              f"{p['median_ci'][1]:+.3f}   share>0 {p['share_above_zero']:.2f}   balanced "
              f"{p.get('balanced_median')}   viable={p.get('viable')}   "
              f"({p['n_species_defined']} species)")
    for k, v in rep.get("baseline_shapes", {}).items():
        if not v["converged"]:
            print(f"  WARNING: {k} shape fit did not converge ({v['message']}); rerun with more "
                  "--shape-iters before reading its comparisons")
    if rep.get("completeness_gaps"):
        print(f"  WARNING: predictors with neither a result nor a reason: "
              f"{rep['completeness_gaps']}")
    for p_, why in rep.get("unavailable", {}).items():
        print(f"  unavailable: {p_}: {why}")
    for sname, cap in rep.get("captured", {}).items():
        print(f"  captured share of available change ({sname}): " + ", ".join(
            f"{k} {v['pooled']:+.3f}" for k, v in cap.items() if not k.startswith("no_change")
            and k != "intercept" and "_k" not in k))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help="default: <desk_output_dir>/gp_species")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pooled", action="store_true",
                    help="skip the cross-species amplitude-prior arms")
    ap.add_argument("--no-baselines", action="store_true",
                    help="desk / no_change / intercept / oracle only (no spacetime, covariate)")
    ap.add_argument("--n-fit", type=int, default=3000,
                    help="training rows the baselines' shared lengthscales are fitted on")
    ap.add_argument("--shape-iters", type=int, default=400,
                    help="cap on L-BFGS-B iterations for the shared shape; it stops when converged")
    ap.add_argument("--k-nn", default="32,8",
                    help="neighbours per test row for the local baselines; the first is primary, "
                         "the rest are a support-sensitivity check")
    ap.add_argument("--k-max", type=int, default=4000)
    ap.add_argument("--species", default="out_of_community", choices=SPECIES_MODES,
                    help="which species are graded; 'community' is the consistency run, written "
                         "to <run>/gp_species_community")
    ap.add_argument("--thin", default="0.3,0.1,0.03",
                    help="training fractions for the data-poor arm; 'none' to skip")
    ap.add_argument("--species-list", default=None,
                    help="file of species codes (one per line) to grade, the SAME across runs "
                         "that are compared (e.g. the three tempho overlays)")
    args = ap.parse_args()
    opts = {"baselines": not args.no_baselines, "pooled": not args.no_pooled,
            "n_fit": args.n_fit,
            "shape_iters": args.shape_iters,
            "k_nn": tuple(int(k) for k in args.k_nn.split(",") if k),
            "k_max": args.k_max, "species": args.species,
            "thin": (() if args.thin.strip().lower() in ("", "none", "0")
                     else tuple(float(f) for f in args.thin.split(",") if f))}
    if args.species_list:
        with open(args.species_list, encoding="utf-8") as fh:
            opts["species_codes"] = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    run(out_dir=args.out_dir, n_boot=args.n_boot, seed=args.seed, opts=opts)


if __name__ == "__main__":
    main()
