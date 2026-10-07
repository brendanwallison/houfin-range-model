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
    esk_oracle      the observed community's ESK projection in place of z_ema -- a CEILING,
                    and an optimistic one: it is computed from the same routes as the truth

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
from .validate_bbs_routes import EPOCH_EARLY, EPOCH_MODERN, MIN_EPOCH_YEARS, epoch_gate

#: Change and level, both RMSE-based, then the probabilistic ones.
PREDICTORS = ("desk", "no_change", "intercept", "esk_oracle")
#: Skill baselines for desk. intercept is meaningless for CHANGE (it predicts none, like
#: no_change), so the change table uses no_change only.
LEVEL_BASELINES = ("no_change", "intercept")
CHANGE_BASELINES = ("no_change",)
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

    The chain is ``validate_bbs_routes.load_observed``'s with only the species list changed: route
    QC, route->cell, ``SpeciesTotal`` summed per cell-year and divided by QC route-years,
    densified against coverage so a surveyed absence is a real zero. Any change to that chain
    must happen there, not here.
    """
    import pandas as pd

    from src.config_utils import load_data_config, target_points_dir
    from src.data.preprocess import bbs
    from src.data.preprocess.bbs_community import (build_community_matrix, densify_community,
                                                   route_grid_map)

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

    species = comm + ev
    code_ix = {c: i for i, c in enumerate(species)}
    obs_all, coverage = bbs.load_usca_observations(aou_filter=None, return_coverage=True)
    routes = bbs.load_routes()
    land_mask, _, transform, crs, nx, ny = bbs.load_grid_reference(bbs.MASK_PATH)
    route_cells = route_grid_map(routes, transform, crs, nx, ny, land_mask)
    mean_df, cov_df = build_community_matrix(obs_all, coverage, xw, route_cells)
    sp = mean_df["species_code"].astype(str).str.lower()
    mean_df = mean_df[sp.isin(code_ix)]
    X_raw, keys, dropped = densify_community(
        mean_df["row"].to_numpy(), mean_df["col"].to_numpy(), mean_df["year"].to_numpy(),
        mean_df["species_code"].astype(str).str.lower().map(code_ix).to_numpy(),
        mean_df["mean_count"].to_numpy(),
        cov_df["row"].to_numpy(), cov_df["col"].to_numpy(), cov_df["year"].to_numpy(),
        len(species))
    if X_raw.shape[1] != len(species):
        raise ValueError("all-species matrix does not have the pinned column count")
    layout = {"community": comm, "evaluation": ev, "n_community": len(comm),
              "n_evaluation": len(ev), "focal_excluded": focal, "crosswalk": xdiag,
              "presence_triples_outside_coverage": int(dropped), "community_csv": community_csv}
    print(f"[gp-species] {len(ev)} evaluation species (+{len(comm)} community columns for the "
          f"oracle) over {len(keys):,} surveyed cell-years; {focal} excluded")
    return X_raw, keys, layout


# ----------------------------- splits -----------------------------

#: Held-out row groups, mirroring the existing suite's holdouts. 0 = not scored.
GROUPS = {1: "space", 2: "time", 3: "space_time"}


def row_splits(keys, holdout, buffer, block_cells, withheld_years=(), common_years=()):
    """``(is_train, group, block_id)`` per row. Pure.

    Training rows are cells neither held out nor buffer, in years the run did NOT withhold --
    exactly what the DESK checkpoint trained on. ``withheld_years`` is ``desk.trend.holdout_years``
    (the ``desk_tempho_*`` runs withhold the early decades from every cell); leaving those years in
    training would hand every GP the very years DESK never saw.

    Scored rows fall in three groups, the holdouts of the existing suite:

    * ``space`` (1): held-out block cells in trained years -- spatial extrapolation only. Every
      such row has training cells around it observed in the SAME year.
    * ``time`` (2): training cells in withheld years -- temporal extrapolation only. No cell is
      observed in those years.
    * ``space_time`` (3): held-out block cells in withheld years -- both at once.

    ``common_years`` (``common_holdout_years``) restricts groups 2 and 3 to the window every run
    withholds, so the tempho runs are comparable; withheld years outside it are neither trained
    on nor scored. Block ids follow ``augment.blocked_holdout``'s tiling (origin 0,0).
    """
    keys = np.asarray(keys)
    r, c, yr = keys[:, 0], keys[:, 1], keys[:, 2]
    ho = np.asarray(holdout, bool)[r, c]
    bf = np.asarray(buffer, bool)[r, c] if buffer is not None else np.zeros(len(keys), bool)
    wh = np.isin(yr, np.asarray(list(withheld_years), int))
    win = np.isin(yr, np.asarray(list(common_years), int)) if len(common_years) else wh
    is_train = ~ho & ~bf & ~wh
    group = np.zeros(len(keys), "int8")
    group[ho & ~wh] = 1
    group[~ho & ~bf & wh & win] = 2
    group[ho & wh & win] = 3
    b = max(1, int(block_cells))
    nbx = int(np.asarray(holdout).shape[1] + b - 1) // b
    block_id = (r // b) * nbx + (c // b)
    return is_train, group, block_id.astype("int64")


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


def pooled_skill(block_sse_model, block_sse_base, n_boot=1000, seed=0, alpha=0.05):
    """Median species skill and share of species above zero, with a two-way bootstrap. Pure.

    Inputs are ``(n_blocks, n_species)`` SSE tables. Each draw resamples held-out BLOCKS (the unit
    the split drew, so spatial correlation inside a block stays inside the draw) and, separately,
    SPECIES (the population the claim is about). Skill is recomputed from the resampled block sums,
    not averaged from per-block skills, so a block with little data weighs little.
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
    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot).astype("float64")
    med, share = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    for i in range(n_boot):
        s = skill_from_sse(W[i] @ sm, W[i] @ sb)
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


def epoch_change(values, e_rows, m_rows):
    """Per-cell change, modern epoch mean minus early epoch mean. ``(n_cells, S)``. Pure.

    Averages log1p values, NOT raw counts then log1p as ``epoch_mean_observed`` does. Here the
    GP predicts log1p row by row, and averaging the truth any other way would compare two
    different functionals of the same rows -- the mismatch this codebase measured at -0.28.
    """
    v = np.asarray(values, "float64")
    e = np.stack([v[np.asarray(r)].mean(0) for r in e_rows])
    m = np.stack([v[np.asarray(r)].mean(0) for r in m_rows])
    return e, m


def direction_by_species(obs_e, obs_m, pred_e, pred_m):
    """``species_change_agreement`` per species, ACROSS CELLS. ``[dict or None]``. Pure.

    The existing function scores species within a row; given one species' column it scores cells
    within a species, with the same majority-direction correction and the same abstain-on-zero
    rule. That transposition is the whole adaptation.
    """
    from .validate_baselines import species_change_agreement
    # Snap predicted change below ``tol`` to exactly zero. A predictor that is constant over
    # years (no_change, intercept) gives epoch means that differ only by float rounding, and
    # species_change_agreement read those 1e-16 differences as committed directions -- scoring
    # an abstaining null as a confident wrong guess (-0.54 in the first run).
    tol = 1e-9
    d = np.asarray(pred_m, "float64") - np.asarray(pred_e, "float64")
    pred_m = np.where(np.abs(d) < tol, pred_e, pred_m)
    out = []
    for s in range(obs_e.shape[1]):
        r = species_change_agreement(obs_e[:, s:s + 1], obs_m[:, s:s + 1],
                                     pred_e[:, s:s + 1], pred_m[:, s:s + 1])
        out.append(r)
    return out


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


def desk_z_both(config, keys):
    """DESK's z at every key, RAW and output-EMA'd. ``(Z_raw, Z_ema, info)``.

    DESK trains through a learned causal output EMA (demographic lag, half-life ~10 yr) and is
    supervised on z_ema, but the cube exports RAW z because the population model downstream
    supplies the lag itself. A static GP has no dynamics, so neither is exactly the deployed
    object: raw z is the literal deployed feature with no lag supplied (a lower bound), z_ema is
    raw z plus one particular lag. Both are graded. Same encode as
    ``validate_bbs_routes.desk_z_ema`` -- every involved cell over the contiguous span from
    ``ema_warmup_start``, since the scan cannot be applied to an isolated cell-year -- done once
    and returned both ways.
    """
    from .desk_training import apply_output_ema
    from .model_arch import check_basis_matches
    from .validate_spacetime import encode_points

    run_dir = config["paths"]["desk_output_dir"]
    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    _zd = (config.get("desk", {}) or {}).get("z_dir")
    if _zd:
        check_basis_matches(dm, _zd, context="gp-species")
    ema_on = bool(dm["output_ema"]) if "output_ema" in dm.files else False
    hl = float(dm["ema_half_life"]) if "ema_half_life" in dm.files else float("nan")
    warm = int(dm["ema_warmup_start"]) if "ema_warmup_start" in dm.files else 1940
    keys = np.asarray(keys)
    cells = np.unique(keys[:, :2], axis=0)
    years = list(range(min(warm, int(keys[:, 2].min())), int(keys[:, 2].max()) + 1))
    grid = np.empty((len(cells) * len(years), 3), dtype="int32")
    for t, y in enumerate(years):
        grid[t * len(cells):(t + 1) * len(cells), :2] = cells
        grid[t * len(cells):(t + 1) * len(cells), 2] = y
    Z, ok = encode_points(config, grid)
    L = Z.shape[1]
    raw = Z.reshape(len(years), len(cells), L)
    valid = ok.reshape(len(years), len(cells))
    if not (ema_on and np.isfinite(hl)):
        raise ValueError(f"desk_meta has output_ema={ema_on}, ema_half_life={hl}: this "
                         "checkpoint has no learned lag, so z_ema cannot be reconstructed")
    ema = apply_output_ema(raw, hl, valid=valid)
    info = {"ema_half_life": hl, "ema_warmup_start": warm, "encode_years": [years[0], years[-1]]}
    return (rows_from_stack(raw, years, cells, keys), rows_from_stack(ema, years, cells, keys),
            info)


def spacetime_inputs(keys, cell_km):
    """``[x_km, y_km, year]`` per key, on the model grid. Pure."""
    k = np.asarray(keys, "float64")
    return np.column_stack([k[:, 1] * cell_km, k[:, 0] * cell_km, k[:, 2]])


def cooccurrence_similarity(Y_eval, Y_comm, top=5):
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

def fit_predict_all(D, tr, te, opts, rng, verbose=True):
    """Every predictor fitted on training rows ``tr`` and predicting held-out rows ``te``.

    ``D`` holds per-row arrays over a common row set (``Y``, ``Z``, ``Z_nc``, ``F_st``,
    ``F_cov``, optionally ``Z_oracle``) and ``groups`` for the test rows. Returns
    ``(preds {name: (mean, var)}, fits)``. Shared by the main run and every thinning fraction, so
    the two cannot differ in anything but the data.
    """
    Ytr = D["Y"][tr].astype("float64")
    preds, fits = {}, {}
    m = gpk.fit(D["Z"][tr], Ytr)
    fits["desk"] = m
    preds["desk"] = gpk.predict(m, D["Z"][te])
    preds["no_change"] = gpk.predict(m, D["Z_nc"])
    if D.get("Z_raw") is not None:
        mr = gpk.fit(D["Z_raw"][tr], Ytr)
        fits["desk_raw"] = mr
        preds["desk_raw"] = gpk.predict(mr, D["Z_raw"][te])
        preds["no_change_raw"] = gpk.predict(mr, D["Z_nc_raw"])
    # Pooled arms: the same kernels with every species' amplitude under one cross-species prior
    # (gp_kernels.fit_pooled_scales). Applied to every model, so pooling cannot tilt a comparison.
    if opts.get("pooled", True):
        mp = gpk.fit(D["Z"][tr], Ytr, pool=True)
        fits["desk_pooled"] = mp
        preds["desk_pooled"] = gpk.predict(mp, D["Z"][te])
        if D.get("Z_raw") is not None:
            mrp = gpk.fit(D["Z_raw"][tr], Ytr, pool=True)
            fits["desk_raw_pooled"] = mrp
            preds["desk_raw_pooled"] = gpk.predict(mrp, D["Z_raw"][te])
    n = len(tr)
    vy = np.maximum(Ytr.var(0) * (1 + 1 / n), gpk.VAR_FLOOR)
    preds["intercept"] = (np.broadcast_to(Ytr.mean(0), (len(te), Ytr.shape[1])),
                          np.broadcast_to(vy, (len(te), Ytr.shape[1])))
    if D.get("Z_oracle") is not None:
        mo = gpk.fit(D["Z_oracle"][tr], Ytr)
        fits["esk_oracle"] = mo
        preds["esk_oracle"] = gpk.predict(mo, D["Z_oracle"][te])
    if opts.get("baselines", True):
        fit_rows = tr if len(tr) <= opts["n_fit"] else np.sort(rng.choice(tr, opts["n_fit"],
                                                                          replace=False))
        th_cov = np.full(D["F_cov"].shape[1], np.log(np.sqrt(max(D["F_cov"].shape[1], 1))))
        arms = [("spacetime", "spacetime", "F_st", np.log([300.0, 20.0])),
                ("covariate", "covariate", "F_cov", th_cov)]
        # The covariate GP matched to RAW z (input-side EMA only), so desk_raw has a like-for-like
        # rival. Main run only: it is a third shape fit, and thinning is about data, not lag.
        if opts.get("raw_baseline", True) and D.get("F_cov_raw") is not None:
            arms.append(("covariate_raw", "covariate", "F_cov_raw", th_cov))
        for label, kind, Fkey, th0 in arms:
            t0 = time.perf_counter()
            shape = gpk.fit_shared_shape(kind, D[Fkey][fit_rows], D["Y"][fit_rows],
                                         th0, n_iter=opts["shape_iters"], verbose=verbose)
            # The per-species scales from the fit subsample, but the MEAN from all training rows,
            # as for desk.
            shape["ybar"] = Ytr.mean(0)
            fits[label] = shape
            for k in opts["k_nn"]:
                name = label if k == opts["k_nn"][0] else f"{label}_k{k}"
                preds[name] = gpk.predict_local(shape, D[Fkey][tr], Ytr, D[Fkey][te],
                                                D["groups"], k=k, k_max=opts["k_max"])
            if opts.get("pooled", True):
                ps = gpk.pool_shape_scales(shape, D[Fkey][fit_rows], D["Y"][fit_rows])
                ps["ybar"] = Ytr.mean(0)
                fits[f"{label}_pooled"] = ps
                preds[f"{label}_pooled"] = gpk.predict_local(
                    ps, D[Fkey][tr], Ytr, D[Fkey][te], D["groups"], k=opts["k_nn"][0],
                    k_max=opts["k_max"])
            if verbose:
                print(f"[gp-species] {label} baseline fitted and predicted in "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    return preds, fits


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


# ----------------------------- run -----------------------------

def _load_masks(run_dir, buffer_floor=None):
    ho_p, bf_p = os.path.join(run_dir, "holdout_cells.npy"), os.path.join(run_dir,
                                                                          "buffer_cells.npy")
    if not os.path.exists(ho_p):
        raise FileNotFoundError(f"{ho_p} missing: this run has no holdout to grade on")
    ho = np.load(ho_p)
    if not ho.any():
        raise ValueError(f"{ho_p} is empty -- holdout_frac=0 (the production run?). Point "
                         "paths.desk_output_dir at a run trained with a holdout.")
    if os.path.exists(bf_p):
        return ho, np.load(bf_p), "saved buffer_cells.npy"
    # Older checkpoints (the desk_tempho_* runs) did not save the buffer. Rebuild it the way
    # training did -- a Chebyshev ring of width spatial_kernel // 2 (floored by buffer_floor)
    # around the held-out cells -- rather than letting buffer cells pass as training cells.
    from .augment import _shift2d
    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    k = int(dm["spatial_kernel"]) if "spatial_kernel" in dm.files else 0
    width = max(k // 2, int(buffer_floor or 0))
    near = np.zeros_like(ho)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            near |= _shift2d(ho, dy, dx)
    return ho, near & ~ho, f"rebuilt: Chebyshev width {width} (spatial_kernel {k})"


def _predictor_scores(Y, mean, var, block_id):
    """Per-row scores reduced to per-block sums, so any later bootstrap can reuse them."""
    blocks, sse = block_sums((Y - mean) ** 2, block_id)
    _, lpd = block_sums(gaussian_lpd(Y, mean, var), block_id)
    _, crps = block_sums(gaussian_crps(Y, mean, var), block_id)
    _, c50 = block_sums(interval_coverage(Y, mean, var, 0.5), block_id)
    _, c90 = block_sums(interval_coverage(Y, mean, var, 0.9), block_id)
    return blocks, {"sse": sse, "lpd": lpd, "crps": crps, "cov50": c50, "cov90": c90}


def _change_tables(preds, Yte, e_rows, m_rows, cell_block):
    """Per-block SSE of predicted vs observed epoch change, and direction tables, per predictor."""
    obs_e, obs_m = epoch_change(Yte, e_rows, m_rows)
    d_obs = obs_m - obs_e
    sse, dirs = {}, {}
    for name, (mu, _v) in preds.items():
        pe, pm = epoch_change(np.asarray(mu), e_rows, m_rows)
        sse[name] = block_sums((pm - pe - d_obs) ** 2, cell_block)[1]
        dirs[name] = direction_by_species(obs_e, obs_m, pe, pm)
    return sse, dirs


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
    for d, rivals in (("desk_pooled", ("spacetime_pooled", "covariate_pooled")),
                      ("desk_raw_pooled", ("covariate_raw_pooled",))):
        if d in names:
            pairs += [(d, p) for p in rivals if p in names]
    return pairs


def _score_comparisons(names, sse_by_pred, n_boot, seed, skip_refs=()):
    """Pooled skill for every comparison pair on one set of SSE tables. ``(report, per_species)``."""
    out, per = {}, {}
    for a, b in comparisons(names):
        if b in skip_refs or a not in sse_by_pred or b not in sse_by_pred:
            continue
        pooled, sk = pooled_skill(sse_by_pred[a], sse_by_pred[b], n_boot, seed)
        out[f"{a}_vs_{b}"] = pooled
        per[f"{a}_vs_{b}"] = sk
    return out, per


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


def run(config=None, out_dir=None, n_boot=1000, seed=0, opts=None):
    t0 = time.perf_counter()
    config = config or load_config()
    opts = {"baselines": True, "pooled": True, "n_fit": 3000, "shape_iters": 400,
            "k_nn": (32, 8),
            "k_max": 4000, "thin": (0.3, 0.1, 0.03), **(opts or {})}
    rng = np.random.default_rng(seed)
    run_dir = config["paths"]["desk_output_dir"]
    out_dir = out_dir or os.path.join(run_dir, "gp_species")
    os.makedirs(out_dir, exist_ok=True)
    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    latent = int(dm["latent_dim"])
    tr_cfg = (config.get("desk", {}) or {}).get("trend", {}) or {}
    block_cells = int(tr_cfg.get("block_cells", 6))
    withheld = [int(y) for y in (tr_cfg.get("holdout_years") or [])]
    common = [int(y) for y in (tr_cfg.get("common_holdout_years") or [])]
    cell_km = float(opts.get("cell_km") or _cell_km())

    from src.data.preprocess.bbs_community import log1p_community
    X_raw, keys, layout = load_all_species(config)
    nc = layout["n_community"]
    Ylog = log1p_community(X_raw)
    Y_eval, X_comm = Ylog[:, nc:], Ylog[:, :nc]

    ho, bf, buffer_note = _load_masks(run_dir, tr_cfg.get("buffer_floor"))
    is_train, group, block_id = row_splits(keys, ho, bf, block_cells, withheld, common)
    yr = keys[:, 2]
    # Training cells with a withheld early epoch also need their MODERN rows predicted (in-sample)
    # so their change can be scored: the "time" change set.
    time_cells = {(int(a), int(b)) for a, b in keys[group == 2, :2]}
    in_time_cell = np.array([(int(a), int(b)) in time_cells for a, b in keys[:, :2]], bool)
    insample = (is_train & in_time_cell & (yr >= EPOCH_MODERN[0]) & (yr <= EPOCH_MODERN[1]))
    pred = (group > 0) | insample
    use = is_train | pred
    print(f"[gp-species] withheld years: {withheld[:1] + ['...'] + withheld[-1:] if withheld else 'none'}"
          f"; scored rows: " + ", ".join(f"{GROUPS[g]} {int((group == g).sum()):,}"
                                         for g in GROUPS) + f"; buffer {buffer_note}")

    # One encode for everything: rows to fit and predict, plus the modern epoch of every predicted
    # cell for the no-change reference.
    ref_keys = modern_reference_keys(keys[pred])
    want = np.concatenate([keys[use], ref_keys]).astype("int32")
    want_u, inv = np.unique(want, axis=0, return_inverse=True)
    Zu_raw, Zu, zinfo = desk_z_both(config, want_u)
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
    # forms), finite no-change z.
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
    dropped_species = [c for c, k in zip(layout["evaluation"], keep_sp) if not k]
    Y_eval = Y_eval[:, keep_sp]
    layout = {**layout, "evaluation": [c for c, k in zip(layout["evaluation"], keep_sp) if k]}
    print(f"[gp-species] dropped {len(dropped_species)} species with no training detections; "
          f"{int(keep_sp.sum())} evaluated")
    print(f"[gp-species] {len(tr_rows):,} training rows, {len(te_rows):,} predicted rows "
          f"({int((row_group == 0).sum()):,} in-sample modern rows of time-group cells; "
          f"{n_drop} dropped: outside the covariate footprint or no modern z)")

    Z_oracle, oracle_note = None, None
    try:
        from .esk_kernel import project_points_to_z
        Z_oracle = project_points_to_z(X_comm, config["desk"]["z_dir"], latent)
        if Z_oracle is None:
            oracle_note = "no ESK projection saved in desk.z_dir"
    except Exception as exc:                         # a missing ceiling must not sink the run
        oracle_note = f"esk_oracle unavailable: {exc}"
    if oracle_note:
        print(f"[gp-species] {oracle_note}")

    bid = block_id[te_rows]
    D = {"Y": Y_eval, "Z": Z, "Z_nc": Z_nc_all[te_rows], "Z_raw": Z_raw,
         "Z_nc_raw": Z_nc_raw[te_rows], "F_st": F_st, "F_cov": F_cov, "F_cov_raw": F_cov_raw,
         "Z_oracle": Z_oracle, "groups": bid}
    preds, fits = fit_predict_all(D, tr_rows, te_rows, opts, rng)
    Yte = Y_eval[te_rows].astype("float64")
    Ytr = Y_eval[tr_rows]
    names = list(preds)
    tk = keys[te_rows]
    ev = layout["evaluation"]

    rep = {"run_dir": run_dir, "basis": config["desk"].get("z_dir"),
           "ema": zinfo, "best_epoch_of_checkpoint": int(dm["best_epoch"])
           if "best_epoch" in dm.files else None,
           "withheld_years": withheld, "common_holdout_years": common, "buffer": buffer_note,
           "selection_caveat": ("the checkpoint's epoch was selected on a held-out kernel metric "
                                "over these same blocks: a mild optimistic bias for desk"),
           "covariate_gp_caveat": ("the covariate GP sees each cell's own covariates; desk also "
                                   "sees a 5x5 neighbourhood through its convolution"),
           "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opts.items()},
           "layout": {k: v for k, v in layout.items() if k not in ("community", "evaluation")},
           "rows": {"train": int(len(tr_rows)), "predicted": int(len(te_rows)),
                    "dropped": n_drop, "in_sample_modern": int((row_group == 0).sum()),
                    **{GROUPS[g]: int((row_group == g).sum()) for g in GROUPS}},
           "dropped_zero_training_detections": {"n": len(dropped_species),
                                                "species": dropped_species},
           "pooled_priors": {k: v["prior"] for k, v in fits.items() if "prior" in v},
           "oracle_note": oracle_note or
           "esk_oracle is computed from the same routes as the truth: an optimistic ceiling",
           "baseline_shapes": {k: shape_summary(v) for k, v in fits.items() if "theta" in v},
           "primary": {}, "change": {}, "level": {}, "probabilistic": {}, "direction": {},
           "epoch_gate": {}}
    per_species = {}

    # ---- level, per held-out group (never on in-sample rows) ----
    level_sse = {}
    for g, gname in GROUPS.items():
        sel = np.where(row_group == g)[0]
        if len(sel) == 0:
            continue
        scores = {}
        for name, (mu, var) in preds.items():
            _, scores[name] = _predictor_scores(Yte[sel], np.asarray(mu)[sel],
                                                np.asarray(var)[sel], bid[sel])
        level_sse[gname] = {k: v["sse"] for k, v in scores.items()}
        rep["level"][gname], per = _score_comparisons(names, level_sse[gname], n_boot, seed)
        per_species.update({f"level_skill_{gname}_{k}": v for k, v in per.items()})
        n_g = float(len(sel))
        prob = {}
        for name, sc in scores.items():
            prob[name] = {
                "median_lpd_per_row": float(np.median(sc["lpd"].sum(0) / n_g)),
                "median_crps": float(np.median(sc["crps"].sum(0) / n_g)),
                "coverage50_pooled": float(sc["cov50"].sum() / (n_g * sc["cov50"].shape[1])),
                "coverage90_pooled": float(sc["cov90"].sum() / (n_g * sc["cov90"].shape[1]))}
        for name in [p for p in scores if p != "desk"]:
            dl = (scores["desk"]["lpd"].sum(0) - scores[name]["lpd"].sum(0)) / n_g
            prob[f"desk_minus_{name}_lpd_per_row"] = {
                "median": float(np.median(dl)), "share_above_zero": float((dl > 0).mean())}
        rep["probabilistic"][gname] = prob

    # ---- change, per change set ----
    sets = change_sets(tk, row_group, withheld)
    change_store, change_sse = {}, {}
    for sname, rows in sets.items():
        cells, e_loc, m_loc, gate = epoch_gate(tk[rows], EPOCH_EARLY, EPOCH_MODERN,
                                               MIN_EPOCH_YEARS)
        rep["epoch_gate"][sname] = gate
        if len(cells) == 0:
            continue
        e_rows = [rows[np.asarray(r)] for r in e_loc]
        m_rows = [rows[np.asarray(r)] for r in m_loc]
        cell_block = bid[[r[0] for r in e_rows]]
        sse, dirs = _change_tables(preds, Yte, e_rows, m_rows, cell_block)
        change_sse[sname] = sse
        rep["change"][sname], per = _score_comparisons(names, sse, n_boot, seed,
                                                       skip_refs=("intercept",))
        per_species.update({f"change_skill_{sname}_{k}": v for k, v in per.items()})
        rep["direction"][sname], per = _direction_report(dirs)
        per_species.update({f"direction_skill_{sname}_{k}": v for k, v in per.items()})
        change_store[sname] = {"cells": cells, "e_rows": e_rows, "m_rows": m_rows,
                               "cell_block": cell_block}
    primary_set = "space_time" if withheld else "space"
    if "desk_vs_no_change" in rep["change"].get(primary_set, {}):
        rep["primary"] = {"metric": f"per-species RMSE skill on held-out same-cell change "
                                    f"({primary_set}), desk vs no_change",
                          "set": primary_set, **rep["change"][primary_set]["desk_vs_no_change"]}
    else:
        rep["primary"] = {"note": f"no {primary_set} cell passes the epoch gate"}

    # ---- extrapolation degree, per predicted row and per change cell ----
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
    cmax, ctop = cooccurrence_similarity(Ytr, X_comm[tr_rows])
    table = {"species_code": np.array(ev), "n_train_detections": det_tr,
             "n_heldout_detections": det_te, "prevalence_train": det_tr / max(len(tr_rows), 1),
             "cooc_max": cmax, "cooc_top5": ctop,
             "s2_desk": fits["desk"]["s2"], "n2_desk": fits["desk"]["n2"],
             "fit_ok_desk": fits["desk"]["ok"], **per_species}
    import pandas as pd
    pd.DataFrame(table).to_csv(os.path.join(out_dir, "per_species.csv"), index=False)
    with open(os.path.join(out_dir, "layout.json"), "w", encoding="utf-8") as fh:
        json.dump({"community": layout["community"], "evaluation": ev}, fh, indent=1)

    # Per-row predictions, groups, extrapolation degrees, and per-set change bookkeeping, so the
    # noise ceiling, the regressions and any re-pooling run without re-encoding.
    change_arrays = {}
    for sname, cs in change_store.items():
        e_rows, m_rows = cs["e_rows"], cs["m_rows"]
        change_arrays.update({
            f"change_cells_{sname}": cs["cells"], f"change_cell_block_{sname}": cs["cell_block"],
            f"change_cell_dist_km_{sname}": np.array([dist_km[r[0]] for r in e_rows]),
            f"change_cell_novelty_{sname}": np.array([np.nanmean(novelty[r]) for r in e_rows]),
            f"change_cell_reach_{sname}": np.array([np.mean(reach[r]) for r in e_rows]),
            f"change_early_rows_{sname}": np.array([np.asarray(r) for r in e_rows], dtype=object),
            f"change_modern_rows_{sname}": np.array([np.asarray(r) for r in m_rows],
                                                    dtype=object),
            **{f"change_sse_{sname}_{k}": v for k, v in change_sse[sname].items()}})
    np.savez_compressed(
        os.path.join(out_dir, "heldout_predictions.npz"),
        keys=tk, block_id=bid, y=Yte.astype("float32"), species=np.array(ev),
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
            primary_set)

    rep["elapsed_s"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=_json_default)
    _print_summary(rep)
    return rep


def run_thinning(D, tr_rows, te_rows, Yte, scored, e_rows, m_rows, opts, seed, out_dir,
                 change_set):
    """Refit every predictor on a thinned training set; score on the SAME held-out rows.

    Level is scored on ``scored`` (all held-out groups, no in-sample rows); change on the primary
    change set. Saves per-species SSE per predictor per fraction, with each species' remaining
    training detections -- the x-axis of the data-poor curve. Skill against no_change on change
    needs no refit of the reference: no_change predicts zero change at every fraction.
    """
    fracs = [1.0] + [float(f) for f in opts["thin"]]
    out = {"fractions": fracs, "n_train_rows": [], "level_sse": {}, "change_sse": {},
           "n_train_detections": []}
    have_change = len(e_rows) > 0
    if have_change:
        oe, om = epoch_change(Yte, e_rows, m_rows)
        d_obs = om - oe
        out["change_sse_zero"] = (d_obs ** 2).sum(0)
    for i, f in enumerate(fracs):
        rng = np.random.default_rng(seed + 1000 * (i + 1))
        tr = thin_rows(tr_rows, f, rng)
        print(f"[gp-species] thinning {f:g}: {len(tr):,} training rows", flush=True)
        preds, _ = fit_predict_all(D, tr, te_rows,
                                   {**opts, "k_nn": opts["k_nn"][:1], "raw_baseline": False},
                                   rng, verbose=False)
        out["n_train_rows"].append(int(len(tr)))
        out["n_train_detections"].append((D["Y"][tr] > 0).sum(0))
        for name, (mu, _v) in preds.items():
            mu = np.asarray(mu)
            out["level_sse"].setdefault(name, []).append(
                ((Yte[scored] - mu[scored]) ** 2).sum(0))
            if have_change:
                pe, pm = epoch_change(mu, e_rows, m_rows)
                out["change_sse"].setdefault(name, []).append(((pm - pe - d_obs) ** 2).sum(0))
    arrays = {"fractions": np.array(fracs), "n_train_rows": np.array(out["n_train_rows"]),
              "n_train_detections": np.stack(out["n_train_detections"]),
              "change_set": np.array(change_set)}
    for kind in ("level_sse", "change_sse"):
        for name, v in out[kind].items():
            arrays[f"{kind}_{name}"] = np.stack(v)
    if "change_sse_zero" in out:
        arrays["change_sse_zero"] = out["change_sse_zero"]
    np.savez_compressed(os.path.join(out_dir, "thinning.npz"), **arrays)
    summary = {}
    for name, v in out["change_sse"].items():
        sk = [skill_from_sse(s_, out["change_sse_zero"]) for s_ in v]
        summary[name] = [float(np.nanmedian(s_)) if np.isfinite(s_).any() else None for s_ in sk]
    return {"fractions": fracs, "n_train_rows": out["n_train_rows"], "change_set": change_set,
            "median_change_skill_vs_no_change": summary,
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
              f"{p['median_ci'][1]:+.3f}   share>0 {p['share_above_zero']:.2f}  "
              f"CI {p['share_above_zero_ci'][0]:.2f}...{p['share_above_zero_ci'][1]:.2f}   "
              f"({p['n_species_defined']} species defined, {p['n_species_undefined']} undefined)")
    for k, v in rep.get("baseline_shapes", {}).items():
        if not v["converged"]:
            print(f"  WARNING: {k} shape fit did not converge ({v['message']}); rerun with more "
                  "--shape-iters before reading its comparisons")
    for sec in ("change", "level"):
        for grp, pairs in rep[sec].items():
            for k, v in pairs.items():
                if "median" in v:
                    print(f"  {sec:6s} {grp:10s} {k:34s} median {v['median']:+.3f}  "
                          f"share>0 {v['share_above_zero']:.2f}")


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
    ap.add_argument("--thin", default="0.3,0.1,0.03",
                    help="training fractions for the data-poor arm; 'none' to skip")
    args = ap.parse_args()
    opts = {"baselines": not args.no_baselines, "pooled": not args.no_pooled,
            "n_fit": args.n_fit,
            "shape_iters": args.shape_iters,
            "k_nn": tuple(int(k) for k in args.k_nn.split(",") if k),
            "k_max": args.k_max,
            "thin": (() if args.thin.strip().lower() in ("", "none", "0")
                     else tuple(float(f) for f in args.thin.split(",") if f))}
    run(out_dir=args.out_dir, n_boot=args.n_boot, seed=args.seed, opts=opts)


if __name__ == "__main__":
    main()
