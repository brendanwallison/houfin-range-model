"""Route-level BBS validation: DESK vs a no-change null, in similarity space.

WHY THIS EXISTS. Every other DESK validation grades against an IDW-interpolated target: the
BBS trend product is USGS's published inverse-distance surface (see ``bbs_trend.py``), further
smoothed by ``trend.smooth_sigma_cells``. So the IDW baseline that beats DESK on held-out cells
is approximately the target's own construction operator -- the metric rewards mimicking a
smoother, and no number computed against it can settle whether DESK has real temporal skill.

Raw BBS route counts escape that circularity: they are the observations the IDW surface was
interpolated FROM. Restricting evaluation to cell-years that were *actually surveyed* cannot be
won by reproducing an interpolator.

WHAT IT MEASURES. One question: does DESK beat a no-change null at reproducing observed
community turnover at surveyed sites -- overall, and separately in a modern and an early window.

    S_true = Ruzicka(log1p observed community)          <- raw BBS, routes averaged per cell-year
    S_desk = cosine Gram of DESK z_ema
    S_nc   = cosine Gram of DESK z_ema at each cell's MODERN year, copied to all its years
    cka_gain = CKA(S_true, S_desk) - CKA(S_true, S_nc)

BOTH MODEL-SIDE MATRICES USE THE SAME FUNCTIONAL (cosine on DESK z), so the ONLY difference
between them is temporal variation. That is deliberate and load-bearing. An earlier version built
the null as Ruzicka on the observed modern community; because truth is also Ruzicka, that null was
scored in truth's own metric while DESK was scored in cosine, and the difference of the two
similarity FUNCTIONALS swamped the temporal signal (measured: a temporally-neutral model scored
-0.28 purely from the metric mismatch). The observed-space null is still reported, as
``cka_nochange_observed``, but it is a separate diagnostic and is NOT differenced against
``cka_desk``.

THE GAIN IS THE READOUT, NOT ABSOLUTE CKA. A mixed similarity matrix is dominated by
between-cell (spatial) structure -- the same reason ``Val`` MSE says little about temporal skill.
The no-change null has ZERO temporal variation by construction, so differencing against it
isolates exactly the temporal component. A large ``cka_desk`` with ``cka_gain <= 0`` means DESK
reproduced the map and nothing about change.

WHY SIMILARITY SPACE AND NOT Z SPACE. Ruzicka is invariant when *both* arguments are rescaled,
but the ESK landmarks are frozen at grid-product magnitude, so ``K(route, landmark)`` is not.
Projecting route counts into ``esk/spacetime`` would require inventing a per-species BBS->eBird
scale factor, and if it were off that artifact would dominate the result. Comparing similarity
STRUCTURES via ``linear_cka``/``mantel_r`` is rotation-invariant and scale-robust and introduces
no new estimated quantity.

    python -m src.community_encoder.train_DESK.validate_bbs_routes
"""
import argparse
import json
import os
import time

import numpy as np

from src.config_utils import load_config

from .desk_training import apply_output_ema
from .validate_spacetime import linear_cka, mantel_r, ruzicka_rect

# Window definitions. 1966, not 1965: BBS begins in 1966, so a 1965 bound would silently be a
# 1966 bound and misreport the window in the output.
MODERN_WINDOW = (2010, 2025)
EARLY_WINDOW = (1966, 1980)


# ----------------------------- pure: observed community -----------------------------

def densify_community(row, col, year, species_index, mean_count,
                      cov_row, cov_col, cov_year, n_species):
    """Long-form BBS community triples → dense per-surveyed-cell-year matrix (pure).

    ``community_matrix.npz`` stores PRESENT species only; absences are implicit. The
    authoritative row set is therefore the COVERAGE table (``cov_*``), not the presence
    triples: a surveyed cell-year where a species went unrecorded is a genuine zero, and a
    surveyed cell-year where *nothing* was recorded is a real all-zero row, not a missing one.
    Building rows from the presence triples instead would silently drop exactly the cell-years
    that carry the strongest turnover signal.

    Returns ``(X (N, n_species) float32 raw counts, keys (N, 3) int32 [row, col, year],
    n_dropped)`` ordered by ``(row, col, year)``. Presence triples for cell-years absent from
    coverage are dropped (they failed QC) and reported as ``n_dropped``.
    """
    keys = np.stack([np.asarray(cov_row, "int64"),
                     np.asarray(cov_col, "int64"),
                     np.asarray(cov_year, "int64")], axis=1)
    if keys.shape[0] == 0:
        return np.zeros((0, n_species), "float32"), np.zeros((0, 3), "int32")
    # Deduplicate + sort the coverage rows so the output order is deterministic and a repeated
    # cell-year in cov_* cannot create two rows for one site-year.
    keys = np.unique(keys, axis=0)
    order = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(keys)}

    X = np.zeros((keys.shape[0], int(n_species)), dtype="float32")
    dropped = 0
    for r, c, y, s, v in zip(np.asarray(row), np.asarray(col), np.asarray(year),
                             np.asarray(species_index), np.asarray(mean_count)):
        i = order.get((int(r), int(c), int(y)))
        if i is None:                       # present species in an uncovered cell-year
            dropped += 1
            continue
        si = int(s)
        if 0 <= si < X.shape[1]:
            X[i, si] += float(v)            # += so crosswalk lumps accumulate, matching bbs_community
    return X, keys.astype("int32"), dropped


def log1p_community(X):
    """``log1p(clip(x, 0, None))`` -- the transform the spacetime ESK was fit with.

    Matches ``trend_community.py`` (``ruzicka_log1p``, default True). Ruzicka is scale-sensitive
    per-argument, so the transform must match what training used or the similarity structure is
    not the same quantity.
    """
    return np.log1p(np.clip(np.asarray(X, "float64"), 0.0, None)).astype("float32")


def modern_reference_rows(keys, modern_window=MODERN_WINDOW):
    """Map each cell to its most recent surveyed row inside ``modern_window`` (pure).

    Returns ``(nc_src (N,) int64, keep (N,) bool)`` where ``nc_src[i]`` is the row index holding
    cell ``i``'s modern observed community and ``keep[i]`` is False for every row of a cell with
    NO surveyed year in the window. Those cells are dropped from EVERY matrix (truth included) --
    the same ``has_rec`` gate ``zspace_reconstruction`` applies, at cell granularity. Without the
    gate a no-change null cannot be defined for those sites, and silently keeping them in truth
    while excluding them from the null would compare two different row sets.
    """
    keys = np.asarray(keys)
    n = keys.shape[0]
    lo, hi = int(modern_window[0]), int(modern_window[1])
    best = {}
    for i in range(n):
        r, c, y = int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])
        if lo <= y <= hi:
            cell = (r, c)
            if cell not in best or y > best[cell][0]:
                best[cell] = (y, i)
    nc_src = np.full(n, -1, dtype="int64")
    for i in range(n):
        hit = best.get((int(keys[i, 0]), int(keys[i, 1])))
        if hit is not None:
            nc_src[i] = hit[1]
    return nc_src, nc_src >= 0


def cosine_gram(Z):
    """Row-normalized ``Z @ Z.T`` → ``(n, n)`` cosine similarity.

    Cosine, not raw dot: measured ``||Z||^2`` is 0.73-0.81 rather than the kernel contract's
    1.0, so a raw dot would attribute most of the apparent structure to that norm deficit. This
    is the same convention ``temporal_turnover_agreement`` uses for temporal comparisons.
    """
    Z = np.asarray(Z, "float64")
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / np.where(nrm > 0, nrm, 1.0)
    return Zn @ Zn.T


def bucket_metrics(S_true, S_desk, S_nc, S_nc_obs=None):
    """CKA + Mantel of DESK and the no-change null against truth, plus the gains (pure).

    ``S_desk`` and ``S_nc`` must be built with the SAME similarity functional (both cosine on
    DESK z) so their difference isolates temporal variation. ``S_nc_obs`` is the optional
    observed-space null (Ruzicka on the modern observed community); it is reported but never
    differenced against ``cka_desk``, because truth is also Ruzicka and that comparison would be
    scoring the two models with different functionals.
    """
    cka_d, cka_n = linear_cka(S_true, S_desk), linear_cka(S_true, S_nc)
    man_d, man_n = mantel_r(S_true, S_desk), mantel_r(S_true, S_nc)
    out = {
        "cka_desk": cka_d, "cka_nochange": cka_n, "cka_gain": cka_d - cka_n,
        "mantel_desk": man_d, "mantel_nochange": man_n, "mantel_gain": man_d - man_n,
        "n_rows": int(np.asarray(S_true).shape[0]),
    }
    if S_nc_obs is not None:
        # Diagnostic only -- same-functional as truth, so not comparable to cka_desk.
        out["cka_nochange_observed"] = linear_cka(S_true, S_nc_obs)
        out["mantel_nochange_observed"] = mantel_r(S_true, S_nc_obs)
    return out


def stratified_sample(keys, n_sample, rng, windows=(MODERN_WINDOW, EARLY_WINDOW)):
    """Sample row indices, guaranteeing each named window gets a share (pure).

    A uniform sample would be swamped by the modern decades (BBS coverage grows ~monotonically),
    leaving the early window too thin to report. Splits the budget evenly across the windows plus
    one "other" stratum, then backfills from the remainder if a stratum is short.
    """
    keys = np.asarray(keys)
    n = keys.shape[0]
    if n <= n_sample:
        return np.arange(n)
    yrs = keys[:, 2]
    strata, claimed = [], np.zeros(n, bool)
    for lo, hi in windows:
        m = (yrs >= lo) & (yrs <= hi) & ~claimed
        strata.append(np.where(m)[0])
        claimed |= m
    strata.append(np.where(~claimed)[0])

    per = max(1, n_sample // len(strata))
    picked = []
    for idx in strata:
        take = min(per, idx.size)
        if take:
            picked.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(picked) if picked else np.zeros(0, "int64")
    if out.size < n_sample:                                  # backfill from whatever is left
        rest = np.setdiff1d(np.arange(n), out, assume_unique=False)
        if rest.size:
            extra = rng.choice(rest, size=min(n_sample - out.size, rest.size), replace=False)
            out = np.concatenate([out, extra])
    return np.sort(out)


# ----------------------------- IO / driver -----------------------------

def load_observed(config):
    """Load + densify the BBS community matrix → ``(X_log, keys, meta)``."""
    path = config["bbs"]["community_matrix"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BBS community matrix not found at {path}. Build it with:\n"
            "    python -m src.data.preprocess.bbs_community\n"
            "(that module owns the route->cell aggregation; do not reimplement it here)")
    d = np.load(path, allow_pickle=True)
    species = [str(s) for s in d["species_codes"]]
    X_raw, keys, dropped = densify_community(
        d["row"], d["col"], d["year"], d["species_index"], d["mean_count"],
        d["cov_row"], d["cov_col"], d["cov_year"], len(species))

    # The log1p flag lives in points_meta.json, NOT the ESK meta.json. Assert rather than
    # assume: if training used raw counts, log1p here would silently compare a different space.
    zt = config.get("bbs", {}).get("z_dir", "")
    pm_path = os.path.join(zt, "points_meta.json") if zt else ""
    log1p_flag = True
    if pm_path and os.path.exists(pm_path):
        with open(pm_path, "r", encoding="utf-8") as fh:
            log1p_flag = bool(json.load(fh).get("ruzicka_log1p", True))
        if not log1p_flag:
            raise ValueError(
                f"{pm_path} reports ruzicka_log1p=false; the ESK basis was fit on RAW counts. "
                "Comparing a log1p similarity structure against it is not like-for-like.")
    else:
        print(f"[bbs-routes] WARNING: no points_meta.json at {pm_path or '<bbs.z_dir unset>'}; "
              "assuming ruzicka_log1p=true (the trend_community default)")

    meta = {"n_species": len(species), "n_surveyed_cell_years": int(keys.shape[0]),
            "presence_triples_outside_coverage": int(dropped),
            "ruzicka_log1p": log1p_flag,
            "year_range": [int(keys[:, 2].min()), int(keys[:, 2].max())] if keys.size else []}
    return log1p_community(X_raw), keys, meta


def desk_z_ema(config, keys):
    """DESK ``z_ema`` for every row of ``keys`` → ``(N, latent)``.

    The EMA is a CAUSAL scan, so a single (cell, year) cannot be smoothed in isolation. Encodes
    the involved cells over the contiguous span ``ema_warmup_start .. max(year)``, runs the scan
    along the year axis, then gathers the requested (cell, year) rows. Grades ``z_ema`` because
    that is the quantity the trainer supervised -- the cube deliberately exports raw z (the
    population model below it supplies demographic lag), so raw is the wrong comparand here.
    """
    from .validate_spacetime import encode_points

    out_dir = config["paths"]["desk_output_dir"]
    dm = np.load(os.path.join(out_dir, "desk_meta.npz"), allow_pickle=True)
    ema_on = bool(dm["output_ema"]) if "output_ema" in dm.files else False
    hl = float(dm["ema_half_life"]) if "ema_half_life" in dm.files else float("nan")
    warm = int(dm["ema_warmup_start"]) if "ema_warmup_start" in dm.files else 1940

    keys = np.asarray(keys)
    cells = np.unique(keys[:, :2], axis=0)
    y_max = int(keys[:, 2].max())
    years = list(range(min(warm, int(keys[:, 2].min())), y_max + 1))

    # One (cell, year) request per cell per year in the span; encode_points batches by year, so
    # this costs one whole-grid forward per year regardless of how many cells we ask for.
    grid = np.empty((len(cells) * len(years), 3), dtype="int32")
    for t, y in enumerate(years):
        s = t * len(cells)
        grid[s:s + len(cells), :2] = cells
        grid[s:s + len(cells), 2] = y
    t0 = time.perf_counter()
    Z, ok = encode_points(config, grid)
    print(f"[bbs-routes] encoded {len(cells)} cells x {len(years)} yr "
          f"({years[0]}-{years[-1]}) in {time.perf_counter() - t0:.1f}s", flush=True)

    L = Z.shape[1]
    stack = Z.reshape(len(years), len(cells), L)
    valid = ok.reshape(len(years), len(cells))
    if ema_on and np.isfinite(hl):
        stack = apply_output_ema(stack, hl, valid=valid)
        print(f"[bbs-routes] applied output-EMA (half-life={hl:.2f} yr, "
              f"alpha={1.0 - 2.0 ** (-1.0 / hl):.3f}) over {len(years)} years")
    else:
        print(f"[bbs-routes] WARNING: desk_meta has output_ema={ema_on}, ema_half_life={hl}; "
              "grading RAW z (no EMA available in this checkpoint)")

    cell_ix = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    year_ix = {y: t for t, y in enumerate(years)}
    Zout = np.full((keys.shape[0], L), np.nan, dtype="float32")
    for i in range(keys.shape[0]):
        Zout[i] = stack[year_ix[int(keys[i, 2])], cell_ix[(int(keys[i, 0]), int(keys[i, 1]))]]
    return Zout, {"output_ema_applied": bool(ema_on and np.isfinite(hl)),
                  "ema_half_life": hl if np.isfinite(hl) else None,
                  "ema_warmup_start": warm, "encode_years": [years[0], years[-1]]}


def run(config=None, n_sample=4000, seed=0):
    """Driver: observed vs no-change vs DESK, bucketed, written to ``desk_output_dir``."""
    config = config or load_config()
    rng = np.random.default_rng(seed)

    X_log, keys, meta = load_observed(config)
    print(f"[bbs-routes] {meta['n_surveyed_cell_years']} surveyed cell-years, "
          f"{meta['n_species']} species, years {meta['year_range']}")

    # Site gate: drop every row of any cell lacking a modern survey (no definable null).
    nc_src, keep = modern_reference_rows(keys, MODERN_WINDOW)
    n_cells_all = np.unique(keys[:, :2], axis=0).shape[0]
    X_log, keys, nc_src = X_log[keep], keys[keep], nc_src[keep]
    remap = -np.ones(len(keep), dtype="int64")
    remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
    nc_src = remap[nc_src]                                   # reindex into the kept rows
    n_cells_kept = np.unique(keys[:, :2], axis=0).shape[0]
    print(f"[bbs-routes] site gate: kept {int(keep.sum())}/{len(keep)} rows, "
          f"{n_cells_kept}/{n_cells_all} cells "
          f"({n_cells_all - n_cells_kept} dropped for no {MODERN_WINDOW[0]}-{MODERN_WINDOW[1]} survey)")
    if keys.shape[0] < 3:
        raise SystemExit("[bbs-routes] fewer than 3 rows survive the site gate; nothing to compare")

    sel = stratified_sample(keys, int(n_sample), rng)
    X_s, keys_s, nc_s = X_log[sel], keys[sel], nc_src[sel]
    keys_nc = keys[nc_s]                                     # each row's MODERN (cell, year)
    X_nc_s = X_log[nc_s]                                     # for the observed-space diagnostic
    print(f"[bbs-routes] sampled {len(sel)} rows for the {len(sel)}x{len(sel)} matrices")

    # Encode the sampled rows AND their modern references together: same cells, same year span,
    # so this is still one whole-grid forward per year. A sampled row's modern reference is
    # usually NOT itself in the sample, so it has to be requested explicitly.
    n_s = keys_s.shape[0]
    Z_both, zmeta = desk_z_ema(config, np.vstack([keys_s, keys_nc]))
    Z_s, Z_nc = Z_both[:n_s], Z_both[n_s:]

    finite = np.isfinite(Z_s).all(1) & np.isfinite(Z_nc).all(1)
    if finite.sum() < 3:
        raise SystemExit("[bbs-routes] DESK returned non-finite z for nearly every row "
                         "(covariate footprint mismatch?); nothing to compare")
    if not finite.all():
        print(f"[bbs-routes] dropping {int((~finite).sum())} rows with non-finite DESK z")
        X_s, keys_s, X_nc_s = X_s[finite], keys_s[finite], X_nc_s[finite]
        Z_s, Z_nc = Z_s[finite], Z_nc[finite]

    S_true = ruzicka_rect(X_s, X_s)
    S_desk, S_nc = cosine_gram(Z_s), cosine_gram(Z_nc)       # SAME functional -> temporal only
    S_nc_obs = ruzicka_rect(X_nc_s, X_nc_s)                  # diagnostic, different functional

    # Underpowered-comparison guard: if observed communities are all near-identical there is no
    # turnover to predict and every CKA will be ~1 regardless of model quality.
    iu = np.triu_indices_from(S_true, k=1)
    med_off = float(np.median(S_true[iu])) if iu[0].size else float("nan")
    print(f"[bbs-routes] median off-diagonal observed similarity {med_off:.4f} "
          f"(near 1.0 => underpowered)")

    ho_path = os.path.join(config["paths"]["desk_output_dir"], "holdout_cells.npy")
    has_ho = os.path.exists(ho_path)
    if has_ho:
        ho = np.load(ho_path)
        is_ho = ho[keys_s[:, 0], keys_s[:, 1]]
    else:
        print(f"[bbs-routes] WARNING: no {ho_path}; train/heldout split unavailable, "
              "reporting pooled only (every prior metric was pooled at ~83% training points)")
        is_ho = np.zeros(keys_s.shape[0], bool)

    yrs = keys_s[:, 2]
    windows = {"all": np.ones(len(yrs), bool),
               "modern": (yrs >= MODERN_WINDOW[0]) & (yrs <= MODERN_WINDOW[1]),
               "early": (yrs >= EARLY_WINDOW[0]) & (yrs <= EARLY_WINDOW[1])}
    splits = {"pooled": np.ones(len(yrs), bool)}
    if has_ho:
        splits.update({"train": ~is_ho, "heldout": is_ho})

    report = {"config": {"n_sample": int(n_sample), "seed": int(seed),
                         "modern_window": list(MODERN_WINDOW), "early_window": list(EARLY_WINDOW)},
              "observed": meta, "desk": zmeta,
              "site_gate": {"rows_kept": int(keep.sum()), "rows_total": int(len(keep)),
                            "cells_kept": int(n_cells_kept), "cells_total": int(n_cells_all),
                            "cells_dropped_no_modern": int(n_cells_all - n_cells_kept)},
              "median_offdiag_observed_similarity": med_off,
              "buckets": {}}

    for sname, smask in splits.items():
        for wname, wmask in windows.items():
            m = smask & wmask
            if m.sum() < 3:
                report["buckets"][f"{sname}/{wname}"] = {"n_rows": int(m.sum()),
                                                         "skipped": "fewer than 3 rows"}
                continue
            ix = np.where(m)[0]
            g = np.ix_(ix, ix)
            report["buckets"][f"{sname}/{wname}"] = bucket_metrics(
                S_true[g], S_desk[g], S_nc[g], S_nc_obs=S_nc_obs[g])

    out_dir = config["paths"]["desk_output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "bbs_route_validation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    np.savez_compressed(os.path.join(out_dir, "bbs_route_validation.npz"),
                        keys=keys_s, S_true=S_true.astype("float32"),
                        S_desk=S_desk.astype("float32"), S_nc=S_nc.astype("float32"),
                        S_nc_obs=S_nc_obs.astype("float32"), is_heldout=is_ho)

    print(f"\n{'bucket':<18}{'n':>7}{'cka_desk':>11}{'cka_nc':>10}{'cka_gain':>11}{'mantel_gain':>13}")
    for k, v in report["buckets"].items():
        if "skipped" in v:
            print(f"{k:<18}{v['n_rows']:>7}  skipped ({v['skipped']})")
        else:
            print(f"{k:<18}{v['n_rows']:>7}{v['cka_desk']:>11.4f}{v['cka_nochange']:>10.4f}"
                  f"{v['cka_gain']:>+11.4f}{v['mantel_gain']:>+13.4f}")
    print(f"\n[bbs-routes] report -> {out}")
    print("[bbs-routes] cka_gain > 0 means DESK beats no-change on GENUINELY OBSERVED data; "
          "<= 0 is a real negative result, not a bug.")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sample", type=int, default=4000,
                    help="rows in the similarity matrices (n^2 memory; default 4000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(n_sample=args.n_sample, seed=args.seed)


if __name__ == "__main__":
    main()
