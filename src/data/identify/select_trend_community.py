"""Select the reference community from the species present in BOTH trend products.

The community-encoder trend target needs species that have (a) a USGS BBS trend
raster ``tr{AOU}.tif`` and (b) an eBird Status & Trends *trends* product, with
House Finch excluded (it is the transfer target). This is a **different set** than
the weekly community, which instead required 52-week-complete eBird abundance.

Pipeline::

    candidate pool (reference_community_ranked.csv: axis values + migration class)
      -> crosswalk to AOU (scientific-name join, bbs_crosswalk)            [local]
      -> keep those whose AOU has a tr{AOU}.tif in the BBS trend dir        [local]
      -> keep those with a non-empty eBird trends listing (REST) + season   [network, cached]
      -> per-axis top-N with a migration rank penalty (community_axes)
      -> print + write community_trend.csv

**The gates come before the cut.** The old rule walked a single composite ranking
applying the gates inline and stopped at 100 -- correct for a scalar ranking, wrong
for per-axis selection, where each axis would otherwise lose a different unbalanced
number of species to the gates. So the whole pool is gated first, then each axis
takes its top-N from the survivors, and the per-axis gate attrition is reported.

Because gating first means probing every candidate rather than only the ~120 the
old walk reached, the eBird REST listing is cached to CSV. That also makes
selection offline and repeatable, which matters if the rule is going to be
iterated -- which is the point of making it explicit.

The eBird presence test reuses the same REST listing the trends downloader uses
(``acquire.ebird.list_trend_objkeys``); it is the practical stand-in for the
``ebirdst_runs$has_trends`` flag and also yields each species' modelled season.

    python -m src.data.identify.select_trend_community --bbs-species <SpeciesList.csv>
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from src.config_utils import load_config, load_data_config
from src.data.identify import community_axes as ca
from src.data.identify.bbs_crosswalk import build_crosswalk, read_community_codes

_TR_RE = re.compile(r"tr0*(\d+)\.tif$", re.IGNORECASE)
_SEASON_RE = re.compile(r"/trends/[a-z0-9]+_([a-z]+)_ebird-trends_", re.IGNORECASE)

# Axis-value columns the pool artifact must carry (written by identify/avonet.py).
POOL_VALUE_COLS = ("migration", "phylo_distance", "trait_distance", "urban_tolerance")


def bbs_trend_aou_set(trend_dir):
    """Set of int AOU codes that have a ``tr{AOU}.tif`` in ``trend_dir``."""
    aou = set()
    if not os.path.isdir(trend_dir):
        return aou
    for name in os.listdir(trend_dir):
        m = _TR_RE.search(name)
        if m:
            aou.add(int(m.group(1)))
    return aou


def _season_from_objkey(objkey):
    m = _SEASON_RE.search(objkey)
    return m.group(1).lower() if m else None


def code_to_aou_map(matched, bbs_aou_set):
    """eBird ``species_code`` -> a single AOU that HAS a trend raster (or None).

    ``matched`` is the crosswalk frame ``[aou, species_code, sci_norm]``. A code
    may map to several AOUs (a lump); prefer one whose ``tr{AOU}.tif`` exists.
    """
    out = {}
    for code, grp in matched.groupby("species_code"):
        aous = [int(a) for a in grp["aou"].tolist()]
        with_trend = [a for a in aous if a in bbs_aou_set]
        out[code] = (with_trend[0] if with_trend else None,
                     aous[0] if aous else None,
                     grp["sci_norm"].iloc[0])
    return out


def load_pool(ranked_path, exclude=None):
    """The candidate pool as a dict of aligned lists, ready for ``community_axes``.

    Requires the axis-value columns; a pool artifact predating them is refused
    rather than silently falling back to the composite rank, because that fallback
    would reinstate the old selection rule with no visible symptom.
    """
    df = pd.read_csv(ranked_path)
    missing = [c for c in POOL_VALUE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{ranked_path} lacks {missing}. It predates per-axis selection -- "
            f"re-run `python -m src.data.identify.avonet` to rewrite the candidate "
            f"pool with the axis values and the migration class.")
    codes = read_community_codes(ranked_path, top_n=None, exclude=exclude)
    df = (df.dropna(subset=["species_code"])
            .assign(species_code=lambda d: d["species_code"].astype(str).str.lower())
            .drop_duplicates("species_code")
            .set_index("species_code"))
    keep = [c for c in codes if c in df.index]
    sub = df.loc[keep]
    pool = {"species_code": keep, "migration": sub["migration"].astype(int).to_numpy()}
    for col in POOL_VALUE_COLS[1:]:
        pool[col] = sub[col].astype("float64").to_numpy()
    return pool


def subset_pool(pool, codes):
    """The pool restricted to ``codes``, preserving pool order."""
    want = set(codes)
    idx = [i for i, c in enumerate(pool["species_code"]) if c in want]
    out = {"species_code": [pool["species_code"][i] for i in idx]}
    for k, v in pool.items():
        if k != "species_code":
            out[k] = np.asarray(v)[idx]
    return out


def gate_pool(pool, c2a, ebird_has_trends_fn, verbose=True):
    """Restrict the pool to species present in BOTH trend products.

    Returns ``(gated_pool, attrs, skipped)`` where ``attrs[code]`` carries the
    ``aou``/``season``/``sci_norm`` the output CSV needs.
    """
    kept, attrs = [], {}
    skipped = {"no_aou": [], "no_bbs_trend": [], "no_ebird_trend": []}
    for code in pool["species_code"]:
        aou_t, aou_any, sci = c2a.get(code, (None, None, None))
        if aou_any is None:
            skipped["no_aou"].append(code)
            continue
        if aou_t is None:                        # no BBS trend raster for this species
            skipped["no_bbs_trend"].append(code)
            continue
        objkey = ebird_has_trends_fn(code)       # network: only for BBS-present candidates
        if not objkey:
            skipped["no_ebird_trend"].append(code)
            continue
        kept.append(code)
        attrs[code] = {"aou": aou_t, "season": _season_from_objkey(objkey) or "",
                       "sci_norm": sci}
    if verbose:
        print(f"[select] gated pool: {len(kept)} of {len(pool['species_code'])} candidates "
              f"in both trend products")
        print(f"[select]   dropped {len(skipped['no_bbs_trend'])} without a BBS trend raster, "
              f"{len(skipped['no_ebird_trend'])} without an eBird trends product, "
              f"{len(skipped['no_aou'])} with no AOU crosswalk")
    return subset_pool(pool, kept), attrs, skipped


def gate_attrition(pool, gated_pool, n_per_axis, p, axes=ca.AXES):
    """Per axis, which of its ungated top-N the gates removed.

    Gating before the cut means each axis silently backfills from deeper in the
    pool. That is the correct behaviour -- but it has to be visible, or an axis
    starved by the gates looks identical to one that was well served.
    """
    survivors = set(gated_pool["species_code"])
    out = {}
    for ax in axes:
        ungated = ca.select_axis(pool["species_code"], pool[ax["column"]],
                                 pool["migration"], ax["ascending"], n_per_axis, p)
        lost = [c for c in ungated["codes"] if c not in survivors]
        out[ax["name"]] = {"n_lost": len(lost), "lost": lost}
    return out


def _probe_cache_load(path, year):
    """``{code: objkey_or_empty}`` from a probe cache written for ``year``."""
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "year" in df.columns:
        df = df[df["year"].astype(int) == int(year)]
    return {str(r.species_code).lower(): ("" if pd.isna(r.objkey) else str(r.objkey))
            for r in df.itertuples()}


def _probe_cache_save(path, year, probes):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([{"species_code": c, "year": int(year), "objkey": v or ""}
                  for c, v in sorted(probes.items())]).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbs-species", default=None, help="BBS SpeciesList CSV (AOU + Genus/Species).")
    ap.add_argument("--ebird-taxonomy", default=None)
    ap.add_argument("--ranked", default=None, help="reference_community_ranked.csv (candidate pool)")
    ap.add_argument("--trend-dir", default=None, help="Dir of BBS tr{AOU}.tif (default: config).")
    ap.add_argument("--n-per-axis", type=int, default=None,
                    help="top-N per axis (default: config bbs.community_axes.n_per_axis)")
    ap.add_argument("--max-migratory-frac", type=float, default=None,
                    help="target ceiling on the migratory fraction of the union")
    ap.add_argument("--penalty", type=float, default=None,
                    help="fix the rank penalty instead of solving for the target")
    ap.add_argument("--year", type=int, default=None, help="eBird trends version year (default: config, 2022).")
    ap.add_argument("--probe-cache", default=None,
                    help="CSV cache of the eBird trends REST probe (default: alongside --ranked)")
    ap.add_argument("--refresh-probes", action="store_true",
                    help="ignore the probe cache and re-query eBird")
    ap.add_argument("--out", default=None, help="Output community CSV (default: community_trend_list).")
    args = ap.parse_args()

    dcfg = load_data_config()
    cfg = load_config()
    dr = dcfg["datasets_root"]
    acfg = cfg.get("bbs", {}).get("community_axes", {}) or {}
    n_per_axis = args.n_per_axis or int(acfg.get("n_per_axis", 30))
    max_migr = args.max_migratory_frac if args.max_migratory_frac is not None \
        else float(acfg.get("max_migratory_frac", 0.20))
    p_max = int(acfg.get("penalty_max", 120))
    ebird_tax = args.ebird_taxonomy or os.path.join(dr, "avonet", "eBird_taxonomy.csv")
    ranked = args.ranked or dcfg.get("species_list") or os.path.join(dr, "avonet", "reference_community_ranked.csv")
    bbs_species = args.bbs_species or os.path.join(dr, "bbs_2026_release", "SpeciesList.csv")
    trend_dir = args.trend_dir or os.path.join(dr, dcfg["sciencebase"]["out_subdirs"]["bbs_trends"])
    out = args.out or dcfg.get("community_trend_list") or os.path.join(dr, "avonet", "community_trend.csv")
    year = args.year or int(dcfg.get("ebird_trends_version_year", 2022))
    probe_cache = args.probe_cache or os.path.join(os.path.dirname(ranked) or ".",
                                                  "ebird_trends_probe.csv")

    pool = load_pool(ranked)
    pc = ca.composition(pool["migration"])
    print(f"[select] candidate pool: {len(pool['species_code'])} species; "
          + ", ".join(f"{ca.MIGRATION_LABELS[c]} {pc['fracs'][c]:.0%}" for c in (1, 2, 3)))

    matched, _ = build_crosswalk(bbs_species, ebird_tax, ranked, top_n=None,
                                 community_codes=pool["species_code"])
    bbs_aou = bbs_trend_aou_set(trend_dir)
    print(f"[select] {len(bbs_aou)} BBS trend rasters in {trend_dir}")
    c2a = code_to_aou_map(matched, bbs_aou)

    # eBird presence via the trends REST listing, cached across runs.
    from src.data.acquire.ebird import list_trend_objkeys, resolve_key
    probes = {} if args.refresh_probes else _probe_cache_load(probe_cache, year)
    if probes:
        print(f"[select] loaded {len(probes)} cached eBird trend probes from {probe_cache}")
    key = None

    def has_ebird_trend(code):
        nonlocal key
        if code not in probes:
            if key is None:
                key = resolve_key()
            oks = list_trend_objkeys(code, year, key)
            probes[code] = oks[0] if oks else ""
        return probes[code]

    gated, attrs, skipped = gate_pool(pool, c2a, has_ebird_trend)
    _probe_cache_save(probe_cache, year, probes)

    gc = ca.composition(gated["migration"])
    print(f"[select] gated composition: "
          + ", ".join(f"{ca.MIGRATION_LABELS[c]} {gc['fracs'][c]:.0%}" for c in (1, 2, 3)))

    if args.penalty is not None:
        sel = ca.select_union(gated, n_per_axis, args.penalty)
    else:
        sel, trace = ca.solve_penalty(gated, n_per_axis, max_migr, p_max=p_max)
        print(f"[select] solved the rank penalty over p=0..{p_max}: "
              f"{trace[0]['migratory_frac']:.0%} migratory at p=0 -> "
              f"{trace[-1]['migratory_frac']:.0%} at p={trace[-1]['p']}")
    print(ca.format_selection(sel))
    if not sel.get("target_met", True):
        print(f"[select] WARNING migratory fraction {sel['composition']['fracs'][3]:.0%} "
              f"still exceeds the {max_migr:.0%} target at p={p_max}; raise "
              f"bbs.community_axes.penalty_max or relax the target.")

    for name, a in gate_attrition(pool, gated, n_per_axis, sel["penalty"]).items():
        if a["n_lost"]:
            print(f"[select] gate attrition on {name}: {a['n_lost']}/{n_per_axis} of the "
                  f"ungated top-N lack a trend product ({a['lost'][:6]}"
                  + (" ..." if a["n_lost"] > 6 else "") + "); backfilled from deeper")

    # mean_rank is the synthesized round-robin order over the union: a union has no
    # natural total order, but three readers sort the community on this column.
    rows = []
    for i, code in enumerate(sel["codes"], start=1):
        a = attrs[code]
        mig = int(gated["migration"][gated["species_code"].index(code)])
        rows.append({"species_code": code, "aou": a["aou"], "season": a["season"],
                     "sci_norm": a["sci_norm"], "mean_rank": i,
                     "category": "|".join(sel["category"][code]), "migration": mig})
    df = pd.DataFrame(rows, columns=["species_code", "aou", "season", "sci_norm",
                                     "mean_rank", "category", "migration"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[select] wrote {len(df)} community species -> {out}")

    n_multi = sum(1 for c in sel["category"].values() if len(c) > 1)
    slots = n_per_axis * len(ca.AXES)
    print(f"[select] {n_multi} species were chosen by more than one axis "
          f"(union {len(df)} of a possible {slots})")

    # Community SIZE is now an outcome, not a target: the old rule walked deeper
    # until it had exactly top_n, whereas four fixed-N axes drawn from a gated pool
    # overlap more as the pool shrinks. Say so, with the knob, rather than leaving
    # a quietly smaller community to be discovered downstream.
    if len(df) < 0.8 * slots:
        print(f"[select] NOTE the union is {len(df)}, well short of the {slots} axis "
              f"slots -- a gated pool of {len(gated['species_code'])} forces the axes to "
              f"overlap. Raise bbs.community_axes.n_per_axis to grow the community.")


if __name__ == "__main__":
    main()
