"""Select the reference community present in BOTH trend products.

The community-encoder trend target needs species that have (a) a USGS BBS trend
raster ``tr{AOU}.tif`` and (b) an eBird Status & Trends *trends* product, taken
as the top-N by House-Finch similarity (the avonet ranking) with House Finch
itself excluded (it is the transfer target). This is a **different set** than the
weekly community, which instead required 52-week-complete eBird abundance.

Pipeline::

    ranked eBird codes (reference_community_ranked.csv, focal dropped)
      -> crosswalk to AOU (scientific-name join, bbs_crosswalk)           [local]
      -> keep those whose AOU has a tr{AOU}.tif in the BBS trend dir       [local]
      -> keep those with a non-empty eBird trends listing (REST) + season  [network]
      -> take the top-N by rank; print + write community_trend.csv

The eBird presence test reuses the same REST listing the trends downloader uses
(``acquire.ebird.list_trend_objkeys``); it is the practical stand-in for the
``ebirdst_runs$has_trends`` flag and also yields each species' modelled season.

    python -m src.data.identify.select_trend_community --bbs-species <SpeciesList.csv>
"""
import argparse
import os
import re

import pandas as pd

from src.config_utils import load_config, load_data_config
from src.data.identify.bbs_crosswalk import build_crosswalk, read_community_codes

_TR_RE = re.compile(r"tr0*(\d+)\.tif$", re.IGNORECASE)
_SEASON_RE = re.compile(r"/trends/[a-z0-9]+_([a-z]+)_ebird-trends_", re.IGNORECASE)


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


def select(ranked_codes, ranked_rank, c2a, ebird_has_trends_fn, top_n, verbose=True):
    """Walk ranked codes; keep those present in both products; stop at ``top_n``.

    ``c2a`` maps code -> (aou_with_trend | None, any_aou | None, sci_norm).
    ``ebird_has_trends_fn(code)`` returns the eBird trends objkey (truthy) or None.
    Returns ``(selected_rows, skipped)`` where each row is a dict.
    """
    selected, skipped = [], {"no_aou": [], "no_bbs_trend": [], "no_ebird_trend": []}
    for code in ranked_codes:
        if len(selected) >= top_n:
            break
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
        selected.append({
            "species_code": code, "aou": aou_t,
            "season": _season_from_objkey(objkey) or "",
            "sci_norm": sci, "mean_rank": ranked_rank.get(code, float("nan")),
        })
        if verbose:
            print(f"  [{len(selected):>3}] {code:<8} AOU {aou_t:<6} "
                  f"{selected[-1]['season']:<11} rank {selected[-1]['mean_rank']:.1f}")
    return selected, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbs-species", default=None, help="BBS SpeciesList CSV (AOU + Genus/Species).")
    ap.add_argument("--ebird-taxonomy", default=None)
    ap.add_argument("--ranked", default=None, help="reference_community_ranked.csv")
    ap.add_argument("--trend-dir", default=None, help="Dir of BBS tr{AOU}.tif (default: config).")
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--year", type=int, default=None, help="eBird trends version year (default: config, 2022).")
    ap.add_argument("--out", default=None, help="Output community CSV (default: community_trend_list).")
    args = ap.parse_args()

    dcfg = load_data_config()
    cfg = load_config()
    dr = dcfg["datasets_root"]
    top_n = args.top_n or cfg.get("bbs", {}).get("community_top_n", 100)
    ebird_tax = args.ebird_taxonomy or os.path.join(dr, "avonet", "eBird_taxonomy.csv")
    ranked = args.ranked or dcfg.get("species_list") or os.path.join(dr, "avonet", "reference_community_ranked.csv")
    bbs_species = args.bbs_species or os.path.join(dr, "bbs_2026_release", "SpeciesList.csv")
    trend_dir = args.trend_dir or os.path.join(dr, dcfg["sciencebase"]["out_subdirs"]["bbs_trends"])
    out = args.out or dcfg.get("community_trend_list") or os.path.join(dr, "avonet", "community_trend.csv")
    year = args.year or int(dcfg.get("ebird_trends_version_year", 2022))

    # Ranked codes (focal excluded) + their ranks; crosswalk the FULL community to AOU.
    ranked_codes = read_community_codes(ranked, top_n=None)
    ranked_rank = (pd.read_csv(ranked).dropna(subset=["species_code"])
                   .assign(species_code=lambda d: d["species_code"].astype(str).str.lower())
                   .set_index("species_code")["mean_rank"].to_dict())
    matched, _ = build_crosswalk(bbs_species, ebird_tax, ranked, top_n=None,
                                 community_codes=ranked_codes)

    bbs_aou = bbs_trend_aou_set(trend_dir)
    print(f"[select] {len(bbs_aou)} BBS trend rasters in {trend_dir}")
    c2a = code_to_aou_map(matched, bbs_aou)

    # eBird presence via the trends REST listing (only for BBS-present candidates).
    from src.data.acquire.ebird import list_trend_objkeys, resolve_key
    key = resolve_key()
    _cache = {}

    def has_ebird_trend(code):
        if code not in _cache:
            oks = list_trend_objkeys(code, year, key)
            _cache[code] = oks[0] if oks else None
        return _cache[code]

    print(f"[select] walking {len(ranked_codes)} ranked species -> top {top_n} present in both products:")
    selected, skipped = select(ranked_codes, ranked_rank, c2a, has_ebird_trend, top_n)

    n = len(selected)
    print(f"\n[select] selected {n}/{top_n} community species present in BOTH trend products.")
    print(f"[select] skipped: {len(skipped['no_bbs_trend'])} without a BBS trend raster, "
          f"{len(skipped['no_ebird_trend'])} without an eBird trends product, "
          f"{len(skipped['no_aou'])} with no AOU crosswalk.")
    if n < top_n:
        print(f"[select] WARNING only {n} species qualify (< requested {top_n}); ranked list exhausted.")

    df = pd.DataFrame(selected, columns=["species_code", "aou", "season", "sci_norm", "mean_rank"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[select] wrote {len(df)} community species -> {out}")


if __name__ == "__main__":
    main()
