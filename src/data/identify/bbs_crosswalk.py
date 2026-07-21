"""Crosswalk BBS species (AOU codes) to eBird species codes, restricted to the
reference community.

BBS counts are keyed by numeric AOU codes; the eBird community (and everything
downstream) is keyed by eBird ``SPECIES_CODE``. There is no shared key, so we
bridge on the **scientific name**: normalize both sides and join. Restricting to
the reference community (top-N from ``reference_community_ranked.csv``) keeps the
match space small and the diagnostics interpretable.

Handled carefully (per plan): **lumps** — several AOU rows mapping to one eBird
code are all kept (their counts get summed at ingest, B2); **splits** — one AOU
mapping to several eBird codes is flagged loudly; and we always print
``n_matched / n_community / n_unmatched`` with the unmatched community members,
never dropping silently.

    python -m src.data.identify.bbs_crosswalk --bbs-species <SpeciesList.csv> [--top-n 100]
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.config_utils import load_config, load_data_config
from src.data.identify.avonet import normalize_name


def normalize_bbs_species(bbs_df):
    """BBS species table → DataFrame ``[aou:int, sci_norm:str]`` (defensive).

    Accepts either explicit ``Genus``/``Species`` columns (BBS ``SpeciesList``)
    or a single scientific-name column; column names are matched case-insensitively.
    """
    cols = {c.lower().strip(): c for c in bbs_df.columns}
    aou_col = cols.get("aou")
    if aou_col is None:
        raise KeyError(f"BBS species table has no AOU column; got {list(bbs_df.columns)}")
    out = pd.DataFrame({"aou": pd.to_numeric(bbs_df[aou_col], errors="coerce")})
    if "genus" in cols and "species" in cols:
        sci = (bbs_df[cols["genus"]].astype(str).str.strip() + " "
               + bbs_df[cols["species"]].astype(str).str.strip())
    else:
        sci_col = next((cols[k] for k in cols
                        if "scientific" in k or k in ("sci_name", "sciname", "latin")), None)
        if sci_col is None:
            raise KeyError("BBS species table has neither Genus/Species nor a "
                           f"scientific-name column; got {list(bbs_df.columns)}")
        sci = bbs_df[sci_col].astype(str)
    out["sci_norm"] = sci.apply(normalize_name)
    return out.dropna(subset=["aou"]).astype({"aou": int})


def load_ebird_taxonomy(path):
    """eBird taxonomy → DataFrame ``[species_code(lower), sci_norm]``."""
    tax = pd.read_csv(path)
    if "SCI_NAME" not in tax.columns and "SCIENTIFIC_NAME" in tax.columns:
        tax = tax.rename(columns={"SCIENTIFIC_NAME": "SCI_NAME"})
    tax = tax[["SPECIES_CODE", "SCI_NAME"]].copy()
    tax["species_code"] = tax["SPECIES_CODE"].astype(str).str.lower()
    tax["sci_norm"] = tax["SCI_NAME"].apply(normalize_name)
    return tax[["species_code", "sci_norm"]].dropna(subset=["sci_norm"])


def read_community_codes(ranked_path, top_n=None, exclude=None):
    """Top-``top_n`` eBird ``species_code`` (best mean_rank first) from the ranked CSV.

    ``exclude`` defaults to the config focal species (the transfer target) so the BBS
    community can never include it, even from a stale ranked list (defense-in-depth;
    avonet already drops it at the source).
    """
    if exclude is None:
        from src.config_utils import load_data_config
        f = str(load_data_config().get("focal_species_code") or "").strip().lower()
        exclude = {f} if f else set()
    excl = {str(e).lower() for e in exclude}
    df = pd.read_csv(ranked_path)
    df = df.dropna(subset=["species_code"]).sort_values("mean_rank")
    codes = [c for c in df["species_code"].astype(str).str.lower().tolist() if c not in excl]
    return codes[:top_n] if top_n else codes


def crosswalk(bbs_norm, ebird_tax, community_codes):
    """Join normalized BBS species to eBird codes within the community (pure).

    Returns ``(matched, diag)``: ``matched`` is ``[aou, species_code, sci_norm]``
    for community species with an AOU; ``diag`` reports match counts, the
    unmatched community members, and any split AOUs.
    """
    community = list(dict.fromkeys(str(c).lower() for c in community_codes))
    m = bbs_norm.merge(ebird_tax, on="sci_norm", how="inner")
    m = m[m["species_code"].isin(community)].copy()

    # Splits: one AOU resolving to >1 eBird code (via ambiguous sci names).
    per_aou = m.groupby("aou")["species_code"].nunique()
    split_aous = sorted(per_aou[per_aou > 1].index.tolist())

    matched_codes = set(m["species_code"])
    unmatched = [c for c in community if c not in matched_codes]
    diag = {
        "n_community": len(community),
        "n_matched": len(matched_codes),
        "n_unmatched": len(unmatched),
        "unmatched_codes": unmatched,
        "split_aous": split_aous,
        "n_aou_rows": int(m["aou"].nunique()),
    }
    return m[["aou", "species_code", "sci_norm"]].drop_duplicates().reset_index(drop=True), diag


def build_crosswalk(bbs_species_path, ebird_taxonomy_path, ranked_path, top_n=None,
                    community_codes=None):
    """Load inputs, run the crosswalk, print diagnostics; return ``(matched, diag)``.

    ``community_codes`` (if given) overrides the ranked-CSV top-N selection -- pass the
    eBird stack's actual species so the BBS community matches the eBird blocks the
    amplitude step modulates (top-N-by-rank and top-N-complete diverge otherwise, freezing
    common species). Falls back to ``read_community_codes(ranked_path, top_n)``.
    """
    bbs_norm = normalize_bbs_species(_read_species_table(bbs_species_path))
    ebird_tax = load_ebird_taxonomy(ebird_taxonomy_path)
    community = community_codes if community_codes is not None else \
        read_community_codes(ranked_path, top_n)
    matched, diag = crosswalk(bbs_norm, ebird_tax, community)
    print(f"[crosswalk] matched {diag['n_matched']}/{diag['n_community']} community "
          f"species ({diag['n_aou_rows']} AOU rows); {diag['n_unmatched']} unmatched.")
    if diag["unmatched_codes"]:
        print(f"[crosswalk] UNMATCHED community codes: {diag['unmatched_codes']}")
    if diag["split_aous"]:
        print(f"[crosswalk] WARNING split AOUs (one AOU -> multiple codes): {diag['split_aous']}")
    return matched, diag


def _read_species_table(path):
    """Read a BBS species list defensively (encoding + delimiter fallbacks)."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.read_csv(path, encoding="latin-1")  # last resort, surfaces the error


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbs-species", required=True, help="BBS SpeciesList CSV (AOU + Genus/Species)")
    ap.add_argument("--ebird-taxonomy", default=None)
    ap.add_argument("--ranked", default=None, help="reference_community_ranked.csv")
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--out", default=None, help="write matched crosswalk CSV here")
    args = ap.parse_args()

    dr = load_data_config()["datasets_root"]
    cfg = load_config()
    top_n = args.top_n if args.top_n is not None else cfg.get("bbs", {}).get("community_top_n")
    ebird_tax = args.ebird_taxonomy or os.path.join(dr, "avonet", "eBird_taxonomy.csv")
    ranked = args.ranked or os.path.join(dr, "avonet", "reference_community_ranked.csv")

    matched, _ = build_crosswalk(args.bbs_species, ebird_tax, ranked, top_n)
    out = args.out or os.path.join(dr, "bbs", "aou_ebird_crosswalk.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    matched.to_csv(out, index=False)
    print(f"[crosswalk] wrote {len(matched)} rows -> {out}")


if __name__ == "__main__":
    main()
