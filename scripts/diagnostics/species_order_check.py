"""Do the ESK basis and the BBS validation put the same species in the same COLUMN?

basis_coverage_gap.py found species present in >half of one domain and absent from the other, in
both directions, while total abundance and sparsity matched and each domain was internally
coherent (best neighbour 0.645 for landmarks, 0.648 for BBS). A column permutation produces
exactly that: it preserves row sums and nonzero counts, keeps each domain self-consistent, and
destroys cross-domain overlap.

The two sides build their column layout in different places and in different ways:

  basis      bbs_community_points.species_order(community_csv)
               -> [str(c).lower() for c in csv["species_code"]]        CSV row order, LOWERCASED
  validation validate_bbs_routes.load_observed
               -> list(dict.fromkeys(crosswalk["species_code"]))       CROSSWALK order, as-is

This prints both and diffs them. It is the check points_meta.json was supposed to perform --
skipped every run because trend.points_dir points at the stale esk_spacetime directory.

Run: cd $HOUFIN_REPO && python scripts/diagnostics/species_order_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd  # noqa: E402

from src.community_encoder.train_DESK.config_utils import load_config          # noqa: E402


def main():
    cfg = load_config(os.environ.get("ESK_DESK_CONFIG") or None)
    dcfg = cfg["data"] if "data" in cfg else cfg
    tc = cfg.get("desk", {}).get("trend", {}) or {}
    community_csv = tc.get("community_trend_list") or dcfg["community_trend_list"]
    print(f"community list: {community_csv}")

    # --- the BASIS's column order ---------------------------------------------------------
    from src.community_encoder.train_DESK.bbs_community_points import species_order
    basis = species_order(community_csv)

    # --- the VALIDATION's column order ----------------------------------------------------
    from src.data.preprocess.bbs_community import build_crosswalk
    from src.data.preprocess import bbs_data as bbs
    codes = [str(c) for c in pd.read_csv(community_csv)["species_code"].tolist()]
    dr = dcfg["datasets_root"]
    bbs_species = cfg.get("bbs", {}).get("species_list") or \
        os.path.join(bbs.BBS_PARENT_DIR, "SpeciesList.csv")
    crosswalk, _ = build_crosswalk(
        bbs_species, os.path.join(dr, "avonet", "eBird_taxonomy.csv"),
        os.path.join(dr, "avonet", "reference_community_ranked.csv"),
        community_codes=codes)
    valid = list(dict.fromkeys(crosswalk["species_code"]))

    print(f"\nbasis      order: {len(basis)} species, first 12 {basis[:12]}")
    print(f"validation order: {len(valid)} species, first 12 {valid[:12]}")

    same_len = len(basis) == len(valid)
    identical = same_len and all(a == b for a, b in zip(basis, valid))
    ci_identical = same_len and all(str(a).lower() == str(b).lower()
                                    for a, b in zip(basis, valid))
    same_set = set(map(str.lower, map(str, basis))) == set(map(str.lower, map(str, valid)))

    print(f"\nsame length            : {same_len}  ({len(basis)} vs {len(valid)})")
    print(f"identical order        : {identical}")
    print(f"identical ignoring case: {ci_identical}")
    print(f"same SET of species    : {same_set}")

    if not ci_identical:
        n_moved = sum(1 for a, b in zip(basis, valid) if str(a).lower() != str(b).lower())
        print(f"\n!! {n_moved} of {min(len(basis), len(valid))} positions hold a DIFFERENT species.")
        print("   Every z-space number computed against BBS is then comparing one species'")
        print("   abundance to another's. First 15 disagreements (position: basis vs validation):")
        shown = 0
        for i, (a, b) in enumerate(zip(basis, valid)):
            if str(a).lower() != str(b).lower():
                print(f"     col {i:>3}: basis={a!r:<12} validation={b!r}")
                shown += 1
                if shown >= 15:
                    break
        if same_set:
            pos = {str(s).lower(): i for i, s in enumerate(valid)}
            perm = [pos[str(s).lower()] for s in basis]
            print(f"\n   The SETS match, so this is a pure PERMUTATION -- the same 96 species in a")
            print(f"   different order. Reindexing one side by it makes the domains comparable.")
            print(f"   basis column i holds validation column perm[i]; first 12: {perm[:12]}")
    else:
        print("\n   Orders agree, so a column permutation is NOT the explanation and the")
        print("   coverage gap has another cause. Next suspect: the basis applies temporal_ema")
        print("   before log1p (bbs_community_points) and load_observed does not.")

    only_basis = set(map(str.lower, map(str, basis))) - set(map(str.lower, map(str, valid)))
    only_valid = set(map(str.lower, map(str, valid))) - set(map(str.lower, map(str, basis)))
    if only_basis or only_valid:
        print(f"\n   basis-only species     : {sorted(only_basis)}")
        print(f"   validation-only species: {sorted(only_valid)}")


if __name__ == "__main__":
    main()
