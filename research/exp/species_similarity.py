"""E059g: every community and dev species' similarity to the focal species (House Finch) on the community's own
selection axes, for stratifying the out-of-community evaluation.

    python research/exp/species_similarity.py --cache <any research cache (for the species lists)> --out <csv>

The reference community was chosen by src/data/identify (avonet.py + community_axes.py) on four axes: patristic
distance to the focal, distance in standardized AVONET morphospace (11 traits), and both tails of urban tolerance,
with a migration penalty. reference_community_ranked.csv holds those values only for the gated candidate pool (139 of
the 273 dev species). This recomputes phylogenetic and trait distance for ALL species with avonet.py's own functions:
- traits are standardized with the MEAN and SD of the same urban-species universe avonet.py uses, so distances are
  on the same scale;
- the phylogenetic distance comes from the same tree and focal tip.
It is checked against the ranked CSV where both exist. Urban tolerance exists only for the urban-dataset species.
Migration is AVONET's class. Read-only on data.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import dendropy
    import pandas as pd
    from src.data.identify import avonet as av
    meta = json.load(open(os.path.join(a.cache, "meta.json"), encoding="utf-8"))
    groups = {"community": meta["community"], "dev": meta["dev_species"]}
    bl_all = pd.read_csv(av.BL_PATH, encoding="latin1")
    urban = pd.read_csv(av.URBAN_PATH)
    eb = pd.read_csv(av.EBIRD_CROSSWALK_PATH)
    if "SCI_NAME" not in eb.columns and "SCIENTIFIC_NAME" in eb.columns:
        eb = eb.rename(columns={"SCIENTIFIC_NAME": "SCI_NAME"})
    eb["SPECIES_CODE"] = eb["SPECIES_CODE"].str.lower()
    eb["sci_norm"] = eb["SCI_NAME"].apply(av.normalize_name)
    bl_all["sci_norm"] = bl_all["Species1"].apply(av.normalize_name)
    # the urban universe avonet.py standardizes traits over
    urban["species_code"] = urban["species_code"].str.lower()
    urban = urban.merge(eb[["SPECIES_CODE", "sci_norm"]], left_on="species_code", right_on="SPECIES_CODE", how="left")
    uni = bl_all.loc[bl_all["sci_norm"].isin(set(urban["sci_norm"].dropna()))]
    mu, sd = uni[av.TRAIT_COLS].mean(), uni[av.TRAIT_COLS].std(ddof=0)
    if not np.allclose(sd, uni[av.TRAIT_COLS].std(), rtol=0.01):
        pass
    # match avonet.standardize's convention exactly
    probe = av.standardize(uni.copy(), av.TRAIT_COLS)
    ddof = 0 if np.allclose(probe[av.TRAIT_COLS].std(ddof=0), 1.0, atol=1e-6) else 1
    sd = uni[av.TRAIT_COLS].std(ddof=ddof)
    Z = (bl_all[av.TRAIT_COLS] - mu) / sd
    focal = Z.loc[bl_all["Avibase.ID1"] == av.FOCAL_ID].iloc[0]
    bl_all["trait_distance"] = np.sqrt(((Z - focal) ** 2).sum(1))
    cw = av.load_crosswalk(av.CROSSWALK_PATH)
    bl_all = bl_all.merge(cw[["Species1", "Species3"]], on="Species1", how="left")
    bl_all["Species3_underscored"] = bl_all["Species3"].str.replace(" ", "_")
    tree = dendropy.Tree.get(path=av.PHYLO_PATH, schema="nexus", preserve_underscores=True)
    pd_map = av.compute_phylo_distances(tree, av.derive_focal_phylo_label(bl_all, cw))
    bl_all["phylo_distance"] = bl_all["Species3_underscored"].map(pd_map)
    code2sci = dict(zip(eb["SPECIES_CODE"], eb["sci_norm"]))
    by_sci = bl_all.drop_duplicates("sci_norm").set_index("sci_norm")
    ranked = pd.read_csv(os.path.join(os.path.dirname(av.SPECIES_LIST_PATH), "reference_community_ranked.csv")
                         ).set_index("species_code")
    rows = []
    for g, codes in groups.items():
        for code in codes:
            s = code2sci.get(code)
            r = by_sci.loc[s] if s in by_sci.index else None
            rows.append({"species_code": code, "group": g,
                         "phylo_distance": float(r["phylo_distance"]) if r is not None else np.nan,
                         "trait_distance": float(r["trait_distance"]) if r is not None else np.nan,
                         "migration": float(r["Migration"]) if r is not None else np.nan,
                         "urban_tolerance": float(ranked.loc[code, "urban_tolerance"]) if code in ranked.index else np.nan,
                         "in_ranked_pool": code in ranked.index})
    df = pd.DataFrame(rows)
    chk = df[df["in_ranked_pool"]].set_index("species_code")
    rk = ranked.reindex(chk.index)
    dp = float(np.nanmax(np.abs(chk["phylo_distance"] - rk["phylo_distance"])))
    dt = float(np.nanmax(np.abs(chk["trait_distance"] - rk["trait_distance"])))
    df.to_csv(a.out, index=False)
    d = df[df.group == "dev"]
    print(f"[species-similarity] {len(df)} species -> {a.out}; dev with phylo {int(d.phylo_distance.notna().sum())}, "
          f"trait {int(d.trait_distance.notna().sum())}, migration {int(d.migration.notna().sum())} of {len(d)}; "
          f"check vs ranked pool ({int(df.in_ranked_pool.sum())} species): max |d phylo| {dp:.2e}, max |d trait| {dt:.2e}")


if __name__ == "__main__":
    main()
