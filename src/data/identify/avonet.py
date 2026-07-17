#!/usr/bin/env python3
"""
AVONET + phylogeny + urban tolerance pipeline

Filtering:
- Species set is defined by presence in the urban intensity dataset
  (via eBird taxonomy crosswalk)

Adds:
- Urban tolerance distance (6 indices)
- Equal weighting across morphology, phylogeny, and urban tolerance
- eBird species code (SPECIES_CODE)
"""

import os
import numpy as np
import pandas as pd
import dendropy

from src.config_utils import load_data_config
_CFG = load_data_config()
_DR = _CFG["datasets_root"]

# Configuration

BL_PATH = f"{_DR}/avonet/TraitData/AVONET1_BirdLife.csv"
CROSSWALK_PATH = f"{_DR}/avonet/PhylogeneticData/BirdLife-BirdTree crosswalk.csv"
PHYLO_PATH = f"{_DR}/avonet/PhylogeneticData/HackettStage1_0001_1000_MCCTreeTargetHeights.nex"

URBAN_PATH = f"{_DR}/urban_avian/spp_urban_indices.csv"
# eBird taxonomy crosswalk, fetched by acquire/avonet.py from the eBird API.
EBIRD_CROSSWALK_PATH = f"{_DR}/avonet/eBird_taxonomy.csv"

# The clean, ordered species list the eBird downloader consumes; the two wide
# analysis tables land alongside it. All config-driven (previously written to
# the current working directory).
SPECIES_LIST_PATH = _CFG.get(
    "species_list", f"{_DR}/avonet/reference_community_ranked.csv"
)
_OUT_DIR = os.path.dirname(SPECIES_LIST_PATH) or "."
OUTPUT_FILTERED = os.path.join(_OUT_DIR, "AVONET_Filtered_ByUrbanSpecies.csv")
OUTPUT_COMPARISON = os.path.join(_OUT_DIR, "AVONET_Comparison_WithPhylogeny_Urban.csv")

FOCAL_ID = "AVIBASE-89431E9F"

TRAIT_COLS = [
    "Beak.Length_Culmen",
    "Beak.Length_Nares",
    "Beak.Width",
    "Beak.Depth",
    "Tarsus.Length",
    "Wing.Length",
    "Kipps.Distance",
    "Secondary1",
    "Hand-Wing.Index",
    "Tail.Length",
    "Mass",
]

URBAN_COLS = [
    "Mean.UA",
    "X90th.UA",
    "Block.size.UA",
    "Mean.NL",
    "X90th.NL",
    "Habitat.Use.NL",
]

# Utilities

def normalize_name(x):
    if pd.isna(x):
        return np.nan
    return x.strip().lower().replace("_", " ")

def standardize(df, cols):
    df = df.copy()
    for c in cols:
        mu = df[c].mean()
        sd = df[c].std()
        df[c] = 0.0 if sd == 0 or np.isnan(sd) else (df[c] - mu) / sd
    return df

def euclidean_distance(df, focal_row, cols, prefix):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    f = focal_row[cols].apply(pd.to_numeric, errors="coerce")

    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    diffs = X.sub(f.values, axis=1)

    dist = np.sqrt(np.sum(diffs.to_numpy() ** 2, axis=1))

    out = diffs.add_prefix(f"{prefix}_Diff_")
    out[f"{prefix}.Distance"] = dist
    return out.reindex(df.index)

# Crosswalks

def load_crosswalk(path):
    cw = pd.read_csv(path)
    cw = cw.dropna(subset=["Species3"])
    return cw.drop_duplicates("Species1", keep="first")

def derive_focal_phylo_label(bl, crosswalk):
    row = bl.loc[bl["Avibase.ID1"] == FOCAL_ID]
    if row.empty:
        raise ValueError("Focal species not found after filtering.")
    sp1 = row.iloc[0]["Species1"]
    sp3 = crosswalk.loc[crosswalk["Species1"] == sp1, "Species3"]
    if sp3.empty:
        raise ValueError("Focal species not found in BirdTree crosswalk.")
    return sp3.iloc[0].replace(" ", "_")

# Phylogeny

def compute_phylo_distances(tree, focal_label):
    """Patristic distance from the focal tip to every taxon: single-source, O(n).

    We only need distances *to the focal species*, so we traverse the tree as an
    undirected graph outward from the focal node, summing edge lengths -- one
    O(n) pass. (The previous ``tree.phylogenetic_distance_matrix()`` computed the
    full O(n^2) all-pairs matrix, ~44M distances for this tree, then discarded
    every row but the focal's.)
    """
    from collections import deque

    node = tree.find_node_with_taxon_label(focal_label)
    if node is None:
        raise ValueError("Focal species not found in phylogeny.")

    dist = {}
    seen = {node}
    queue = deque([(node, 0.0)])
    while queue:
        cur, d = queue.popleft()
        if cur.taxon is not None:
            dist[cur.taxon.label] = d
        neighbors = [(cur.parent_node, cur.edge.length)]  # edge to parent
        neighbors += [(c, c.edge.length) for c in cur.child_nodes()]  # edges to children
        for nb, length in neighbors:
            if nb is None or nb in seen:
                continue
            seen.add(nb)
            queue.append((nb, d + (length or 0.0)))
    return dist

# Main

def main():
    print("Working directory:", os.getcwd())

    # Load data
    bl = pd.read_csv(BL_PATH, encoding="latin1")
    urban = pd.read_csv(URBAN_PATH)
    ebird = pd.read_csv(EBIRD_CROSSWALK_PATH)
    # The eBird API taxonomy names the column SCIENTIFIC_NAME; older Cornell CSVs
    # used SCI_NAME. Normalize so downstream code can rely on SCI_NAME.
    if "SCI_NAME" not in ebird.columns and "SCIENTIFIC_NAME" in ebird.columns:
        ebird = ebird.rename(columns={"SCIENTIFIC_NAME": "SCI_NAME"})

    # Define species universe via urban dataset
    urban["species_code"] = urban["species_code"].str.lower()
    ebird["SPECIES_CODE"] = ebird["SPECIES_CODE"].str.lower()

    ebird["sci_norm"] = ebird["SCI_NAME"].apply(normalize_name)
    bl["sci_norm"] = bl["Species1"].apply(normalize_name)

    # Merge ebird taxonomy info into urban dataset
    urban = urban.merge(
        ebird[["SPECIES_CODE", "sci_norm"]],
        left_on="species_code",
        right_on="SPECIES_CODE",
        how="left"
    )

    urban_species = set(urban["sci_norm"].dropna().unique())

    bl = bl.loc[bl["sci_norm"].isin(urban_species)].copy()
    print(f"Filtered AVONET to {len(bl)} species present in urban dataset.")

    if FOCAL_ID not in bl["Avibase.ID1"].values:
        raise RuntimeError("Focal species excluded by urban-species filter.")

    bl.to_csv(OUTPUT_FILTERED, index=False)

    # Morphology
    crosswalk = load_crosswalk(CROSSWALK_PATH)
    bl = bl.merge(crosswalk[["Species1", "Species3"]], on="Species1", how="left")
    bl["Species3_underscored"] = bl["Species3"].str.replace(" ", "_")

    bl_morph = standardize(bl, TRAIT_COLS)
    focal_row = bl_morph.loc[bl_morph["Avibase.ID1"] == FOCAL_ID].iloc[0]

    morph_block = euclidean_distance(bl_morph, focal_row, TRAIT_COLS, "Trait")
    bl = pd.concat([bl, morph_block], axis=1)

    # Urban tolerance
    # We include "SPECIES_CODE" in the merge columns here
    bl = bl.merge(
        urban[["sci_norm", "SPECIES_CODE"] + URBAN_COLS],
        on="sci_norm",
        how="left"
    )

    bl = bl.dropna(subset=URBAN_COLS).copy()
    bl = standardize(bl, URBAN_COLS)

    # A single urban-tolerance score (mean of the standardized indices; higher =
    # more urban-tolerant), then EXTREMENESS from the median so that BOTH the most
    # and the least urban-tolerant species score well -- not only those similar to
    # the urban-tolerant house finch. This axis is deliberately focal-independent:
    # "Urban.Distance" here is distance from the median tolerance, so the tails of
    # the gradient rank best (see the rank block below).
    bl["Urban.Tolerance"] = bl[URBAN_COLS].mean(axis=1)
    bl["Urban.Distance"] = (bl["Urban.Tolerance"] - bl["Urban.Tolerance"].median()).abs()

    # Phylogeny
    focal_phylo = derive_focal_phylo_label(bl, crosswalk)
    tree = dendropy.Tree.get(
        path=PHYLO_PATH,
        schema="nexus",
        preserve_underscores=True
    )

    phylo_dist = compute_phylo_distances(tree, focal_phylo)
    bl["Phylo.Distance"] = bl["Species3_underscored"].map(phylo_dist)
    bl = bl.dropna(subset=["Phylo.Distance"])

    # Rank-based combination. Morphology + phylogeny reward PROXIMITY to the focal
    # (ascending: smaller distance = better rank). Urban tolerance rewards
    # EXTREMENESS (descending: larger distance-from-median = better rank), so the
    # community spans both the most and least urban-tolerant species.
    rank_cols = ["Trait.Distance", "Urban.Distance", "Phylo.Distance"]

    for c in ["Trait.Distance", "Phylo.Distance"]:
        bl[f"{c}.Rank"] = bl[c].rank(method="average", ascending=True)
    bl["Urban.Distance.Rank"] = bl["Urban.Distance"].rank(method="average", ascending=False)

    bl["Mean.Rank"] = bl[[f"{c}.Rank" for c in rank_cols]].mean(axis=1)

    # Sort by mean rank
    bl = bl.sort_values("Mean.Rank")
    os.makedirs(_OUT_DIR, exist_ok=True)
    bl.to_csv(OUTPUT_COMPARISON, index=False)

    print(f"Saved rank-based comparison table (with eBird SPECIES_CODE) to {OUTPUT_COMPARISON}")

    # Clean species list for the eBird downloader
    # Minimal, ordered artifact: species_code + mean_rank, most-similar first.
    # Selection (top-N / threshold) is deliberately left to download time, so
    # the full ranked list is written here rather than a pre-cut subset.
    species_list = (
        bl[["SPECIES_CODE", "Mean.Rank"]]
        .dropna(subset=["SPECIES_CODE"])
        .rename(columns={"SPECIES_CODE": "species_code", "Mean.Rank": "mean_rank"})
    )
    species_list.to_csv(SPECIES_LIST_PATH, index=False)
    print(f"Saved {len(species_list)} ranked species codes to {SPECIES_LIST_PATH}")

if __name__ == "__main__":
    main()