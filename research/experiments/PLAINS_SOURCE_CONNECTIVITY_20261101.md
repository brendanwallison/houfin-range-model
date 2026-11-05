# Great Plains sink prevalence and source bridges

All 101 checked standard-yearling fits in CROSS_MODEL_PROCESS_AUDIT_20261101.json. Land cells have equal weight; no survey or fitted-population weighting. Plains masks here are latitude 25-49N, primary 100-95W and broader 105-95W. These are geographic bands, not formal ecoregions.

Source means strictly >1 for the specified metric. Potential lambda excludes density/Allee suppression, density-dependent disease and dispersal losses; C is origin-attributed breeding contribution; local replacement is the fitted-state no-movement population ratio. Finite unoccupied-cell C can be numerically fragile and is not evidence of an established source population.

A bridge is one connected component of source cells joining both endpoint sets. Four-neighbor means shared edges; eight-neighbor also permits corner contact. Plains tests join the westernmost and easternmost ~0.8-degree strips of the band, with the entire path confined to the band. Continental interior joins longitude <115W to >=80W. Coastal endpoints use modeled land cells adjacent to border-connected non-land, west of 120W and east of 81W; these are grid shoreline proxies, not a separate validated ocean coastline product. Primary paths are confined to 25-49N; additional full_domain tests retain those endpoint latitudes but allow paths anywhere on the modeled land grid. No FFT wraparound or dispersal jumps are used. Annual tests prevent era averaging from fabricating a persistent source path. Threshold >1.01 and >1.05 era sensitivities are saved in JSON.

Modern interpretation: all 59 fits have majority-sink Plains under C (64.7-90.5% of primary-band cells) and actual-state local replacement (88.4-98.4%), but only 21/59 do under potential lambda (27.9-62.1% sinks). Potential-lambda coast-to-coast paths occur in 38/59 era maps with shared-edge connectivity, and all 59 if diagonal contacts count. The stricter lambda>1.05 threshold leaves 35/59 shared-edge paths. Neither C nor local replacement gives a coast-to-coast path in any era or annual map tested. A majority of sink cells does not prevent a narrow source path: modern primary-Plains C paths occur in 58/59 fits, despite sink dominance.

Example map: [later-state native-departure middle scenario](SOURCE_BRIDGE_MAP_20261101.png). Its edge-connected potential-lambda path is explicitly saved in SOURCE_BRIDGE_PATH_20261101.npz. Source connectivity does not establish positive population-growth capacity after Allee/density suppression or validate historical colonization; dispersal can also cross sink gaps.

## Modern era, every model version

| Fit | Metric | Primary Plains sink share | Broad Plains sink share | Primary Plains bridge (4/8 neighbors) | Broad Plains bridge (4/8) | Continental interior bridge (4/8) | Coastal bridge (4/8) |
|---|---|---:|---:|---|---|---|---|
| step29_yb_native | potential_lambda | 51.8% | 44.5% | yes/yes | yes/yes | yes/yes | yes/yes |
| step29_yb_native | C | 87.6% | 83.9% | yes/yes | no/no | no/no | no/no |
| step29_yb_native | local_replacement | 96.3% | 97.1% | no/no | no/no | no/no | no/no |
| step29_yb_native_s42 | potential_lambda | 52.1% | 44.6% | yes/yes | yes/yes | yes/yes | yes/yes |
| step29_yb_native_s42 | C | 87.4% | 83.7% | yes/yes | no/no | no/no | no/no |
| step29_yb_native_s42 | local_replacement | 96.3% | 96.8% | no/no | no/no | no/no | no/no |
| step31_nest | potential_lambda | 46.8% | 41.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step31_nest | C | 89.5% | 86.1% | yes/yes | no/no | no/no | no/no |
| step31_nest | local_replacement | 94.2% | 95.0% | no/no | no/no | no/no | no/no |
| step31_nest_bloose | potential_lambda | 49.5% | 43.2% | yes/yes | yes/yes | yes/yes | yes/yes |
| step31_nest_bloose | C | 89.5% | 86.9% | yes/yes | no/no | no/no | no/no |
| step31_nest_bloose | local_replacement | 94.2% | 95.2% | no/no | no/no | no/no | no/no |
| step31_nestonly | potential_lambda | 46.3% | 40.8% | yes/yes | yes/yes | yes/yes | yes/yes |
| step31_nestonly | C | 88.9% | 85.6% | yes/yes | no/no | no/no | no/no |
| step31_nestonly | local_replacement | 94.2% | 94.9% | no/no | no/no | no/no | no/no |
| step32_cbc_yb | potential_lambda | 49.5% | 42.2% | yes/yes | yes/yes | no/yes | no/yes |
| step32_cbc_yb | C | 88.7% | 84.4% | yes/yes | no/no | no/no | no/no |
| step32_cbc_yb | local_replacement | 95.5% | 96.3% | no/no | no/no | no/no | no/no |
| step32_cbchi_yb | potential_lambda | 55.5% | 46.4% | yes/yes | yes/yes | no/yes | no/yes |
| step32_cbchi_yb | C | 86.1% | 78.8% | yes/yes | no/no | no/no | no/no |
| step32_cbchi_yb | local_replacement | 98.2% | 98.6% | no/no | no/no | no/no | no/no |
| step32_cbclo_yb | potential_lambda | 49.7% | 42.3% | yes/yes | yes/yes | no/yes | no/yes |
| step32_cbclo_yb | C | 88.2% | 84.1% | yes/yes | no/no | no/no | no/no |
| step32_cbclo_yb | local_replacement | 95.3% | 96.3% | no/no | no/no | no/no | no/no |
| step33_nestcbc_abs_s41 | potential_lambda | 60.0% | 51.7% | yes/yes | yes/yes | no/yes | no/yes |
| step33_nestcbc_abs_s41 | C | 81.8% | 76.1% | yes/yes | no/no | no/no | no/no |
| step33_nestcbc_abs_s41 | local_replacement | 96.1% | 96.7% | no/no | no/no | no/no | no/no |
| step33_nestcbc_natdep_s41 | potential_lambda | 44.2% | 36.3% | yes/yes | yes/yes | yes/yes | yes/yes |
| step33_nestcbc_natdep_s41 | C | 82.4% | 77.5% | yes/yes | no/no | no/no | no/no |
| step33_nestcbc_natdep_s41 | local_replacement | 94.5% | 93.3% | no/no | no/no | no/no | no/no |
| step33_nestcbc_natdep_s42 | potential_lambda | 45.5% | 37.2% | yes/yes | yes/yes | yes/yes | yes/yes |
| step33_nestcbc_natdep_s42 | C | 82.6% | 77.8% | yes/yes | no/no | no/no | no/no |
| step33_nestcbc_natdep_s42 | local_replacement | 94.2% | 93.3% | no/no | no/no | no/no | no/no |
| step33_nestcbc_s41 | potential_lambda | 46.8% | 41.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step33_nestcbc_s41 | C | 89.5% | 86.1% | yes/yes | no/no | no/no | no/no |
| step33_nestcbc_s41 | local_replacement | 94.7% | 95.3% | no/no | no/no | no/no | no/no |
| step33_nestcbc_s42 | potential_lambda | 46.8% | 41.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step33_nestcbc_s42 | C | 89.5% | 86.3% | yes/yes | no/no | no/no | no/no |
| step33_nestcbc_s42 | local_replacement | 94.7% | 95.3% | no/no | no/no | no/no | no/no |
| step34_abs_broodhi | potential_lambda | 54.2% | 47.4% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_broodhi | C | 87.9% | 83.1% | yes/yes | no/no | no/no | no/no |
| step34_abs_broodhi | local_replacement | 92.1% | 92.1% | no/no | no/no | no/no | no/no |
| step34_abs_broodlo | potential_lambda | 62.1% | 53.0% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_broodlo | C | 85.3% | 78.4% | yes/yes | no/no | no/no | no/no |
| step34_abs_broodlo | local_replacement | 93.7% | 94.0% | no/no | no/no | no/no | no/no |
| step34_abs_comp_natdep_s41 | potential_lambda | 27.9% | 23.1% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_abs_comp_natdep_s41 | C | 64.7% | 68.2% | yes/yes | yes/yes | no/no | no/no |
| step34_abs_comp_natdep_s41 | local_replacement | 97.4% | 97.8% | no/no | no/no | no/no | no/no |
| step34_abs_comp_natdep_s42 | potential_lambda | 45.3% | 37.1% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_abs_comp_natdep_s42 | C | 68.2% | 68.2% | yes/yes | yes/yes | no/no | no/no |
| step34_abs_comp_natdep_s42 | local_replacement | 96.3% | 96.2% | no/no | no/no | no/no | no/no |
| step34_abs_comp_s42 | potential_lambda | 57.9% | 49.7% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_comp_s42 | C | 80.8% | 76.7% | yes/yes | no/no | no/no | no/no |
| step34_abs_comp_s42 | local_replacement | 96.1% | 96.4% | no/no | no/no | no/no | no/no |
| step34_abs_foundhi | potential_lambda | 60.0% | 52.2% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_foundhi | C | 84.7% | 77.8% | yes/yes | no/no | no/no | no/no |
| step34_abs_foundhi | local_replacement | 95.8% | 96.6% | no/no | no/no | no/no | no/no |
| step34_abs_foundlo | potential_lambda | 59.7% | 51.3% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_foundlo | C | 80.8% | 75.3% | yes/yes | no/no | no/no | no/no |
| step34_abs_foundlo | local_replacement | 96.6% | 97.1% | no/no | no/no | no/no | no/no |
| step34_abs_hazard_natdep_s41 | potential_lambda | 27.9% | 22.4% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_abs_hazard_natdep_s41 | C | 65.0% | 67.0% | yes/yes | yes/yes | no/no | no/no |
| step34_abs_hazard_natdep_s41 | local_replacement | 96.3% | 96.7% | no/no | no/no | no/no | no/no |
| step34_abs_hazard_natdep_s42 | potential_lambda | 44.7% | 36.5% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_abs_hazard_natdep_s42 | C | 67.6% | 67.9% | yes/yes | yes/yes | no/no | no/no |
| step34_abs_hazard_natdep_s42 | local_replacement | 96.1% | 96.1% | no/no | no/no | no/no | no/no |
| step34_abs_hazard_s41 | potential_lambda | 58.7% | 50.6% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_hazard_s41 | C | 81.6% | 76.1% | yes/yes | no/no | no/no | no/no |
| step34_abs_hazard_s41 | local_replacement | 96.3% | 96.7% | no/no | no/no | no/no | no/no |
| step34_abs_hazard_s42 | potential_lambda | 59.2% | 50.8% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_hazard_s42 | C | 81.1% | 75.6% | yes/yes | no/no | no/no | no/no |
| step34_abs_hazard_s42 | local_replacement | 96.3% | 96.8% | no/no | no/no | no/no | no/no |
| step34_abs_natdep_s41 | potential_lambda | 41.1% | 33.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_abs_natdep_s41 | C | 71.1% | 70.0% | yes/yes | yes/yes | no/no | no/no |
| step34_abs_natdep_s41 | local_replacement | 96.1% | 96.6% | no/no | no/no | no/no | no/no |
| step34_abs_natdep_s42 | potential_lambda | 31.6% | 25.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_abs_natdep_s42 | C | 66.6% | 68.4% | yes/yes | yes/yes | no/no | no/no |
| step34_abs_natdep_s42 | local_replacement | 96.6% | 97.5% | no/no | no/no | no/no | no/no |
| step34_abs_s42 | potential_lambda | 58.9% | 51.0% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_s42 | C | 80.8% | 75.2% | yes/yes | no/no | no/no | no/no |
| step34_abs_s42 | local_replacement | 96.8% | 97.2% | no/no | no/no | no/no | no/no |
| step34_abs_v2_hazard_s41 | potential_lambda | 62.1% | 52.2% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_v2_hazard_s41 | C | 83.2% | 76.4% | yes/yes | no/no | no/no | no/no |
| step34_abs_v2_hazard_s41 | local_replacement | 95.8% | 96.6% | no/no | no/no | no/no | no/no |
| step34_abs_v2_hazard_s42 | potential_lambda | 60.5% | 51.0% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_v2_hazard_s42 | C | 78.7% | 74.1% | yes/yes | no/no | no/no | no/no |
| step34_abs_v2_hazard_s42 | local_replacement | 96.3% | 97.1% | no/no | no/no | no/no | no/no |
| step34_abs_v2_natdep_s41 | potential_lambda | 46.1% | 38.0% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_abs_v2_natdep_s41 | C | 79.2% | 75.1% | yes/yes | no/no | no/no | no/no |
| step34_abs_v2_natdep_s41 | local_replacement | 89.5% | 89.8% | yes/yes | no/no | no/no | no/no |
| step34_abs_v2_natdep_s42 | potential_lambda | 46.3% | 37.0% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_abs_v2_natdep_s42 | C | 74.5% | 71.9% | yes/yes | no/yes | no/no | no/no |
| step34_abs_v2_natdep_s42 | local_replacement | 94.5% | 94.3% | no/no | no/no | no/no | no/no |
| step34_abs_v2_s41 | potential_lambda | 62.1% | 52.4% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_v2_s41 | C | 83.4% | 76.4% | yes/yes | no/no | no/no | no/no |
| step34_abs_v2_s41 | local_replacement | 95.8% | 96.6% | no/no | no/no | no/no | no/no |
| step34_abs_v2_s42 | potential_lambda | 60.5% | 51.0% | yes/yes | yes/yes | no/yes | no/yes |
| step34_abs_v2_s42 | C | 78.7% | 74.1% | yes/yes | no/no | no/no | no/no |
| step34_abs_v2_s42 | local_replacement | 96.6% | 97.3% | no/no | no/no | no/no | no/no |
| step34_broodhi | potential_lambda | 46.8% | 42.1% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_broodhi | C | 90.5% | 87.4% | no/no | no/no | no/no | no/no |
| step34_broodhi | local_replacement | 94.7% | 95.7% | no/no | no/no | no/no | no/no |
| step34_broodlo | potential_lambda | 46.6% | 41.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_broodlo | C | 88.7% | 84.9% | yes/yes | no/no | no/no | no/no |
| step34_broodlo | local_replacement | 92.6% | 93.6% | no/no | no/no | no/no | no/no |
| step34_foundhi | potential_lambda | 45.5% | 40.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_foundhi | C | 88.4% | 84.4% | yes/yes | no/no | no/no | no/no |
| step34_foundhi | local_replacement | 94.2% | 94.8% | no/no | no/no | no/no | no/no |
| step34_foundlo | potential_lambda | 46.8% | 41.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_foundlo | C | 89.5% | 86.1% | yes/yes | no/no | no/no | no/no |
| step34_foundlo | local_replacement | 94.7% | 95.3% | no/no | no/no | no/no | no/no |
| step34_rel_comp_natdep_s41 | potential_lambda | 43.7% | 36.0% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_comp_natdep_s41 | C | 82.4% | 78.3% | yes/yes | no/no | no/no | no/no |
| step34_rel_comp_natdep_s41 | local_replacement | 93.9% | 93.1% | no/no | no/no | no/no | no/no |
| step34_rel_comp_natdep_s42 | potential_lambda | 35.0% | 31.0% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_comp_natdep_s42 | C | 82.4% | 79.9% | yes/yes | no/yes | no/no | no/no |
| step34_rel_comp_natdep_s42 | local_replacement | 93.4% | 93.0% | no/no | no/no | no/no | no/no |
| step34_rel_comp_s41 | potential_lambda | 46.1% | 41.0% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_comp_s41 | C | 89.5% | 86.3% | yes/yes | no/no | no/no | no/no |
| step34_rel_comp_s41 | local_replacement | 94.5% | 95.2% | no/no | no/no | no/no | no/no |
| step34_rel_hazard_natdep_s41 | potential_lambda | 43.4% | 35.8% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_hazard_natdep_s41 | C | 82.4% | 78.0% | yes/yes | no/no | no/no | no/no |
| step34_rel_hazard_natdep_s41 | local_replacement | 93.9% | 93.1% | no/no | no/no | no/no | no/no |
| step34_rel_hazard_natdep_s42 | potential_lambda | 42.9% | 35.6% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_hazard_natdep_s42 | C | 81.8% | 77.8% | yes/yes | no/no | no/no | no/no |
| step34_rel_hazard_natdep_s42 | local_replacement | 93.7% | 92.8% | no/no | no/no | no/no | no/no |
| step34_rel_hazard_s41 | potential_lambda | 45.5% | 40.5% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_hazard_s41 | C | 89.5% | 86.1% | yes/yes | no/no | no/no | no/no |
| step34_rel_hazard_s41 | local_replacement | 94.7% | 95.3% | no/no | no/no | no/no | no/no |
| step34_rel_hazard_s42 | potential_lambda | 46.6% | 41.3% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_hazard_s42 | C | 89.5% | 85.9% | yes/yes | no/no | no/no | no/no |
| step34_rel_hazard_s42 | local_replacement | 94.5% | 95.2% | no/no | no/no | no/no | no/no |
| step34_rel_v2_hazard_s41 | potential_lambda | 48.4% | 41.9% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_v2_hazard_s41 | C | 87.1% | 84.2% | yes/yes | no/no | no/no | no/no |
| step34_rel_v2_hazard_s41 | local_replacement | 92.4% | 94.3% | yes/yes | no/no | no/no | no/no |
| step34_rel_v2_hazard_s42 | potential_lambda | 44.7% | 40.0% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_v2_hazard_s42 | C | 88.4% | 84.8% | yes/yes | no/no | no/no | no/no |
| step34_rel_v2_hazard_s42 | local_replacement | 93.2% | 94.4% | yes/yes | no/no | no/no | no/no |
| step34_rel_v2_natdep_s41 | potential_lambda | 41.1% | 34.3% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_v2_natdep_s41 | C | 81.3% | 76.9% | yes/yes | no/no | no/no | no/no |
| step34_rel_v2_natdep_s41 | local_replacement | 91.1% | 91.6% | yes/yes | no/no | no/no | no/no |
| step34_rel_v2_natdep_s42 | potential_lambda | 40.3% | 34.3% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_v2_natdep_s42 | C | 84.2% | 80.3% | yes/yes | no/no | no/no | no/no |
| step34_rel_v2_natdep_s42 | local_replacement | 92.6% | 93.4% | yes/yes | no/no | no/no | no/no |
| step34_rel_v2_s41 | potential_lambda | 47.9% | 41.8% | yes/yes | yes/yes | yes/yes | yes/yes |
| step34_rel_v2_s41 | C | 87.4% | 84.8% | yes/yes | no/no | no/no | no/no |
| step34_rel_v2_s41 | local_replacement | 92.6% | 94.4% | yes/yes | no/no | no/no | no/no |
| step35_est1951_ctl_hi_s41 | potential_lambda | 53.4% | 46.5% | yes/yes | yes/yes | no/yes | no/yes |
| step35_est1951_ctl_hi_s41 | C | 87.4% | 83.4% | yes/yes | no/no | no/no | no/no |
| step35_est1951_ctl_hi_s41 | local_replacement | 90.0% | 90.7% | no/no | no/no | no/no | no/no |
| step35_est1951_ctl_lo_s41 | potential_lambda | 53.9% | 47.0% | yes/yes | yes/yes | no/yes | no/yes |
| step35_est1951_ctl_lo_s41 | C | 87.4% | 82.5% | yes/yes | no/no | no/no | no/no |
| step35_est1951_ctl_lo_s41 | local_replacement | 89.7% | 90.9% | no/no | no/no | no/no | no/no |
| step35_est1951_ctl_mid_s41 | potential_lambda | 55.0% | 47.5% | yes/yes | yes/yes | no/yes | no/yes |
| step35_est1951_ctl_mid_s41 | C | 87.6% | 82.1% | yes/yes | no/no | no/no | no/no |
| step35_est1951_ctl_mid_s41 | local_replacement | 90.8% | 91.4% | no/no | no/no | no/no | no/no |
| step35_est1951_ctl_mid_s42 | potential_lambda | 57.6% | 49.7% | yes/yes | yes/yes | no/yes | no/yes |
| step35_est1951_ctl_mid_s42 | C | 87.6% | 82.2% | yes/yes | no/no | no/no | no/no |
| step35_est1951_ctl_mid_s42 | local_replacement | 91.6% | 92.0% | no/no | no/no | no/no | no/no |
| step35_est1951_nat_hi_s41 | potential_lambda | 48.4% | 38.1% | yes/yes | yes/yes | yes/yes | yes/yes |
| step35_est1951_nat_hi_s41 | C | 81.3% | 75.1% | yes/yes | no/no | no/no | no/no |
| step35_est1951_nat_hi_s41 | local_replacement | 93.2% | 92.0% | no/no | no/no | no/no | no/no |
| step35_est1951_nat_lo_s41 | potential_lambda | 49.5% | 39.9% | yes/yes | yes/yes | yes/yes | yes/yes |
| step35_est1951_nat_lo_s41 | C | 78.9% | 74.2% | yes/yes | no/no | no/no | no/no |
| step35_est1951_nat_lo_s41 | local_replacement | 88.4% | 88.7% | no/no | no/no | no/no | no/no |
| step35_est1951_nat_mid_s41 | potential_lambda | 47.9% | 37.4% | yes/yes | yes/yes | yes/yes | yes/yes |
| step35_est1951_nat_mid_s41 | C | 76.3% | 72.2% | yes/yes | no/no | no/no | no/no |
| step35_est1951_nat_mid_s41 | local_replacement | 93.4% | 92.2% | no/no | no/no | no/no | no/no |
| step35_est1951_nat_mid_s42 | potential_lambda | 42.4% | 34.6% | yes/yes | yes/yes | yes/yes | yes/yes |
| step35_est1951_nat_mid_s42 | C | 75.0% | 72.4% | yes/yes | no/yes | no/no | no/no |
| step35_est1951_nat_mid_s42 | local_replacement | 94.7% | 93.9% | no/no | no/no | no/no | no/no |
| step36_est1951_nonest_ctl_s41 | potential_lambda | 55.8% | 46.4% | yes/yes | yes/yes | no/yes | no/yes |
| step36_est1951_nonest_ctl_s41 | C | 85.5% | 82.1% | yes/yes | no/no | no/no | no/no |
| step36_est1951_nonest_ctl_s41 | local_replacement | 96.6% | 97.5% | no/no | no/no | no/no | no/no |
| step36_nonest_ctl_s41 | potential_lambda | 55.8% | 46.4% | yes/yes | yes/yes | no/yes | no/yes |
| step36_nonest_ctl_s41 | C | 83.7% | 81.4% | yes/yes | no/no | no/no | no/no |
| step36_nonest_ctl_s41 | local_replacement | 97.9% | 98.7% | no/no | no/no | no/no | no/no |
| step36_nonest_ctl_s42 | potential_lambda | 56.1% | 46.9% | yes/yes | yes/yes | no/yes | no/yes |
| step36_nonest_ctl_s42 | C | 83.9% | 81.6% | yes/yes | no/no | no/no | no/no |
| step36_nonest_ctl_s42 | local_replacement | 98.2% | 98.7% | no/no | no/no | no/no | no/no |
| step36_nonest_nat_s41 | potential_lambda | 38.9% | 30.6% | yes/yes | yes/yes | yes/yes | yes/yes |
| step36_nonest_nat_s41 | C | 68.9% | 70.6% | yes/yes | yes/yes | no/no | no/no |
| step36_nonest_nat_s41 | local_replacement | 98.4% | 98.3% | no/no | no/no | no/no | no/no |
| step36_nonest_nat_s42 | potential_lambda | 46.6% | 37.6% | yes/yes | yes/yes | yes/yes | yes/yes |
| step36_nonest_nat_s42 | C | 71.8% | 71.9% | yes/yes | yes/yes | no/no | no/no |
| step36_nonest_nat_s42 | local_replacement | 96.8% | 97.1% | no/no | no/no | no/no | no/no |
| step38_cbc1951_broods_free_s41 | potential_lambda | 23.9% | 24.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step38_cbc1951_broods_free_s41 | C | 88.7% | 88.7% | no/no | no/no | no/no | no/no |
| step38_cbc1951_broods_free_s41 | local_replacement | 96.3% | 96.7% | no/no | no/no | no/no | no/no |
| step38_cbc1951_circle2_free_s41 | potential_lambda | 20.3% | 21.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step38_cbc1951_circle2_free_s41 | C | 88.2% | 87.9% | no/no | no/no | no/no | no/no |
| step38_cbc1951_circle2_free_s41 | local_replacement | 95.5% | 96.1% | no/no | no/no | no/no | no/no |
| step38_cbc1951_free_s41 | potential_lambda | 23.7% | 23.6% | yes/yes | yes/yes | yes/yes | yes/yes |
| step38_cbc1951_free_s41 | C | 87.6% | 87.8% | no/no | no/no | no/no | no/no |
| step38_cbc1951_free_s41 | local_replacement | 95.5% | 96.3% | no/no | no/no | no/no | no/no |
| step38_cbc1951_free_s42 | potential_lambda | 22.9% | 22.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step38_cbc1951_free_s42 | C | 87.1% | 87.5% | no/no | no/no | no/no | no/no |
| step38_cbc1951_free_s42 | local_replacement | 95.3% | 96.1% | no/no | no/no | no/no | no/no |
| step38_cbc1951_lo_s41 | potential_lambda | 23.4% | 22.6% | yes/yes | yes/yes | yes/yes | yes/yes |
| step38_cbc1951_lo_s41 | C | 87.9% | 87.8% | no/no | no/no | no/no | no/no |
| step38_cbc1951_lo_s41 | local_replacement | 96.1% | 96.6% | no/no | no/no | no/no | no/no |
| step38_cbc1951_mid_s41 | potential_lambda | 22.6% | 22.2% | yes/yes | yes/yes | yes/yes | yes/yes |
| step38_cbc1951_mid_s41 | C | 86.6% | 86.7% | no/no | no/no | no/no | no/no |
| step38_cbc1951_mid_s41 | local_replacement | 94.7% | 95.7% | no/no | no/no | no/no | no/no |
| step38_cbc1951_mid_s42 | potential_lambda | 21.6% | 21.9% | yes/yes | yes/yes | yes/yes | yes/yes |
| step38_cbc1951_mid_s42 | C | 86.3% | 85.9% | no/no | no/no | no/no | no/no |
| step38_cbc1951_mid_s42 | local_replacement | 95.3% | 96.1% | no/no | no/no | no/no | no/no |
| step38_cbc1951_noncbc_free_s41 | potential_lambda | 60.0% | 51.7% | yes/yes | yes/yes | no/yes | no/yes |
| step38_cbc1951_noncbc_free_s41 | C | 81.8% | 76.1% | yes/yes | no/no | no/no | no/no |
| step38_cbc1951_noncbc_free_s41 | local_replacement | 96.3% | 96.8% | no/no | no/no | no/no | no/no |
| step38_cbc1951_noncbc_free_s42 | potential_lambda | 59.5% | 51.1% | yes/yes | yes/yes | no/yes | no/yes |
| step38_cbc1951_noncbc_free_s42 | C | 80.8% | 75.6% | yes/yes | no/no | no/no | no/no |
| step38_cbc1951_noncbc_free_s42 | local_replacement | 96.6% | 97.1% | no/no | no/no | no/no | no/no |
| step39_adult_priority_fixed_s41 | potential_lambda | 0.0% | 0.1% | yes/yes | yes/yes | yes/yes | yes/yes |
| step39_adult_priority_fixed_s41 | C | 38.7% | 60.6% | yes/yes | no/yes | no/no | no/no |
| step39_adult_priority_fixed_s41 | local_replacement | 31.1% | 53.2% | yes/yes | no/yes | no/no | no/no |
| step39_adult_priority_fixed_s42 | potential_lambda | 0.0% | 0.1% | yes/yes | yes/yes | yes/yes | yes/yes |
| step39_adult_priority_fixed_s42 | C | 42.4% | 62.0% | yes/yes | no/yes | no/no | no/no |
| step39_adult_priority_fixed_s42 | local_replacement | 36.1% | 55.3% | yes/yes | yes/yes | no/no | no/no |
| step39_kfloor1p25_s41 | potential_lambda | 59.5% | 51.2% | yes/yes | yes/yes | no/yes | no/yes |
| step39_kfloor1p25_s41 | C | 81.1% | 76.1% | yes/yes | no/no | no/no | no/no |
| step39_kfloor1p25_s41 | local_replacement | 96.8% | 97.2% | no/no | no/no | no/no | no/no |
| step39_kfloor1p25_s42 | potential_lambda | 59.5% | 51.2% | yes/yes | yes/yes | no/yes | no/yes |
| step39_kfloor1p25_s42 | C | 80.0% | 75.5% | yes/yes | no/no | no/no | no/no |
| step39_kfloor1p25_s42 | local_replacement | 96.8% | 97.3% | no/no | no/no | no/no | no/no |
| step39_kfloor4p0_s41 | potential_lambda | 57.1% | 49.6% | yes/yes | yes/yes | no/yes | no/yes |
| step39_kfloor4p0_s41 | C | 87.4% | 82.3% | yes/yes | no/no | no/no | no/no |
| step39_kfloor4p0_s41 | local_replacement | 91.1% | 91.7% | no/no | no/no | no/no | no/no |
| step39_kfloor4p0_s42 | potential_lambda | 59.7% | 51.3% | yes/yes | yes/yes | no/yes | no/yes |
| step39_kfloor4p0_s42 | C | 78.9% | 75.1% | yes/yes | no/no | no/no | no/no |
| step39_kfloor4p0_s42 | local_replacement | 96.8% | 97.3% | no/no | no/no | no/no | no/no |
| step39_shared_fixed_s41 | potential_lambda | 0.5% | 0.4% | yes/yes | yes/yes | yes/yes | yes/yes |
| step39_shared_fixed_s41 | C | 44.2% | 63.3% | yes/yes | yes/yes | no/no | no/no |
| step39_shared_fixed_s41 | local_replacement | 37.4% | 56.7% | yes/yes | yes/yes | no/no | no/no |
| step39_shared_fixed_s42 | potential_lambda | 0.5% | 0.4% | yes/yes | yes/yes | yes/yes | yes/yes |
| step39_shared_fixed_s42 | C | 45.3% | 63.0% | yes/yes | yes/yes | no/no | no/no |
| step39_shared_fixed_s42 | local_replacement | 39.2% | 57.4% | yes/yes | yes/yes | no/no | no/no |
| step39_veit_fixed_s41 | potential_lambda | 59.5% | 51.1% | yes/yes | yes/yes | no/yes | no/yes |
| step39_veit_fixed_s41 | C | 81.1% | 75.6% | yes/yes | no/no | no/no | no/no |
| step39_veit_fixed_s41 | local_replacement | 96.6% | 97.1% | no/no | no/no | no/no | no/no |
| step39_veit_fixed_s42 | potential_lambda | 58.9% | 50.8% | yes/yes | yes/yes | no/yes | no/yes |
| step39_veit_fixed_s42 | C | 81.1% | 75.6% | yes/yes | no/no | no/no | no/no |
| step39_veit_fixed_s42 | local_replacement | 96.6% | 97.1% | no/no | no/no | no/no | no/no |
| step40_rangev2_historical_s41 | potential_lambda | 61.3% | 54.0% | yes/yes | yes/yes | no/yes | no/yes |
| step40_rangev2_historical_s41 | C | 79.2% | 73.4% | yes/yes | no/no | no/no | no/no |
| step40_rangev2_historical_s41 | local_replacement | 96.1% | 96.4% | no/no | no/no | no/no | no/no |
| step40_rangev2_historical_s42 | potential_lambda | 60.3% | 53.0% | yes/yes | yes/yes | no/yes | no/yes |
| step40_rangev2_historical_s42 | C | 78.9% | 73.1% | yes/yes | no/no | no/no | no/no |
| step40_rangev2_historical_s42 | local_replacement | 96.1% | 96.6% | no/no | no/no | no/no | no/no |
| step40_rangev2_invasion_s41 | potential_lambda | 63.2% | 55.5% | yes/yes | yes/yes | no/no | no/no |
| step40_rangev2_invasion_s41 | C | 80.8% | 77.3% | yes/yes | no/no | no/no | no/no |
| step40_rangev2_invasion_s41 | local_replacement | 95.3% | 94.4% | no/no | no/no | no/no | no/no |
| step40_rangev2_invasion_s42 | potential_lambda | 62.9% | 55.3% | yes/yes | yes/yes | no/no | no/no |
| step40_rangev2_invasion_s42 | C | 80.5% | 76.4% | yes/yes | no/no | no/no | no/no |
| step40_rangev2_invasion_s42 | local_replacement | 95.3% | 94.8% | no/no | no/no | no/no | no/no |
| step41_rational_adult20_s41 | potential_lambda | 38.7% | 33.2% | yes/yes | yes/yes | yes/yes | yes/yes |
| step41_rational_adult20_s41 | C | 82.6% | 79.4% | yes/yes | no/no | no/no | no/no |
| step41_rational_adult20_s41 | local_replacement | 85.5% | 86.9% | yes/yes | no/no | no/no | no/no |
| step41_rational_adult20_s42 | potential_lambda | 39.2% | 33.7% | yes/yes | yes/yes | yes/yes | yes/yes |
| step41_rational_adult20_s42 | C | 82.6% | 79.7% | yes/yes | no/no | no/no | no/no |
| step41_rational_adult20_s42 | local_replacement | 86.6% | 87.8% | yes/yes | no/no | no/no | no/no |
| step41_rational_shared_s41 | potential_lambda | 42.6% | 36.3% | yes/yes | yes/yes | no/yes | no/yes |
| step41_rational_shared_s41 | C | 84.2% | 81.7% | yes/yes | no/no | no/no | no/no |
| step41_rational_shared_s41 | local_replacement | 86.8% | 88.1% | yes/yes | no/no | no/no | no/no |
| step41_rational_shared_s42 | potential_lambda | 43.7% | 37.0% | yes/yes | yes/yes | no/yes | no/yes |
| step41_rational_shared_s42 | C | 84.5% | 82.1% | yes/yes | no/no | no/no | no/no |
| step41_rational_shared_s42 | local_replacement | 88.2% | 88.4% | yes/yes | no/no | no/no | no/no |
| step41_veit_adult20_s41 | potential_lambda | 57.9% | 49.3% | yes/yes | yes/yes | no/yes | no/yes |
| step41_veit_adult20_s41 | C | 78.4% | 73.7% | yes/yes | no/no | no/no | no/no |
| step41_veit_adult20_s41 | local_replacement | 97.9% | 98.5% | no/no | no/no | no/no | no/no |
| step41_veit_adult20_s42 | potential_lambda | 57.6% | 49.2% | yes/yes | yes/yes | no/yes | no/yes |
| step41_veit_adult20_s42 | C | 78.4% | 73.7% | yes/yes | no/no | no/no | no/no |
| step41_veit_adult20_s42 | local_replacement | 97.9% | 98.5% | no/no | no/no | no/no | no/no |
| step42_power2_rational_adult20_s41 | potential_lambda | 48.4% | 39.9% | yes/yes | yes/yes | yes/yes | yes/yes |
| step42_power2_rational_adult20_s41 | C | 83.7% | 78.7% | yes/yes | no/no | no/no | no/no |
| step42_power2_rational_adult20_s41 | local_replacement | 92.4% | 93.1% | yes/yes | no/no | no/no | no/no |
| step42_power2_rational_adult20_s42 | potential_lambda | 49.5% | 40.4% | yes/yes | yes/yes | yes/yes | yes/yes |
| step42_power2_rational_adult20_s42 | C | 83.7% | 78.9% | yes/yes | no/no | no/no | no/no |
| step42_power2_rational_adult20_s42 | local_replacement | 92.6% | 93.3% | yes/yes | no/no | no/no | no/no |
| step42_power2_veit_s41 | potential_lambda | 60.3% | 51.1% | yes/yes | yes/yes | no/yes | no/yes |
| step42_power2_veit_s41 | C | 82.6% | 77.3% | yes/yes | no/no | no/no | no/no |
| step42_power2_veit_s41 | local_replacement | 98.2% | 98.6% | no/no | no/no | no/no | no/no |
| step42_power2_veit_s42 | potential_lambda | 61.8% | 51.5% | yes/yes | yes/yes | no/yes | no/yes |
| step42_power2_veit_s42 | C | 83.4% | 79.2% | yes/yes | no/no | no/no | no/no |
| step42_power2_veit_s42 | local_replacement | 97.6% | 97.8% | no/no | no/no | no/no | no/no |
| step43_factor_p1_rational_adult20_s41 | potential_lambda | 39.2% | 33.8% | yes/yes | yes/yes | yes/yes | yes/yes |
| step43_factor_p1_rational_adult20_s41 | C | 82.9% | 79.5% | yes/yes | no/no | no/no | no/no |
| step43_factor_p1_rational_adult20_s41 | local_replacement | 86.1% | 87.5% | yes/yes | no/no | no/no | no/no |
| step43_factor_p1_rational_adult20_s42 | potential_lambda | 38.9% | 33.5% | yes/yes | yes/yes | yes/yes | yes/yes |
| step43_factor_p1_rational_adult20_s42 | C | 82.6% | 79.4% | yes/yes | no/no | no/no | no/no |
| step43_factor_p1_rational_adult20_s42 | local_replacement | 86.1% | 87.4% | yes/yes | no/no | no/no | no/no |
| step43_factor_p2_rational_adult20_s41 | potential_lambda | 47.6% | 39.4% | yes/yes | yes/yes | yes/yes | yes/yes |
| step43_factor_p2_rational_adult20_s41 | C | 83.2% | 78.5% | yes/yes | no/no | no/no | no/no |
| step43_factor_p2_rational_adult20_s41 | local_replacement | 92.1% | 92.9% | yes/yes | no/no | no/no | no/no |
| step43_factor_p2_rational_adult20_s42 | potential_lambda | 49.2% | 40.5% | yes/yes | yes/yes | no/yes | no/yes |
| step43_factor_p2_rational_adult20_s42 | C | 83.7% | 78.8% | yes/yes | no/no | no/no | no/no |
| step43_factor_p2_rational_adult20_s42 | local_replacement | 92.4% | 93.1% | yes/yes | no/no | no/no | no/no |
| step43_factor_p2_veit_s41 | potential_lambda | 58.4% | 49.4% | yes/yes | yes/yes | no/yes | no/yes |
| step43_factor_p2_veit_s41 | C | 80.5% | 76.5% | yes/yes | no/no | no/no | no/no |
| step43_factor_p2_veit_s41 | local_replacement | 97.6% | 98.1% | no/no | no/no | no/no | no/no |
| step43_factor_p2_veit_s42 | potential_lambda | 59.5% | 50.3% | yes/yes | yes/yes | no/yes | no/yes |
| step43_factor_p2_veit_s42 | C | 79.7% | 76.0% | yes/yes | no/no | no/no | no/no |
| step43_factor_p2_veit_s42 | local_replacement | 97.9% | 98.2% | no/no | no/no | no/no | no/no |

## Summary by era and metric

| Era | Metric | Primary sink-share range | Broad sink-share range | Primary bridge fits (4/8) | Broad bridge fits (4/8) | Interior bridge fits (4/8) | Coastal bridge fits (4/8) |
|---|---|---|---|---|---|---|---|
| 1951-1965 | potential_lambda | 4.5%-98.4% | 7.1%-85.8% | 5/5 | 4/4 | 4/4 | 4/4 |
| 1951-1965 | C | 81.8%-100.0% | 62.4%-93.0% | 2/2 | 0/1 | 0/0 | 0/0 |
| 1951-1965 | local_replacement | 87.6%-99.5% | 66.1%-98.0% | 0/0 | 0/0 | 0/0 | 0/0 |
| 1966-1985 | potential_lambda | 2.6%-96.6% | 1.5%-81.3% | 43/43 | 4/4 | 4/4 | 4/4 |
| 1966-1985 | C | 73.4%-99.2% | 55.8%-92.1% | 3/4 | 0/3 | 0/0 | 0/0 |
| 1966-1985 | local_replacement | 82.9%-98.7% | 62.6%-97.2% | 2/4 | 1/3 | 0/0 | 0/0 |
| 2010-2025 | potential_lambda | 0.0%-63.2% | 0.1%-55.5% | 101/101 | 101/101 | 57/99 | 57/99 |
| 2010-2025 | C | 38.7%-90.5% | 60.6%-88.7% | 93/93 | 10/15 | 0/0 | 0/0 |
| 2010-2025 | local_replacement | 31.1%-98.4% | 53.2%-98.7% | 20/20 | 3/4 | 0/0 | 0/0 |

## Annual bridges (1951-2025)

| Metric | Endpoint test | Fits with at least one bridge year (4/8) | Model-years with bridges (4/8) |
|---|---|---|---|
| potential_lambda | plains_primary | 101/101 | 5100/5167 |
| potential_lambda | plains_broad | 101/101 | 2387/2648 |
| potential_lambda | continental_interior | 61/99 | 1354/2274 |
| potential_lambda | coastal_endpoints | 61/99 | 1354/2274 |
| potential_lambda | continental_interior_full_domain | 61/99 | 1354/2274 |
| potential_lambda | coastal_full_domain | 61/99 | 1354/2274 |
| C | plains_primary | 101/101 | 2722/2834 |
| C | plains_broad | 18/23 | 254/382 |
| C | continental_interior | 1/4 | 3/62 |
| C | coastal_endpoints | 1/4 | 1/38 |
| C | continental_interior_full_domain | 1/4 | 3/62 |
| C | coastal_full_domain | 1/4 | 1/38 |
| local_replacement | plains_primary | 99/99 | 1371/1434 |
| local_replacement | plains_broad | 7/12 | 85/126 |
| local_replacement | continental_interior | 0/4 | 0/23 |
| local_replacement | coastal_endpoints | 0/4 | 0/18 |
| local_replacement | continental_interior_full_domain | 0/4 | 0/23 |
| local_replacement | coastal_full_domain | 0/4 | 0/18 |
