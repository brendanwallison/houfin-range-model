# Overnight progress, November 3, 2026, 00:12 CST

Four newly completed accepted fits since the November 2 evening update: both seeds of the smaller standalone capacity floor, seed 41 of the larger standalone floor, and seed 41 of gradual juvenile departure with adult departure fixed to one fifth of juvenile departure. Seed 42 of the larger floor is running. These comparisons retain the 1951 established-state initialization, shared habitat field, weak absolute clumped Allee effect, NestWatch, and original absence supervision; they do not add CBC count likelihood or a native departure scalar.

## Correction: lambda inflation means historical increase

The user-defined pathology is excessive change in potential growth through time. The modern population-weighted level table below describes a separate diagnostic and does not measure lambda inflation. Calling a high modern level "lambda inflation" in the conversational update was incorrect.

The following uses equal weights on the same land cells within each region in every model and both eras. Early is 1902-1915 and modern is 2010-2025. Entries are absolute changes in mean lambda, modern minus early, from the biological replay's lambda0 field.

| Configuration | All land | East | Great Plains | West |
|---|---:|---:|---:|---:|
| Original baseline, seed 41 | +.758 | +.973 | +.621 | +.646 |
| Smaller floor, seed 41 | +.761 | +.975 | +.627 | +.649 |
| Smaller floor, seed 42 | +.761 | +.975 | +.627 | +.650 |
| Larger floor, seed 41 | +.843 | +1.156 | +.724 | +.700 |
| Gradual departure with reduced adult departure, seed 41 | +.953 | +1.266 | +.928 | +.754 |

Whole-land early-to-modern means: baseline .414 to 1.172; smaller floor seed 41 .414 to 1.174; smaller floor seed 42 .415 to 1.175; larger floor .373 to 1.216; gradual departure .468 to 1.421. Thus the smaller floor leaves temporal inflation essentially unchanged, while the larger floor and gradual departure increase it in all three regions. This conclusion follows from temporal differences, independently of their modern population-weighted levels. These are fitted historical fields, not independent observations of early growth.

## Results and tradeoffs

| Modern diagnostic | Matched original departure control, seed 41 | Smaller floor, seed 41 | Larger floor, seed 41 | Gradual departure with reduced adult departure, seed 41 |
|---|---:|---:|---:|---:|
| Population-weighted median adult survival | .567 | .570 | .416 | .442 |
| Population-weighted median juvenile/adult survival ratio | .473 | .466 | .975 | .976 |
| Population-weighted median maximum reproductive output | 8.38 | 8.47 | 6.32 | 6.41 |
| Population-weighted mean potential annual growth multiplier | 3.016 | 3.017 | 3.270 | 3.436 |
| Population share with juvenile/adult survival ratio above .95 | 0% | 0% | 65.6% | 67.1% |
| Median fitted invasion arrival, years relative to observation | -3 | -3 | -3 | -3 |
| Median arrival under frozen-environment, release-only counterfactual | -26 | -26 | -27 | -26 |

Potential growth is calculated with the actual yearling-breeding life cycle, before mating, crowding, disease and movement losses. It is not realized growth. The maximum reproductive output falls in the last two fits, but potential growth increases because juvenile survival rises. Legacy cached metrics use a different growth definition in some fields; this table uses the biological replay's lambda0 field, not the legacy metric. Negative timing means early; all timing comparisons use the same saved 10% threshold. No invasion-timing improvement is established.

Smaller-floor seed 42 closely reproduces seed 41: adult survival .569, juvenile/adult ratio .468, maximum output 8.46, potential-growth mean 3.018. This is replicated stability for that perturbation, not evidence of general identification.

## Geographic establishment and source-sink processes

Fractions below are modern regional land assigned positive potential growth but no viable density because of the K/Allee interaction. Genuine demographic sinks are counted separately. These regional supports follow the contribution audit; the bridge table below uses its explicitly narrower latitude restriction.

| Region | Original control | Smaller floor | Larger floor | Gradual departure |
|---|---:|---:|---:|---:|
| East | .46% | .46% | 0% | 0% |
| Great Plains, 100-95W | 1.91% | .85% | .42% | .21% |
| West | 1.76% | 1.61% | .29% | .24% |

The larger floor substantially removes this mechanism without making the Plains uniformly good habitat: fundamental demographic sinks still occupy 63.8% of the contribution audit's Plains land, versus 65.7% in control. The gradual-departure fit lowers this fraction to 48.5%, a consequential habitat change rather than only a redistribution among vital rates. At positive modern survey records, K/Allee nonviability falls from 1.54% in control to .057% with the larger floor; the smaller floor leaves 1.60%. A floor above the fixed mating scale does not mathematically remove every positive-growth nonviable cell.

| Modern realized measure, East / Plains / West | Original control | Smaller floor | Larger floor | Gradual departure |
|---|---|---|---|---|
| Origin-attributed breeding contribution | .990 / .879 / .988 | .990 / .878 / .988 | .990 / .860 / .988 | .993 / .924 / .987 |
| Successful export per starting breeder | .263 / .388 / .494 | .261 / .384 / .495 | .229 / .270 / .508 | .133 / .146 / .215 |
| Local replacement excluding movement | .894 / .821 / .832 | .894 / .821 / .830 | .942 / .869 / .909 | .960 / .916 / .943 |

These are population/exposure-weighted regional summaries, not claims that maps agree. The small floor barely changes aggregate processes. The larger floor changes Plains retention/exchange and local replacement despite similar eastern/western regional contribution totals. Gradual departure retains far more local production and reduces successful exchange substantially, but it does not nearly eliminate exchange as strict overflow did.

On equal-weight land cells within 25-49N and 100-95W, modern Plains contribution-sink shares are 81.1%, 81.1%, 87.4%, and 82.6%, respectively; local-replacement sink shares are 96.6%, 96.8%, 91.1%, and 85.5%. None of these four modern era maps has a coast-to-coast source bridge under contribution or actual-state local replacement, with either shared-edge or diagonal connectivity. Gradual departure DOES produce a shared-edge bridge in potential growth; control and both floors have only diagonal potential-growth bridges. This distinction must accompany any statement about preserving the Plains barrier. Annual and historical-era results remain in the connectivity artifact and are not replaced by these modern summaries.

Successful off-cell movement distances for larger floor versus control, East / Plains / West, are 247 / 254 / 227 km versus 237 / 242 / 215 km. Smaller floor changes distances little. The gradual-departure distance/finite-propagule follow-up is not yet present in the collected movement report; missing is not a pass. These are successful permanent modeled movements, not seasonal movements or all-bird distances.

## Prediction, coverage and interpretation

Raw temporal/spatial held-out log scores change from -1.787/-1.431 in control to -1.787/-1.429 with the small floor, -1.778/-1.415 with the larger floor, and -1.774/-1.400 with gradual departure. Larger values are better, but paired uncertainty is not calculated in this update, and these modest score gains do not outweigh the whole-pathology evidence by themselves.

Historical habitat/growth shifts remain; fitted and frozen-environment invasion timing are not improved. Both larger-floor and gradual-departure fits revive the juvenile survival ceiling and retain high potential growth. The shared field, fixed kernels and absence of additional native departure reduce some escape routes structurally, but do not establish that field amplitudes, movement survival, historical extrapolation or observation conversion are identified. Initial abundance, age split and within-cell support remain conditional assumptions. No new HMC geometry/sampling evidence is available for these completed variants. No fit is certified across the complete checklist.

The useful result is that the larger floor blocks much of the tiny-capacity route while preserving genuinely poor habitat, exposing a different compensation in survival. This supports the already queued combinations with altered crowding, reproductive variation and local settlement; it does not show that the floor alone reduces lambda inflation. Gradual departure is far less extreme than strict overflow but is not yet a lower-growth solution.

## Compute status

At 00:12 CST, A100-small job 3544518 remains active, with about 5h51m remaining. The larger-floor seed-42 fit is running. Three full-node requests remain pending: development 3548096 is now estimated for 01:50 CST, while regular requests 3547292 and 3548331 have no start estimate. These estimates can move. The four crowding-curvature fits and eighteen floor-combination fits remain pending; the expected overnight parallel throughput has not yet materialized. Persistent local collection and resource watchers 1154 and 1156 are running; the three completed floor movement follow-ups succeeded.

Evidence: FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json, CONTRIBUTION_RESULTS_20261101.md, CROSS_MODEL_PROCESS_AUDIT_20261101.md, PLAINS_SOURCE_CONNECTIVITY_20261101.md, MOVEMENT_ESTABLISHMENT_RESULTS_20261101.md, and live later_state_status.json. Artifacts retain the historical filename dates but are updated by watchers. This note records the snapshot and explicitly distinguishes completed and missing checks.