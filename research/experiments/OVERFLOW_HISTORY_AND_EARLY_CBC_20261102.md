# Prior movement tests, candidate bases and historical CBC occurrence track

**Implemented November 2:** the exact overflow/adult-priority alternatives, capacity-floor sensitivities and historical CBC detection arms are now staged and published as 16 immutable fits. See [execution details](OVERFLOW_EARLY_EXECUTION_20261102.md). The historical discussion below explains the design; it is no longer a pending-only proposal.

## What was tested before

Checked the October 30 dispersal synthesis, current GRID54 ladder and all 60 original age-model experiment notes in `/home/breallis/houfin/work/houfin/research/GRID54_LADDER_20261029/code/research/age_model/experiments`. These records contain close predecessors, but no recorded test of the exact bounded excess-only, sequential adult-priority allocator proposed on November 2. This is a statement about audited records, not a claim that no private/unrecorded branch could exist.

| Record | Tested mechanism | Finding/relevance |
|---|---|---|
| E012 / run26 | Adult logit offset on a common departure curve | Adult-specific retention was tested; it did not allocate a finite retention budget to adults before young. |
| E037, later E041/E043 | Natal-departure constraint and additional adult fidelity | Related age-specific tests; the October 30 synthesis records adult offsets down to -7.4. The departure layer could become an excessive clamp on kernels that already include local retention. |
| E013 / run27 | Smooth above-K crowding arm added to a free-sign logistic departure curve | With settlement beta~.19 and little gain; without settlement beta~6.98, loss gain~1,025 nats. Both retained sparse-cell exodus. This was not an excess-only retention-capacity allocator. |
| E008 / E074 | Capacity-directed settlement, all birds or dispersers only, several block sizes | Destination redistribution, not outgoing excess/adult-priority allocation. Better-looking MAPs did not prevent refinement contortions. |
| E014 | Fitness-directed settlement; low-quality cells filled only by spillover | An explicit prior spillover idea. Perfect sorting flattened vital-rate fields and transferred spatial structure to K. This is a failure mode the new branch must check. |
| E051 and current ladder | Veit-form p~min(N/K,1), both ages | Current positive density dependence is already tested, but it allows substantial departure well below capacity. |

E014 also documents a large historical source/sink change: K-directed settlement shifted the Plains from roughly 37% fundamental sinks to roughly 1%, while overpredicting Plains counts. That supports the user's warning against inferring cross-version stability from correlations or regional means alone.

Sources in the present repository: [October 30 synthesis](DISPERSAL_EXPRESSIVENESS_PLAN_2026-10-30.md), [GRID54 ladder](GRID54_LADDER_2026-10-29.md). Original note filenames: `E012_departure_by_age.md`, `E013_crowding_departure.md`, `E014_settlement_weight.md`, `E037_natal_dispersal.md`, `E043_kernel_is_dispersal.md`, `E051_veit_lewis.md`, `E074_settlement_revisited.md`.

## Which model versions to use

Use the matched **1951 established-state, no-CBC, no-native-departure** controls, `step38_cbc1951_noncbc_free_s41/s42`, as the principal source/configuration parents. They retain yearling breeding, one shared habitat field, NestWatch, no founder backcast pin and no release-cell exemption. They have modern adult survival ~.56 and juvenile/adult survival ~.47. Those summaries motivate a useful baseline, not biological clearance.

The first controlled comparison should hold the initialized state identical across departure rules, at one explicitly conditional reference near the fitted no-CBC state. This prevents different initial abundances from buying away the movement change. Compare:

1. Existing Veit-form departure.
2. Smooth excess-only departure, shared retention across ages.
3. The same excess-only law with adult-first retention.

Use the same seeds, Allee, habitat representation, observation likelihoods, kernel shapes and fixed smoothing width. Then allow initial abundance to refit/profile in matched followups. This separates the stricter overflow effect from age priority and initializer compensation. The current all-years CBC count likelihood is not part of this first movement comparison. Uncertain NestWatch annual conversion is a later sensitivity rather than another simultaneous primary change.

For this first numerical comparison, C can be the current K in its existing total-population/pre-departure density currency. This tests population retention at capacity without claiming a number of individual breeding territories. A breeder-territory interpretation instead needs a matched adult-equivalent capacity and explicit survival/vacancy timing; these interpretations must not be silently mixed.

## Absolute Allee and aggregation are separate choices

The proposed parents already use **absolute Allee with `aggregation_k=3`**. The mate-finding multiplier is:

    m(N) = 1 - (1 + gamma*N/3)^(-3)

Absolute means that the argument is density N rather than N/K. The finite-k form averages encounter success over a gamma distribution of subcell densities; k→infinity gives the Poisson-encounter form, k=1 the hyperbolic form. It is smooth and broader than the Poisson form, but remains a strong Allee effect. It neither removes establishment thresholds nor explicitly represents spatial flock arrivals, territory occupancy or demographic stochasticity. Keep k and the current Allee scale fixed in the first departure comparison.

Original history: [E079 Allee-form experiment](lambda_audit_20261026/notes/E079_allee_form.md). A prior planning note asserted that demographic stochasticity would necessarily double-count this aggregation; that does not follow from an encounter average alone and is not adopted here.

## A separate pre-BBS CBC occurrence track

No current focal demographic run implements the proposed 1902–1966 presence/range-envelope supervision. The old founder backcast used selected historical counts; the new joint-count branch uses 1951 onward through modern years. Earlier community/DESK CBC-presence experiments are a different target and excluded the focal species from their community training products.

A new raw CBC product has now been prepared independently for winter labels 1902–1966: **18,311 surveyed circle-years**. It preserves positives, recorded-count uncertainty, effort missingness and circle locations. It is not a verified absence map. Dataset: `/mnt/e/Datasets/houfin/joint_observations_20261102/cbc_1902_1966`; coverage receipt: [EARLY_CBC_COVERAGE_20261102.json](EARLY_CBC_COVERAGE_20261102.json). For the eastern >=80W band, 1902–1939 has 2,887 survey rows and no focal detections, but only 103 rows have measured party-hours; 1951–1966 has 2,901 rows, 234 detections and 2,815 with measured effort. This sharp historical quality difference must be retained.

Recommended first branch:

- Preserve BBS/NestWatch as the modern abundance/reproductive likelihoods. Use CBC detected/not-detected observations only for the pre-BBS period, with actual dates and an effort/reporting-aware observation link. Do not use the CBC count magnitude or its modern count trajectory in this branch.
- Keep the winter observation target distinct from breeding establishment. Positive winter records can be seasonal visitors. A simple detection link to the annual state plus a conservative seasonal footprint is an explicit approximation, not an identified full occupancy/detection model from one visit per circle-year.
- Use actual surveyed nondetections, with imperfect detection/reporting represented. Unsurveyed pixels remain unknown in the data likelihood. Early missing finch records on incomplete lists must not be declared certain absences.
- Treat a buffered observed range as an envelope sensitivity. Select buffers at least commensurate with the 54-km grid and circle footprint; compare them rather than presenting one chosen radius as observed truth. Construct separate native-west and introduced-east components, from training observations only. A single coast-to-coast convex hull would bridge the Plains by construction. Presence inside a hull is not automatic, and lack of records outside it is not proof of absence.
- If adding zeros in unsurveyed exterior pixels, label that arm a soft historical-range regularizer, normalize its effective weight by area/time blocks and report buffer/weight sensitivity. Do not manufacture thousands of independent surveyed absences or double-count the CBC observations used to draw the envelope.
- With a 1951 established-state start, use pre-1940 data for the modeled native population/pre-introduction range, and 1951-to-pre-BBS data for introduced-population spread. The local introduced population existed in 1940–1950 but is intentionally not simulated in this branch; its observations cannot be compared to an artificial zero state. Retain them for initialization-history checks, or explicitly left-truncate/exclude the relevant introduced-region likelihood.

CBC winters ending January 1966 correspond principally to the preceding postbreeding year; align dates explicitly so the historical block does not accidentally become a second modern abundance likelihood. Temporal partitioning and using detection instead of counts reduce the direct count-scale conflict, but early constraints still affect shared demographic parameters. BBS transfer and all pathology diagnostics remain evaluation criteria.

This is an export/design result, not a claim that an early-only CBC fit or a range-envelope likelihood has been implemented or queued. First isolate this observation change from the departure-law change; combine successful versions only after their separate effects are understood.

## Literature basis

- [MacKenzie et al. (2002)](https://www.usgs.gov/publications/estimating-site-occupancy-rates-when-detection-probabilities-are-less-one): nondetection is not certain absence; separating occupancy and detection requires suitable information/design. A Bernoulli observation model here should not be sold as identifying both automatically.
- [Phillips et al. (2009)](https://doi.org/10.1890/07-2153.1): sampling bias and background/pseudo-absence selection affect presence-only inference. This supports using actual survey coverage and auditing envelope assumptions rather than blind zero filling.
- [Able & Belthoff (1998)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1689500/): eastern House Finches are partial migrants; winter occurrence is not automatically local breeding establishment.
