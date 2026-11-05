# Configuration assessment correction — November 1, 2026

**Interpretation corrected:** SOURCE_SINK_DIAGNOSTIC_RECONCILIATION_20261101.md distinguishes an expected Plains barrier from the low-K/absolute-Allee artifact. The mask below already selects positive potential growth with no positive equilibrium; its prevalence is not by itself a biological ranking. Existing viability and barrier-operator diagnostics were omitted from the earlier interpretation, and actual demographic/immigration budgets remain unfinished.

The afternoon response called absolute Allee plus native departure a strong parsimonious candidate on survival and western sink summaries. That was an incomplete configuration assessment. The regional establishment check below materially weakens that recommendation. No current configuration has been demonstrated to meet the full pathology checklist.

## Great Plains establishment

Recomputed from the saved modern `allee_dead_mask` and `K_modern` for every accepted real-data fit with cached maps. Full results: REGIONAL_ESTABLISHMENT_20261101.json. Reproduction script: research/exp/regional_establishment_audit.py, with --root, --audit and --out arguments. The whole-domain results reproduce the existing positive-record audit. These use the existing cached definition, not a newly estimated establishment probability. An Allee-dead diagnostic does not mean the simulated cell is necessarily unoccupied: immigration can sustain counts. That possibility is precisely why favorable population-weighted summaries cannot settle this concern.

Primary region matches the existing project scorecard: longitude 100°W–95°W, all modeled latitudes, 472 modeled land cells and 1,115 positive modern BBS records (2010–2025). Records include repeated surveys and are not independent occupied cells.

| Configuration | Allee-dead land cells | Positive records in those cells |
|---|---:|---:|
| Relative Allee reference, seed 41 | 0/472 (0%) | 0/1,115 (0%) |
| Absolute Allee reference, seed 41 | 6/472 (1.27%) | 4/1,115 (0.36%) |
| Absolute + native departure, seed 41 | 23/472 (4.87%) | 59/1,115 (5.29%) |
| Absolute + native departure, seed 42 | 55/472 (11.65%) | 106/1,115 (9.51%) |
| Absolute + DESK v2 + native departure, seed 41 | 5/472 (1.06%) | 5/1,115 (0.45%) |
| Absolute + DESK v2 + native departure, seed 42 | 12/472 (2.54%) | 45/1,115 (4.04%) |

A broader Plains sensitivity, 105°W–95°W and 25°N–49°N, contains 787 cells and 2,213 positive records. Absolute/native seeds 41/42 have 5.21%/10.93% Allee-dead cells and 3.71%/6.64% positive records in those cells. The absolute reference has 1.65% and 1.08%, respectively. Thus the deterioration is not an artifact of choosing only the narrow longitudinal band. Neither definition is a formal Great Plains ecoregion polygon, and these summaries do not establish connectivity or a continuous barrier across the region.

## Later founder injection

The accepted modern absolute/native comparisons do not test the proposed later initialization of an already established population within the release cell. They retain the current release setup and ten-year founder Allee exemption. The recorded founder-count brackets change the amount, not the timing or the established-state interpretation. The earlier CBC-derived founder values are not estimates of a later initialized population.

The live shared queue contains no later-injection script. Old step5_inj1950, step5_inj1955 and step5_inj1960 scripts are in `queue/withdrawn`; the old ladder says 1950/1955 were withdrawn following an inj1960 test. None has an accepted refinement in the refreshed archive. That older test is not a matched evaluation on the current yearling/shared-field/absolute-Allee/NestWatch configuration and does not resolve the user's concern about artificial release-cell establishment.

Therefore the proposed later established-state branch remains untested in the current configuration comparison. Its absence is a work gap, not an unfavorable scientific result. A useful matched test must initialize a localized established state before outside-cell spread, remove the release-cell exemption, bracket plausible abundance/within-cell occupancy rather than rely on a founder prior, and replace rather than duplicate the historical CBC backcast constraint. Exact initialization dates and spatial support need historical evidence; a calendar shift alone is not an adequate implementation of that hypothesis.

## All fifteen pathologies: absolute Allee plus native departure

| Concern | Current assessment and evidence |
|---|---|
| P1 Historical habitat/growth contortion | Unresolved. Eastern land-mean lambda rises .346→1.342 and .419→1.394 from early to modern eras. Shared fields restrict flexibility but do not independently validate that change. |
| P2 Invasion too fast | Unresolved. Fitted 10%-threshold arrivals remain median 3 years early; frozen release-only diagnostics are 37 years early. These counterfactuals do not reconstruct historical habitat. |
| P3 Excess reproduction/local growth | Not cleared. Realized annual fecundity at nest locations is 3.80/4.00, but population-weighted modern median Fmax is approximately 10.0/11.6. These are different quantities. Nest constraints are modern-skewed and annual conversion remains uncertain. |
| P4 Juvenile survival ceiling | Improved for resident survival: median Sj/Sa=.479/.437; no modern model-population share above .95 in these two runs. Effective mover survival remains unmeasured. |
| P5 Demographic compensation | Unresolved. Allee law, annual reproductive conversion, basis and seed can give different demographic explanations with similar count prediction. |
| P6 Immigration-supported sinks | Western fitted-population sink share improves to 2.24%/1.58%. Actual immigration replacement deficits are missing. Plains fundamental-sink land share is still 51.0%/42.7%; fundamental sinks and Allee-dead cells are distinct diagnostics. |
| P7 Tiny K/habitat holes | Not cleared. Plains land-median K is .0262/.0178 relative units, and .81% of positive Plains records have K below .05 route-count units in each seed. Spatial continuity and occupied-location density support still need assessment. |
| P8 Allee-dead zones/establishment | Worsened in the Plains: 4.87%/11.65% of cells and 5.29%/9.51% of positive records. Must accompany the survival/sink improvements in any recommendation. Perturbed establishment and barrier connectivity are not yet evaluated. |
| P9 Founder count, timing and exemption | Unresolved. Existing count brackets retain the release mechanism. Later established-state initialization is missing, and the ten-year exemption remains. |
| P10 Movement distances and tails | Unresolved. Fixed kernel bands restrict one source of flexibility, but actual departure-weighted permanent distances/tails have not been evaluated. |
| P11 Exodus/regathering | Native departure changes the mechanism; settlement attraction is disabled. Departure curves and spatial redistribution still require measurement. A disabled attraction mechanism does not clear departure pathologies. |
| P12 Effective mover/stayer survival | Unresolved. No free decoupled Q in these configurations removes one escape route, but mover-weighted annual survival and transit loss remain missing. |
| P13 Projection flexibility | Shared field is structurally simpler. Field geometry, extrapolation and basis sensitivity still matter; v2/native's seed-41 juvenile ceiling is evidence of a different compensation. |
| P14 Scale inflation | Parameter norms are inventoried, but a common-unit field-amplitude and coefficient-compensation assessment is missing. Fixed observer scale does not settle this. |
| P15 Flexible corrections bypassing spread | Extra trend/correction mechanisms are disabled or zero in these current candidates. This removes those specific routes structurally; it does not establish that the remaining habitat history is biologically credible. |

The defensible conclusion is a tradeoff: native departure improves western sink and resident-survival summaries while worsening Plains establishment in the current basis. DESK v2 alleviates the establishment diagnostic but can restore survival compensation. None is an established preferred whole configuration. Later initialization, immigration requirements, effective mover survival, permanent distances and invasion timing remain central missing comparisons.

Subsequent configuration recommendations must carry the complete P1–P15 assessment, including regional land and positive-observation context and explicit missing diagnostics. New predictive gains or a small set of favorable weighted summaries do not replace that assessment.
