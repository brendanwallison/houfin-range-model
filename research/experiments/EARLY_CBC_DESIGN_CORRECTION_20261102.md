# Historical CBC supervision: design correction

The replacement is now implemented, validated and queued: [four range-supervision fits](HISTORICAL_RANGE_IMPLEMENTATION_20261102.md). The original effort-model fits below remain held and are not republished. The requested [old/new pseudozero sanity check](HISTORICAL_PSEUDOZERO_COMPARISON_20261102.md) is complete.

The four unstarted `step39_early_*` fits are preserved in `queue/held_early_design`. Movement/K-floor fits and the existing A100 work remain authorized and queued. Frozen source/input files are retained for provenance, not silently changed.

The early-CBC encounter model used missing hours replaced by 20 plus an estimated missing-effort coefficient. Actual retained coverage is:

| Summer years | Rows | Missing effort | Training rows |
|---|---:|---:|---:|
| 1902–1919 | 2,237 | 2,236 | 1,762 |
| 1920–1939 | 3,485 | 3,151 | 2,924 |
| 1940–1950, western rows retained | 551 | 215 | 448 |
| 1951–1962 | 6,354 | 224 | 5,377 |
| 1963–1965 | 2,113 | 43 | 0 |

That missingness adjustment is an assumption about encounter opportunity, not recovered effort. Circle effects and a source scale do not independently identify detection in early mostly-zero records. Feeder fields are present in the export, but completeness/coding in early decades has not been established; numeric zero must not automatically imply verified absence of feeder effort.

The envelope factor explicitly shares the fitted CBC scale: for an exterior zero, its loss is approximately `0.1 * exp(cbc_log_scale) * N`, with N on the scaled population readout. The range constraint therefore interacts with observed CBC data through both population parameters and calibration. It uses annual population rather than the CBC seasonal readout, another explicit difference in interpretation. These are not independent hard absences.

Separate western/eastern hulls avoid an artificial continental hull, but a 700 km buffer only restricts gross range over-expansion. It does not constrain the initialized 1951 population to the release cell or resolve initial delayed spread. Cumulative detections through a five-year bin relax the constraint retrospectively within that bin. This is particularly unsuitable as the sole supervisor of the early lag.

The intended historical constraints must distinguish: (a) the established introduced state at initialization, confined to its supported footprint; (b) subsequent dated expansion/absence evidence; (c) winter observations that may occur away from permanent breeding populations. Existing initialization notes record first next-cell CBC occurrence in 1953. That is recorded occurrence evidence, not proof of a particular first colonization date or perfect detection elsewhere. It also means confinement cannot simply be imposed for all years up to 1966.

Withholding 1963–1965 and spatial blocks was an observation-model validation choice. It reduces supervision precisely where the requested historical constraint should use the available evidence. A corrected primary constrained fit should use the accepted historical evidence through 1965, with separate withheld-data sensitivity fits for validation. Synthetic range constraints need their own explicit ecological scale/tolerance, independent of freely fitted CBC detection calibration. Broad hull/buffer bounds can supplement dated near-founder constraints; they do not replace them.
