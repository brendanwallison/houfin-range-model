# Does regional change coherence grow with temporal span?

Registered 2026-10-26 before inspecting any results from this extension.

E014 supplies one spatial covariance curve for the contrast 1966–1986 versus
2005–2025. Its roughly 0.56–0.59 value below 10 km is neither an estimate of a
spatial decay rate nor a comparison of short versus long temporal spans. E004's
temporal variograms measure total turnover, not the regional share of that
turnover. E028/E028b find modest fading of local deviations over approximately
11 years, including within unchanged-observer strata; that is compatible with,
but does not establish, increasing regional coherence over longer spans.

Question: do nearby routes' shared change and their share of total change
increase from 5 to 10, 20, 30, 40 and 50 years?

Use E014's route counts, reference species, log1p mean counts, Ruzicka kernel,
protocol quality screen and independent ABBA year halves. No DESK model or focal
species response is fitted. Initial epochs are four years wide, with exactly
two surveyed years in each half. Anchor 2016–2019 permits all six spans without
overlapping epochs; anchor 2006–2009 permits spans through 40 years as an era
sensitivity check. Four-year epochs deliberately trade precision for a
non-overlapping 5-year contrast; this is not a numerical replication of E014's
21-year epochs and three-year halves.

For each anchor, report an available panel at each span and a fixed panel of
routes observed in every required epoch. The latter uses the same route pairs
across spans. Also report a fixed panel restricted to spans 5–30 years to check
whether requiring 1960s coverage drives a result. Do not compare ratios across
different panels as though they identified a temporal trend.
Also use pairwise fixed panels (5 years versus each longer span), which retain
more routes than requiring complete coverage in every intervening epoch.

For every distance bin, report cross-route change covariance C, pair-matched
mean same-route change energy V, C/V, and V−C. Also retain the original E014
denominator (mean energy across all eligible routes). Symmetrize both cross-half
orientations. C/V is an aggregate kernel covariance ratio, not Pearson
correlation, a measured decay rate, or a universal predictable/irreducible split.
Do not clip negative estimates or ratios above one. Keep continental mean change
in this regional-sharing diagnostic, as E014 did; this is not centered predictor
skill. Report support even where there are too few pairs to interpret a curve.

Spatial uncertainty: endpoint-weighted 6×6 grid-block multiplier bootstrap,
using the same block draws for every span in a fixed panel. Resample each block
with nonnegative mean-one exponential weights; a pair's weight is the product
of its two endpoint weights (one weight for within-block pairs). Report paired
span contrasts; these intervals are exploratory, not independent pairwise
sampling intervals. Observer stability is a separate sensitivity restricted to
routes with the same observer across the two epochs and within each epoch;
report its reduced support, particularly at long spans. Unchanged observer
identity does not exclude observer drift or other shared measurement biases.

Interpretation: increasing C/V, accompanied by increasing C relative to V−C on
fixed panels, supports increasingly dominant regional structure. Flat C/V does
not imply no regional predictability; C itself can grow. A result on 1966 onward
breeding observations cannot establish 1902–1940 winter behavior or unobserved
early summer behavior. Coherence also does not establish out-of-time predictive
skill of a fitted regional reconstruction.

Status: completed locally on CPU. No training run, GPU allocation or model
selection was involved. This is an exploratory observation diagnostic.

## Results

First reproduced the original E014 estimator from the newly loaded count cache:
5589 route-epochs, 1513 routes, same-route energy 0.35522376768325875, and
below-10-km ratio 0.5581380731193527 (original 0.5581380731193526). The nearby
40-year coherence estimate had only 37 route pairs. This reproduction verifies
the cache/operator connection; it does not validate an ecological interpretation.

The new fixed panels through 30 years have sufficient support starting at
30–60 km. Every row within a column uses exactly the same routes and route pairs.

| Temporal span | Endpoint 2016–2019: 211 routes, 70 pairs | Endpoint 2006–2009: 199 routes, 62 pairs |
|---|---:|---:|
| 5 years | 0.069 [−0.047, 0.225] | 0.007 [−0.109, 0.252] |
| 10 years | 0.184 [0.101, 0.292] | 0.186 [0.092, 0.307] |
| 20 years | 0.216 [0.163, 0.283] | 0.277 [0.189, 0.352] |
| 30 years | 0.325 [0.248, 0.393] | 0.380 [0.305, 0.444] |

Entries are C/V with exploratory 95% spatial-block multiplier intervals. The
paired 30-minus-5-year difference is +0.255 [0.104, 0.376] for the 2016 anchor
and +0.374 [0.126, 0.496] for the 2006 anchor. Both 60–150 km and 150–300 km
also show increases on these fixed panels. The anchor checks partly reuse early
records, so they are not fully independent replications.

This is not solely a denominator effect. In the 2016 anchor's 30–60 km bin,
shared covariance grows from 0.0043 to 0.1076; pair-matched total energy grows
from 0.0622 to 0.3315. However, V−C also grows, from 0.0579 to 0.2239. Regional
sharing grows relatively; these results do not show local differences shrinking
in absolute size or identify a restoring dynamical mechanism.

Requiring every epoch through 50 years leaves only 16 routes. The more useful
long-span checks therefore compare each longer span against five years on its
own fixed panel, using the 2016 endpoint:

| Matched contrast at 30–60 km | Routes / pairs | 5-year C/V | Longer-span C/V | Paired difference |
|---|---:|---:|---:|---:|
| 40 versus 5 years | 328 / 127 | 0.042 | 0.413 | +0.371 [0.224, 0.525] |
| 50 versus 5 years | 76 / 45 | 0.232 | 0.491 | +0.259 [0.016, 0.482] |

These are different panels: they do not establish a single monotonic 5–50-year
curve. The 50-year result has much less support. At below 10 km, the matched
5-versus-40-year panel has four pairs and the 5-versus-50-year panel has two;
we cannot give a reliable nearest-neighbor lag curve from this strict design.

Observer sensitivity: at 60–150 km, the 2016 anchor's matched, unchanged-observer
5-versus-20-year panel has 221 routes / 249 pairs. C/V rises from 0.083 to
0.169, paired difference +0.086 [0.005, 0.180]. The 2006 anchor has 137 routes /
135 pairs, 0.031 to 0.202, difference +0.172 [−0.003, 0.342]. This weakens an
explanation solely in terms of observer replacements, but is not decisive.
Support becomes poor for unchanged-observer comparisons at close distances over
30 years, and nearly vanishes at 40–50 years. Ecological change cannot be cleanly
separated from observer drift, protocol/route effects or other persistent biases.

After inspecting the very small panels, added a reporting guard: retain counts
and point estimates but suppress uncertainty intervals unless a bin has at least
30 pairs and ten endpoint blocks. This was not an original preregistered rule;
it prevents apparently precise intervals from one or a few blocks. Thresholds do
not make supported bins independent, or their uncertainty definitive.

## Interpretation and next validation

The earlier E014 result did not answer the question about temporal span. The new
matched diagnostic supports the user's proposed pattern: longer-span observed
change is more regionally shared, especially from five to 20–30 years. Therefore
the below-10-km number must not be used as a universal predictable-change ceiling
or a reason to dismiss community-level spatial borrowing.

That does not establish a reliable reconstruction method. Coherent observations
can include coherent measurement effects; continental mean change contributes to
this uncentered kernel statistic; and a regional signal measured at two endpoints
is not necessarily predictable when one endpoint is withheld. The next modeling
validation must test recovery of these measured shared changes with withheld
observations and the existing endpoint/modern-memorization controls. This test
does not answer early-century winter behavior or unobserved early summer behavior.

Artifacts: [full results](change_coherence_20261026/result.json),
[summary](change_coherence_20261026/summary.json),
[E014 reproduction](change_coherence_20261026/E014_reproduction.json),
[figure](change_coherence_20261026/coherence_by_span.png).
Implementation: `research/exp/final_route_counts_cache.py` and
`research/exp/final_change_coherence.py`. Two independent Gram-matrix/limiting-case
checks passed in `tests/test_change_coherence.py`. The cached original counts live
under `.research_runtime/change_coherence_20261026/counts`; metadata record source
hashes, reference species and observation-file sizes/mtimes. Result JSON records
the analysis source hash, panel route-key hashes and bootstrap settings.
