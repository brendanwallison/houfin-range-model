# Interpretation of five completed fits and four checkpointed fits

## Scope and definitions

These are conditional tests of the original density-dependent departure model,
yearling breeding, fixed weak absolute/clumped mate limitation, a K floor four
times its fixed mating scale, and a nominal established population injected in
1951. They retain the original pseudozeros and NestWatch conversion fixed at
2.5 annual attempts. They do not incorporate a new historical CBC likelihood,
uncertain annual reproductive conversion, or native departure.

Lambda inflation here means historical increase in potential local annual
growth, not a high modern growth level. Potential growth is adult survival plus
juvenile survival times maximum reproduction, before crowding, mate limitation
and movement. Tables below use modern minus 1902–1915 mean, with modern defined
as 2010–2025. All comparisons use identical equal-weight land cells between
25°N and 49°N: East ≥95°W, Plains 100°W–95°W, West west of 100°W.
Positive modern BBS-cell comparisons independently show the same main pattern.
These are point estimates, not posterior identification results.

## Historical growth change

| Fit | East | Plains | West | Status |
|---|---:|---:|---:|---|
| Shared control seed 41 | 1.425 | .854 | .902 | Complete |
| Shared control seed 42 | 1.252 | .730 | .822 | Complete |
| New shared control seed 43 | 1.264 | .732 | .830 | Complete |
| Private K, prior SD .10 | 1.462 | .952 | .682 | Complete |
| Private K, prior SD .25 | 1.296 | 1.206 | .756 | Complete |
| All four fields, prior correlation .99 | 1.580 | .997 | .669 | Complete |
| Regional annual correction, prior SD 1% | 1.196 | .683 | .777 | Complete |
| Independent cell-year correction, prior SD .5% | 1.238 | .697 | .816 | Refinement unfinished, 1,450 iterations |
| Independent cell-year correction, prior SD 1% | 1.489 | .892 | .913 | Refinement stopping criterion reached; export/replay unfinished |

The new shared control reproduces seed 42 closely. Seed 41 occupies a different
demographic mode, so intervention comparisons require matched restarts/seeds;
the table does not establish every difference as a causal treatment effect.

Regional corrections reduce historical growth change about 5–7% relative to
controls 42/43 in all three regions. This is a modest benefit, not a resolution.
Small field separation helps the West but increases the historical change in
the Plains; some versions also worsen the East. The pattern persists when
starting the comparison in 1966, so it is not solely early extrapolation.

## What the independent corrections actually are

There are 237,490 independent standard-Normal latent innovations: 55 years
(1966–2020) times 4,318 land cells. After the annual transition, both age classes
in a cell receive the same multiplier exp(sigma*u - sigma²/2). This preserves
age proportions and exact zeros. The multiplier is one outside the active
window. They are direct cell-time abundance corrections, without regional
smoothing or shared regional modes. The nominal .5%/1% describes prior spread,
not the fitted adjustment.

At the saved refinement points, typical fitted annual log adjustments are only
.00219% (.5% case) and .00894% (1% case). Largest single adjustments are about
.115% and .439% in log units. Median accumulated log adjustments over all 55
years are approximately −.069% and −.275%, respectively. Most of that median
negative accumulation is the fixed mean-one lognormal centering term, not a
learned common decline. These corrections have barely been used by the fitted
model; this is not evidence that all ways of representing process error fail.

The .5% checkpoint resembles the seed-42/43 control regime. The 1% checkpoint
has larger growth changes in all three regions, broadly resembling the more
inflated seed-41 regime. Its stopping rule was an objective decrease below
.05 over 20 iterations after 1,252 iterations. That does not prove a well
identified optimum. A saved terminal optimizer state is not a completed
export, population replay, prediction assessment, or pathology audit.

I decoded only the necessary Normal-valued components, accounting explicitly
for the observer-effects zero-sum support transform. Against two completed
fits, decoded components matched their exported values and reconstructed
potential growth matched saved trajectories to maximum absolute error
7.11e−15. No optimization or full population simulation was run for this check.

The other checkpoints (regional .5%, MAP step 375/400; independent 2%, MAP step
75/400) are too early for a useful biological conclusion.

### Optimizer versus prior shrinkage follow-up

Inspection of the frozen model confirms independent standard-Normal priors for
every cell-year innovation. Cached optimizer gradients at the last evaluated
points have innovation-block RMS .000144 (.5% case) and .000198 (1% case),
versus fitted standardized innovation RMS .00359 and .00741. These gradients
belong to the last evaluated point before the final saved update, not a new
gradient evaluation at the saved point. Global gradient norms remain about
46 and 33. Neither result proves convergence, but the small correction-block
gradients argue against a large unexploited first-order opportunity in that
block at the current parameter basin. Both corrections moved during refinement.

An independent prior charges each cell-year separately. At sigma=.01, a coherent
1% log adjustment across 1,000 cells for ten years costs approximately 5,000
negative-log-prior units; a .1% adjustment costs about 50 (ignoring the tiny
centering offset). The regional model can represent a coherent pattern with
far fewer penalized coefficients. Equal marginal annual prior SD therefore
does not imply equal prior support for a coherent historical correction.
Point optimization also does not average over process uncertainty. These are
strong explanations for small fitted corrections, alongside local-basin and
conditioning concerns; tiny fitted values alone cannot distinguish them.

## Tradeoffs in the completed fits

Field separation lowers maximum reproduction: population-weighted medians
fall from about 8.68 in control 43 to 5.51/private-.10, 4.74/private-.25 and
6.10/all-fields-.99. But juvenile/adult stayer-survival ratios rise from .455
to .563, .645 and .618. Lower reproduction alone is not a clean improvement.

The nominally small priors allow appreciable fitted separation. Private K's
effective coefficient norm is 47% or 111% of the shared coefficient norm in
the .10/.25 cases. The shared vector itself shrinks. Those are coefficient
comparisons, not percentages of geographic variance, but they show why prior
labels cannot stand in for fitted field separation.

The Plains remains predominantly a sink at the fitted population state:
roughly 94–97% of cells fail local replacement, and 78–84% have breeding-origin
contribution below one. Nevertheless, potential-growth sink area falls from
59.7% in control 43 to 28.4% in private-.25. Private-.25 also creates an
edge-connected coast-to-coast potential-growth source path at threshold one.
No completed case has a coast-to-coast source path under breeding-origin
contribution or actual-state local replacement, using either edge or diagonal
connectivity. A bridge across the narrow Plains band alone is a different test.

Positive-growth cells with no viable K/Allee equilibrium occupy 2.1% of Plains
cells in control 43, versus 5.0%, 7.1%, 5.0% in the three separation tests and
1.8% under regional corrections. The tested floor has not eliminated this
route. This is distinct from learning genuinely unfavorable habitat where
growth itself cannot sustain a population; the latter is expected in the Plains.

Higher-level processes are not invariant to the field changes. On identical
observed western cells, private-.25 changes mean local replacement from .701
to .830 and immigrant share from .660 to .466. Twenty-four of 662 western
cells reverse contribution class across the broad .95–1.05 neutral band.
Regional corrections change these maps much less and yield no direct
contribution reversals across that band in any region. Stability of a point
estimate comparison does not establish posterior identifiability.

None resolves invasion timing: fitted fronts remain about three years early
under the project's 10% occupancy threshold; frozen-environment release-only
fronts are about 25–27 years early. Those thresholds are relative to each
fit's density, so this is not proof of identical absolute abundance fronts.
For regional 1%, saved fixed-parameter zero-correction replays change eastern
totals little (about −1.35% to +.62% at checked years) and do not change the
reported median front dates. Its modest growth benefit does not appear to
come from replacing the invasion with large abundance additions.

Saved prediction summaries show some spatial improvement under field
separation, with smaller gains after observer marginalization; temporal
scores change little. Paired real-record uncertainty has not been calculated.
Total objectives across different latent dimensions are not comparable scores.

## What can and cannot be concluded

Regional corrections are the clearest modest reduction in historical growth
change among the completed new tests. Field separation exposes an important
compensation: reproduction can fall while historical growth, Plains intrinsic
sources and K/Allee failures worsen. Independent cell-year corrections as
currently fitted are far too small to demonstrate the proposed mechanism;
their differing growth regimes also require basin sensitivity checks.

Outstanding checks include accepted export/replay of the independent cases,
their complete source/sink and prediction diagnostics, matched intervention
replicates, permanent movement distances, finite-propagule establishment,
new-fit HMC geometry, paired predictive uncertainty, and sensitivity to
founder/history supervision and uncertain reproductive conversion. No model
here has passed a complete holistic audit.

Evidence: `FIVE_FIT_ANALYSIS_20261103.json`,
`PARTIAL_INDEPENDENT_ANALYSIS_20261103.json`; read-only analysis scripts
`research/exp/analyze_five_fits.py` and
`research/exp/analyze_partial_independent_points.py`. Scientific calculations
were performed on TACC; the local GPU was not used.
