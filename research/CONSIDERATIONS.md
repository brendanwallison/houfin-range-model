# Considerations: what success means and how to read results

A living list of the interpretive points raised in this project, mostly by the user in conversation, each with its
source. There is rarely one objective answer: these are the considerations to hold in balance. **Every
pre-registration lists the items it must respect and how; every interpretation checks itself against them**
(PROTOCOL.md). When the user raises a new point, add it here the same day, with its source.

Items are numbered so records can cite them (e.g. "considerations: G3, M1, M6").

## G. What success means for the downstream model

- **G1 An honest backcast, not minimal or maximal movement.** Accurate where the information allows, with the
  uncertainty carried into the readout. Neither "move less" nor "move more" is a goal in itself.
  (user, 2026-10-09, after the over/under-movement discussion; DIGEST)
- **G2 Over- vs under-movement (the user's three points, 2026-10-09).**
  1. Over-estimated similarity (e.g. a past place looking more like somewhere else than it was) may be more
     harmful than under-estimated similarity, so excess movement might look like the safer error;
  2. but excess movement in the WRONG direction moves a place toward some other place or time, creating exactly
     that over-similarity (found: DESK's backcasts lean toward climate-analog places, E017);
  3. keep "how much it moves" separate from "how much it moves given how right its direction is"; responses lag;
     spatial turnover may reflect an equilibrium that temporal change has not reached, so extrapolating time from
     space is always fraught.
  Consequence: report direction (correlation), magnitude for its accuracy (calibration), and where the movement
  points (analog pull) separately.
- **G3 Division of labour in the age model.** Habitat (H = z.beta) should carry habitat; the population part
  (vital rates, dispersal, disease) carries a species' own dynamics. A habitat representation that does NOT
  track species-specific change is desirable: otherwise an invasion front could be fitted as "habitat improved"
  (hypothesis D3). (user + discussion, 2026-10-10; E030, E031)
- **G4 But too little habitat change is a bias too.** Shrinking habitat change beyond what its accuracy justifies
  pushes real habitat change (farmland spread or abandonment) into the dynamics -- the mirror-image confound.
  The right amount of shrinkage is measurable (scale change by its calibration slope). (discussion, 2026-10-10)
- **G5 Shrinkage fixes magnitude, not direction.** An error pointing the wrong way (toward climate analogs) gets
  smaller when shrunk but still points the wrong way. (discussion, 2026-10-10)
- **G6 What production actually ships.** The age model consumes RAW instantaneous z, first 24 of 64 dims, 1902-2025.
  The production checkpoint is the state at EPOCH 200 of a 500-epoch schedule (config/overlays/production.json:
  stop_at_epoch 200; no validation set) -- not epoch 490. In the training logs its direction of change is already near
  the late plateau at epoch 200, while its movement is between the selected epochs (65-162) and epoch 490 (E025).
  In 1902-1939 the only data are pre-invasion pseudo-zeros. (downstream_contract.md; production.json; E025)
- **G7 Understanding before score.** The short-term goal is experiments that improve understanding, not "improving
  DESK". Interventions run as tests of a mechanism. Do not jump to mitigation (recalibrate, hold habitat at its
  modern state) or to the age-model sensitivity fit before the exploration has earned it. (user, 2026-10-08/09)
- **G8 Production has 50-60 years of history; validation spans are shorter.** Span and reach are confounded in
  the tempho runs; test them separately. (user, 2026-10-09; E023/E024/E029)
- **G9 No special years.** eBird is no longer a metric; 2025 is not special and may be withheld like any year.
  (user, 2026-10-09)
- **G10 Rank.** 24 dims were chosen because DESK seemed to capture nothing meaningful beyond ~24 (maybe ~12);
  going to 64 is possible if DESK captures something there. (user, 2026-10-09; E020: it adds little)

## M. How to read a metric

- **M1 Squared-error skill against "no change" = (var predicted / var true) x (2k - 1)**, k = the slope of truth on
  prediction. A prediction beats no change only if k > 1/2, whatever its correlation; the best any rescaling can do
  is corr^2. Always report correlation (information), k (scaling) and skill separately: "worse than no change" can
  be pure over-scaling. (E030)
- **M2 Noise-free estimation.** Cross-half (ABBA) covariances remove survey noise within an epoch; they do NOT
  remove observer effects, which both halves share. (E004-E013; A10, A14)
- **M3 Observers.** ~30% of measured "true change" is observer turnover (E021); where observers changed, ~55-60% of
  the fading of local deviations is the change of observer (E028/E028b, three eras). A change of observer moves
  every species on a route together, inflating any "shared change" measure. Check observer-continuous cells.
- **M4 Centred vs uncentred.** Centred (place-specific: each species' mean change across places removed) and
  uncentred (as the suite scores: includes the species' continent-wide trend) answer different questions;
  continent-wide trends are shared with a species' associates (E031) and invisible to centred scores.
- **M5 The suite asks habitat to predict a species' WHOLE change.** The age model does not ask that of habitat
  (G3); ~90% of a species' 40-year change is not shared even with its closest associates (E031). Low "information"
  in that score is partly expected and is not by itself a habitat failure.
- **M6 Shared vs unique, not regional vs local.** Habitat change is often regional, so removing a species' regional
  trend removes habitat-driven change too; test habitat-likeness by sharing with the species that live alongside
  it, with observers held fixed. (user, 2026-10-10; E031)
- **M7 Define the scales.** "Change" is between multi-year averages; state the gap between window centres (8-46
  years across experiments). "Regional" = smooth over a few hundred km (300 km maps; 150 km in E028); "local" =
  one 27 km cell. (user, 2026-10-10; DIGEST "TERMS")
- **M8 Errors-in-variables.** Noisy features (e.g. a half-window oracle) times large spatial coefficients amplify
  noise into apparent failure; compare noise-free versions. (E011 skeptic, E030)
- **M9 Epoch selection.** Selection is on a spatial score and noisy (an argmin of a series that swings 10-30%);
  a selected epoch can be early (65, 74) with immature temporal behaviour. Compare models at matched training
  state (epoch 490) as well as at the selected epoch. (E025, E029b)
- **M10 Seeds and pooling.** Seed spread is ~0.02-0.03 in backcast correlation. Pool windows with one shared
  bootstrap for paired comparisons; one window or one seed is not a test. (E024 skeptic)
- **M11 Placebos.** Every claimed temporal-signal gain needs a placebo without the claimed content (position x
  year; static environment x linear year; random directions). Several apparent gains were spatial interpolation or
  a trend. (E011, E016, E027c)
- **M12 Survey composition.** Year means over surveyed rows track which places were surveyed (very few in
  1966-67); use full-grid or fixed-set averages for trajectories. (E027c skeptic)
- **M13 Forking paths.** Count contrasts; a "+-0.02 band" on a point estimate is not an equivalence test -- report
  the CI bound of what is excluded; fix the span/window/state of a decision rule in advance. (E027c skeptic)
- **M14 Prevalence.** Rare species carry almost no change information and dominate medians over species; report
  by prevalence tier. (E030)
- **M15 Great Plains separately.** DESK is weakest there and the pseudo-zeros bite there. (plan)
- **M16 Broken baselines.** The covariate GP's fit is broken (A13); the spacetime_sum GP is a degenerate damped-
  persistence fit (A5b) -- quote neither as a habitat benchmark.

## P. Process

- **P1 Every finding carries its WHY**, marked established / likely / unknown, with the cheapest test for an
  unknown. (user, 2026-10-10)
- **P2 Plain English for the user**; define terms on first use. (user, 2026-10-10)
- **P3 Pre-register before results**; if a rule turns out ambiguous, say so rather than choosing post hoc.
- **P4 A skeptic round before acting on a surprising result**; several of this project's readings were overturned
  that way (DIGEST "OVERTURNED").

## Added 2026-10-10 (afternoon)
- **M17 Readouts that learn within-place change can learn a species' own trend.** Over the trained years anything
  that drifts with time in a place (DESK z, or position x year) lets a readout learn each species' regional trend and
  extrapolate it. That predicts the score well, so always compare with a position x time placebo. (E033b)
  CORRECTED (user, 2026-10-11): the placebo measures PREDICTABILITY, not CAUSE. A regional trend may be regional
  habitat change (M6) as much as a species' own dynamics; matching the placebo does not make a signal "the
  population model's job". Calling a trend dynamics needs evidence that it is not habitat-driven:
  - unshared with co-occurring species (M6, E031);
  - unaligned with regional covariate change;
  - or present where no habitat changed.
  At the COMMUNITY level (ESK coordinates), change is shared across co-occurring species by construction, so a
  regional community trend is, if anything, habitat-like (or observer-driven, M3/M18).
- **M18 Observer turnover ADDS change a regional trend cannot predict** rather than creating the trend: the
  trend-predictable part is the same where observers stayed (E034b + skeptic). "Shared" or "regional" structure is not
  automatically an observer artifact -- check, don't assume; a drift common to all observers is still not excluded.
- **M19 Nulls must keep spatial structure.** Shuffling a spatially smooth field across cells gives a far too narrow
  null; use spatial shifts or random directions/rotations. (E035 skeptic: p95 0.006 -> ~0.10)
- **M20 Random-feature placebos need an adequate basis and several draws.** One draw of 24 features was the best of
  five; 96 features changed the size of the effect. (E033b skeptic)
- **M21 "Skill >= 0" is a low bar when information is near zero**: any shrink between 0 and twice the optimum gives it.
  Report the information (corr^2) next to any "stops the harm" claim. (E033 skeptic)
- **M22 Saturation.** DESK's z stops following covariate drift before ~1960 while covariates keep changing; a
  backcast's distance from today can plateau, so compare windows on the same part of the curve. (E035 skeptic)

## Added 2026-10-10 (evening)
- **M23 Leave-one-out is not transfer.** When a row's own place's other years stay in the conditioning set, a
  leave-one-out score measures interpolation within a place (every covariate-GP variant scored ~0.77 that way while
  scoring -0.16 to +0.34 on new places). Judge a predictor for unsurveyed places only on held-out places. (E037)
- **M24 A fitting sample decides what a model learns.** Blocks of neighbouring places sit at their own regional level
  (a continental-mean likelihood can then only absorb the offset with a flat, inflated kernel); whole places make the
  likelihood about year-to-year variation within places; a place's repeated years treated as independent make its
  lasting quirk look like signal. For a predictor of NEW places, fit on continental samples with a few years per place
  and model the place explicitly. (E037 v2b / v6 / v8 / v10)
- **M25 A family's name is not its content.** Before attributing to "land-use classes" or "moisture", check which
  channels dominate the change: LUH's 33 channels are 12 class fractions, 2 secondary-land states and 19 management
  channels, and the management trends dominate; 48 of 60 "moisture" channels depend on temperature. (E035d skeptic)
- **M26 A uniform trend is a clock.** Covariate change that is the same everywhere carries only time; a model can turn
  it into place-specific change through place-specific responses, which looks like "the covariates explain change".
  Swap-attributions must separate the uniform part from the spatial pattern, and smoothing (e.g. 300 km) keeps a
  uniform shift intact, so "regional beats local" can be built in. (E035d skeptic; E027c) BUT a near-uniform
  trend can also be a real driver that changed almost everywhere (fertilizer intensification): separate the two by
  WHERE the response sits -- a clock responds wherever the channel is defined, a driver where it acts (cropland).
- **M27 An analog direction inherits its channel mix.** Built from all 302 channels it is mostly climatic, so any
  covariate-only model's climate response aligns with it; compare families' "lean" against per-family directions,
  and rest error claims on a part's correlation with the truth, not only on its lean. (E035d skeptic)
- **P5 Interactive runs carry the jobs' thread caps** (OMP/MKL/OPENBLAS=6): an uncapped pytest beside four capped
  jobs pushed the load to 76 on 32 cores and slowed everything. Waiters watch jobs queued after they start.
- **M28 One yardstick for every arm.** A metric built from a model's own cached inputs (the analog direction from its
  F_cov) silently changes when the treatment changes the inputs (static channels): E038's static-temperature models
  were graded against a direction with no temperature change, which alone halves a lean. When a treatment alters
  inputs, build the metric from a fixed reference (the control's real covariates) and grade every arm on it; and
  measure at matched training states (selected AND late), with paired CIs. (E038 skeptic, 2026-10-11)

## Added 2026-10-13
- **G11 Season of the habitat representation.** House Finches are year-round residents; BBS (breeding season) was used
  for convenience, and winter communities may describe the habitat a resident experiences -- winter survival, feeders,
  the 1994+ eye-disease spread through feeders -- better than breeding communities. Test, never on House Finch data
  (contract): proxy residents (sedentary dev species, preferably feeder-associated and finch-like), comparing
  CBC-community vs BBS-community features for their abundance and change, scored in both seasons. Weigh against the
  age model being fitted to breeding-season observations: a representation can describe the habitat better and still
  match the observation season worse. (user, 2026-10-13)
- **M30 Completeness is relative to the site.** Some places are genuinely richer than others, so a sampling-
  completeness or effort covariate taken from the lists themselves (list length, benchmark species) must be measured
  against what THAT site records under good, modern sampling -- otherwise a rich site reads as a well-sampled one.
  Corollaries: sites with no good-era reference borrow a neighbour's (weaker; report the share); and the modern
  reference itself carries real turnover since the early era, which a completeness covariate can absorb as if it
  were incompleteness. (user, 2026-10-13; E065)

## Added 2026-10-12
- **M29 A deeper series inherits its curators' limits.** Audubon's own CBC trend products start in 1966-67, chosen
  for sample size and consistent methods; pre-1966 effort (party-hours) is unreliable. Pre-1966 CBC "truth" would
  bring effort and observer change larger than BBS's (M3), in a winter community ESK was not built on, and BBS-CBC
  calibration only covers the years we can already test. Judge any pre-1966 validation source by (i) whether its
  own curators trust that era, (ii) season and community match, (iii) whether it tests what the downstream uses in
  1902-1939 (absences beyond range edges). Candidates: Grinnell resurveys (site occupancy ~1900-1940 vs 2000s,
  California and Great Basin), Breeding Bird Census plots (1937+), historical range edges. (user, 2026-10-12)

## Added 2026-10-14 (uncertainty and reformulation thread)
- **G11 Excess dissimilarity is the safer error, but injected uncertainty must not take temporal learning away from
  DESK.** An additive generic temporal block fitted by marginal likelihood can explain away DESK's change. Uncertainty
  should sit on top of DESK's (or delta's) own change: errors-in-variables shaped by the representation (the user
  found option 2 appealing), sized by measured error, growing with reach and novelty. (user, 2026-10-14; E072 parked)
- **G12 Reformulate for extrapolation, not interpolation.** A persistent spatial field plus deviations driven only by
  covariate change, with no time or position-by-time input, extrapolates as far as the covariates do. Early-20th-
  century supervision will at best be weak (CBC), so the structure must carry the extrapolation. Understanding comes
  before performance. (user, 2026-10-14; E073)
- **M31 LUH3 irrigation over the High Plains is wrong in level AND trend.** The covariates' land-use stream is raw LUH3
  (no transform). Irrigated cropland (sum over crop types of crop fraction x irrigated fraction) in a box over the High
  Plains aquifer core (lat 32-43, lon -104 to -97; 1,031 cells) is 7.9 M acres in 1930, 11.0 M in 1949, 15.4 M in 1970,
  17.1 M in 1980 -- growth x1.6 from 1949 to 1980. USGS (aquifer footprint) reports 2.1 M (1949), 6.1 M (1959), 9.0 M
  (1969), 13.7 M (1980) -- x6.5, the Ogallala boom. The box is wider than the aquifer, but LUH3 puts several times too
  much irrigation there before the war and misses most of the post-war growth. Also: LUH3's irrigated fraction is
  IDENTICAL across the five crop types in every cell (one cell-level fraction copied). Likely cause (not verified): a
  historical national total allocated by a modern irrigation map. Consequence: DESK sees substantial High Plains
  irrigation in 1920-40 and little change after; the irrigation channels should not be trusted for the Great Plains
  backcast (E073's no_clock arm already excludes management). Candidate replacement: county irrigated acreage from the
  Census of Agriculture. (2026-10-15, user question; checked against USGS fact sheets, e.g. pubs.usgs.gov/fs/2007/3029)
