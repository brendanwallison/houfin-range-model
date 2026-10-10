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
- **G6 What production actually ships.** The age model consumes RAW instantaneous z, first 24 of 64 dims, 1902-2025;
  the production checkpoint is the FINAL-epoch state (no validation set), which moves more and leans less on
  space-for-time than every selected-epoch model graded (E025). In 1902-1939 the only data are pre-invasion
  pseudo-zeros. (downstream_contract.md; E025)
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
