# E004 / E009 -- dispersal pinned to prior mean; joint plausible demography (pre-registered 2026-10-16)

Motivation (from E001/E002 + baseline checks on run_23a): region x era obs/fit ratios show the
observed eastern invasion is MORE SIGMOID than the fit -- east 1966-79 0.41 (fit ahead), 1980-93 1.32
(fit behind); midwest 1980-93 1.62. The fit sharpens the curve with three levers at once: strong
Allee (slow start), suppressed dispersal (intercept -0.58 vs prior 2 +- 1), lambda0 ~2 (fast surge).

E009 f3n003: pin Fmax 3 AND n50 0.03 together (the plausible-demography corner).
- If cost << 630 + 757 = 1387 (sub-additive): the two demands are one signal (the curve shape, H10).
- If ~additive: independent demands (fill-in vs front) -> separate Phase 2 mechanisms.
E004 d20 / d20f3: pin dispersal_logit_intercept = 2.0 (prior mean), alone and with Fmax 3.
- Prediction (H10): large cost on east 1966-79 zeros (front leaks ahead), n50 RISES further
  (Allee takes over the braking), lambda0 unchanged or up.
- If cheap (< 30 nats): the suppressed dispersal is a soft optimum -> drop from the story.
