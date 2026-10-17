# E028 -- the Phase 1 candidate: warm vs cold (falsification) (pre-registered 2026-10-17)

Candidate = overlay map_run35_candidate: block-3 K-weighted ideal-free settlement + free departure
curve; single historical release; firm weak-Allee prior; learned juvenile kernel tail; disease as
added mortality (Hill on current density, resistance term); breeding-season observation; demographic
priors from House Finch biology (fecundity median 5/adult, Sj/Sa median 0.6; Sa unchanged).
E027 up1 (Sa .65, F 6, Sj .39, lambda0 1.60, n50 .08) cost -1 nat vs the free fit: the data are flat
along the ridge, so the priors set the split.
- cand_warm: candidate from up1, demography FREE (priors only).
- cand_cold: MAP from scratch (quick90) -> refine -> full viz (38_model_experiment body).
Falsification: cold lands within ~50 nats of warm with similar demography -> not a warm-start artifact.
Plausibility targets: Sa ~.55-.7, Sj ~.3-.45, F ~4-6, lambda0 ~1.5-1.6, n50 < ~.1; plains corridor
true-sink share > 0 and Allee-dead small; NE crash/recovery vs Hochachka prevalence.

2026-10-17 01:31: first cold attempt FAILED harmlessly -- scripts/tacc/38_model_experiment sources env.sh,
which cds to the MAIN checkout (main-branch code, no research features); re-queued as a task that runs
map -> refine -> viz explicitly from the research worktree.
E027 up2 (Sa .60, F 6, Sj .39, lambda0 1.52): +57 -- lambda0 ~1.5 still cheap; ~1.3 costs ~1800.

cand_warm (2026-10-17): U 195422.4; FREE demography drifted back: Sa .56, Sj .43 (ratio .76), F 9.2,
lambda0 1.78, n50 .10, rec .04. CORRECTION to E027: up1's -1 nat was prior-assisted -- the old
fecundity prior (median 2.1, sd .5) penalised F 8.2 ~40 nats more than F 6; in LIKELIHOOD terms the
data prefer F ~8-9 over F 6 / Sa .65 by ~40 nats (small, but it beats a sd-0.75 prior). A plausible
split therefore has to be carried by the prior. Queued: cand_tight (fecundity sd 0.3 around 5) and
the full viz/readout of cand_warm. Cold start running (MAP ~40 min).

cand_warm viz (Great Plains corridor, CONUS): true sink .09, Allee-dead .08, viable .83, median lam_f 1.38
(vs 23a .39/.25/.36/1.18; ifd3 .02/.26/.72/1.52; sr .03/.30/.67/1.67). Continental Allee-dead 11% (mostly
boreal). Disease (survival form): peak added mortality ~40-50% NE, ~0 native west, resistance 4%.
NOTE: 09_disease_diagnostics titles still say "fraction of K removed" -- viz labels predate target=survival.
cand_tight (fecundity sd 0.3 @ 5): U 195465.2 (+43 vs cand_warm incl. prior change), Sa .62 Sj .43
(ratio .70) F 6.8 lambda0 1.70 n50 .10 -- vaguely plausible.

FALSIFICATION RESULT (cand_cold, 2026-10-17 03:2x): cold start (MAP quick90 -> refine 6510 its -> viz) lands in
a DIFFERENT, worse mode: U 196888.4 (+1466 vs cand_warm 195422.4). Cold mode: ordinary POSITIVE density-
dependent departure (int .81 / slope +1.19), founders ~390 birds (T 9.74), mid-band (155-483 km) up, tail
down (logw 2.51/0.80), Sa .40, Sj .37, F 8.7, lambda0 1.36. Warm mode: sparse-cell exodus (-7.61/-9.74),
~70 founders, tail up, Sa .56, F 9.2, lambda0 1.78.
Reading: the posterior is MULTIMODAL in the dispersal regime. The exodus mode (inherited along the warm
lineage from ifd3) is ~1470 nats better but unreachable from a cold start; the best model rests on the
inverse-density-dependent exodus (flagged by the user as questionable). Phase 2 must settle the dispersal
functional form -- justify the exodus (conspecific attraction) or find what it substitutes for; the cold
mode's lambda0 1.36 suggests part of the warm lambda0 demand is tied to the exodus regime.
