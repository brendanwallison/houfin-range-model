# Dispersal mechanisms at 54 km -- synthesis (2026-10-31)

One place for every dispersal arm run since the 54 km ladder began. Source entries: GRID54_LADDER_2026-10-29.md
(steps 8a, 14b, 20-30), GRID54_LADDER_2026-10-30_MISSPEC_SUITE.md (warp / resets), DISPERSAL_EXPRESSIVENESS_PLAN_
2026-10-30.md (design, implementation map), GRID54_SREAL_RESCORE_2026-10-30.md (paired held-out).
Reference: step 3 (relative Allee k = 3, single habitat field, observer effect; averaged-27 km Z): lambda change +.743
(seed range .742-.766), frozen-habitat release-only front 30 yr early, fitted front 8 / 4 yr early.
Paired S-real = held-out real BBS rows in withheld blocks, arm minus step 3 (seed floor +-.005).

## 1. What was tried and what happened
| family | arm | guards (null / recovery / cross-leak) | real-data value | lambda change | paired S-real | verdict |
|---|---|---|---|---:|---|---|
| kernel shape | 8a learned band weights (mass-preserving) | n/a (2 seeds agree) | juvenile mass 5% / 35% / 60% in 0-155 / 155-483 / >483 km (prior 37 / 42 / 21%); mean ~625 km (prior ~330) | +.537 / +.550 | +.007 / +.006 | reduces inflation at no held-out cost, but implausible (banding: ~25% stay, pooled mean 109 km); acts as a hidden loss -- juveniles dumped into sparse / sink cells; fecundity 11 -> 7.2; frozen front unchanged |
| kernel shape | 14b era-varying band weights | n/a | small era deviations | +.554 | -- | nothing beyond 8a |
| kernel geometry | 4 dir x 4 bands | n/a | -- | +.748 | -- | resolution is not the lever (fronts identical) |
| kernel scale | MDD 200 / 110 km (Belthoff & Gauthreaux direction) | n/a | -- | +.786 / +.689 | pending | compensated: fecundity 13.9 / 14.8, founders log +.5 / +.6 |
| path mortality | J1 distance hazard | null pass; RECOVERY FAIL (.03 -> .002) | h = 0 | = step 3 | -- | unidentifiable: planted cost absorbed by fecundity (9.4 -> 8.3) and founders |
| path mortality | J2 water-crossing hazard | null pass; RECOVERY FAIL (1.0 -> .14) | hw .04 | = step 3 | -- | unidentifiable (fecundity 9.4 -> 8.6); W is mostly a coastal-destination effect |
| path mortality | loose band cost (reference) | -- | .015 / .009 / .004 | = step 3 | -- | the data do not want band mortality |
| path mortality | loose decoupled journey survival (reference) | -- | Q offset +3.1, slope x20 | artefact | -- | pathology: resident juvenile survival collapses (in cores ~93% of juveniles depart, so Q is their survival) |
| path mortality | terrain hazard | not built | -- | -- | -- | same class as J1/J2 |
| path mortality | compounding (endpoint-bottleneck) path survival | queued (TACC 0385 guards, 0390 arm) | -- | -- | -- | built for realism (user: adopt on realism even if neutral); 88 coastal/island segment pairs are roundoff |
| settlement | conspecific attraction | null pass; RECOVERY FAIL (.3 -> .008); founders pass; Allee stalled | a = .006 | +.698 | -- | unidentifiable |
| movement choice | directional choice (softmax over habitat ahead) | running (TACC 0375 / 0380); one cross-leak already passed | -- | -- | -- | pending |
| departure | native / partial-migration departure (E077 port) | null pass; recovery .6 -> .78 (detected, biased high; bar .75); fecundity cross-leak pass | r = .65 | +.761 | pending | the first dispersal mechanism the data clearly USE (western sedentariness); does not reduce inflation; provisional |
| state correction | displacement warp at 1964 | takeover (skeleton front -20 / -16) | heavy use | +.217 (unconverged) | -.065 | diagnostic: the 1940-64 spread is 10-20 yr too fast; not a model |
| state correction | free / anchored resets 1964-66 | pass trivially | unused | ~ step 3 | ~0 | the fit does not want to rewrite the pre-BBS state |

## 2. Cross-cutting conclusions
1. IDENTIFIABILITY (established for this design): every mechanism that removes or redirects dispersers at the front
   -- distance hazard, water hazard, settlement attraction -- failed its recovery guard. Planted effects went into the
   free compensators (fecundity, founders, adult survival). The one mechanism detected (native departure) has a
   distinctive SPATIAL signature (western native range only), which BBS counts can see.
2. DIRECTION OF PRESSURE (likely): the real data push toward MORE successful long-distance dispersal (8a fat tail;
   loose decoupling raised journey survival), not less -- consistent with sustaining the large "western sink" share
   (46-55% of the occupied West) by immigration. A Plains barrier and that western-sink reading pull in opposite
   directions.
3. THE FRONT-SPEED DEFECT IS UNTOUCHED: no dispersal arm slowed the frozen-habitat release front (30 yr early in every
   arm). Only the warp (a patch) did. Shorter kernels and costs were compensated by fecundity and founders.
4. NOT THE CONFOUND ALONE: fitted to a static-habitat outside truth (step 28), the age model kept lambda flat in every
   region, so the real-data inflation reflects structure in the real data (likely: the real front timing vs the
   model's growth + dispersal), not the equilibrium confound acting through our structure.
5. WHAT WOULD MAKE DISPERSAL TESTABLE: pinning the compensators with data -- fecundity (NestWatch, with the
   yearling-breeding life cycle and broods anchored; FECUNDITY_CONSTRAINT_SCOPING_2026-10-31.md), founders (CBC-anchored
   release counts), survival (Bird Banding Lab). Without them, recovery guards for front mechanisms will keep failing.

## 3. Pending
- Paired S-real for native departure, kernel scale, yearling breeding, v2 native (running locally).
- Directional choice guards + arm (TACC, running); compounding guards + arm (TACC, next).
- Step 28b slow-eastern-front static truth: refine diverged; diagnosis in progress.
- Decision inputs for the base: native Z (adopted), yearling breeding (step 29 rule), compounding (realism).

## 4. Paired S-real added (2026-10-31 15:00; rescore_run.sh)
vs step 3 (averaged Z): attraction +.013 [+.004, +.024]; native departure -.006 [-.011, -.001]; MDD 200 km -.026
[-.047, -.009]; MDD 110 km -.025 [-.057, +.008] (east -.06) -- the shorter, banding-consistent kernels FIT WORSE.
vs step3_native (same native Z): yearling breeding -.007 [-.023, +.015] (both seeds; suitable land .10 -> .44 vs
.14 -> .50) -- strictly just outside the pre-registered +-.005 floor, CI spans 0: borderline, not adopted yet;
step3_native_v2 -.012 [-.022, -.004] vs v1 -- retraining DESK + field (production) moves S-real by ~.01, larger than
MAP-seed noise. 8a on native v2 +.050 vs step 3 (cf. native step 3 +.056).

## 5. Compensator constraints -- started 2026-10-31 15:20 (were scoped only; user: "what happened to this?")
- Fecundity: NestWatch joint nest-likelihood block on the step29_yb_native base (broods anchored by region), with
  guards incl. the key test: J1's planted front cost redone WITH the nest block on -- does it become recoverable?
  (step 31, in build).
- Founders: CBC-anchored release prior / pin in model units (step 32, in build).
- Survival: Bird Banding Lab encounter scoping + submodel design (SURVIVAL_BBL_SCOPING_2026-10-31.md, in progress).

## 6. Founders and banding evidence (2026-10-31 evening)
Founders (step 32 queued, TACC 0397): CBC-derived band 17-352 birds (median 77; 1 model unit = ~103 birds at the release
cell). Fitted founders: native delayed-breeding fits ~0.2-2.4 birds (implausibly few -- founders act as a compensator);
yearling base ~44 birds (inside the band); MDD 110 / 200 km ~170-190 birds. Arms: CBC pin on the yearling and native
bases + low / high brackets.
Bird Banding Lab (SURVIVAL_BBL_SCOPING_2026-10-31.md; 1,000,051 HOFI bandings, ~2,600 dead recoveries): adult survival
~.53 (yearling base fits Sa ~.40 -- too low; native delayed base ~.61). Breeding-season natal dispersal (n 316): 40% at
the banding block, mean 62 km, 9.8% > 155 km, 2.8% > 483 km -- vs the Veit & Lewis prior 63% / 21% and 8a's 95% / 60%
beyond those distances (report-location bias would have to be 6-20x to reconcile). Summer-winter displacement: median 0 km
West vs 147-194 km East -- matching the native-departure result and the likely source of V&L's fat tail (their
recaptures were not season-restricted). Implication (likely): the juvenile kernel tail is too fat; shorter kernels fit
BBS worse today because free fecundity / founders compensate -- a BBL-anchored kernel should be retested once
compensators are constrained.
