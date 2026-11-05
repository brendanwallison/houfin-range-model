# Misspecification suite at 54 km (pre-registered 2026-10-30, before running)

User (2026-10-30): misspecification needs a suite, not one run -- the outlet may be too small (century skeleton,
no room for departures) or the term may take over (worse). Step 10 (GRID54_LADDER, sigma .05/.10/.20 on step 3) is
part B below. Prior evidence: E068/68b (data demand it), E070 (noise shrunk to 29% recovery, held-out flat, MAP
takeover of eastern growth), E073 (detrended noise does not trade with persistent terms), E080 (persistent habitat
change over-used). Tooling: research/age_model/exp/simulate_observations.py (NB2 counts on the real design at a known
point; fit with HOUFIN_SYNTHETIC_OBS).

## A. Calibration on synthetic data (truths drawn on the real 54 km design; each fitted with step 3 + detrended noise)
Run at sigma .10 and .20 (the two sizes most at risk of takeover).
- A1 NULL: truth = step 3 refined point, no noise. Pass: realised RMS u within the 70n level (<= .25 of the prior),
  lambda change, F, Sa/Sj within the seed band (+-.01 lambda, +-5% F), front unchanged.
- A2 RECOVERY: truth = step 3 + a known transient slowdown (negative eps at the eastern front, 1950-70, consistent
  with detrending). Pass: detected at the right place and time (>= 2x null RMS there); skeleton within the A1 band.
- A3 TAKEOVER: truth = step 3 with the fecundity intercept lowered so the release front is ~10 yr slower, no noise
  (the skeleton CAN represent it). Pass: the fit recovers the lower F (within 15%) and the noise stays at the A1
  level. Failing = the noise absorbs what the skeleton should carry: takeover, quantified.
- A4 PERSISTENT: truth = step 3 + a known regional habitat trend. Pass: detrended noise stays at the A1 level
  (the trend goes to habitat, not noise).
## B. Real data, sigma profile (step 10 running): .05 / .10 / .20, plus .40 if .20 passes T1/T2 (takeover onset).
## C. Form controls (real data, step 3 base)
- C1 observation-only: year x region effects on expected counts (no effect on dynamics). If it buys as much as
  process noise, the misfit is in the counts, not the dynamics.
- C2 persistent smooth habitat correction (E067 form) as the known-bad comparator: shows how much lambda inflation a
  persistent outlet adds.
## D. Base and replicates: the selected sigma on step 9 (best arm); MAP seeds 42/43 on the selected arm.
## Gates for every real-data arm (as step 10): T1 skeleton-only front within 10 yr of BBS and < 10% eastern cells never
reaching 10%; T2 noise explains < 25% of log-density variance (1966-2020 occupied); held-out (marginal) within .01 of
the no-noise twin; plus lambda change, demography, noise maps by era/region.
Decision: a sigma is usable only if its A1-A4 guards pass AND the real-data arm passes T1/T2. The answer to "does an
outlet reduce lambda inflation" is read only from usable arms.
Compute: ~16 fits; synthetic arms and replicates go to the TACC A100 worker (keeps it stocked), real-data arms local.

## REVISION (2026-10-30, before any suite fit was read): screen the FORM first, calibrate only the shortlist
User: the plan over-tests very few approaches; treat the form of the correction as the thing to probe. Sections A-D
above are deferred; A (synthetic null / recovery / takeover / trend) and D run only on shortlisted forms.
Screen (one fit per form, step 3 base unless noted, same readouts for all: lambda change, held-out marginal T/S,
T1 skeleton-only front + T2 noise share via scratchpad front_speed.py `skeleton_no_noise`, where/when the term acts):
| id | acts on | form | status |
|---|---|---|---|
| S-pn05/10/20 | abundance | yearly detrended noise, n_freq 5, from 1940 (= step 10) | running |
| S-reset1966 | state | one smooth regional state reset at 1966, sigma .5, n_freq 8 (multiple shooting, 1 boundary) | queued (fit code 9ad8dc6) |
| S-reset3 | state | resets at 1966 / 1985 / 2005, sigma .5, n_freq 8 | queued |
| S-lam | growth rate | transient (detrended) regional deviation of log fecundity / lambda, compounding through growth | to build |
| S-obs | counts | year x region effects on expected counts (dynamics untouched) | to build (E025 is continental) |
| S-disp | spread | era-varying long-distance dispersal share | to build |
| S-front | establishment | low-density / front-local colonisation correction | to build |
| S-H | habitat | persistent smooth correction (E067), known-bad comparator | exists |
Shortlist rule: lambda change at least .1 below the no-term twin, T1 and T2 pass, held-out marginal within .01 of the
twin (or better). Shortlisted forms then get A1-A4 and a size profile.
Screen status (2026-10-30): every form built, tested and queued (local pueue via research/bin/grid54_retry.sh; TACC
queue tasks 14-20). Fit code: resets 9ad8dc6, S-lam 4e20e5f, S-obs / S-disp / S-front 6ace0a8 (all default-off;
the step 3 configuration is unchanged by every switch). Arms: step10_pn05/pn10/pn20, step11_reset1966, step11_reset3,
step12_lam10 (fecundity deviation, sigma .1), step13_obsreg (counts, sigma .2, n_freq 5, detrended), step14_dispera
(8a + era-varying band weights, knots 1940-2020 by 20 y, scale .5; twin = 8a), step15_lowdens (noise weighted
h/(h + N/K), h .05, sigma .2). S-H (persistent E067 correction) is the remaining comparator.
S-H built: step16_sthab = step 3 + E067 correction (15 cosines x linear from 1966, w ~ N(0, .1)), fit code 5ea86c8.
Its 54 km basis (st_correction_basis.npz in the obsroute input) needed a builder fix: the snapshot's kernel builder makes
POINT kernels, but the 54 km Z_disp used cell-integrated kernels -- the builder's operator guard caught it (median rel
err 6e-5 > 1e-5); with the input's own stored kernels (inverse FFT) it reproduces Z_disp to 1e-8. Queued (pueue 929,
TACC task 21). The full screen (9 arms + S-H) is now queued; nothing remains to build.

## Screen result: S-reset3 (step11_reset3; resets 1966 / 1985 / 2005, sigma .5, n_freq 8; pueue 923; 6,154 refine it)
| | lambda change | T marginal | S marginal | fecundity | adult surv. logit | founders log |
|---|---:|---:|---:|---:|---:|---:|
| step 3 (twin) | +.743 | -1.844 | -1.219 | 11.0 | .22 | -1.07 |
| reset3 | +.452 | -1.844 | -1.232 | 3.7 | 1.27 | +.52 |
Guards (front_speed.py skeleton_no_noise): T1 FAILS -- with the resets removed, the eastern front arrives 40 yr early
(1%) / 14 yr early (10%) vs the BBS (fitted with resets: -10 / -4). T2 FAILS -- the resets' share of log-density
variance (1966-2020 occupied) is 1.59 (bar .25); on average the resets multiply density by e^1.40 ~ 4x.
Reading (established for this arm): TAKEOVER. The fit cut fecundity from 11.0 to 3.7 and raised adult survival, then
used the resets to do the demographic work: knock back early front birds (the skeleton alone floods the East 40 yr
early) and add abundance later (+1.4 log units). The lambda reduction (-.29) is bought by moving growth out of the
skeleton into the resets, not by a better skeleton. Held-out: T equal, S -.013. Not shortlisted at sigma .5.
Open: a reset of this size (land-RMS e^.5 per boundary, 80 maps) is large; a smaller sigma could behave differently --
the single-reset arm (1966) reads next; a size profile is warranted only if a smaller reset passes T1/T2.

## Reset FORMULATIONS (pre-registered 2026-10-30, after reset3's takeover; user: constrain resets, vary their mathematics)
reset3 took over in two ways: it SUPPLIED growth (mean +1.4 log; fecundity 11 -> 3.7) and it HELD BACK the front the
skeleton would otherwise flood. Shrinking sigma limits how much a reset can do, not what it may do; these arms change
what it may do. All: ONE boundary at 1964 (just before the BBS; the whole observed era is skeleton), step 3 base,
same readouts and guards T1 / T2. Code: conserve + skeleton_tau 5ea86c8..(HEAD of 2026-10-30 commit "process_noise.conserve"),
state_warp 232b9a9.
| arm | reset form | rules out |
|---|---|---|
| step17_r64_free | log-multiplicative, sigma .5, n_freq 8 | nothing (reference; reset1966 = same at 1966) |
| step17_r64_total | + continental count conserved | net creation / removal of birds |
| step17_r64_regional | + count conserved in 8-cell (432 km) blocks | creation, and moving birds between regions |
| step17_r64_freepen | + skeleton-consistency penalty tau .3 | large skeleton departures anywhere, 1940-2020 |
| step17_r64_regpen | regional + penalty | both |
| step18_warp64 | DISPLACEMENT: state resampled at x - d(x), d smooth (35 maps, 3-cell RMS), total preserved | any change but relocation |
| step18_warp64pen | warp + penalty tau .3 | relocation beyond what the skeleton can carry forward |
Penalty: forward run without the correction; each cell-year 1940-2020 adds -0.5 w (d / .3)^2, d = log-density
difference (n0 1e-3), w = soft occupancy -- so takeover is costed in the fit, not only detected afterwards.
Predictions: conservation removes the growth substitution (fecundity stays near 11) but may still let the reset trim
the front by moving birds back (T1 is the test); the warp is the most direct "front too far ahead" correction -- if
the century-error hypothesis is right, warp64 cuts lambda inflation while T1/T2 pass; the penalty arms bound the
skeleton gap by construction (T2 should pass) and show how much lambda reduction survives when takeover is costly.
Next formulation (to build): observation-anchored 1964 state (blend toward the first BBS years' abundance, one weight).
| step19_anchor64 | OBSERVATION-ANCHORED: at the end of 1964 the state becomes (1 - alpha s) N + alpha s A, A = mean TRAINING count 1966-68 / pop_scalar, s = surveyed; alpha ~ Beta(2, 2); no spatial parameters (fit code 1b05fb4) | inventing any pattern the first surveys do not show |
| step19_anchor64pen | anchor + penalty tau .3 | as above, and large skeleton departures afterwards |
Real-input check (54 km obsroute): 760 of 4,318 land cells surveyed in 1966-68 (training rows only), 64 with finches;
A median 0, p99 2.5, max 15.7. The 696 surveyed zeros are what can clear an over-advanced front.
Queued locally (pueue 935-943, priority above the earlier screen) and on TACC (tasks 022-030, which sort first).

## Screen result: S-reset1966 (step11_reset1966; ONE free reset at 1966, sigma .5, n_freq 8; pueue 932)
lambda change +.784 (.455 -> 1.239; step 3 +.743), T marginal -1.844 (= step 3), S marginal -1.218 (-1.219),
fecundity 9.9, founders log -1.15. Guards PASS trivially: skeleton-only front -9 / -4 yr = fitted; T2 share .014, mean
log full - skeleton +.04. The fit barely used the reset. Reading (likely): given freedom to rewrite the state at the
start of the observed era, the fit does not want to -- consistent with inj1960 (no change) -- so the pressure behind
the lambda inflation sits INSIDE the BBS era (reset3 used its 1985 / 2005 resets heavily), not in error accumulated
before 1966. Consequence for the queue: the constrained 1964 log-multiplicative variants (total / regional / penalty)
constrain a correction the fit does not use; they are moved behind the displacement and anchored arms (pueue switch;
TACC files renamed 0191-0194), which change the reset's mechanics rather than its limits.

## Screen result: S-r64_free (one free reset at 1964, sigma .5, n_freq 8; pueue 935)
lambda change +.775, T marginal -1.843, S marginal -1.218 (step 3: +.743 / -1.844 / -1.219); fecundity 9.7. Guards
pass trivially (skeleton-only front -9 / -5 = fitted; T2 .028, mean log diff 0). Same as reset1966: the fit does not
use a free pre-BBS reset. Established for both boundaries (1964, 1966).

## Screen results, batch of 2026-10-30 afternoon (TACC bundles 011/013 + local warp; readouts read_batch.sh)
| arm | lambda change | T marg | S marg (all) | fecundity | T1 skeleton front 1% / 10% | T2 share | verdict |
|---|---:|---:|---:|---:|---|---:|---|
| step 3 (seeds 41/42/43) | +.743 / +.742 / +.766 | -1.844 | -1.219 | 10.5-11.0 | -- | -- | twin |
| pn05 (yearly noise .05) | +.467 | -1.875 | -1.252 | 5.5 | -8 / -3 | .013 | guards pass; demography moved |
| pn10 | +.680 | -1.841 | -1.219 | 4.4 | -6 / +1 | .094 | pass |
| pn20 | +.452 | -1.846 | -1.214 | 2.5 | -6.5 / +5 | .438 | T2 FAILS (takeover) |
| lam10 (fecundity deviation .1) | +.562 | -1.871 | -1.250 | 11.1 | -7 / -3 | .010 | pass |
| sthab (persistent E067 correction) | +.643 | -1.838 | -1.220 | 11.0 | -- | -- | comparator |
| warp64 (UNCONVERGED, 16,000 it cap) | +.217 | -1.863 | -1.261 | 6.4 | -20 / -16 | .364 | T1 + T2 FAIL |
| amp8 (step 4 arm) | +.572 | -1.863 | -1.251 | 11.4 | -- | -- | -- |
Seed spread of lambda change (3 seeds): .742-.766. Reading (likely): yearly abundance noise moves the fit into a
low-fecundity basin (F 2.5-5.5 vs 11) rather than carrying timing; lambda change is not monotone in sigma (.47 / .68 /
.45). The fecundity deviation keeps F at 11 and takes .18 off lambda change with a 1% share -- the only in-era form that
passes both guards with a sizeable reduction. The warp is used heavily: the skeleton alone fills the East 16-20 yr early,
i.e. the 1940-64 spread is too fast and the warp patches it once (takeover by the pre-registered rule; points at the
pre-BBS front speed as the defect to fix in the skeleton -- the dispersal changes). Held-out comparisons here use T
marginal / S-all, which B0 showed are weak; all arms are being rescored on S-real (SDM_BASELINE_B0_2026-10-30.md).

## Screen results, 2026-10-31 early (TACC 0300 seeds + 012 survivors)
| arm | lambda change | T marg | S marg (all) | T1 skeleton front 1% / 10% | T2 share | verdict |
|---|---:|---:|---:|---|---:|---|
| lam10 seed 41 / 42 / 43 | +.562 / +.658 / +.662 | -1.871 / -1.838 / -1.836 | -1.250 / -1.220 / -1.219 | -7 / -3 | .010 | basin-sensitive: seed 41 alone paid the held-out cost |
| 8a seed 41 / 42 | +.537 / +.550 | -1.841 / -1.840 | -1.215 / -1.216 | -- | -- | robust |
| obsreg (count year x region, sigma .2) | +.471 | -1.875 | -1.243 | -- | -- | counts-only term cuts lambda change .27 (pending S-real) |
| lowdens (noise at low density, sigma .2) | +.244 | -1.829 | -1.218 | -88 / -11 | .542 | TAKEOVER (T1 + T2 fail) |
Readings (likely): the fecundity deviation's lambda reduction is ~.08 in the two seeds that match step 3 on held-out (the
.18 of seed 41 came with a worse basin); a counts-only (observation) term reducing lambda change by .27 says part of the
inflation compensates regional-year count swings the dynamics cannot carry -- to be checked on paired S-real before any
reading; low-density noise takes over the front. S-real rescoring queued locally (rescore_run.sh).
Paired S-real (vs step 3; 90% CI) for these: lam10 s41 -.048 [-.097, -.006] / s42 -.000 [-.009, +.005] / s43 +.000
[-.006, +.005] (T marg s42 +.006, s43 +.008, CIs > 0); 8a s41 +.007 / s42 +.006 [-.001, +.014]; obsreg -.037
[-.078, -.005]; lowdens +.002 (but takeover). Established: the fecundity deviation is held-out NEUTRAL (seed 41 was a
worse basin, not the term); its robust lambda reduction is ~.08 (seeds 42/43), modest. 8a's .2 reduction and
neutrality replicate. The count-only term costs real held-out skill -> diagnostic only. Shortlist (both guards pass,
held-out neutral): 8a (dispersal weights; lambda -.20), lam10 (fecundity deviation; lambda -.08).
