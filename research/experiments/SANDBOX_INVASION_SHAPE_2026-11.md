# Invasion-shape sandbox (started 2026-11-03)

Plan: C:\Users\brend\.claude\plans\codex-currently-has-a-serene-peach.md. Question: how much historical habitat change
(alpha*) does each mechanism still need to reproduce the observed lag-then-acceleration of the eastern invasion, with the
West held behind the Plains and plausible rates? Code: research/exp/invasion_sandbox/. Outputs:
/mnt/e/Datasets/houfin/grid54/sandbox/. Considerations: G3, G4, G13, P6, M12 (and G1: honest backcast, not minimal
habitat change). Host caveat: the local box is likely unstable until the BIOS fix (houfin-host-instability); results
below are tiny single-thread numpy runs, re-run after the fix before any decision rests on them.

Observed anchors (project record, GRID54_LADDER step 5/32 CBC; E046 notes; Veit & Lewis 1996): radius from Jones Beach
54 km in 1955 (year 15), 100 km in 1956, ~200-250 km by 1965 (year 25), Mississippi (~1500 km) by ~1990 (year 50); late
radial speed ~40-50 km/yr.

## L1 -- Veit & Lewis 1996 reproduced (2026-11-03; run_vl_validation.py; L1/VL1996_VALIDATION.json)
Paper parameters (s .67, c 1.5, delta 1.5; Weibull-form kernels juvenile a .468 b 41.056 km, adult a .859 b 76.893 km,
27% / 22% non-movers; p(N) = min(N/N2*, 1); .5 birds/km^2 in 10 km; edge at .1 birds/km^2).
| model | R yr 15 / 25 / 50 (km) | lag (R > 100 km) | late speed yr 30-50 |
|---|---|---|---|
| observed | 54 / ~225 / ~1500 | 16 | ~40-50 |
| 1-D r .15, 1500 km domain (paper) | 55 / 180 / 1500 (domain) | 21 | 58 (capped) |
| 1-D r .15, 6000 km domain | 55 / 175 / 1835 | 21 | 76 |
| 1-D r 0 / .25 | 660 / 3045 / 6000; 10 / 55 / 575 | 8; 33 | -- |
| 2-D r .15 | 10 / 20 / 51 | none | 1 |
| 2-D r .01 | 51 / 142 / 1282 | 22 | 51 |
| 2-D r 0 | 117 / 520 / 2121 | 15 | 58 |
Established (for this model class): the paper's qualitative result reproduces -- r 0 no lag, r .15 lag then
acceleration matching the 1955 / 1965 anchors in 1-D, r .25 far too slow. In 2-D the same Allee threshold stalls the
release (dilution of arrivals over area, as the paper anticipated); a ~10x weaker threshold (r ~ .01-.02) reproduces the
lag AND the late speed with uniform, unchanging habitat. Implementation note: the paper's kernel means (329 / 98 km) are
those of the exp(-(|z|/b)^a) form (mean b G(2/a)/G(1/a)), not a Weibull distance distribution (b G(1+1/a), ~93 km); the
2-D kernel uses the paper's distance distribution (a bug in the first 2-D attempt, fixed before reading results).
Their peak annual growth is 1.24 (low-density potential, without Allee, ~1.42) -- far below the age-model fits'
potential lambda (~2.5-3+, P3).

## D1 -- does high low-density growth remove the time-homogeneous solution? (pre-registered 2026-11-03, before running)
Model: the age model's yearling cycle (proto.hofi_step), 2-D, uniform habitat, V&L kernels and non-movers, Veit
departure min(N/K, 1), clumped mate-finding (k 3) with absolute n50 (birds/km^2), Beverton-Holt crowding at power 1
with c = Fmax Sj / (1 - Sa) - 1 (equilibrium K), K 3.76 birds/km^2, release .5 birds/km^2 in 10 km.
Baseline mapped from V&L: Sa = Sj = .67, Fmax 1.12 (recruits per bird .75 at low density, no Allee), lambda0 = Sa + Sj Fmax
= 1.42. Arms: lambda0 in {1.42, 1.8, 2.2, 2.6, 3.0} by raising Fmax (Sa, Sj fixed) and, separately, the fits' typical
split (Sa .55, Sj .30, Fmax set to the same lambda0); n50 on a log grid .0005-.2 birds/km^2; juvenile kernel scale x
{.33, .6, 1} (mean 110 / 200 / 329 km; adult unchanged).
Readout: R at years 15 / 25 / 50, lag year, late speed (yr 30-50); "matches" = R15 in [30, 100], R25 in [140, 350],
R50 in [1000, 2200] km (generous: grid, 2-D idealisation, coastline ignored).
Prediction: at lambda0 1.42 a band of n50 matches (as L1); as lambda0 rises the matching band narrows and vanishes --
any n50 strong enough to hold year 15-25 makes a die-or-explode front whose late radius overshoots, while a shorter
kernel slows both phases (as the step 26 fits showed); if a match survives at lambda0 >= 2.6, then high growth alone is
NOT what removes the lag in the fits, and the gap must be elsewhere (54 km cell averaging of arrivals, the fitted n50,
founders, K map, or the relative-to-K Allee of the older steps).
What it cannot show: fitted rates vary with habitat; this is a capability map in an idealised landscape.

## D1 first read (partial, 112 / 360 runs; vl split lambda0 1.42-2.2) -- prediction FAILING so far
The matching band does not vanish as lambda0 rises; it MOVES: lambda0 1.42 matches at n50 ~.001-.004 birds/km^2,
1.8 at ~.02-.04, 2.2 at ~.07-.12 (kernel x .33/.6). A stronger mate-finding threshold buys back the lag that higher
growth removes -- the lambda-vs-Allee ridge (age-model CONSIDERATIONS C3) shows up as a family of equally good invasion
curves. Implication (likely): high low-density growth ALONE does not rule out a time-homogeneous lag in a uniform
landscape; the fits lose it somewhere else. Full read when D1 completes.

## Stochastic twin, first look (2026-11-03; 3 seeds; 10 km cells)
Integer birds (binomial departure / survival, Poisson arrivals and births, no breeding below 2 birds) spread FASTER
than the deterministic model at V&L rates (n50 .005: R yr 25 400-1000 km vs 155 deterministic): integer arrivals
concentrate birds above the mating threshold where the deterministic step smears them below it. The simple
"deterministic tail causes the early lead" hypothesis is NOT supported in this form; what an Allee parameter does
depends on how mate finding is tied to the spatial unit. (The max-distance radius is satellite-dominated in stochastic
runs; r90 / r_area readouts added.)

## D2 -- cell size (pre-registered 2026-11-03, before running)
Each D1 matching (split, lambda0, kernel) combination (middle of its n50 band) re-run at 10 / 27 / 54 km cells with the
same ~157 founders placed in the release cell, radius = equal-area radius; plus n50 x {1/30, 1/10, 1/3, 1, 3}.
Prediction: at 54 km the release and arrivals are diluted ~30x relative to 10 km cells, so the D1 n50 kills the release
or stalls the front for lambda0 >= 1.8 (their n50s are large); a ~10-30x smaller n50 restores the lag only for the
lowest lambda0, if at all, because the front-cell dilution and the release dilution scale differently. If instead
matches survive at 54 km with modest n50 rescaling, cell size is not why the fits lose the lag, and the next suspects
are the fitted n50 / relative Allee form, the K map (arrivals into low-K cells) and the West constraint.

## D2 RESULT (2026-11-03; D2/D2_resolution.json; vl split, lambda0 1.42-2.2 matches) -- prediction FAILED
At 27 and 54 km cells the D1 matches mostly SURVIVE at the same n50 (lambda0 1.42: all three kernels match at 54 km,
e.g. 91 / 253 / 1277 km at years 15 / 25 / 50; lambda0 1.8: all three match at 54 km). Coarse cells make early spread
slightly FASTER (one cell step = 54 km; equal-area radius floor ~30 km), not slower. Matches become fragile at lambda0
2.2 (54 km: 30 / 140 / 1015 or 91 / 367 / 2101 km). The 10 km rows differ from D1 only because D2 places all founders in
one cell (1.57 birds/km^2 over 100 km^2) -- founder placement alone shifts the curve. Established (idealised uniform
landscape): cell size is not what removes the lag for lambda0 <= ~1.8. Next suspects, all in the REAL fitted
landscape: the fitted Allee strength / relative-to-K form (step 3 n50/K .0197), the K map (arrivals into low-K cells),
heterogeneous growth, and the West constraint -- R0 on real checkpoints (after the host fix).

## R0 -- real checkpoint diagnosis, step3_obsroute (pre-registered 2026-11-03; spec specs/R0_step3.json; runs after the host fix)
Harness: harness.py via sbx_env.sh (CPU float64, capture-replay of the refined point). Validation arms first: fitted and
frozen-modern release-only must reproduce runs/step3_obsroute/out/front_speed.json (-8/-4 and -30/-26 at 1% / 10%) --
if not, stop and fix the harness. Then: alpha .25 / .5 / .75 (ref 1966-2025), frozen at 1951-65 and all-year means,
Fmax x .5 / .25 and n50 x 10 / x 100 under frozen modern habitat, West-only frozen / fitted, and a canary repeat
(final-state SHA-256 must equal the first frozen arm's).
Predictions: (1) the alpha sweep is monotone, and the observed lag needs alpha >~ .75 (inflation is doing the lag work);
(2) frozen at 1951-65 habitat the release stalls (deep early-East sink) -- the circularity of the modern-mean
counterfactual made explicit; (3) Fmax x .5 alone does not create a lag (shifts both phases, as step 26 suggested);
(4) n50 x 10 slows the early phase more than the late phase (the D1 ridge), x 100 kills the release; (5) West-only
frozen crosses the Plains before 1940 (ladder front_speed native-seed arm).

## R1 -- first mechanism cards on step3 (pre-registered 2026-11-03; spec specs/R1_step3_cards.json; after R0)
A1 absolute n50 .01-.3 birds/km^2, A2 density floor .01 / .05 birds/km^2, B1 East-only juvenile band ramp (outer-band
log-weights -2 / -4 rising to the fitted kernel over 1940-65 or 1950-80, via per-origin directional weights,
mass-preserving), C1 a shared (East + West) logistic K or Fmax rise centred 1965 on frozen all-year habitat, D1 disease
off, and one combination (A1 .1 + B1 at alpha .5). Read each against R0's frozen release-only and fitted arms on: T2
radius (CBC 1955-65, BBS 1975-95 with the ~10 yr BBS lag), arrival offsets by cohort, east total, potential lambda.
Predictions: A1 at .1-.3 creates a lag but the release-cell exemption and K map decide whether it dies (the D1 ridge
says a matching band exists in a uniform landscape); A2 slows mostly the early phase; B1 slows 1940-65 without slowing
the late phase; C1 accelerates East and West together (the T5 western-front test of G15); none alone reaches the
observed shape with alpha 0 -- the question is how far each lowers alpha*.

## D1 RESULT (2026-11-03; D1/D1_summary.json, 360 runs, 10 km cells) -- prediction FAILED; a narrower finding instead
A time-homogeneous model matches the lag AND the late radius at EVERY lambda0 from 1.42 to 3.0, for both survival
splits, if the mate-finding threshold rises with growth: n50 ~.001-.004 (lambda0 1.42), ~.02-.04 (1.8), ~.07 (2.2),
~.12 (2.6), ~.2 birds/km^2 (3.0). What changes is the WIDTH of the solution: 4 grid steps (a ~5x n50 band) at 1.42,
2 at 1.8, a single step (< ~1.7x) from 2.2 up; with the full 330 km juvenile kernel high lambda0 overshoots late (e.g.
3.0: 67 / 333 / 2879 km). The survival split (Sa = Sj = .67 vs Sa .55 / Sj .30 at equal lambda0) barely matters.
Established (uniform landscape): high low-density growth does not remove the time-homogeneous solution; it makes it a
knife edge -- a small change in n50 (or in local growth, K, founders) flips the release between dying and exploding.
Likely mechanism for the fits (not tested): with heterogeneous habitat and K, a single knife-edge n50 cannot hold for
every cell and era, and a gradient optimiser near a die-or-explode transition (cf. FINAL_ALLEE_LOCAL_AUDIT precision
sensitivity) finds smooth habitat gating instead. This links P3 (high growth level) to P1 (historical inflation)
through the width of the Allee solution: lowering potential growth widens the band. Cheapest tests: R1 A1 sweep on
the real checkpoint (does any n50 give the lag with frozen habitat, and how wide is the band?), and the same sweep
with Fmax x .5 (does lower growth widen it in the real landscape?).

## Host validation after the BIOS fix (2026-11-03 09:18 boot; BIOS 4601, microcode 0x133, RAM 4000 MT/s)
host_stress.py: 30 single-threaded workers x 30 min (~88 GB resident; float64 BLAS, FFT convolution,
scipy.ndimage.label, 0.8 GB/worker memory pattern write+verify): 7,436 iterations, 0 mismatches, 0 worker deaths,
0 kernel segfaults (host/stress_30min.jsonl). Passed. A first launch did not run (shell chain bug: `grep -c` exit 1
short-circuited `&&`), not a host event. Canary hashes continue in every round.

## R0 RESULT (2026-11-03 ~10:00; R0_step3/rows.jsonl; 16 arms, ~3 s each, CPU float64)
Harness VALIDATED: fitted -4 yr and frozen-modern release-only -26 yr at the 10% threshold (front_speed.json: -4 / -26);
canary repeat final-state SHA-256 identical (3c1a05520bd0...). Two harness fixes before reading: step 3 is the
DELAYED-breeding cycle (c = F Sa Sj/(1-Sa) - 1; potential lambda = dominant eigenvalue), now chosen from the captured
yearling flag; relative thresholds now come from the first (reference) arm -- per-arm thresholds collapse when a
release dies (three arms showed a spurious identical -52 yr).
Absolute threshold (.1 birds/km^2), p90 radius from the release cell, 1955 / 1965 / 1975 / 1985 / 1995 km
(observed T2: CBC ~54-100 / 267; BBS 233 / 550 / 1422):
| arm | R90 | arrival offset by cohort (66-74 / 75-84 / 85-94 / 95-09) |
|---|---|---|
| fitted | 0 / 0 / 121 / 508 / 1350 | +6 / +4 / +2 / -1 |
| frozen 1966-2025 mean | 146 / 616 / 1404 / 1596 / 1573 | -16 / -19 / -19 / -22 |
| frozen 1902-2025 mean | 0 / 223 / 272 / 600 / 310 | -8 / -7 / -1 / never |
| frozen 1951-65 mean | release never spreads | -- |
| alpha .25 / .5 / .75 (ref modern) | 114/452/1188/1574 ; 54/282/871/1414 ; 0/72/357/1005/1515 | ~-15 ; ~-10 ; ~-3 (uniform) |
| Fmax x .5 (frozen modern) | 0 / 171 / 535 / 1191 / 1459 | -6 / -8 / -6 / -10 |
| Fmax x .25 ; n50 x10 ; n50 x100 | release dies | -- |
| West only, frozen modern | crosses the Plains before 1955 (East filled) | -39 ... -64 |
| West only, fitted | West never enters the East | -- |
Established for step 3 (sandbox, demography held at the fit): (1) the user's diagnosis holds in the real checkpoint --
a FIXED habitat that matches the early phase (the all-year mean: 223 km in 1965 vs 267 observed) stalls late (310 km
in 1995 vs 1422), and one that matches the late speed (the modern mean) is ~20 yr early; no fixed habitat level fits
both, so the fit needs most of its historical change: alpha* ~ .75-.9 here. (2) Partial habitat change shifts the
whole front uniformly (offsets constant across cohorts), it does not change its shape. (3) The low-density cliffs are
sharp in the real landscape: Fmax x .5 -> ~8 yr early, x .25 -> extinct; n50 x 10 -> extinct (R1 maps between).
(4) Without habitat change the native West crosses the Plains before 1955; with the fitted habitat it never enters
the East -- the same lever holds both. Note: the fitted eastern population in 1965 is ~180 birds in this unit
conversion against 4,079 birds counted on 53 CBC circles in 1966. WITHDRAWN 2026-11-03 (user challenge): step 3 is
BBS-supervised only (CBC appears in its code only in a smoke test), and this compared model units -> "birds" (E063
convention, BBS-based, breeding season) with RAW winter CBC counts (different survey, season, winter inflation per
Weiss 1997, no CBC observation model). Not a valid comparison. A within-BBS check (crude: summed state over real BBS
rows, observation model ignored) points the OTHER way for 1966-79: observed / model = 12.5 East 1966-79 vs 29-33 East
1980-2025 and 37-51 West, i.e. the fitted early BBS-era East is relatively HIGH, not empty. Pre-1966 East density in
BBS units is unconstrained by any data this fit sees. Proper check pending: the fit's own expected counts per row.

## R1 partial (step 3, habitat frozen at the modern mean, release only)
| arm | R90 1955/65/75/85/95/2005 km | offset (10%, by cohort) |
|---|---|---|
| absolute n50 .01 | 241/894/1459/1588/1557/1071 | -21/-24/-26/-28 |
| absolute n50 .03 | 121/386/954/1405/1460/1028 | -17/-19/-17/-17 |
| absolute n50 .1 | 0/74/241/366/613/439 | -4/-2/-1/-14 (stalls) |
| absolute n50 .3 | release dies | -- |
Established (step 3 demography): an Allee brake strong enough to give the early lag (n50 .1: 74 km in 1965 vs 267 obs;
241 in 1975 vs 233) leaves the late front far too slow (613 km in 1995 vs 1422) -- the user's "early too fast / late
too slow" in the real landscape for a population-process brake, as for a fixed habitat. The D1 uniform-landscape ridge
does not survive here at step 3's growth: no n50 matches both phases. Floor arms killed the release itself (founders <<
any floor once spread over a 54 km cell); rerun with the release cell exempt. B1 band ramps need per-year kernel
weights the step-3 code lacks: moved to the yearling base step39_veit_fixed_s41 (R1c).

## R1b (step 3) and R1c (step39_veit_fixed_s41, yearling, 1940 release) RESULTS
Canaries bit-identical in both; no segfaults. Offsets: 10% threshold, cohorts 66-74 / 75-84 / 85-94 / 95-09.
- Step 3, frozen modern habitat, ONE brake, no habitat change: Fmax x .4 -> -3 / -5 / -4 / -9 (fitted: +2 / -1 / -3 /
  -6.5); R90 1965 / 75 / 85 / 95 = 54 / 270 / 617 / 1151 (fitted 0 / 121 / 508 / 1350; obs 267 / 233 / 550 / 1422).
  Fmax x .45: ~-9; x .35: +3 (too late); x .3: stalls. Relative n50 x 5 ~ Fmax x .4 (-5 / -5 / -3 / -9).
  So a time-homogeneous brake can match the BBS-era ARRIVAL timing about as well as the fitted inflation -- in
  first-arrival terms. Its radius is short late (1151 vs 1422 in 1995) and it changes abundance (unchecked): R2.
- Absolute n50 with half fecundity: .03 -> -4 / -5 / -3 / -7.5; .1 -> release dies. The cliff stays sharp.
- A2 floors (.01 / .05 birds/km^2, release cell exempt) stop ALL spread: every newly colonised 54 km cell starts below
  ~30 birds IN THE PROJECT CONVERSION -- read: the deterministic front is carried by densities far below the settled
  level. Established in model units only. The "birds" reading rests on the E063 / ladder step 32 convention: the BBS
  400 m count radius (protocol), an ASSUMED detection of .5 (range .25-1), and route-to-cell extrapolation that is not
  validated for House Finch. Correction 2026-11-03 after the user's challenge: "sub-bird" / "no real population" was
  overstated. The general phenomenon is known (the atto-fox problem, Mollison 1991); our magnitude is not. R2
  maps .0003-.003.
- C1 shared logistic K x 2 or Fmax x 1.5 rise (t_mid 1965) on frozen ALL-YEAR habitat: -11 / -13.5, and the late front
  stalls (R90 902 / 315 km in 1995) -- as specified this shared rise is too weak/late to replace the eastern opening.
- D1 disease off (fitted): small late effect (R90 2005 1619 vs 1245); not a lag lever.
- Yearling base (step39): frozen modern + 1940 release -35 yr; alpha .25 / .5 / .75 / .9 -> -31 / -24 / -13 / -4
  (alpha* ~ .9); absolute n50 .03 -> -16, .06 -> extinct (cliff); B1 East band ramps (outer bands e^-2 / e^-4 rising to
  the fitted kernel over 1940-65, 1950-80, or e^-4/e^-8) -> -31 to -34: removing most long-distance mass early barely
  slows the front. Established (this base): kernel tail length is not the early-lag lever (agrees with step 26).

## R2 (pre-registered 2026-11-03, before running; specs R2_step3 / R2_step39)
Fmax x .35-.5 under frozen modern habitat (release only), x .40 / .42 with the West included and West-only, density
floors .0003 / .001 / .003 birds/km^2, Fmax x .4 on the FITTED habitat, canary. New readout: sum of model state over the
real BBS rows by region x era relative to the reference (fitted) arm, and the row-level corr of log1p(state) vs
log1p(count). Predictions: (1) the low-F arms lose eastern abundance in the BBS era (sinks where Fmax drops below
replacement) -- model_over_reference < .7 East 1980-2009; (2) with the West included, frozen modern + low F still lets
the West cross the Plains early (the West held only by habitat gating); (3) a floor of .0003-.001 slows the early
phase more than the late (an establishment threshold), .003 stops spread as before.

## R2 RESULT (2026-11-03; R2_step3 / R2_step39; canaries identical, 0 segfaults)
Abundance = model state summed over real BBS rows / the fitted arm's, eras 1966-79 / 80-93 / 94-09 / 10-25.
| step 3 arm | offsets (cohorts) | East abundance vs fitted | East corr (log1p) | Plains / West abundance 10-25 |
|---|---|---|---|---|
| fitted | +2 / -1 / -3 / -6.5 | 1 / 1 / 1 / 1 | .62 / .82 / .70 / .64 | 1 / 1 |
| frozen modern F x .42 | -5 / -7 / -6 / -10 | 6.3 / 1.3 / .60 / .31 | .42 / .76 / .69 / .59 | -- |
| frozen modern F x .40 | -3 / -5 / -4 / -9 | 4.4 / 1.1 / .55 / .30 | .47 / .78 / .69 / .59 | -- |
| frozen modern F x .37 | 0 / -1 / 0 / -7 | 1.8 / .82 / .45 / .27 | .58 / .79 / .66 / .58 | -- |
| F x .40 with the West | -4 / -6 / -5 / -10 | 4.8 / 1.2 / .57 / .30 | -- | .13 / .72 |
Established (step 3, demography otherwise as fitted): (1) a fixed habitat with one time-homogeneous brake (Fmax x
.37-.40) can reproduce first-arrival timing about as well as the fitted inflation, but NOT the abundance trajectory:
the 1966-79 East is 2-4x too full and the modern East falls to ~30% -- the user's "late too slow" holds for local
fill-in / abundance even where it fails for first arrival; the inflation supplies late front AND late abundance growth.
(2) The same low-F fixed habitat holds the native West behind the Plains with no temporal change (West-only arm never
enters the East), at the cost of a collapsing Plains population late (.13 of fitted). (3) Yearling base (step39):
Fmax x .35-.5 under frozen modern habitat stays 17-26 yr early -- the step-3 "F x .4" match is specific to the
delayed-breeding demography (likely: step39's higher adult survival keeps the front fast even with low F; unchecked).
(4) Density floors .0003 / .001 birds/km^2 slow step 3 by <= ~5 yr and step39 not at all; .003 stops spread. An
establishment threshold does not create the lag. (5) F x .4 on the FITTED habitat: release dies (habitat gating plus
low F).
Implication: the fit is using one lever (H(t)) for two jobs -- slow early arrival and a steep late rise in abundance.
Next (R3): can REAL capacity growth in occupied habitat (shared East+West K rise; feeders / suburbs, G15) restore the
late abundance while low F keeps arrival timing? And an F rise for contrast (equivalent to inflation, but shared).

## R3 (pre-registered 2026-11-03, before running; specs R3_step3 / R3_step39)
Step 3: Fmax x .4 under frozen modern habitat plus a SHARED logistic K rise (x2 / x3 / x5 by the 2000s; centre 1975 or
1990; tau 8) -- a stand-in for real capacity growth in settled landscapes (feeders, suburbs; G15); with the West; East-only
K for contrast; and a shared Fmax rise .4 -> 1.0 (centre 1975) as the "inflation-like" contrast. Prediction: the K rise
restores late East abundance toward the fitted (model_over_reference -> ~1 for 1994-2025) while barely moving first
arrival (K enters departure only through N/K, which LOWERS emigration -- so arrival may even slow); the F rise restores
abundance but speeds the late front (the inflation route). Step39: conspecific attraction in settlement (a .1-1,
juveniles or all, eps .1; fit code a267c24 form) under frozen modern habitat + 1940 release. Prediction: attraction
slows first arrival by a few years at most (step 24: recovery guard passed, fitted a ~.013) -- not the lag lever.

## R3 RESULT (2026-11-03; canaries identical; 0 segfaults)
Step 3, frozen modern habitat, release only. East abundance vs fitted by era (66-79 / 80-93 / 94-09 / 10-25).
| arm | offsets | East abundance | R90 1995 km |
|---|---|---|---|
| F x .4 | -3 / -5 / -4 / -9 | 4.4 / 1.1 / .55 / .30 | 1151 |
| F x .4 + shared K x3 rise (t 1975) | -1 / -3 / -3 / -10 | 4.5 / 2.2 / 1.3 / .66 | 1028 |
| F x .4 + shared K x5 rise (t 1975) | 0 / -1 / -2 / -10 | 3.4 / 2.8 / 1.9 / 1.0 | 954 |
| F x .37 + shared K x3 rise (t 1975) | +4 / +2 / +1 / -7 | 1.1 / 1.3 / .93 / .53 | 730 |
| F x .4 + shared Fmax rise to 1.0 (t 1975) | -6 / -9 / -10 / -14 | 9.0 / 1.9 / .90 / .56 | 1477 |
| F x .4 + K x3, West included | -2 / -4 / -3 / -10 | 5.1 / 2.3 / 1.3 / .66 | West 1.7-2.2x fitted |
Established (sandbox, step 3): with NO habitat change, low fixed fecundity plus a capacity rise reproduces arrival
timing within ~4 yr and keeps East abundance within ~x1.3 of the fit through 2009 (F .37 + K x3). The K rise slows the
front slightly (departure depends on N/K) while lifting abundance -- it decouples fill-in from spread, which a fecundity
rise (inflation-like) cannot: that speeds the late front. Costs: the late radius is short (730 vs 1422 km in 1995), the
modern East is at ~half the fitted level, and a uniform continental K rise doubles the West -- capacity growth would
have to be concentrated (settlement / suburbs / feeders), which is a testable, data-bearing hypothesis (G15) rather
than a free H(t). Step39 attraction (a .1-1, juveniles or all): no lag (-27 throughout), as predicted; at a = 1 late
years became NaN (numerical instability in that form; recorded via the new nonfinite counter).

## R5 inverse diagnostic (pre-registered 2026-11-03, before running; inverse.py, step 3)
Fixed habitat (frozen modern), release only; fit an 8-knot (1940-2025) East multiplier on K, or on Fmax, plus a constant
Fmax scale, by Nelder-Mead to arrival offsets 0 (sd 3), CBC p90 radius 1959 / 1965 (153 / 267 km, log sd .3) and East
abundance = the fitted arm's per era (log sd .2). Harness regression: refactored Sandbox reproduces R0 hashes exactly.
Prediction: K version -> low early K rising steeply 1965-95 (capacity growth after establishment) with F scale ~.4;
Fmax version -> low early F rising after ~1960 (an inflation-like reproductive rise), i.e. the same need as H(t) but in
one rate. If neither reaches the fitted arm's loss, a time-varying single rate cannot replace H(t)'s spatial pattern.

### R5 result (2026-11-03): inconclusive, two design flaws; rerun as R5b
- **K inverse never moved any knot.** All 8 multipliers stayed at 1.000 ± 2e-4 over 371 evaluations; only the Fmax scale moved, to .376. Cause (established): scipy's default Nelder-Mead simplex perturbs zero-valued entries by only 2.5e-4, and the log-knots start at 0. The Fmax run moved a little only because its scale started away from 0. It reached .80 → 1.47, a noisy shape with no clear step (loss 181 against the fitted arm's 209).
- **The CBC radius term was flat.** At 0.1 birds/km², R90 sits at the 10 km floor in 1959, and in 1965 for most arms, the fitted arm included. Only 1 distinct value appeared in the K run and 27 in the Fmax run. That term (about 111 of the loss) carried no shape information.
- **Side finding (likely; the unit conversion is approximate):** the fitted step-3 arm has no eastern cell at 0.1 birds/km² or above in 1965, while CBC circles spanned about 267 km. In absolute density the fitted model is behind early, consistent with lambda inflation suppressing early growth (G13).
- **Fixes in R5b:** an explicit simplex with step .3 in log for every parameter; the CBC radius on a new `abs_0.01_birds_km2` curve; maxiter 600. Pre-registration unchanged.
- Host: no segfaults, and every final-state SHA was distinct (no stuck evaluations).

## Observation-unit readout (2026-11-03): abundance in the fit's own BBS terms, no unit conversion
`Sandbox.observe` re-traces the model at the refined point with the forward call returning the replayed state, so
route / observer / year effects, the breeding-season mix and the NB2 likelihood apply as in the fit. Validation
(established): on the unedited fit it reproduces `expected_obs` and the per-row log-likelihood exactly (max diff 0.0,
123,191 real rows). New per region x era fields: expected_sum, obs_over_expected, loglik, zero_share_obs / _pred.
Canaries c13f3f7ddb69 / 3c1a05520bd0 reproduced. West and Plains rows in release-only arms are meaningless (the native
population is removed).

| East, observed / expected (zero share obs / pred) | 1966-79 | 1980-93 | 1994-2009 | 2010-25 | East log-lik |
|---|---|---|---|---|---|
| fitted | .78 (.98/.98) | 1.13 (.70/.75) | 1.12 (.45/.54) | 1.13 (.49/.71) | -61,061 |
| frozen modern, release only | .016 (.98/.48) | .36 (.70/.36) | 1.11 (.45/.53) | 1.97 (.49/.78) | -124,239 |
| frozen modern, Fmax x .4 | .14 (.98/.93) | .93 (.70/.72) | 1.92 (.45/.69) | 3.89 (.49/.87) | -79,229 |

Read (established, step 3, frozen modern habitat):
- In BBS units the fitted East is calibrated within about 1.3x in every era. In 1966-79 it runs slightly high (obs/exp
  .78), confirming the withdrawal above: the fitted early BBS-era East is not empty.
- Frozen habitat: the East is about 60x too dense in 1966-79 and 2x too sparse by 2010-25. "Early too fast" holds in
  abundance, and the modern fill-in is too low.
- Fmax x .4, which matched arrival timing (R2): still about 7x too dense in 1966-79, and 1.9x / 3.9x too sparse in
  1994-2009 / 2010-25, at a cost of 18,000 East log-lik units. In data units, the time-homogeneous brake buys arrival
  timing with exactly the "early too much, late too little" abundance shape.
- Separate finding (likely): even the fitted model predicts too many zero routes in 2010-25 (.71 vs .49 East, .70 vs
  .53 West, .80 vs .67 Plains), while its count totals are right. Modern birds are spread over more routes than the
  model puts them on, so the modern occupancy is too patchy (the concentration / occupancy structure, P-checklist
  candidate). Mechanism unknown.
Next: R5c reruns the inverse fit with the East BBS log-likelihood plus the arrival offsets as its loss, dropping the
conversion-based CBC radius term.

## R5c RESULT (2026-11-03; gradient inverse, grad_inverse.py; loss = East BBS log-lik, fit's own observation model)
Gradient check: autodiff matches central differences (1591.87 vs 1591.87; -13082.6 vs -13082.5). About 3.5 s per
value + gradient. Fitted reference -61,060.
- Fmax version (100-iteration cap, not formally converged; 7 min): effective birth rate relative to the fit's, at
  1940 / 55 / 65 / 75 / 85 / 95 / 2010 / 25 = .14 / .39 / .48 / .97 / 1.21 / 1.04 / 2.49 / 2.24. A steady rise of about
  16x over 70 years. -61,884, within 824 of the fit. Offsets +3 / 0 / -3 / -8; obs / expected .90 / 1.05 / 1.24 / 1.34.
- K version (converged, 4.5 min): -67,620, much worse. Knots hit the bounds (x20 in 1965, x.02 in 1975 and 2025): no
  sensible capacity-only path. Not interpretable as a shape.
Read (established for this setup): with habitat fixed, the best one-rate history is a smooth, large, continuing rise
in eastern birth rate from 1940 to about 2010, i.e. lambda inflation by another name, and it gets close to the fit's
score. No step near 1960 and no 1970s trough; R5b's trough is therefore likely an artefact of its conversion-based CBC
term and fitted-arm abundance targets (likely). Capacity alone cannot substitute. Time-only multipliers cannot tell
"improves where finches have been longest" from "improves everywhere on the calendar": next test, a multiplier on
local colonisation age vs calendar year.

## R6 (pre-registered 2026-11-03, before running): local settling-in vs calendar time
Question: R5c's fixed-habitat fit wants eastern birth rate to rise about 16x over 1940-2010. Is that rise better
described by time since each cell was colonised (local mechanism: new colonies start as sinks and improve, per Badyaev
2009; feeder or urban learning; Allee recovery) or by calendar year (continent-wide change: settlement, feeders,
climate)? Same setup as R5c (step 3, habitat frozen at the 1966-2025 mean, release only, loss = East BBS log-lik,
gradient L-BFGS, 200-iteration cap).
- Arm CAL: Fmax x m(calendar year), 8 knots 1940-2025 (R5c rerun with the same cap).
- Arm AGE: Fmax x m(years since the cell's colonisation), 8 knots at ages 0 / 2 / 5 / 10 / 15 / 25 / 40 / 85. The
  colonisation year is the first year the cell's density reaches 1% of the reference arm's modern occupied median
  (unit-free). It depends on the state, so it is solved by fixed-point iteration: fit m given colonisation years,
  re-simulate, update the years, refit (up to 5 rounds, until under 2% of cells change). Uncolonised cells use age 0.
- Same parameter count (8 knots + Fmax scale) in both.
Predictions: if the local mechanism carries the rise, AGE beats CAL by a clear margin (> 200 log-lik units) and m(age)
rises over the first 10-25 years; if CAL wins, the rise is a calendar-time change, so a shared driver (or habitat
change, which the full fit uses) is needed. A tie means the BBS-era East cannot tell them apart, because colonisation
age and calendar year are confounded along a single expanding front (my expectation: they are partly separable
because the core and the front differ in age at the same calendar year).
Guards: AGE knots at their bounds; fixed point not settling; offsets; obs / expected by era.

## Front fullness, unit-free (2026-11-03; front_fullness.py on R0_step3 saved states)
When an eastern cell is first colonised (density >= frac x the reference arm's modern occupied median), how full is
the fullest already-colonised cell within 2 grid steps the year before, relative to its own 2010-25 median? Median
ratio by arrival era:

| arm | frac .001 | frac .01 | frac .1 |
|---|---|---|---|
| fitted, 1966-79 / 1980-93 | .002 / .11 | .048 / .019 | .54 / .18 |
| frozen modern, release only, 1940-65 / 1966-79 | .004 / .31 | .035 / .19 | .39 / .55 |
| frozen all-years (the slow arm), 1940-65 / 1966-79 | .078 / .81 | .64 / .96 | 1.96 / 2.02 |

Read:
- Established: the number depends strongly on how "colonised" is defined. "The front is carried by densities far
  below settled" holds at the 1% definition (sources about 2-5% full in the fitted model's main expansion), but only
  about 18% at the 10% definition, which is closer to BBS detectability. This replaces the withdrawn birds-per-km²
  claim; no absolute statement is warranted.
- Robust across all three definitions: fast fronts (the fit, frozen modern) advance from sources well below their
  own settled level, while the slow arm (the one that matches the early lag) advances only from sources at or above
  it. The slow arm behaves like a pushed front, the fast ones like pulled fronts (likely; a proper pushed / pulled
  test needs the growth-rate decomposition). Settled levels in the fitted arm include habitat improvement, which
  biases its ratios low.

### R6 result (2026-11-03): CAL done; AGE fixed point did not settle, so it was replaced by R6b
- CAL (converged, 72 evals, 3.3 min): -61,886 (the fit: -61,060). Effective Fmax x .27 / .29 / .50 / 1.05 / 1.20 /
  1.03 / 2.50 / 2.37 at 1940 ... 2025. Reproduces R5c.
- AGE fixed point: -64,043 / -62,904 / -65,971 / -66,049 over rounds 0-3, with 27-59% of eastern cells changing
  colonisation year each round. It oscillates instead of converging, because refitting moves the front, which moves
  every age. Not a valid comparison (established design flaw).
- R6b replaces it: the colonisation clock is carried inside the scan (clock_forward.py: a soft years-occupied clock
  in the unused occupancy slot; East Fmax x g(clock); c recomputed). Everything is differentiable and fitted in one
  pass. Checks: identity with g = 1 to 1.4e-13 relative; autodiff matches finite differences (1545.34 / 1545.35,
  -3334.68 / -3334.67); start value 79,228.7 equals the Fmax x .4 arm; readout equals the objective. Same
  pre-registration as R6 (AGE vs CAL, margin > 200 log-lik).

### R6b RESULT (2026-11-03; in-scan clock; East BBS -log-lik, lower is better; the fit: 61,060)
| arm | -log-lik | offsets 66-74 / 75-84 / 85-94 / 95-09 | East obs / expected by era |
|---|---|---|---|
| CAL: Fmax x m(calendar year) | 61,886 | +3 / 0 / -3 / -8 | .92 / 1.05 / 1.24 / 1.33 |
| AGE: Fmax x g(years occupied) | 64,041 (200-iteration cap) | +1 / -1 / -1 / -5.5 | .37 / .94 / 1.21 / 1.38 |
| AGE + CAL (17 parameters) | 61,528 (cap) | +2 / -1 / -4 / -7 | 1.04 / 1.14 / 1.25 / 1.33 |

- AGE alone: effective Fmax x .003 at ages 0-2, .16 at 5, 2.6 at 10, then jagged (.56 / 3.0 / 1.8 / .28 at 15 / 25 /
  40 / 85). New cells barely breed, so a cell must fill by immigration before its clock starts: a pushed front, the
  "new colonies are sinks" form (Badyaev 2009). It gives the best arrival timing of the three, but the early East is
  2.7x too dense and the score is 2,155 worse than CAL.
- AGE + CAL: the age term flips sign (new cells x 6.9 / 2.3 at ages 0 / 2, mature about .5-.9). The calendar part
  keeps the R5c rise (x .16 in 1940 to x 4.5 in 2025). It gains 358 over CAL alone and is still 468 short of the fit.
- Pre-registered test (AGE beats CAL by > 200): FAILED. CAL beats AGE by 2,155. The age effect is not stable in
  sign once calendar time is present, so it does not read as a mechanism (likely).
- Read (established for this setup: step 3, fixed habitat, one clock definition): the fixed-habitat birth-rate rise is
  mainly a calendar-time change, not local settling-in. Under the pre-registration that points to a shared driver,
  or to the habitat change the full fit uses. The Plains / West test (does one calendar rise also fit the West?)
  follows as R7.
- Caveats: age 0 also covers never-colonised cells (so g(0) acts as an establishment brake or boost); clock
  threshold 1% with soft scale .5 only; AGE and AGE + CAL hit the iteration cap.

## R7 (pre-registered 2026-11-03, before running): one shared calendar rise vs East-only, native West included
Setup: step 3, habitat frozen at the 1966-2025 mean, native West AND the 1940 release both present, loss = minus the
BBS log-likelihood over ALL real rows (East + Plains + West) under the fit's own observation model; L-BFGS with exact
gradients, 200-iteration cap. Fmax x fscale x a calendar multiplier (8 knots 1940-2025; flat before 1940):
- R7a EAST: multiplier on East cells only (West and Plains at fixed habitat and fitted rates).
- R7b SHARED: one multiplier on every cell (a continent-wide driver: settlement, feeders, climate).
- R7c SEPARATE: independent East and not-East multipliers (17 parameters).
Questions and predictions:
1. Can a fixed habitat hold the West west of the Plains before about 1960 at all? Prediction: no. R0 showed that
   without habitat change the native West crosses the Plains before 1955; a low early multiplier (R7b, R7c) may now
   hold it. Watch Plains obs / expected for 1966-79 and the East's arrival offsets (an early western invasion of
   the East shows up as negative offsets).
2. If R7b comes within about 300 of R7c, one shared calendar change suffices (supports CONSIDERATIONS G15, a
   simultaneous western acceleration). If R7c beats R7b by a lot and the West's multiplier stays flat, the rise is
   eastern-specific (eastern adaptation or migration, or an eastern habitat change).
3. Reference: the fitted model's all-row -log-lik (habitat change and inflation) is the target to compare against.
Guards: knots at bounds, Plains zero shares, West obs / expected by era.
R7 implementation note (caught before any result): the first objective summed every real BBS row, including those
outside the 25-49N region bands that the readout and the fitted reference use (124,395 readout vs 126,679 objective
in the smoke test). It is now restricted to the band rows: objective = readout to 1e-8. R5c, R6 and R6b used band-
restricted East rows from the start (their readouts matched their objectives).

### R7 RESULT (2026-11-03; all 25-49N BBS rows; -log-lik, lower is better)
| arm | -log-lik | gap to fit | parameters |
|---|---|---|---|
| fitted model (habitat change + inflation) | 110,647 | 0 | -- |
| frozen habitat, no multiplier | 194,568 | +83,921 | 0 |
| R7a EAST-only rise | 116,837 (cap) | +6,190 | 9 |
| R7b one SHARED rise | 112,262 | +1,615 | 9 |
| R7c SEPARATE East / not-East | 111,711 | +1,064 | 17 |

Effective Fmax multipliers at 1940 / 55 / 65 / 75 / 85 / 95 / 2010 / 25 (flat before 1940):
- R7b all cells: .05 / .54 / .57 / .49 / .80 / 1.09 / 1.74 / 1.84.
- R7c East: .06 / .48 / .67 / .40 / .98 / .98 / 2.48 / 2.37 (about 40x). Not-East: .19 / .01 / .80 / .51 / .63 / 1.17
  / 1.25 / 1.69.
- R7a East: .08 / .52 / .46 / .43 / 1.04 / 1.01 / 2.64 / 2.20, with the West held at a constant x.455.
Offsets (East BBS cohorts) are +2 to +3 / -1 to 0 / -3 / -7 to -8 in every arm, as in the fit (+2 / -1 / -3 / -6.5).

Read:
1. (Established, step 3) A fixed habitat plus regional calendar trends in birth rate reproduces the eastern
   invasion's timing and comes within about 1,000-1,600 log-lik units of the full fit; frozen habitat alone is about
   84,000 worse. Most of what the fitted habitat change buys can be bought with a few regional time trends. Per
   CONSIDERATIONS and the regional-trend memory, this shows predictability, not cause: a regional calendar trend
   can itself be habitat change (M6).
2. (Established) The West needs a rise too. Holding it constant (R7a) costs 4,000 in the West after 1994 and leaves
   the late Plains 4.6x too sparse. Pre-registered threshold for "one shared rise suffices" (within 300 of
   SEPARATE): not met (551), so the change is shared in direction (a western rise of about 2-3x from mid-century)
   but steeper in the East (about 40x from 1940). Supports G15 (a simultaneous western acceleration) in direction,
   not in size.
3. (Likely, artefact) The not-East 1955 knot at x.01 is the fit choking the West in the 1950s so it cannot cross the
   Plains before BBS starts; no data before 1966 constrain it. Every arm suppresses all pre-1940 breeding (x.05-.19),
   the Wikle mu_t / lambda-inflation pattern again.
4. Not tested here: what the regional trends ARE. Next: whether a measured, time-varying covariate (human density,
   housing, or the DESK z change itself) reproduces the multipliers without free time knots (card C1 / C3).

## R8 (pre-registered 2026-11-03, before running): can MEASURED human population stand in for the calendar trends?
Covariate: HYDE population density, from the DESK covariate states (`processed/encoder/states/yearly_states`, the
`hyde` stream, yearly 1902-2025, 27 km), averaged to 54 km over the 2x2 child blocks (the coarsen_model_grid layout:
origin-aligned, padded south / east). z = standardised log1p(density). Built-up intensity (`bui`) is static across
years in these states, so it cannot carry a time trend and is not used. Setup as R7 (step 3, habitat frozen at the
1966-2025 mean, native West + release, all 25-49N BBS rows). Fmax x fscale x exp(bs zbar_cell + bt (z_cell,t -
zbar_cell)), where zbar_cell is the cell's 1966-2025 mean, on every cell:
- R8a STATIC: bs only (spatial pattern of settlement, no time change; 2 parameters).
- R8b SPLIT: bs + bt (adds the measured within-cell change over time; 3 parameters).
- R8c SPLIT + CAL: bs + bt + the R7b shared calendar knots (what calendar time adds beyond population; 11 parameters).
Matched access (memory: match dimensional access): R8a vs R8b isolates what the TIME change in population adds;
R8b vs R7b (9 calendar parameters) asks whether 1 measured temporal coefficient does what 8 free knots do.
Predictions: bt > 0 (finches gain as people settle); R8b recovers a meaningful share of R7b's gain over frozen
habitat (194,568 -> 112,262), but not all of it, because HYDE rises only about 3x while the East needed about 40x.
If bt about 0 or R8b gains little over R8a, settlement growth is not the driver and the calendar trend stays
unexplained.
Caveats in advance: HYDE before 1950 is interpolated from sparse censuses; population density is also already a DESK
covariate, so its spatial pattern partly duplicates the frozen habitat (bs absorbs residual spatial effects).
R8 implementation note: the first static run stopped at its start. L-BFGS-B's first step went to the bounds corner
(raw gradient norm about 24,000; 1.5e20 there), the line search failed, and it returned the start as "converged".
The optimizer now sees the objective / 1e4; reported scores are unscaled. That attempt is kept in
R8_step3/unscaled_attempt. Earlier rounds (R5c-R7) ran unscaled but moved away from their starts and reached
plausible optima. Their numbers stand, though their convergence may be less tight than reported.

### CORRECTION (2026-11-03, user challenge): built-up intensity is NOT static
The R8 pre-registration said `bui` "is static across years". Wrong. My check compared whole-grid medians, which were
dominated by the constant fill outside BUI's CONUS-only coverage (bui_avail is about 37% of the 27 km grid) and by
the many unbuilt cells (cell median 0). Within coverage, the cell mean of bui_q50 is 21 / 72 / 241 / 605 / 1,150 /
1,210 at 1902 / 1940 / 1966 / 1990 / 2015 / 2025 (about 17x since 1940; q90 302 -> 7,610), against roughly 3x for
HYDE population. BUI is the stronger candidate for the eastern rise and is added as R8d.

## R8d (pre-registered 2026-11-03, before running): measured built-up intensity
Same setup and objective as R8. Covariate: log1p(bui_q90) (the within-cell 90th percentile, so towns register in
otherwise rural cells), averaged over the 27 km children that have BUI coverage; standardised over covered 54 km
cells. Cells without coverage (Mexico, Canada, coastal gaps) get z = the covered mean in every year, so their
multiplier is set by fscale alone (no BUI effect there).
- R8d1 BUI SPLIT: Fmax x exp(bs zbar + bt dz) with BUI only (3 parameters), for comparison with R8b (population,
  119,998).
- R8d2 POP + BUI SPLIT: both covariates' static and temporal terms (5 parameters).
Predictions: bt_BUI > 0; R8d1 beats R8b because BUI grows far faster than population; neither reaches R7b (112,262)
unless the eastern rise is mostly settlement growth. q50 instead of q90 as a sensitivity arm if R8d1 is informative.

### R8 RESULT (2026-11-03; all 25-49N BBS rows; -log-lik; the fit 110,647; frozen 194,568; R7b free shared calendar 112,262)
| arm | -log-lik | parameters | temporal coefficient | East obs / expected by era |
|---|---|---|---|---|
| R8a population, static | 130,486 | 2 | -- (bs -.05) | .23 / 1.02 / 1.83 / 3.60 |
| R8b population, split | 119,998 | 3 | bt 3.79 | .45 / 1.03 / 1.55 / 2.15 |
| R8c population split + calendar | 111,848 | 11 | bt .93 | .84 / 1.08 / 1.23 / 1.40 |
| R8d1 BUI (q90), split | 113,175 | 3 | bt 4.29 | .68 / 1.08 / 1.28 / 1.46 |
| R8d2 population + BUI, split | 112,887 | 5 | pop bt -.66, BUI bt 4.88 | .67 / 1.08 / 1.28 / 1.45 |
All converged. Arrival offsets in R8b-R8d2 are +0 to +3 / -2 to 0 / -4 to -3 / -9 to -7, like the fit's.

Read:
1. (Established, step 3, fixed habitat) Measured built-up intensity change, with ONE temporal coefficient, recovers
   98.9% of the free shared calendar trend's gain over frozen habitat ((194,568 - 113,175) / (194,568 - 112,262)),
   and sits 2,528 short of the full fit. Finches breed better where building grows (bt > 0).
2. Population growth is weaker (R8b 119,998). With BUI present, its temporal term turns slightly negative: BUI
   carries the settlement signal. Population's weakness is likely its slower rise (about 3x against about 17x).
3. The static spatial terms are near zero everywhere (bs .03-.16): the frozen habitat already holds the spatial
   pattern of settlement. The gain is temporal (matched access: static vs split).
4. Caveats: BUI covers CONUS only (Mexico, Canada and coastal gaps get no BUI effect); BUI is a DESK covariate, so
   this is plausibly part of what the fitted habitat change already encodes. Not causal: built-up growth is
   correlated with every other post-war change (feeders, ornamental plantings, the observer pool). It shows that a
   measured settlement trajectory has the right shape and timing, which is what a legitimate driver must have.
Production implication (report, not decision): a habitat model in which settlement growth enters directly, or a
covariate path the DESK change already carries, could replace much of the free lambda inflation; the remaining gap
(about 2,500) and the West's details need checking.
Next (GPU float32 screening, 40 iterations): BUI split + calendar (what is left for calendar time), and the BUI q50
channel as a sensitivity arm.

## R9 (pre-registered 2026-11-03, before running): does the FITTED habitat change already carry built-up growth?
User's concern: BUI is a DESK input, so R8's BUI gain may duplicate what the fitted habitat change already encodes.
Records synthesis (DESK line: BUI attribution, urban sensitivity of the reference community, all-species vs
trait-subset kernels) running in parallel. Direct tests on step 3 (all 25-49N rows, GPU float32 screening, 40-iteration
cap; decision arms re-run float64):
- R9a FITTED + BUI split: the R8d1 term on top of the full fitted habitat (alpha 1). Start = the fit exactly.
- R9b FITTED + shared calendar: what free calendar time adds beyond the fit.
- R9c FITTED + population split.
- R9d (descriptive, no fitting): per eastern / western cell, the fitted model's own change in log Fmax, log K and Sa
  (1940 -> 2010) against the BUI and HYDE change (z units); correlations and the variance share.
Predictions and readings: if the fitted habitat already carries BUI, R9a gains little (< about 300) with bt near 0,
and R9d shows a clear positive association between the fitted rate change and BUI change. If R9a gains a lot with
bt > 0, DESK under-represents the House Finch's response to building growth. The reasons could be that the
community is less urban-responsive than the finch, or that the BUI signal is diluted in z.

### R9 RESULT (2026-11-03; GPU float64; starts exactly at the fit, 110,646.5)
| term added on top of the FITTED habitat | -log-lik | gain | coefficient / multiplier |
|---|---|---|---|
| BUI split (R9a) | 110,644 | 3 | bt .10 (4.29 on frozen habitat) |
| shared calendar knots (R9b) | 110,642 | 5 | .92-1.17 at every knot |
| population split (R9c) | 110,641 | 6 | bt .26 (3.79 on frozen habitat) |
(A float32 attempt could not move from the fitted optimum: the surface is steep, a .03 step costs about 1,000, and
float32 score jitter of about 1 unit stalled the line search. Tests that start at an optimum need float64. That
attempt is kept in R9_step3/f32_attempt.)

R9d descriptive (1940 -> 2010, step 3 fitted rates, BUI-covered cells):
| region | mean change log Fmax / log K / Sa | corr with BUI change (Pearson; Spearman) for log Fmax / log K / Sa | R2 (BUI + POP) for log K |
|---|---|---|---|
| East (n 1,034) | +2.39 (x11) / +3.34 (x28) / +.28 | -.20; -.27 / -.33; -.35 / -.34; -.34 | .12 |
| Plains (n 368) | +1.53 / +2.03 / +.16 | -.24 / -.35 / -.35 | .13 |
| West (n 1,293) | +.53 / +.90 / +.07 | +.06 / +.22 / +.31 | .07 |

Read:
1. (Established, step 3) The fitted habitat change leaves no time signal for any birth-rate trend to explain: BUI,
   population and free calendar time each gain 3-6 units on top of it. In the likelihood, BUI growth and the fitted
   habitat change are SUBSTITUTES: either alone explains the temporal rise (BUI on frozen habitat comes within 2,528
   of the fit; with the fitted change present, BUI's coefficient falls from 4.29 to .10).
2. (Established) Spatially they are not the same thing. The fitted change is a broad eastern rise (K x28, Fmax x11)
   that is largest where building grew LEAST in the East and Plains (correlations -.2 to -.35); only in the West
   does it go with building growth (+.06 to +.31). So the fit does not "contain BUI" as a pattern; it gets the same
   temporal gain by a different spatial route.
3. Consistent with the records: z's eastern change is real community change (corr .88-.93), but the dims the House
   Finch fit leans on east of 80W are only partly shared by resident non-invaders (.25, CI includes 0;
   GRID54_Z_EAST_ATTRIBUTION). BUI's community-level share of DESK change is small (12-15% of size, corr .04-.13;
   E035d), and DESK's temporal network gives no nonlinear edge over linear rivals (E039).
So R9 cannot say whether the fitted change is "BUI plus something" or "an invasion-timing proxy". Next (R10): identify
the finch's BUI response where there is no invasion front (the native West), and test whether it transfers.

## R10 (pre-registered 2026-11-03, before running): is the finch's BUI response identified outside the invasion?
Logic: in the East, BUI growth and arrival timing rise together over the same decades, so an eastern BUI coefficient
can be invasion timing in disguise (the lambda-inflation confound). The native West is a long-established, largely
sedentary population with 1966-2025 BBS coverage and BUI growth that varies across cells, so a BUI response
estimated there is not confounded with a front. Setup: step 3, habitat frozen at the 1966-2025 mean, native West +
release, GPU float64, 100-iteration cap.
- R10a WEST-ID: BUI split (bs, bt, fscale) fitted to West rows ONLY; East and Plains are simulated but not scored.
  Readout: bt_West, plus the East and Plains log-lik under that West-identified response (an out-of-region TRANSFER
  test).
- R10b WEST baseline: fscale only, West rows only (what BUI adds within the West).
- R10c REGIONAL: separate bt for East and not-East, with bs and fscale shared, all rows.
Predictions and readings:
- If bt_West is clearly positive (> about 1) and R10a's transferred East log-lik recovers a large share of the
  East's gain under the joint BUI fit, BUI is a coherent finch habitat driver, and the eastern rise is at least
  partly a real settlement effect (supports G15). The principled production route is then a species-level settlement
  term identified where dynamics are not confounded.
- If bt_West is about 0 while bt_East is large (R10c), the eastern BUI effect is invasion timing: a measured
  covariate would just be a cleaner-looking lambda inflation, and should not be adopted on this evidence.
Caveat in advance: the West's BUI growth is smaller (dz about .37 SD against .65 in the East), so its precision is
lower; there are also West-specific changes (irrigation, M31).

### R8 follow-ups (2026-11-03; GPU float32 screening, 40 iterations; float32 noise about 25 units)
- BUI split + shared calendar: 112,023 (BUI alone 113,175 float64). Calendar time adds about 1,150 beyond BUI, and
  BUI's temporal coefficient falls from 4.29 to 1.0 once calendar knots are present. Calendar knots .34 / .30 / .44 /
  .51 / .89 / 1.02 / 1.58 / .80. BUI and a free calendar trend are largely substitutes (as BUI and the fitted habitat
  change are, R9).
- BUI q50 instead of q90: 115,060 (bt 3.06), about 1,900 worse than q90. The within-cell upper tail of building
  (towns in otherwise rural cells) carries more of the signal than the cell median (likely: finches track towns).

### R10 RESULT (2026-11-03; GPU float64; frozen habitat; -log-lik by region)
| arm | East | Plains | West | BUI temporal coefficient |
|---|---|---|---|---|
| fitted model (reference) | 61,060 | 5,696 | 43,890 | -- |
| R10b West baseline: fscale only, West rows scored | 137,572 | 10,242 | 45,031 | -- |
| R10a West-ID: BUI split, West rows scored | 82,865 (transfer) | 7,629 | 44,389 | 3.03 |
| R8d1 joint BUI split, all rows scored | 62,139 | 5,923 | 45,112 | 4.29 |
| R10c regional: separate East / not-East bt, all rows | 62,138 | 5,931 | 45,102 | East 4.34, not-East 4.20 |
Arrival offsets: R10b -39 / -48 / -64 / -68 (the West crosses early); R10a +11 / +4 / -12 / -15; R10c +2 / -1 / -3 / -8.

Read:
1. (Established, step 3, fixed habitat) The finch's response to building growth is identified without an invasion
   front. Fitted on the native West alone it is clearly positive (3.03) and gains 642 there, 56% of the West's
   frozen-to-fitted gap. Fitted jointly, East and not-East give the same coefficient (4.34 vs 4.20). Pre-registered
   reading "bt_West clearly positive": MET.
2. Transfer: the West-identified response, with nothing fitted to the East, recovers 71% of the East's gap to the
   fit (137,572 -> 82,865, fit 61,060) and stops the early western crossing. The timing shape is wrong, though:
   early cohorts are 11 years late and late cohorts 15 years early.
3. THE CATCH (established by arithmetic): as an Fmax multiplier, bt about 4.3 times the East's BUI change (+.65 SD,
   1940 -> 2020) is exp(2.8), i.e. a 16x rise in eastern fecundity, and about 5x in the West. That is lambda inflation
   (G13) with a measured covariate in place of free time. A 16x fecundity rise is not biologically credible. Settlement
   plausibly adds HABITAT (capacity, possibly winter survival via feeders), not per-capita fecundity. So R8-R10 show
   the TIMING and SHARED-ACROSS-REGIONS shape a settlement driver would need, not a credible mechanism.
4. Data caveat (from COVARIATE_ERA_QUALITY): HISDAC-US is reconstructed from the year-built field of SURVIVING
   structures; survivorship bias "increases towards early points in time" (demolitions unrecorded), and recall
   "drops almost logarithmically" back in time. Early BUI is therefore biased low and BUI growth biased high, most for
   the oldest eras. That bias points the same way as inflation (size unknown).

## R11 (pre-registered 2026-11-03, before running): settlement as CAPACITY, not fecundity
Same setup as R8d1 / R10c (frozen habitat, West + release, all rows, GPU float64). The BUI split term multiplies K
(carrying capacity) instead of Fmax; fscale stays a constant Fmax scale. c is unchanged (it depends on Fmax, Sa, Sj).
- R11a BUI -> K split (bs, bt, fscale).
- R11b BUI -> K split + a calendar Fmax multiplier (what free fecundity time still has to do).
- R11c survivorship sensitivity: BUI -> K with BUI change before 1940 frozen at its 1940 value (removes the
  most-biased early growth).
Predictions: R11a is worse than the Fmax version (113,175), because capacity alone cannot lift low-density growth
(R5c's K-only fit failed on release-only data). If it comes within about 2,000 of it with a plausible K rise
(< about 10x), capacity growth is a credible carrier of the settlement effect. If not, a credible mechanism needs a
low-density growth change (Allee / establishment) tied to settlement, not settlement alone.

### R11 RESULT (2026-11-03; GPU float64; frozen habitat, all rows)
| arm | -log-lik | East / Plains / West | East median change 1940 -> 2020 | offsets |
|---|---|---|---|---|
| BUI -> Fmax (R8d1, reference) | 113,175 | 62,139 / 5,923 / 45,112 | Fmax about x16 | +2 / -1 / -3 / -8 |
| R11a BUI -> K | 154,185 | 93,619 / 8,041 / 52,526 | K x8.7 | -57 / -63 / -73 / -70 |
| R11c BUI -> K, BUI frozen before 1940 | 154,187 | same | K x8.7 | -54 / -62 / -74 / -75 |
| R11b BUI -> K + calendar Fmax | 112,074 (cap) | 61,837 / 5,779 / 44,457 | Fmax x6.2, K x1.13 (BUI bt .20) | +3 / 0 / -3 / -7 |
Read (established, step 3, fixed habitat): settlement as capacity FAILS (41,000 worse than the Fmax version; the
pre-registered bar was 2,000). Fronts run 57-75 years early: a larger K raises the settled level but not low-density
growth, which sets front speed. With a free calendar Fmax term present, BUI -> K is dropped (bt .20) and free time
does all the work (x6.2 fecundity). The survivorship freeze changes nothing here. The timing information in the data
is about LOW-DENSITY GROWTH, so a credible settlement mechanism must act there: winter survival at feeders (card A6;
eastern feeder dependence, Aldrich & Weske 1978) or establishment / Allee relief, not capacity.

## R12 (pre-registered 2026-11-03, before running): settlement as SURVIVAL (feeders), bounded
Same setup. The BUI split term shifts adult survival on the logit scale: Sa' = sigmoid(logit(Sa) + bs zbar + bt dz),
Sj kept <= Sa', c recomputed (K stays the equilibrium). Survival is capped below 1, so this cannot inflate growth
without limit, which is the point of the test.
- R12a BUI -> Sa (logit), fscale constant.
- R12b BUI -> Sa + calendar Fmax (what free fecundity time still has to do).
Predictions: R12a lands between R11a (154k) and R8d1 (113k). A survival route can lift low-density growth, but by at
most (1 - Sa) per year. If R12a is within about 3,000 of R8d1 with an East Sa rise of no more than about .15, feeder
survival is a credible carrier. In R12b, the calendar Fmax rise shrinks well below x6 if survival does real work.

### R12 RESULT (2026-11-03; GPU float64; frozen habitat, all rows)
| arm | -log-lik | East / Plains / West | East median change 1940 -> 2020 | offsets |
|---|---|---|---|---|
| R12a BUI -> Sa (logit) | 118,311 | 65,774 / 6,430 / 46,107 | Sa +.40 | +2 / -1 / -4 / -9 |
| R12b BUI -> Sa + calendar Fmax | 112,304 (cap) | 62,101 / 5,778 / 44,424 | Fmax x7.6, Sa +.02 (BUI bt .14) | +2 / -1 / -3 / -7 |
Read (established, step 3): survival carries more than capacity (118k vs 154k) but needs an eastern adult-survival
rise of +.40, far beyond the pre-registered credibility bar (.15); the late East is 2.5x too sparse. With free
calendar fecundity present, BUI -> Sa is dropped (bt .14), and free time again does the work (x7.6). Across R8-R12:
no single settlement route at a credible size reproduces the eastern rise. The data ask for a 6-16x rise in LOW-
DENSITY growth over 1940-2020, whatever carries it. Settlement growth has the right timing and cross-region
coherence (R10), but in this model the required size is only reachable through fecundity, which is lambda inflation.
Open: (i) these are step-3 rates (averaged 27 km Z); the native 54 km production Z (step3_native, step 23) is the
right base for production-facing reads; (ii) whether a low-density mechanism (establishment / Allee relief tied to
settlement, card A6) gets there with credible parameters.

## NATIVE 54 km base (step3_native, step 23 production Z; 2026-11-03; GPU float64; all 25-49N rows)
Why: step 3 runs on the AVERAGED 27 km Z (GRID54_INPUTS: cube_s64_grid54 "coarsened_from", not a native DESK). The
native production field (field54_prod) is the production-facing base (user, 2026-11-03). Its encoder checkpoint and
native covariate states are not local (only stageF model inputs), so encoder-level attribution needs a TACC copy.
Fitted reference -110,156 (step 3: -110,647); frozen habitat 199,894.

| arm (frozen habitat unless noted) | -log-lik | East / Plains / West | notes |
|---|---|---|---|
| fitted + BUI split (alpha 1) | 110,150 | 60,917 / 5,646 / 43,586 | gain 6; bt .16 |
| fitted + shared calendar (alpha 1) | 110,138 | | gain 18; knots .84-1.13 |
| shared calendar | 113,563 (cap) | 62,912 / 5,867 / 44,785 | x.19 (1940) -> x16.7 (2025) |
| separate East / not-East calendar | 112,632 (cap) | 62,409 / 5,859 / 44,364 | East x.21 -> x22.6 (2010); not-East x.07 -> x7.2 |
| BUI regional (bt East / not-East) | 115,758 | 64,235 / 6,198 / 45,325 | bt 6.31 / 6.40 |
| BUI split (single bt) | 151,140 | | optimizer stuck (fs .15): a nested case of the regional arm, which does far better. INVALID, not a result |
| West-ID: BUI split, West rows only | West 44,738 | East transfer 90,792 | bt_West 4.88 |
| West baseline: fscale only, West rows | West 45,752 | East 144,412 | |
R9d on the native fit (1940 -> 2010): East mean change log Fmax +7.15 (geometric x1,270), log K +2.77, Sa +.18.
Correlation with BUI change, East / Plains / West: log Fmax +.46 / +.48 / +.08; log K +.08 / +.34 / +.28;
Sa -.03 / +.31 / +.33. Step 3 had log Fmax -.20 / -.24 / +.06.

Read:
1. (Established, both bases) The fitted habitat change leaves no time signal for anything else (gains 6-18).
2. (Established, both bases) Measured BUI growth, with one temporal coefficient per region, recovers most of what free
   calendar time does on frozen habitat (native: 97.5% of the shared calendar's gain), with the SAME coefficient East
   and not-East (6.31 / 6.40; step 3 4.34 / 4.20). Identified in the West alone it is positive (4.88; step 3 3.03),
   gains 1,014 there (47% of the West's frozen-to-fitted gap), and transfers 64% of the East's gap (step 3 71%).
3. (Established, the key difference) On the NATIVE Z, the fit's own fecundity change follows BUI growth spatially
   (+.46 East, +.48 Plains); on the averaged Z it ran against it (-.20 / -.24). The native DESK carries more of the
   settlement pattern into the finch's habitat change, or at least into the part the fit uses.
4. (Established, worse) Lambda inflation is far stronger on the native base: the fitted eastern fecundity rises
   about x1,270 1940 -> 2010 (step 3: x11), and the BUI route needs bt about 6.3, i.e. x60 fecundity over 1940-2020.
   As on step 3, the size needed is reachable only through fecundity, so the mechanism problem (R10-R12) stands.

## R13 (pre-registered 2026-11-03, before running): settlement eases MATE FINDING (Allee relief), native base
Mechanism (card A6, likely in the literature): birds concentrated at towns and feeders find mates at lower regional
density, so settlement growth lowers the effective Allee threshold without raising fecundity, survival at density,
or capacity. Implementation: allee_forward.py fork; the encounter rate allee_gamma x G(cell, t), G = exp(bs zbar + bt
dz_BUI) (G > 1 lowers n50 = ln2 / gamma by the factor G). Fecundity stays a constant fscale. Native base
(step3_native), frozen habitat, West + release, all rows, GPU float64, 100 iterations.
- R13a BUI -> Allee.
- R13b BUI -> Allee + calendar Fmax (what free fecundity time still has to do).
Checks: grid of ones = the original forward (identity); objective = readout.
Credibility bounds (set now): a credible Allee relief lowers n50 by at most about 10x over 1940-2020 (G East median
<= 10). Predictions: R13a beats R11a (capacity) clearly, because Allee relief acts at low density where the timing
information is. It can only matter if the native fit's Allee effect actually binds at the front. If its gamma is
large (weak Allee), G has nothing to relieve and R13a about equals the fscale-only fit. In R13b, the calendar Fmax
rise shrinks well below the native shared calendar's x16-88 if Allee relief does real work.

### R13 RESULT (2026-11-03; native base; GPU float64; identity check 0.0; objective = readout)
Native fit's Allee: relative form (gamma . N/K), gamma 35.1, i.e. n50 = ln2 / 35 = .0198 K.
| arm | -log-lik | East / Plains / West | East change 1940 -> 2020 | offsets |
|---|---|---|---|---|
| fscale only (frozen baseline) | 199,892 | 144,366 / 9,769 / 45,757 | -- | -44 / -54 / -69 / -75 |
| R13a BUI -> Allee gamma | 141,010 | 81,003 / 7,397 / 52,610 | G x188,764 (bt at its bound, 20) | -21 / -4 / -17 / -13 |
| R13b BUI -> Allee + calendar Fmax | 113,266 (cap) | 62,709 / 5,852 / 44,705 | Fmax x62, G x.68 (bt -.63) | +2 / 0 / -3 / -8 |
| (BUI -> Fmax regional, native, for reference) | 115,758 | 64,235 / 6,198 / 45,325 | Fmax about x60 | +3 / 0 / -4 / -8 |
Read (established, native base, fixed habitat):
- R13a uses the Allee term as an EARLY BRAKE: gamma tiny when BUI is low (n50 orders of magnitude above K, so no
  breeding at low density), relaxing as building grows, together with a higher constant fecundity (fscale 1.89). That
  is the pushed-front configuration (D1 ridge) with a calendar-like brake. The coefficient hits its bound and the
  implied n50 change (about 2e5x) is far outside the pre-registered credibility bound (10x). It is also 25,000 worse
  than BUI -> Fmax.
- With free calendar fecundity present (R13b), the Allee route is dropped (bt -.63) and free time again does the work
  (x62 fecundity).
- Pre-registered prediction "R13a beats capacity clearly": met (141k vs the step-3 capacity arm's 154k; different
  bases, roughly comparable). "Credible relief within 10x": FAILED.
Across R11-R13 on both bases: capacity, survival and Allee relief tied to settlement each fail at credible sizes.
Within this model, the data's demand (low-density growth rising by an order of magnitude or more between 1940 and
2020, in East and West alike) is met only through fecundity. Either low-density growth really did change that much
(a missing mechanism), or the demand is an artefact of something else in the model's early state. Candidates:
dispersal and settlement structure, observation in the BBS era, or deterministic fractional fronts (the stochastic
twin showed integer birds spread FASTER, the wrong way for this).

## R14 (pre-registered 2026-11-03, before running): how big must a LOW-DENSITY-ONLY change be? (native base)
User's question: is lambda inflation an artefact of rigid growth forms? Test with the most flexible low-density
change the skeleton allows. lowdens_forward.py fork: that year's fecundity becomes Fmax (1 + (G - 1) exp(-(N/K) /
rho)), so G acts only at low relative density (establishment / low-density growth), fades as a cell fills (rho fitted,
.005-2), and leaves the crowding brake c and the settled level alone. Fecundity at density = fscale x fitted.
Frozen habitat, West + release, all rows, GPU float64, 100 iterations. Identity: G = 1 reproduces the original.
- R14a G(calendar), one shared 8-knot spline: the most flexible TIME shape a low-density-only change can take.
- R14b G(calendar), separate East / not-East.
- R14c G(BUI), a 6-knot spline in total BUI z (spatial + temporal; flexible, not linear).
- R14d R14c fitted on West rows only (East as transfer).
Readouts: score; G by era (East median, p10 / p90); rho; offsets; obs / expected.
Decision rule (the user's): if even this flexible low-density curve needs G to change about 10x or more between 1940
and 2020 to approach the fit, the rigid functional form is not the problem. If R14a reaches the native shared
calendar's score (113,563) with G changing well under 10x, much of the inflation is a form artefact and the middle-
ground flexible-growth route is worth promoting. The literature review of low-density growth factors (running) will
be used to judge the credibility of whatever G shape comes out.

### R14 RESULT (2026-11-03; native base; GPU float64; identity check 0.0 after a NaN fix in the fork)
First attempt invalid: identity NaN. Tiny negative post-dispersal densities off land divided by K about 0 made
exp(-N / K / rho) overflow. Densities are now clamped at 0; that attempt is kept in R14_native/nan_attempt.
| arm | -log-lik | East / Plains / West | G East 1940 -> 2020 | rho (fade, x K) |
|---|---|---|---|---|
| R14a shared calendar G | 113,256 | 62,548 / 5,959 / 44,749 | .08 -> 1.70 (x21) | .92 |
| R14b separate East / not-East calendar G | 112,661 (cap) | 62,229 / 5,908 / 44,523 | .12 -> 1.71 (x14) | 1.31 |
| R14c BUI spline G | 199,548 | -- | -- | -- |
| R14d BUI spline G, West rows only | West 45,587 | East transfer 145,701 | G x50-360 | .05 |
(For comparison, calendar on Fmax at every density, native: shared 113,563, separate 112,632.)
- R14c is INVALID: it stopped after 4 evaluations at its start (an optimizer stall like R8's first attempt), not a
  result. R14d's West gain over the fscale-only West baseline is only 165, with absurd G; uninformative.
- Read (established, native base, fixed habitat): the most flexible LOW-DENSITY-ONLY time shape matches or beats the
  same calendar trend on fecundity at every density, but it still needs a 14-21x change in low-density growth between
  1940 and 2020. And the fit does not keep it low-density: it chooses a fade rho of about 1 K, i.e. the boost acts at
  nearly all densities. User's decision rule: "if even a flexible curve needs a ten-fold-plus change, the rigid
  functional form isn't the problem". MET: the functional form is not the problem.

## Realised growth vs literature anchors (2026-11-03; realized_r.py; unit-free)
Anchors (literature agent): ~40-100 founders -> ~280 birds by 1951 (Elliott & Arbib 1953), r = .12-.18 / yr; NE CBC
birds per party-hour .06 (1962) -> .42 (1971), r about .22 (Bock & Lepthien 1976; abstract-only).
| fitted arm, r per year | 1941-51 | 1951-62 | 1962-71 | 1971-85 | 1985-94 |
|---|---|---|---|---|---|
| native, within 150 km of release | .14 | .20 | .31 | .21 | .03 |
| native, north-east box | .08 | .17 | .31 | .33 | .05 |
| native, whole East | .02 | .10 | .23 | .33 | .14 |
| step 3, within 150 km of release | -.06 | .16 | .27 | .19 | .02 |
Read (established for these fits): the native fit's REALISED growth already matches both anchors (about doubling from
the 1940s to the 1960s, as in the literature); step 3's release area declines in the 1940s. The order-of-magnitude
rise the fits put into POTENTIAL rates (fecundity x10+, lambda change +.85) produces only a doubling of realised
growth. So the model dissipates most potential growth early and less later. The question becomes where potential
growth is lost (emigration into poor habitat, journey mortality, crowding, Allee) and what changes it over time,
consistent with step 5's "habitat gating of the front" (inj1960: inflation is driven by the post-1966 period, not the
pre-BBS lag; user correction 2026-11-03: the observed invasion constrains the fit strongly; the release-year test was
already done).

## R15 (2026-11-03; native fitted rates; Z vs Z_disp frozen separately at the 1966-2025 mean; GPU float64)
Question (user: how do middle-ground options relate to Z and Z_disp?): contribution budgets show journey deaths about
equal to ordinary deaths in the West (step29_yb_native: births 2,760, ordinary deaths 1,428, journey deaths 1,396;
East 269 / 226 / 43; CONTRIBUTION_RESULTS_20261101.md:426-429). Is the habitat gating of the front acting through
journey survival (Z_disp) or through local rates (Z)? harness edit_habitat now takes habitat_keys.
| arm | offsets 66-74 / 75-84 / 85-94 / 95-09 | realised r East 1941-51 / 51-62 / 62-71 / 71-85 |
|---|---|---|
| fitted | +2 / -1 / -2 / -6 | .02 / .10 / .23 / .33 |
| freeze Z_disp only | 0 / -3 / -6 / -10 | .01 / .03 / .16 / .29 |
| freeze Z only | -21 / -24 / -30 / -32 | .29 / .19 / .08 / .02 |
| freeze both | -45 / -54 / -70 / -75 | (the West floods the East early) |
| release only: freeze Z_disp only | +4 / +1 / -1 / -5 | .15 / .14 / .34 / .37 |
| release only: freeze Z only | -21 / -24 / -23 / -24 | .63 / .30 / .11 / .04 |
| release only: freeze both | -21 / -25 / -28 / -32 | .65 / .32 / .12 / .02 |
Read (established, native base): the gating acts through LOCAL rates (Z), not journey survival (Z_disp). Freezing
Z_disp leaves the invasion within 0-4 years of the fit; freezing Z gives the familiar early-fast / late-slow shape
(realised r .29-.65 in the 1940s, falling to about 0 by the 1970s). Journey mortality is a large budget item, but its
change over time does not time the invasion. Middle-ground work should target the Z -> local-rate links (per-rate
readouts, a low-density readout from Z), not Z_disp.

## Where potential growth goes (2026-11-03; growth_gap.py, growth_gap_factors.py; native fit, East occupied cells)
Correction to my earlier framing ("the model dissipates 10x potential into 2x realised growth"): part of that is
arithmetic (r = log lambda, lambda = Sa + Sj F), and the rest is the model's own regulation, measured here.
Reproduction: F_eff = Fmax / (1 + c N/K) x mated fraction (relative form, gamma 35); c = Fmax Sa Sj / (1 - Sa) - 1
(delayed breeding), so K stays the equilibrium.
| East occupied cells, medians | 1941-55 | 1956-65 | 1966-75 | 1976-90 | 1991-2010 |
|---|---|---|---|---|---|
| Fmax | 4.4 | 7.2 | 10.5 | 6.4 | 8.2 |
| mated fraction, all occupied cells (N/K .02-.2) | .47 | .69 | .41 | .89 | .98 |
| crowding half-density 1/c (x K) | 1.12 | .30 | .10 | .39 | .23 |
| crowding factor, filling cells (.1-.7 K) | -- | .55 | .32 | .52 | .38 |
| effective F, filling cells (F x crowding x mated) | -- | 4.5 | 3.7 | 3.9 | 3.6 |
| potential / realised lambda, front cells (N/K < .1) | 1.25 / 1.16 | 1.59 / 1.25 | 2.14 / 2.08 | 1.56 / 1.78 | .78 / 1.01 |
| potential / realised lambda, filling cells | -- | 1.91 / 1.33 | 2.38 / 1.30 | 1.75 / 1.34 | 2.07 / .98 |
Read (established, native fit):
1. Above roughly 1/c (10-40% of K), the K-anchored crowding brake cancels fecundity changes almost exactly: effective
   fecundity in filling cells stays at 3.6-4.5 while nominal Fmax swings 6.4-12 (at large Fmax, F_eff about (1 - Sa) /
   (Sa Sj N/K), independent of Fmax).
2. In sparse cells the relative-form mate-finding term removes 30-60% of births early (.41-.69).
3. So Fmax acts almost only at low density, i.e. at the front: it is effectively a FRONT-SPEED knob separate from
   settled abundance (set by K and survival). Lambda inflation is the fit timing the front with its one front-speed
   lever. This explains why K-based settlement mechanisms failed (R11: K moves abundance, not the front) and why
   survival / Allee relief needed implausible sizes (R12, R13). In the 1990s-2000s front cells are sinks (potential
   .78) held up by immigration (realised 1.01); filling cells sit below K (realised .98 at N/K .24; post-1994 disease
   lowers the equilibrium below K).
Implication (report, not decision): the substantive question is what controls FRONT SPEED in reality (dispersal and
establishment at the edge: departure at low density, settlement into empty habitat, establishment probability) and
whether the model has any front-speed lever other than low-density fecundity.

## CORRECTION + replication on the CURRENT base (2026-11-03; user: the project uses ABSOLUTE Allee)
Everything Allee-related above (R13, and the "mate-finding factor .41-.69" in the growth-gap table) is specific to
step3 / step3_native, which use the OLDER RELATIVE Allee (gamma . N/K). The current lineage (steps 41-45) uses
absolute Allee. I chose step 3 for its known fronts and did not re-check its configuration against current practice.
Re-run on step43_factor_p2_veit_s41 (absolute Allee, Veit reproduction; the base of Codex's rescoring script):
- Z vs Z_disp (R15 spec): fitted +3 / -1 / -3 / -6; freeze Z_disp only +2 / -2 / -5 / -9; freeze Z only -28 / -36 /
  -49 / -46; release only: freeze Z_disp only +2 / -2 / -4 / -8, freeze Z only -15 / -19 / -21 / -23. REPLICATED: the
  front is timed through local rates (Z), not journey survival (Z_disp).
- Realised r near the release (150 km): 1941-51 .67, 1951-62 .01, 1962-71 .24, 1971-85 .22. Literature: .12-.18 in
  1941-51; release-cell CBC 37 (1951) -> 1,400 (1966), about .24 / yr (step 32 record). The current base has its boom
  and stall in the wrong order: a fast first decade, then a 1950s stall. Likely cause: the 10-year release Allee
  exemption ending (untested).
- growth_gap.py's potential lambda uses the Leslie formula, which is approximate under Veit reproduction (births
  2 Fmax H from site-limited pairs), so potential vs realised on this base is indicative only. The crowding / Allee
  factor decomposition (growth_gap_factors.py) assumes Beverton-Holt crowding and is not valid for this base.

## Which rate carries lambda inflation? (2026-11-03; lambda_decomp.py; Shapley attribution of the potential-lambda change)
User: fecundity is related to but separate from lambda inflation; how much does the fecundity diagnosis matter?
Potential lambda (Leslie; approximate under Veit reproduction), median over cells; shares of the mean change, Shapley
over all orderings, so they sum to 1.
| base | region | lambda 1940 -> 2010 | Sa share | Sj share | Fmax share | rates 1940 -> 2010 |
|---|---|---|---|---|---|---|
| step43_factor_p2_veit_s41 (current) | East | .21 -> 1.08 | .33 | .39 | .28 | Sa .13->.48, Sj .04->.15, F 1.84->4.06 |
| | Plains | .31 -> .74 | .37 | .38 | .25 | |
| | West | .52 -> 1.00 | .31 | .39 | .30 | |
| step3_native (older, relative Allee) | East | .40 -> 1.90 | .17 | .31 | .53 | Sa .40->.56, Sj .02->.53, F .00->8.51 |
| | Plains | .42 -> 1.08 | .15 | .31 | .53 | |
| | West | .62 -> 1.42 | .16 | .25 | .59 | |
(1966 -> 2010 gives the same split.)
Read (established for these fits, Leslie approximation): on the CURRENT base about 70% of the lambda inflation is
SURVIVAL (eastern adult survival .13 -> .48; .13 is biologically implausible), only about 28% fecundity. On
step3_native fecundity carries about half. So the fecundity / front-speed diagnosis above bears on about half of the
inflation on step3_native and about a quarter on the current base. It explains why capacity mechanisms failed, but it
is not a general account of lambda inflation. On the current base the question is why the fit needs very low early
SURVIVAL, which acts at every density.

### CORRECTION (2026-11-03; records agent + overlay check): what step43_factor_p2_veit_s41 is
grid54_step43_factor_p2_veit_s41.json: reproduction.form standard, yearling_breeding true, crowding_power 2.0,
absolute pinned Allee, release_allee_exemption_years 0, initialization_year 1951 (established state, free initial
abundance), Veit (gradual) DEPARTURE ("veit" in the name is the departure rule). I called it "Veit reproduction"
twice above, and read its 1941-51 realised growth (.67) and the 1950s stall as "the release Allee exemption ending".
Both are WRONG. There is no exemption, and 1941-51 precedes the model's 1951 initialisation, so that window is
meaningless. The 1951-62 realised r .01 near the release, against about .24 from release-cell CBC counts, still
stands as a discrepancy but has no explanation yet. The lambda decomposition (Sa / Sj about 70% of the change) uses
the exact Leslie lambda for standard yearling reproduction (the "approximate" caveat is withdrawn). The growth-gap
factor decomposition can be redone with crowding (N/K)^2.

## R16 (pre-registered 2026-11-03, before running): middle-ground READOUTS BUILT ON Z (current base)
User (repeated): middle-ground options should connect to Z / Z_disp, not to free functions of time or density.
Base step43_factor_p2_veit_s41 (absolute Allee, crowding power 2, yearling; fitted -110,045 on all 25-49N rows;
frozen habitat 178,022). The four existing readouts (Sa, Sj, Fmax, K from Z . beta) are held at FROZEN habitat. ONE
new readout of the model's own time-varying Z (64 dims, standardised) is fitted with exact gradients (GPU float64,
100 iterations, ridge sd .5 on its 64 coefficients, plus fscale):
- R16a THETA, time: crowding curvature theta(x, t) = exp(theta0 + z . b) (zcrowd_forward fork; K stays the
  equilibrium). Identity 4.8e-11.
- R16b THETA, static: the same with each cell's 1966-2025 mean z (space only) -- the matched control.
- R16c LOWDENS, time: low-density boost G = exp(g0 + z . b), fading with N/K at a fitted rho (lowdens_forward).
  Identity 0.0.
- R16d LOWDENS, static: the matched control.
- References on this base: fscale only; shared calendar on Fmax (9 parameters); separate East / not-East.
Readouts: score; how much of the frozen-to-fitted gap each closes; theta / G by era (East median, p10 / p90; West
median); offsets; obs / expected; b norm. Dimensional access: the new readout reaches all 64 dims, as the existing
readouts do (they use the same Z), and the static control matches access exactly.
Predictions: (1) time beats static for both, or Z's change carries nothing at that point of the density response;
(2) if a Z-built readout closes most of the gap with credible ranges (theta within about .5-6; G change within about
3x 1940 -> 2020), routing Z's change to the right part of the density response is a credible middle ground and
the existing readouts' inflation is partly a routing artefact. If it needs extreme values, it is not.

## Dispersal splits / habitat-informed dispersal: what exists (records + code agent, 2026-11-03; verified in code)
- Directional choice (step 27) ALREADY uses Z_disp: weights = directional_choice_weights(Z_disp . (gamma_j beta_sj +
  gamma_a beta_s), kernel mass, kappa), i.e. choice among the 4 directions WITHIN each band by path habitat
  (step44 code age_priors.py:1612-1629). It FAILED its null (kappa .214 vs gate .06) and the water cross-leak
  (.141); its real kappa 3.03 must not be read (DISPERSAL_COLLECTION_2026-10-31.md:5-13).
- Splits: 4 dir x 3 bands (production); 4 x 4 (b44, K = 16, Z_disp on disk) "not the lever", fronts identical; 8
  directions never built (pre-registered gate not met); band log-weights cut lambda change only through an implausibly
  fat tail (8a); era-varying weights add nothing (14b); 4-segment compounding path survival (step 30), with its
  cross-leak pending.
- Uncommitted working-tree code = kernel discretisation + land-share floor patch, not new splits. Nobody is working
  on splits now.
- NEVER TRIED: habitat-informed choice ACROSS bands (softmax over band x direction on the Z_disp signal); departure set
  by ORIGIN habitat; settlement set by DESTINATION habitat on the 54 km base, judged on lambda change / fronts.
  Pitfall: every front mechanism's recovery failed because fecundity / founders / survival absorbed it. Sandbox
  design: hold the fitted rates fixed and read how much free time trend each rule removes.
- Winter Z block: NOT local (stageF inputs carry the summer 64 dims only; Z_gathered = 124 x 4,318 x 64 float32). A
  winter-capacity readout needs field_d_prod_ema_s1 (128-d, about 480 MB) from TACC.

### R16 RESULT (2026-11-03; current base step43_factor_p2_veit_s41; GPU float64; identities 0.0 / 4.8e-11)
References on this base: fitted 110,045 (East / Plains / West 60,764 / 5,658 / 43,623); frozen 178,022; fscale only
177,978; shared calendar 112,615 (+3 / -1 / -3 / -7); separate calendar 111,939 (cap).
| arm (standard readouts frozen; ONE new readout of Z) | -log-lik | gap closed | E / P / W | offsets | readout East median (p10-p90) 1940 -> 2020 |
|---|---|---|---|---|---|
| lowdens, time-varying Z (b = .1 u) | 111,711 | .98 | 61,324 / 5,743 / 44,644 | +2 / -1 / -3 / -6 | G .44 (.25-.85) -> 1.55 (.44-2.7), x3.5; West .31 -> 1.19; rho 1.27 |
| lowdens, static Z, run 1 (cap) | 124,720 | .78 | 70,588 / 7,382 / 46,750 | 0 / -1 / -6 / -11 | G 1.26 (.24-3.9), constant |
| lowdens, static Z, run 2 (b = .1 u) | 148,331 | .44 | 93,234 / 7,287 / 47,810 | -53 / -61 / -70 / -77 | degenerate (fs 14, G .1) |
| lowdens, time-varying Z, first attempt | 178,022 | 0 | -- | -- | optimizer stall: INVALID |
| theta, time-varying Z (b = .1 u) | 113,446 | .95 | 62,127 / 5,926 / 45,394 | +2 / -1 / -3 / -7.5 | theta .2 (.06-.99) -> 2.48 (.68-6.3) |
| theta, time-varying Z, run 1 | 131,303 | .69 | | -10 / -17 / -53 / -44 | |
| theta, static Z | 133,096 | .66 | 74,538 / 7,674 / 50,884 | +4 / 0 / -6 / -10 | theta 3.84 (.23-21), constant |
Read:
1. (Established, current base, frozen standard readouts) Z's OWN time change, routed only through a low-density /
   establishment boost, reproduces the full fit to within about 1,700 (98% of the frozen-to-fitted gap; East within
   560), with the fit's arrival timing, a median boost change of x3.5 (East) / x3.8 (West), and no change in the four
   standard readouts. It does better than a free shared calendar trend (112,615). Pre-registered credibility bound
   (about 3x): borderline.
2. CAVEAT (established): the fitted fade rho = 1.27 K means the boost is still about 45% of (G - 1) at N = K. It is
   only partly low-density, partly a general fecundity term. Its implied potential-lambda change still has to be
   computed against the fit's own (Sa .13 -> .48 etc.).
3. The static controls are NOT robust (78% and 44% from two optimisations). The earlier "static 78%" headline is
   withdrawn. Z's time change matters on this route (time-varying beats the best static run by about 13,000).
4. theta (curvature) works only through extreme early compensatory crowding (theta .06-.99 in 1940): not credible.
Next (plan A1): (i) potential-lambda change implied by the readout; (ii) a tighter-rho variant (rho capped at about
.3 K) to force a genuinely low-density term; (iii) loadings vs the finch-used dims; (iv) static rerun from several
starts; (v) on the fitted habitat with a joint refit.

### R16c (2026-11-03): the low-density Z readout with the fade CAPPED (genuinely low-density term)
| fade cap | -log-lik | gap closed | offsets | G East median (p10-p90) 1940 -> 2020 | G West median |
|---|---|---|---|---|---|
| none (rho 1.27, R16) | 111,711 | .98 | +2 / -1 / -3 / -6 | .44 -> 1.55 (x3.5) | .31 -> 1.19 |
| rho <= .3 K (converged) | 113,366 | .95 | 0 / -1 / -3 / -7 | .25 (.09-.81) -> 2.27 (1.18-4.32) (x9) | .62 -> 1.24 (x2) |
| rho <= .1 K | 157,038 | .31 | -63 ... -89 | degenerate (intercept at bound 4, fs .30) | -- |
Read (established, current base): a genuinely low-density establishment readout of Z's change still closes 95% of
the gap with the fit's timing, freeing the four standard readouts from inflation. But confined to nearly empty
cells it needs a x9 rise in the EAST (beyond the about 3x credibility bar), against x2 in the West. The moderate x3.5
of R16 relied on the term reaching into settled densities. Same pattern as R14, now built on Z: routing the time
signal through low-density growth alone still needs an order-of-magnitude EASTERN change. The East / West asymmetry
points to something eastern-specific (eastern-lineage evolution; the novel climate; front supply via dispersal,
branch B) rather than a continent-wide habitat effect (likely; untested).

## R17 (pre-registered 2026-11-03, before running): habitat-informed dispersal (branch B1) + the capped establishment readout
Question: does better FRONT SUPPLY shrink the x9 eastern establishment change the capped Z readout needs (R16c)?
Mechanism: juveniles choose kernels by the habitat AHEAD. The signal is logit(journey survival Q) of the opposite-
direction path feature at the origin, the fitted path-habitat signal and the same "ahead" mapping as step 27. Per-
origin multipliers v_k = exp(kappa (A_k - mean_p A)) / sum_k p_k exp(...) over ALL 12 band x direction kernels
(choice='band', never tried), or within band only (choice='dir', the step 27 form), so every origin sends the same
total. Passed through the forward's existing directional_weights argument (no fork). kappa FIXED on a grid, not fitted
(the fitted step 27 kappa failed its null: sidestepped by imposing the rule). Signal from the frozen-habitat Q (choice
by spatial habitat, no time change). Arms (current base, standard readouts frozen, lowdens time-varying Z, b = .1 u,
rho <= .3 K, GPU float64, 150 iterations): band kappa .5 / 1 / 2; dir kappa 1. Reference: kappa 0 = R16c (113,366;
East G x9, West x2).
Predictions: if front supply is the eastern deficit, band choice lowers the needed eastern G change (towards the
West's x2) at equal or better score. If G's change and the score are unchanged, smarter dispersal is not the missing
eastern lever (as the ladder found for within-band choice).

## C1: factor decomposition on the CURRENT base (2026-11-03; growth_gap_factors.py, power-aware; step43_factor_p2_veit_s41)
Absolute pinned Allee (gamma on N), crowding power 2, yearling breeding. East occupied cells, medians:
| era | Fmax | Sa | Sj | mated fraction (all occupied; N/K) | crowding (filling cells) | 1/c filling |
|---|---|---|---|---|---|---|
| 1941-55 | 4.3 | .54 | .16 | .23 (.011) | .98 | 2.5 |
| 1956-65 | 4.9 | .63 | .19 | .94 (.195) | .94 | .89 |
| 1966-75 | 5.0 | .67 | .21 | .60 (.032) | .80 | .34 |
| 1976-90 | 4.0 | .48 | .15 | .89 (.256) | 1.00 | c = 0 |
| 1991-2010 | 4.1 | .49 | .15 | 1.00 (.619) | .88 | 1.5 |
Read (established, current base): unlike step3_native, (1) fecundity in occupied cells is flat (4-5): the inflation
sits in survival (lambda_decomp); (2) power-2 crowding is weak below K, so the "crowding cancels fecundity" mechanism
does not apply here; (3) the absolute Allee is the main front brake: 77% of births lost in sparse cells in
1941-55 and 40% in 1966-75; (4) the median filling cell in 1976-90 has c = 0, i.e. replacement F Sj / (1 - Sa) < 1:
a SINK filling by immigration. So on the current base the front is carried by immigration into sink cells and limited
by absolute mate-finding. Fecundity / front-speed reasoning from step3_native should not be transferred.

### R17 RESULT (2026-11-03; current base; GPU float64; lowdens time-varying Z, rho <= .3 K)
| kernel choice (fixed kappa) | -log-lik | gap closed | E / P / W | G East median 1940 -> 2020 | G West |
|---|---|---|---|---|---|
| none (R16c) | 113,366 | .951 | 62,001 / 6,114 / 45,251 | .25 -> 2.27 (x9.2) | x2.0 |
| band, kappa .5 | 113,948 | .943 | 62,208 / 6,306 / 45,434 | .19 -> 1.96 (x10.1) | x1.5 |
| band, kappa 1 | 112,477 | .964 | 61,718 / 5,940 / 44,818 | .35 -> 2.73 (x7.8) | x6.1 |
| band, kappa 2 | 112,744 | .960 | 61,802 / 5,951 / 44,991 | .46 -> 2.83 (x6.2) | x5.9 |
| dir (within band), kappa 1 | 112,350 | .966 | 61,706 / 5,902 / 44,742 | .28 -> 2.58 (x9.1) | x5.6 |
All offsets equal the fit's (0 to +2 / -1 to 0 / -3 / -6 to -7).
Read (established, current base): habitat-informed kernel choice (across bands, never tried before, or within band)
improves the score by at most about 1,000 and does NOT bring the eastern establishment requirement towards the West's
(pre-registered prediction FAILED). With stronger choice the East / West asymmetry disappears (x6.2 vs x5.9 at kappa
2), but by raising the West's requirement, not lowering the East's. Choice redistributes which region carries the
time change; it does not remove it. Agrees with the ladder's verdict, now judged on the requirement itself (which the
ladder never measured). Front supply by kernel choice is not the missing eastern lever.

## R18 (pre-registered 2026-11-03, before running): departure set by ORIGIN habitat (branch B2)
C1 showed the current base's front is carried by immigration into SINK cells and limited by absolute mate-finding. If
birds stayed in good habitat (philopatry) instead of departing in proportion to N/K regardless of habitat, the good
eastern cores would build up and supply denser, mate-finding-capable propagules. Rule (static, frozen-habitat rates):
departure x exp(-kappa_d max(0, log lambda0 - median)), <= 1, via the forward's existing departure_scale argument
(native-departure slot; no fork). Arms (current base, standard readouts frozen, lowdens time-varying Z, b = .1 u, rho
<= .3 K, GPU float64): kappa_d 1, 3; and kappa_d 3 + band choice kappa 1. Reference: R16c (no departure rule).
Prediction: if supply density at the front is the eastern deficit, the eastern G requirement (x9) falls; if not,
unchanged (like R17).

### R18 RESULT (2026-11-03; current base; lowdens time-varying Z, rho <= .3 K)
| departure rule | -log-lik | E / P / W | offsets | G East 1940 -> 2020 | G West |
|---|---|---|---|---|---|
| none (R16c) | 113,366 | 62,001 / 6,114 / 45,251 | 0 / -1 / -3 / -7 | x9.2 | x2.0 |
| philopatry kappa_d 1 | 118,126 | 64,736 / 5,970 / 47,420 | -3 / -1 / -2 / -5 | .49 -> 10.7 (x21.7) | x3.5 |
| philopatry kappa_d 3 | 130,035 | 71,100 / 6,644 / 52,291 | +1 / +1 / -2 / -6 | .43 -> 7.0 (x16.4) | x4.7 |
| philopatry 3 + band choice 1 | 158,642 | 98,918 / 7,749 / 51,975 | -50 / -57 / -71 / -66 | degenerate | -- |
Read (established, current base): origin-habitat philopatry makes the fit WORSE and RAISES the eastern establishment
requirement (x16-22 vs x9). Prediction FAILED, in the opposite direction: fewer dispersers mean a thinner front
supply, so establishment must do more. The model wants MORE eastern dispersal, not less, consistent with the
documented evolution of partial migration in the eastern population (Able & Belthoff 1998: 36% moved > 80 km by the
early 1960s vs about 80% sedentary in the source). Next test: an East-only departure INCREASE over time (needs a
safe fork; departure_scale is static and could push the departure probability above 1).

## R19 (pre-registered 2026-11-03, before running): EAST-only departure increase over time (branch B7; partial-migration evolution)
R18 showed the model wants MORE eastern dispersal. Able & Belthoff 1998: by the early 1960s 36% of eastern birds moved
> 80 km, against about 80% sedentary in the source. Rule (fixed, not fitted): East cells' departure p -> 1 - (1 - p)^s,
s(t) = 1 + a logistic((year - 1955) / 5), so s rises from about 1 (1940) to 1 + a by about 1970 (dep_forward.py fork;
capped: departure never exceeds 1). West and Plains unchanged. Arms (current base, standard readouts frozen, lowdens
time-varying Z, b = .1 u, rho <= .3 K, GPU float64): a = 1, 3. The start value (G = 1) also gives the ramp's effect
alone. Reference: R16c (x9.2 East, x2.0 West).
Prediction: if eastern dispersal evolution is the eastern-specific lever, the eastern G requirement falls towards the
West's (about x2) at equal or better score.

### R19 RESULT (2026-11-03; current base; identity 0.0; lowdens time-varying Z, rho <= .3 K)
| East departure ramp (s by 1970) | start: ramp only, G = 1 | -log-lik | E / P / W | offsets | G East 1940 -> 2020 | G West |
|---|---|---|---|---|---|---|
| none (R16c) | 178,022 | 113,366 | 62,001 / 6,114 / 45,251 | 0 / -1 / -3 / -7 | x9.2 | x2.0 |
| a = 1 (s 1.95) | 177,055 | 112,227 (cap) | 61,671 / 5,916 / 44,641 | +3 / -1 / -3 / -7 | .21 -> 1.75 (x8.3) | x7.2 |
| a = 3 (s 3.86) | 176,423 | 112,283 (cap) | 61,763 / 5,913 / 44,608 | +2 / -1 / -3 / -7 | .21 -> 1.36 (x6.4) | x7.8 |
Read (established, current base): the eastern departure ramp alone gains only 1,000-1,600. With the readout it trims
the eastern requirement modestly (x9.2 -> x6.4) but RAISES the West's (x2 -> x7-8), as kernel choice did (R17).
Across R16c-R19, the robust quantity is a continent-wide rise of about x6-9 in low-density establishment over
1940-2020, whatever the dispersal rules. Its East / West split is NOT identified: it moves with every dispersal
change because the readout is shared across regions through Z. The R16c "eastern-specific x9 vs x2" clue is
WITHDRAWN as not robust. This is the shared, calendar-like change of R7, now routed through establishment.
Prediction (eastern requirement falls towards the West's at equal or better score): FAILED (it converges by raising
the West).

### R16 proxy check (2026-11-03; readout_overlap.py; current base, native field54 Z)
Readout rho <= .3 (R16c) vs the FITTED model's own rates (unfrozen habitat), Pearson over land cell-years:
| fitted field | East: all | East: time part (cell mean removed) | East: 1940 -> 2010 change per cell | West: time part |
|---|---|---|---|---|
| log lambda0 | -.04 | .85 | .44 | .68 |
| log Fmax | -.10 | .74 | .04 | .64 |
| logit Sa | .04 | .92 | .75 | .74 |
| logit Sj | -.04 | .85 | .41 | .68 |
| log K | .29 | .91 | .84 | .68 |
Read (established): the establishment readout's TIME course is the same Z direction the fit already uses for its
habitat change (time-part correlations .85-.92 East with lambda0, Sa and K); spatially it is unrelated (about 0). R16
therefore shows that the same Z time signal can be carried by establishment (needing x6-9) instead of by survival (Sa
.13 -> .48); rerouting moves the inflation, it does not remove it or add information. The open question is the
origin of Z's strong eastern time signal: the observation-fitted regional field R(s, t) vs covariates (branch E,
needs the TACC files). (Dimension indices from GRID54_Z_EAST_ATTRIBUTION were for the averaged Z and do not transfer
to the native field54 Z this base uses.)

## F1: the current base's 1950s stall near the release (2026-11-03; release_area.py; 160 km)
| year | total | occupied cells | lambda0 | mated (core) | N/K | crowding |
|---|---|---|---|---|---|---|
| 1951 (init) | .038 | 2 | 1.13 | .89 | .41 | .96 |
| 1953 | .025 | 3 | 1.20 | .82 | .21 | .98 |
| 1955 | .022 | 2 | 1.27 | .84 | .18 | .98 |
| 1957 | .023 | 1 | 1.34 | .87 | .18 | .97 |
| 1961 | .037 | 1 | 1.46 | .92 | .24 | .94 |
| 1965 | .067 | 4 | 1.61 | .90 | .29 | .87 |
| 1969 | .17 | 8 | 1.78 | .93 | .25 | .86 |
| 1973 | .89 | 15 | 2.05 | .98 | .35 | .67 |
Read (likely; mechanism inferred from the components, not isolated): the core's potential lambda > 1, its mated
fraction .8-.9 and crowding negligible, yet the area declines 1951-55, stalls to about 1963, and the occupied area
shrinks to one cell. The loss is DISPERSAL DRAIN: Veit departure sends about N/K (20-40%) out each year into empty
neighbours, where the absolute Allee leaves mated fraction about .001 (Allee-dead). The lag ends only when the
inflated habitat lifts core lambda above the drain (1.13 -> 2.05 by 1973). Real release-area growth was about .24 / yr
over 1951-66, so the model's early drain is too strong. Ecology: the founders came from a mostly SEDENTARY source
(about 80%; partial migration evolved later, Able & Belthoff 1998), so early eastern departure should be LOW and rise
later. R18 (static philopatry) and R19 (departure only rising) did not test that time course.

## R20 (pre-registered 2026-11-03, before running): sedentary founders -> evolving dispersal (East departure LOW early, rising later)
Rule (fixed, dep_forward.py capped form p -> 1 - (1 - p)^s): East s(t) = s0 + (s1 - s0) logistic((year - 1960) / 5).
Arms: (s0, s1) = (.3, 1), (.3, 2), (.15, 2); with the capped lowdens Z readout (time-varying Z, rho <= .3 K) and
without it (G = 1: the rule's effect on the frozen replay alone). Readouts: score; G East / West change; offsets;
realised r near the release 1951-62 vs CBC about .24.
Prediction: low early drain lets the core grow (realised r near the release rises towards .24) and rising later
dispersal supplies the front, so the establishment requirement falls below x6-9 at equal or better score. If the
requirement does not fall, early drain is not what the inflation substitutes for.

### R20 RESULT (2026-11-03; current base; lowdens time-varying Z, rho <= .3 K; dep_forward capped departure)
| East departure s (early -> late, midpoint 1960) | -log-lik | E / P / W | G East x (1940 -> 2020) | G West x | release-area r 1951-62 / 62-71 |
|---|---|---|---|---|---|
| 1 -> 1 (control; fork identity) | 112,332 | 61,743 / 5,940 / 44,648 | 9.0 | 7.2 | .126 / .196 |
| .3 -> 2 | 112,251 | 61,709 / 5,914 / 44,629 | 8.5 | 8.1 | .147 / .172 |
| .15 -> 2 | 112,302 | 61,751 / 5,913 / 44,638 | 9.1 | 7.9 | .149 / .169 |
| .3 -> 1 | INVALID (non-finite first gradient: (1 - p)^s at p = 1 with s < 1; fixed by clamping 1 - p >= 1e-9; rerun) | | | | |
CBC release-area anchor: about .24 / yr 1951-66.
Read (established, current base): low early departure (sedentary founders) then rising dispersal raises early core
growth only slightly (.126 -> .149, still below .24) and does NOT reduce the establishment requirement (x8.5-9.1 vs
the control's x9.0). Prediction FAILED. OPTIMIZER NOISE (established): the control is the same setup as R16c (start
identical, 178,022) yet reaches 112,332 vs 113,366, with the West's G change x7.2 vs x2.0. So score differences of
about 1,000 between these capped, unconverged runs (R17-R20) are noise, and the East / West split of G is not
identified. Only large changes in the requirement would be informative, and none of R17-R20 produced one.
R20 rerun of .3 -> 1 after the clamp fix (identity 0.0): 112,483; G East x10.1, West x6.1; release-area r 1951-62 .160.
Same result: no reduction of the requirement.

## A3 / C2c (pre-registered 2026-11-03, before running): JOINT refit of the habitat field + a separate ESTABLISHMENT readout
User correction: the single shared field is known project history (ladder step 2), not a finding. Neither the single
field nor the old separate correlated fields ever had an axis specifically for low-density establishment. That
decomposition is the point (the exchangeable refit I built first only recreates the old separate fields; parked).
Design (refit_fields.py, current base): refit the habitat-readout sites (manifold_factor, w_scale_raw, alpha_a / j /
f / k; single field as fitted) on the FULL log posterior (priors + all observations), all other sites held at the
refined point, with exact gradients (GPU float64, 150 iterations).
- CONTROL: the habitat sites alone (the gain from re-optimising those sites).
- ESTAB: the same sites PLUS a separate establishment readout of Z (G = exp(g0 + z . b), fading with N/K at rho <= .3
  K; ridge sd .5 on b; lowdens_forward fork), jointly refitted.
Start check: the single-field start reproduces the fit exactly (BBS -log-lik 110,044.70).
Readouts: log posterior; BBS log-lik; offsets; and the KEY one, the four standard rates' inflation in the East (Sa,
Sj, Fmax 1940 -> 2010 and the potential-lambda change with its Shapley split).
Prediction: if a separate establishment axis is what the field's inflation was standing in for, ESTAB shrinks the
eastern lambda change (now .21 -> 1.08, Sa .13 -> .48) at an equal or better posterior. If the field keeps its
inflation and G also rises, the axis adds without substituting.

### A3 / C2c RESULT (2026-11-03; refit_fields.py; current base; GPU float64; start = the fit exactly)
| arm | -log posterior | BBS -log-lik | East lambda0 1940 -> 2010 | East Sa 1940 -> 2010 | establishment G (East median; West) |
|---|---|---|---|---|---|
| fit (start) | 135,590.8 | 110,044.7 | .207 -> 1.081 | .135 -> .485 | -- |
| CONTROL: habitat sites refitted alone (converged, 5 evals) | 135,590.8 | 110,044.7 | .207 -> 1.081 | .135 -> .485 | -- |
| ESTAB: habitat sites + separate establishment readout, jointly (200-iteration cap) | 135,503.1 | 109,973.1 | .209 -> 1.078 | .138 -> .490 | East .98-1.12 (no trend); West .82 -> 1.15; rho .18 |
Read (established, LOCAL to the fitted optimum): the habitat field is already at its optimum (the control gains
nothing). Adding a jointly fitted establishment axis gains 88 in log posterior / 71 in BBS log-lik, offsets
unchanged, but the axis stays near 1 and does NOT take over the field's inflation (eastern lambda .21 -> 1.08,
Sa .14 -> .49 unchanged). Pre-registered branch "the axis adds without substituting": MET locally.
Limit: this is a local refit from the inflated-field basin. With the field frozen, the establishment basin reaches
BBS about 113,000 (R16c) vs 110,045 here, so a local optimiser will not cross. Next: a homotopy that starts in the
establishment basin (field on frozen Z, establishment on time-varying Z) and moves the field's Z from frozen to real
in steps (alpha .25 / .5 / .75 / 1), refitting jointly at each step: does the field's inflation return as alpha -> 1,
and where does the posterior end vs 135,503?

## A3 homotopy (pre-registered 2026-11-03, before running)
Start in the ESTABLISHMENT basin: the habitat field reads FROZEN Z (alpha 0) with its fitted sites; the establishment
readout starts at the R16c solution (time-varying Z, rho <= .3 K). Jointly refit (habitat sites + establishment
readout, full log posterior) at alpha = 0, then move the field's Z to the real one in steps alpha = .25, .5, .75, 1,
warm-starting each step from the last (refit_fields.py --field_alpha, --init_npz, --init_r16; 150 iterations each).
Readouts per step: log posterior, BBS log-lik, offsets, the field's eastern lambda0 1940 -> 2010 and Sa, and G's
eastern change.
Predictions: if an establishment-carried solution is competitive, the alpha = 1 end point has a log posterior within
a few hundred of the local optimum (135,503), with the field's eastern inflation well below .21 -> 1.08 and G carrying
the time change. If, as alpha -> 1, the field's inflation returns and G flattens, the data prefer the field route; the
establishment axis is then a rerouting option the data do not select.

### CORRECTION NOTE (2026-11-03, plan audit): conversion-dependent claims above
These rest on the unvalidated model-units -> birds/km2 conversion and on CBC-radius comparisons without a CBC
observation model. Treat them as WITHDRAWN as evidence:
- R0 read (1): "the all-year mean matches the early phase (223 km in 1965 vs 267 observed)".
- R1 / R1b / R3: the CBC-radius comparisons ("74 km in 1965 vs 267 obs", "R90 1965 / 75 / 85 / 95 ... obs 267 / 233
  / 550 / 1422").
- R5: the "side finding (likely) ... fitted model behind early" and R5b's 1970s trough (attributed to the CBC term).
The arrival-offset (relative threshold) and BBS observation-unit results are unaffected. Also: "G15" in this record
refers to a PROPOSED CONSIDERATIONS item not yet written (plan G4).

## B9 (pre-registered 2026-11-03, before running): departure FLOOR from near-empty cells
Veit departure p = min(N/K, 1) sends nobody from near-empty cells; R19 / R20 kept p(0) = 0. Rule (fixed): p -> f +
(1 - f) p in all cells (dep_forward floor), f = .1, .3. Current base, standard readouts frozen, lowdens time-varying
Z, b = .1 u, rho <= .3 K; CPU float64 (the GPU is busy with the A3 homotopy). Reference: R20 control (112,332; G East
x9.0, West x7.2; release r .126). Readouts: score, G change, offsets, release-area r 1951-62.
Prediction (uncertain direction): front cells exporting at low density could speed early spread (more scattered
propagules), or deepen the drain under the absolute Allee (F1). If the requirement falls clearly (below about x5),
low-density departure is a lever; if unchanged or worse, not.

### A3 homotopy RESULT (2026-11-03; refit_fields.py --field_alpha; [cur]; GPU float64; 150 iterations per step)
| field Z alpha | -log posterior | BBS -log-lik | field East lambda0 1940 -> 2010 | East Sa | G East / West change 1940 -> 2020 |
|---|---|---|---|---|---|
| 0 (frozen; start = R16c establishment solution) | 137,250 | 111,482 | .78 -> .78 | .38 -> .38 | x8.7 / x4.4 |
| .25 | 136,474 | 110,830 | .57 -> .86 | .29 -> .41 | x5.7 / x3.6 |
| .5 | 135,864 | 110,282 | .39 -> .99 | .21 -> .44 | x3.0 / x3.0 |
| .75 | 135,570 | 110,030 | .28 -> 1.05 | .17 -> .47 | x1.8 / x2.6 |
| 1 (real Z) | 135,491 | 109,957 | .23 -> 1.08 | .15 -> .51 | x1.2 / x2.2 |
(Local joint refit from the fit, C2c: 135,503.) Offsets equal the fit's at every step.
Read (established, [cur]): starting where an establishment axis carries Z's time change, and letting the shared field
see Z's real change, the posterior improves monotonically back to the inflated-field basin (12 better than the local
C2c optimum). The field's inflation returns in full (lambda .23 -> 1.08, Sa .15 -> .51) while the establishment axis
flattens (x1.2 East). Pre-registered outcome: the second branch ("the field's inflation returns and G flattens; the
data prefer the field route; the establishment axis is a rerouting option the data do not select"). The establishment-
carried solution is about 1,750 worse in log posterior when the field cannot use Z's change. A3 closed for this
model structure: an establishment axis built from the same Z cannot pull the time signal away from the shared field.
What would change that is not this axis but what Z's time signal IS (E1-E4).

## C2 (pre-registered 2026-11-03, before running): low-density SURVIVAL anchored, joint refit ([cur])
On [cur] about 70% of the lambda inflation is survival (East Sa .13 -> .48). Rule (survanchor_forward.py): adult
survival in a cell = hi' - (hi' - Sa_field) min(N/K, 1)^q, hi' = max(hi, Sa_field); juvenile survival keeps its ratio;
at N = K survival equals the field's, so K stays the equilibrium. The field can no longer brake a FRONT through
survival; survival only sets settled abundance. hi (one habitat-independent number, start .6) is refitted jointly with
the habitat sites (manifold_factor, w_scale_raw, alphas) on the full log posterior (refit_fields.py --survanchor, q = 1,
GPU float64, 200 iterations). Note: the start already includes the anchor (not the fit).
Readouts: log posterior vs the fit (135,591 / local best 135,491); the field's eastern Sa and lambda0 1940 -> 2010
(do they still inflate?); which rate takes over the time change (Fmax / Sj shares); offsets; fitted hi.
Predictions: if the inflation is survival braking the front, the anchored fit loses little posterior, the field's Sa
time change shrinks, and some of the change moves to fecundity or vanishes. If the posterior drops a lot and offsets go
early, the data need low early low-density survival (a real brake the model has no other way to express).

### C2 RESULT (2026-11-03): the fit OPTED OUT of the anchor (design flaw); rerun with a FIXED anchor
- Start with low-density survival anchored at .6: -log posterior 140,641 (the fit 135,591): the anchor costs about
  5,050.
- Joint refit (200 iterations, converged): hi driven down to .22, where hi' = max(hi, Sa_field) almost never binds.
  The fit returns to the original solution (135,593; East lambda .21 -> 1.08, Sa .13 -> .48, offsets unchanged).
Read (established, [cur]): the data strongly prefer LOW survival in sparse cells (about 5,000 in log posterior) and
escape any optional anchor. NOT the pre-registered test (the anchor must be imposed). C2b: hi FIXED at .45 (about the
modern eastern Sa) and .6, habitat sites refitted: the cost of forbidding low low-density survival, and whether the
field's Sa time change shrinks / moves to fecundity.

### B9 RESULT (2026-11-03; [cur]; CPU float64; lowdens time-varying Z, rho <= .3 K)
| departure floor | -log-lik | E / P / W | offsets | G East x | G West x | release-area r 1951-62 |
|---|---|---|---|---|---|---|
| none (R20 control) | 112,332 | 61,743 / 5,940 / 44,648 | +2 / -1 / -3 / -7 | 9.0 | 7.2 | .126 |
| f = .1 | 112,089 | 61,545 / 5,911 / 44,633 | +3 / 0 / -3 / -7 | 8.6 | 6.7 | .071 |
| f = .3 | 114,382 | 62,235 / 6,397 / 45,750 | +4 / 0 / -3 / -9 | 3.4 | 1.4 | .033 |
Read (established, [cur]): a strong departure floor (birds also leave near-empty cells) is the FIRST dispersal rule
that clearly shrinks the establishment time requirement (East x9 -> x3.4, West x7 -> x1.4; pre-registered bar "below
about x5": MET). But it costs about 2,000 in log-likelihood (twice the optimizer noise) and deepens the early stall
near the release (r .033). Follow-up B9b: the floor as a FITTED parameter (does the data want any floor?).

## B9b (pre-registered 2026-11-03, before running): fitted departure floor
Same as B9 with logit(floor) fitted jointly with the readout (start f = .1, bounds logit -7 .. 2). Prediction: if the
fitted floor stays near 0, the data reject low-density departure as a substitute; if it settles at .1-.3 with a better
score than the control, it is a real lever.

### C2b RESULT (2026-11-03; [cur]; survival anchor FIXED; habitat sites refitted on the full posterior)
| low-density survival | -log posterior (start -> end) | BBS | offsets | field East lambda0 1940 -> 2010 | field East Sa |
|---|---|---|---|---|---|
| not anchored (the fit) | 135,591 | 110,045 | +3 / -1 / -3 / -6 | .21 -> 1.08 | .13 -> .48 |
| anchored at .45 | 135,822 -> 135,642 (+51) | 110,087 | +3 / -1 / -3 / -6 | .20 -> 1.06 | .13 -> .50 |
| anchored at .6 | 140,641 -> 136,093 (+502) | 110,537 | +1 / -1 / -3 / -6 | .23 -> .96 | .17 -> .52 |
Read (established, [cur]): once the habitat field is refitted, anchoring survival in sparse cells costs little (51 at
.45, about 500 at .6; vs 5,050 before refitting) and arrival timing is unchanged. Low survival AT THE FRONT is not
what times the invasion. The field's Sa still rises over time (.13 -> .50), now acting only in filling / settled
cells, so the survival inflation serves SETTLED ABUNDANCE / fill-in ("late too slow"), not front braking. Front timing
then runs through fecundity (x2.2) and the Allee / front mechanics. (Potential lambda here is computed from the field
rates; actual sparse-cell survival is the anchor.)

## B3 (pre-registered 2026-11-03, before running): settlement by DESTINATION habitat (promised three times; finally run)
The forward's own ideal-free settlement (age_forward 3b): within 3 x 3 blocks of cells, ARRIVING dispersers (residents
keep their cells) are reallocated by weight: 'capacity' (shares proportional to K) or 'r0' (proportional to the
breeding-quality R0 = Fmax Sj / (1 - Sa)). Imposed (not fitted; the old fitted versions contorted or failed recovery).
Current base, standard readouts frozen, lowdens time-varying Z, rho <= .3 K, GPU float64, 150 iterations.
Reference: R20 control (112,332; G East x9.0).
Prediction: settling arrivals into better habitat raises front establishment (fewer arrivals wasted in Allee-dead /
sink cells), so the eastern establishment requirement falls. If unchanged, destination choice is not the lever either.

### B3 RESULT (2026-11-03; [cur]; imposed ideal-free settlement of arrivals in 3 x 3 blocks; lowdens readout)
| settlement weight | start | -log-lik | E / P / W | offsets | G East x | G West x |
|---|---|---|---|---|---|---|
| none (R20 control) | 178,022 | 112,332 | 61,743 / 5,940 / 44,648 | +2 / -1 / -3 / -7 | 9.0 | 7.2 |
| capacity | 336,493 | 254,434 | 63,581 / 6,075 / 184,779 | +2 / 0 / -3 / -7 | 16.2 | 24.8 |
| r0 (breeding quality) | 426,383 | 334,335 | 64,116 / 6,081 / 264,139 | +3 / -1 / -3 / -7 | 14.7 | 25.1 |
Read (established, [cur]): full-strength destination settlement WRECKS the native West's fit (West -log-lik x4-6) by
concentrating arrivals in the best cells. In the East it is somewhat worse and RAISES the establishment requirement
(x15-16). Timing unaffected. Agrees with the old settlement records ("contort"), now judged on the requirement too.
Open: a weak (settle_tau) or East-only version.

### F9 RESULT (2026-11-03; [cur]; reruns of earlier invalid arms; GPU float64)
- BUI spline -> LOW-DENSITY boost (R14c design, valid now): 162,940 (gap closed .22), offsets -61 / -69 / -78 / -82;
  G East 3.9 -> 43, rho .06, fscale .35. A low-density boost driven by BUI cannot time the front (it spreads 60-80 yr
  early). Valid negative; replaces the R14c stall.
- BUI split -> Fmax (R8d1 design on [cur], float64): 117,021 (gap closed .90), offsets -2 / -3 / -4 / -5.5, bt 5.41
  (about x16+ fecundity over 1940-2020 for the East's BUI change). Replicates R8 / native on [cur] (F5 partly done).

### F7 RESULT (2026-11-03): frozen-habitat fronts of the step 26 short-kernel fits (promised in round 1; canaries identical)
| fit | fitted offsets | frozen modern | frozen modern, release only | lambda East early / modern |
|---|---|---|---|---|
| step26_mdd200 | +1 / -2 / -2 / -6 | -33 / -42 / -59 / -65 | -25 / -27 / -28 / -32 | .29 / 1.85 |
| step26_mdd110 | -1 / -3 / -5 / -10 | -25 / -31 / -49 / -58 | -25 / -27 / -25 / -28 | .37 / 1.94 |
Read (established): shorter kernels leave the frozen-habitat release front as early as step 3's (about 25-32 yr); kernel
length is not the lag lever (direct measurement, replacing the earlier inference).

### B9b RESULT (2026-11-03; [cur]; departure floor FITTED with the lowdens readout; CPU float64, 200-iteration cap)
Fitted floor .433; -log-lik 111,800 (R20 control 112,332; better by about 530, near the noise edge); E / P / W 61,419 /
5,859 / 44,522; offsets +3 / 0 / -3 / -7; G East x4.3 (control x9.0), West x5.2; release-area r 1951-62 .024.
Read (likely, [cur]): the data SELECT a substantial departure floor (birds leave near-empty cells), and with it the
establishment time requirement roughly halves in the East. First lever the data choose that reduces the requirement.
Cost: a deeper early stall near the release. Decisive next test (B9c): joint refit of the habitat field with a fitted
floor. Does the FIELD's own inflation shrink?

## B9c (pre-registered 2026-11-03, before running): JOINT refit of the habitat field + a fitted departure floor ([cur])
refit_fields.py --floor: the habitat sites (manifold_factor, w_scale_raw, alphas) and logit(floor) are refitted on the
full log posterior (dep_forward, floor in all cells), other sites at the fit; starts floor .1 and .433 (B9b's value);
GPU float64, 200 iterations. Readouts: log posterior vs the fit (135,591 / best local 135,491); fitted floor; the
field's eastern lambda0 and Sa 1940 -> 2010; offsets.
Predictions: if low-density departure is what the inflation substitutes for, the joint fit keeps a substantial floor,
improves the posterior, and the field's eastern inflation shrinks. If the floor goes to about 0 or the inflation is
unchanged, B9b's gain was specific to frozen standard readouts.

### F2 RESULT (2026-11-03; step39_veit_fixed_s41: absolute Allee, standard yearling reproduction, crowding power 1,
### no exemption, release area populated from 1951; CPU float64; canary identical)
| frozen modern, release only | offsets (reached cells) | eastern cells NEVER reached | East potential lambda |
|---|---|---|---|
| Fmax x1 | -16 / -20 / -24 / -30 | .07 | 1.07 |
| x.6 | -13 / -16 / -17 / -20 | .15 | .75 |
| x.4 | -8 / -10 / -7 / -12 | .37 | .59 |
| x.3 | -2 / -3 / +1 / -8.5 | .70 | .51 |
| x.2 | release fails | 1.00 | .43 |
Read (established, step 39): low fecundity DOES slow the front on the yearling base, but only by making most of the
East unreachable. Frozen modern habitat leaves the East just above replacement (1.07), so any uniform cut turns it into
a sink, and the front "slows" by failing to reach much of the region. The offsets are computed on reached cells
only, which is why R1c / R2 read "stays early". This is the D1 knife edge in the real landscape: no fecundity level
both delays the front and fills the East. Fitted step 39: lambda East .26 -> 1.16 (Shapley Sj .45, Fmax .37, Sa .17);
the same 1951-55 release-area decline and 1-cell stall as [cur] (initialised 1951). Readouts should report "never
reached" alongside offsets (the harness has it; summaries omitted it).

### CORRECTION (2026-11-03, from F2): several offset-only "timing matches" hid NEVER-REACHED cells
Share of eastern front cells (first BBS positive) never reached, from the saved rows (fitted step 3: .04):
- R2 Fmax x.4 frozen modern: offsets -3 / -5 / -4 / -9 BUT never .25; x.37: never .36; x.35: never .47.
- R3 F x.4 + K rise (all variants): never .24-.34; F x.37 + K rise: never .48.
- R1b absolute n50 .03 + F x.5: never .48; relative n50 x5: never .25.
So R2's "a time-homogeneous brake matches BBS-era arrival timing about as well as the fitted inflation" and R3's "low
F + a shared K rise gives arrival within about 4 yr" are OVERSTATED. The timing match comes partly from failing to
reach 25-48% of the East (the knife edge; F2). Arms judged on the BBS log-likelihood (R5c onward) penalise unreached
cells through the zero / positive rows, but their summaries did not store the never-reached share; it is now recorded
by harness-based scripts. LESSON: always report "never reached" beside offsets.

### B9c RESULT (2026-11-03; [cur]; habitat field + fitted departure floor, joint, full posterior)
| start floor | fitted floor | -log posterior (start -> end) | BBS | field East lambda0 1940 -> 2010 | Sa |
|---|---|---|---|---|---|
| .1 | .074 | 136,722 -> 135,587 (converged) | 110,044 | .22 -> 1.07 | .14 -> .49 |
| .433 (B9b) | .066 | 265,990 -> 135,587 | 110,047 | .22 -> 1.07 | .14 -> .49 |
(The fit: 135,591; .21 -> 1.08.)
Read (established, [cur]): with the habitat field free, the floor collapses to about .07 from either start, with no
posterior gain and an unchanged inflation. A large floor against the fitted field is enormously costly. B9b's
selected floor (.43) was a SUBSTITUTE for the frozen field, not an improvement on the full model. Pre-registered
second branch ("specific to frozen standard readouts").

## SYNTHESIS (2026-11-03 evening): mechanisms vs the shared field's time signal ([cur])
Three mechanisms looked useful with the standard readouts FROZEN and were each ABANDONED once the shared habitat field
could use Z's real time change: the establishment axis (A3: homotopy back to the inflated basin; axis flattens), the
low-density survival anchor (C2: fittable anchor escaped; fixed anchor costs little and the field still inflates
survival for settled abundance), and the departure floor (B9c: floor -> .07). Kernel choice, philopatry, departure
ramps and destination settlement did not help even with the field frozen (R17-R20, B3). Reading (established for these
mechanisms and this base): the fit's inflation is the shared field passing Z's eastern time signal into Sa / Sj /
Fmax. A mechanism can only compete if the field is denied that signal, and none tested recovers the full fit when it
is (best: an establishment axis at about 1,750 worse; A3 alpha 0). The decisive questions are about the SIGNAL
itself: (1) what Z's eastern time change is (observation-fitted regional field R vs covariates; E1-E4, TACC files);
(2) whether it should be allowed to reach every vital rate (a modelling decision for the user: e.g. let Z's change
reach capacity / settled abundance but not low-density growth or survival at the front).

## Mixed real / frozen rate replays (2026-11-03; mix_replay.py; [cur]; fitted parameters, no refit; Z_disp frozen)
Each arm takes the listed rates from the REAL-Z capture and the rest from the FROZEN-Z capture (c recomputed).
| real-Z rates | BBS -log-lik | share of frozen -> all gap | offsets | never |
|---|---|---|---|---|
| none | 178,022 | 0 | -48 / -57 / -70 / -74 | .08 |
| K only | 133,831 | .67 | -27 / -34 / -44 / -32 | .09 |
| Fmax only | 168,309 | .15 | -32 / -40 / -52 / -51 | .09 |
| Sa only | 186,224 | worse than none | -24 / -32 / -46 / -48 | .08 |
| Sj only | 210,911 | much worse | -43 / -51 / -62 / -66 | .02 |
| all but K | 136,222 | .63 | -5 / -8 / -11 / -15 | .09 |
| all but Fmax | 128,171 | .75 | -8 / -11 / -13 / -17 | .10 |
| all but Sa | 125,361 | .80 | -9 / -13 / -15 / -16 | .10 |
| all but Sj | 119,724 | .88 | -6 / -9 / -10 / -13 | .10 |
| all four | 111,854 | 1 | +2 / -2 / -5 / -9 | .09 |
(The fit with Z_disp real as well: 110,045.)
Read (established, [cur], attribution at fixed parameters; rates interact): the fitted habitat change does two jobs.
(1) CAPACITY's time change carries most of the likelihood (67% alone; largest loss when removed): settled abundance /
fill-in. But alone the front stays 27-44 yr early. (2) FRONT TIMING comes from the growth rates (Sa, Sj, Fmax)
changing TOGETHER: without K's change the timing is already near the fit's (-5 to -15). Any single growth rate alone
fails or worsens the fit (interactions). So lambda inflation (the growth-rate part) is what times the front, and
K's change fits abundance; the single shared field forces them to move together. This is why front mechanisms were
abandoned once the field was free (A3, C2, B9c): the field delivers front timing through the growth rates at no
extra cost. Production-relevant question (for the user, not decided here): should Z's change reach K (abundance) but
not low-density growth, with front timing from an explicit mechanism? R5c-R20 show none of the tested mechanisms
supplies that timing at credible sizes, so the gap would have to be met by something not yet tested (or accepted).

## R21 (pre-registered 2026-11-03, before running): Z's change to CAPACITY only + an establishment readout for timing
The modelling option the mix_replay decomposition suggests: let Z's time change reach K (abundance; 67% of the
likelihood alone, fronts 27-44 yr early) but not the growth rates, and ask an establishment readout of Z to supply
the front timing. zreadout --k_real: K from the real-Z capture; Sa / Sj / Fmax from frozen Z; lowdens readout (rho <=
.3 K, b = .1 u). Arms: readout from time-varying Z; readout from static Z (space only; is a SPATIAL establishment
pattern plus K's change enough to time the front?). GPU float64, 150 iterations. References: K only, no readout
133,831 (mix_replay); the fit 110,045; all four rates real 111,854.
Prediction: if establishment can carry the timing once abundance is handled by K, the time-varying arm approaches
about 112k with fit-like offsets and a G change well below x6-9 (since K now carries the fill-in). If G still needs
about x6+ or the score stays far from the fit, front timing genuinely needs a large low-density growth change.

### R21 RESULT (2026-11-03; [cur]; K from real Z, Sa / Sj / Fmax from frozen Z, + establishment readout)
| arm | -log-lik | E / P / W | offsets | never | G East (1940 -> 2020) | G West |
|---|---|---|---|---|---|---|
| K real only, no readout (mix_replay) | 133,831 | 82,604 / 6,821 / 44,407 | -27 / -34 / -44 / -32 | .09 | -- | -- |
| K real + readout, time-varying Z | 111,682 | 61,571 / 5,801 / 44,310 | -1 / -2 / -3 / -7 | .074 | .46 -> 2.17 (x4.7) | .52 -> 1.14 (x2.2) |
| K real + readout, static Z | 126,022 | 74,373 / 6,218 / 45,430 | -42 / -48 / -56 / -24 | .13 | 16.2 const | 17.0 const |
(All four standard rates real, Z_disp frozen: 111,854; the fit: 110,045.)
Read (established, [cur], replay with fitted parameters): with Z's time change passed to CAPACITY only (abundance),
and NO time change in Sa / Sj / Fmax, an establishment readout of Z supplies the front timing and matches the
all-rates replay (111,682 vs 111,854), with the fit's offsets and never-reached share. The time change needed in
low-density growth falls to x4.7 East / x2.2 West (x9 without K's change), in place of the fit's joint inflation of
three rates (Sa .13 -> .48, Sj x4, Fmax x2). A purely spatial establishment pattern cannot time the front. Pre-
registered prediction (approaches about 112k with fit-like offsets and G well below x6-9): MET. Caveats: x4.7 is still
above the about 3x credibility bar; the readout uses the SAME Z time signal (the proxy question, branch E); this is a
replay at fitted parameters. The decisive next test is a JOINT refit of this decoupled structure (K reads real Z;
growth readouts read frozen Z; + the establishment axis), on the full posterior.

## A9 (pre-registered 2026-11-03, before running): JOINT refit of the DECOUPLED structure
refit_fields.py --estab --decouple: each evaluation runs the model twice under the same habitat parameters (real Z,
only to record K; frozen Z for Sa / Sj / Fmax and the likelihood, with the forward given the real-Z K and the
establishment readout of real Z, rho <= .3 K). Habitat sites (manifold_factor, w_scale_raw, alphas) + the readout are
refitted on the full posterior; establishment starts at R21's solution. GPU float64, 300 iterations. Smoke test: one
iteration gives BBS 111,720 (R21 111,682); growth rates time-constant by construction.
Readouts: log posterior vs the fit (135,591; best local 135,491); BBS; offsets; never reached; G change East / West;
K's eastern change.
Predictions: if the decoupled structure is a real alternative, the joint refit improves on its start (138,109) towards
the fit's posterior, keeps fit-like timing, and G's change stays at or below about x4-5. If the posterior stalls
far from the fit (more than about 1,500 worse), coupling growth to Z's change is genuinely preferred by the data.

### A9 RESULT (2026-11-03; [cur]; decoupled structure, joint refit; 300-iteration cap, not converged)
-log posterior 138,109 -> 136,469 (fit 135,591; best local 135,491: +878 / +978). BBS 110,822 (fit 110,045); E / P / W
61,065 / 5,721 / 44,035. Offsets +2 / -1 / -3 / -7; never reached .051 (fit .04). Growth rates time-CONSTANT by
construction: East Sa .39, Sj .14, Fmax 2.74 (lambda without G .78). Establishment G East .50 -> 2.53 (x5.1), West .83 ->
2.76 (x3.3); rho .30.
Implied eastern LOW-DENSITY potential lambda (Sa + Sj Fmax G, medians, approximate): about .59 (1940) -> 1.37 (2020),
a change of about +.78 vs the fit's +.87 (lambda .21 -> 1.08). (R21's replay estimate of +.61 was optimistic; the
refit raised G.)
Read (established, [cur]): unlike A3 / C2 / B9c, the decoupled structure is NOT abandoned by the joint fit. It stays
within about 900 of the fit's posterior (pre-registered "real alternative" threshold 1,500: MET), with fit-like
timing, because the growth rates cannot take Z's change by construction. Survival becomes plausible and constant (Sa
.39 instead of .13 -> .48). BUT the lambda inflation mostly REMAINS: low-density potential growth must still rise
about 2-2.5x over 1940-2020, now concentrated in one explicit establishment term at the front. Decoupling relocates
and makes the inflation legible; it does not remove it. Whether that rise is real or a proxy is the branch-E question.
A9 continuation (warm start, 400 iterations) launched 2026-11-03.

## R22 (pre-registered 2026-11-03, before running): independent cell-year abundance innovations with a LOOSER prior
User question: Codex's TACC fits (FIVE_FIT_INTERPRETATION_20261103.md) found independent cell-year innovations
(1966-2020, prior SD .5% / 1%) barely used (typical .002-.009% / yr); regional annual corrections cut growth change
only 5-7%. Would a looser prior do better? innov.py: frozen habitat; the forward's own process_noise argument;
eps = sigma u, u ~ N(0, 1) iid per cell-year, window 1951-2020 (this base's initialisation; Codex 1966-2020);
MEDIAN-one. Mean-one centring (-sigma^2/2) adds a systematic per-year decline that compounds: -.35 log over 70 yr at
sigma .1, -3.15 at .3; negligible at Codex's sigmas but a large artificial brake at looser ones. Arms: sigma .05, .1,
.3. Fitted: about 302k innovations + fscale.
Predictions (stated before running): (1) the coherence penalty is about .5 per cell-year at a one-SD coherent shift
at any sigma, so loosening does not make coherent, decades-long front braking cheap; (2) innovations multiply
existing birds and preserve zeros, so they can suppress early populations but not create birds ahead of fronts;
(3) at sigma .3 they will mostly absorb ABUNDANCE misfit (fill-in; modern patchiness), closing much of the gap with
little effect on front timing. Diagnostic value: WHERE they go (front vs settled; East vs West; early vs late).
A9 continuation (warm start, +400 iterations, cap): -log posterior 136,469 -> 136,395 (plateauing about 800 worse than
the fit's 135,591); BBS 110,754; offsets +2 / -1 / -3 / -7; never reached .047 (fit .04); G East .53 -> 2.45 (x4.6),
West .83 -> 2.88 (x3.5); growth rates constant (Sa .41, Sj .145, Fmax 2.71). Conclusion unchanged: a viable,
legible alternative about 800 worse in log posterior that keeps a low-density growth rise of roughly 2-2.5x.

## R23 (pre-registered 2026-11-03, before running): ONE innovation per cell, active only on the wavefront
User: the cell-year prior's coherence cost is what PREVENTS takeover (agreed; my "wrong shape" phrasing was wrong). The
real problem is that the innovations multiply ABUNDANCE (stock). Wanted: a version that targets the wavefront.
frontinnov_forward.py / frontinnov.py: fecundity x exp(sigma u_c exp(-E / tau) exp(-(N/K) / rho)); one u_c ~ N(0, 1)
per land cell (4,318); E = soft colonisation clock (years occupied; 1% threshold); tau 10 yr, rho .3 K, sigma 1. The
correction acts only while a cell is newly colonised AND sparse, is charged once per cell, and cannot touch settled
abundance. Identity (u = 0): 0.0. Main arm: --k_real (Z's change to K only: the R21 / A9 structure; start = K only,
133,831); second arm: K frozen too (all readouts frozen; start 178,022). CPU float64, 200 iterations (the GPU is
running R22).
Readouts: gap closed; offsets; never reached; u by region (mean, sd, share |u| > 1); the u map (where corrections are
wanted).
Predictions: with K real, per-cell front corrections can close most of the K-only -> fit gap, with fit-like timing, IF
front timing is a spatially heterogeneous establishment problem; the u map then shows WHERE (e.g. negative early near
the release / Northeast, positive later inland). If the gap stays large, timing needs a TIME-structured change that
one-per-cell corrections cannot express (u is per cell, but its effect is fixed to each cell's own colonisation
window, so a regional calendar trend is not representable except through the colonisation order).

### R22 RESULT (2026-11-03; [cur]; frozen habitat; median-one independent cell-year abundance innovations, 1951-2020)
| sigma | BBS -log-lik | gap closed (BBS) | prior cost | offsets | never | East mean eps (filling / front, 1966-80) |
|---|---|---|---|---|---|---|
| .05 | 170,305 | .11 | 3,780 | -44 / -53 / -66 / -69 | .11 | -.002 / .000 |
| .1 | 148,301 | .44 | 13,568 | -37 / -45 / -58 / -61 | .13 | -.057 / .000 |
| .3 (cap) | 111,814 | .97 | 14,764 | -33 / -41 / -53 / -55 | .13 | -.21 / -.20 |
Where (sigma .3, East): steady SUPPRESSION of occupied / filling cells 1951-1980 (about -.1 to -.2 per year, front cells
too once occupied), boosts after 1996 (filling +.08, near K +.11 / yr); the West accumulates +.57.
Read (established, [cur]): loose abundance innovations can absorb almost the whole BBS likelihood gap (97% at sigma .3,
for about 14,800 in prior), but leave the fronts 33-55 yr early. They fix how many birds there are, not when birds
arrive: stock multipliers throttle existing populations but cannot stop the low-density spread ahead of the occupied
range. Predictions (2) and (3): MET. The BBS likelihood is dominated by abundance; timing is a smaller, separate
signal, one more reason the fit hides timing in growth-rate inflation. Agrees with the user: the problem is acting on
the stock.

### R24 PRE-REGISTRATION (2026-11-03; [cur]; user's option 2: innovations on ARRIVALS at the Allee knife's edge)
Correction first (user, 2026-11-03): R23's gate (and R22's "front = N/K < .1" bins, and Codex's
process_noise.target = low_density, h/(h + N/K)) use the RATIO N/K. [cur] has an ABSOLUTE mate-finding Allee, so the
low-density regime that matters is absolute density relative to the Allee scale, not N/K. R23 is kept as run (gate
mis-specified for this base); R24 gates on the Allee deficit itself.
Design (arrinnov.py / arrinnov_forward.py): juvenile arrivals x exp(sigma u D), D = 1 - mated_fraction(gamma N_local)
(the base's own Allee form, absolute density). Frozen readouts, K from real Z (R21/A9 structure), Fmax scale free,
prior u ~ N(0,1), sigma 1. Arms: (a) one u per cell; (b) one u per cell-year 1951-2020 (u stays ~0 where D ~ 0).
GPU float64, maxiter 200.
Predictions: if early fronts are a settlement failure of the dispersing trickle, u goes NEGATIVE in East front cells
in the 1950s-70s and the offsets move toward 0 by much more than in R22 (R22 -33/-41/-53/-55 at about the same
likelihood). Risk (user's point: fractions reach everywhere, so the correction is strong and weak at once): suppressing
one year's arrivals only delays establishment by a year, because upstream sources keep sending a trickle; the per-cell
arm (a) is then too weak to hold a front, arm (b) needs coherent negative runs of u (expensive), and the fit is stiff at
the Allee knee (possible stalls). Scored as R23: gap closed, offsets, never reached, u by region (x era).

### R24 first results (2026-11-03)
- (a) per cell, maxiter 200 (CAP hit, not converged; 1.8 min): -log-lik 128,567 (gap closed .22), prior 995, Fmax x.85,
  offsets -13 / -15 / -13 / -16 vs K-real no-correction -27 / -34 / -44 / -32 (R21 row 1); never .090 (vs .09).
  u: East mean -.28 (sd 1.12, 25% |u| > 1), Plains -.21, West ~0. Arrivals suppressed in the East, as predicted.
  Read (likely): settlement corrections at the Allee edge move TIMING far more per unit of likelihood than abundance
  innovations did (R22 sigma .3: 97% of the gap, fronts still -33..-55). Not converged; continuation below.
- (b) per cell-year, default tolerances: stopped after 32 evaluations on scipy's relative-change test (last steps
  improved by ~2e-4 units), gap .018, offsets -25 / -33 / -42 / -30.5 (unchanged). The surface is nearly flat per
  cell-year (the user's "weak" side), but the stop is partly the optimizer's tolerance; rerun with --tight
  (ftol 1e-15, gtol 1e-12, maxiter 300).

### R25 PRE-REGISTRATION (2026-11-03): Allee SHAPE (smoothness) x arrival corrections
Records (agent search, 2026-11-03): the only shape knob is the aggregation k in allee_mated_fraction, 1 - (1 + x/k)^-k
(k = inf Poisson, sharp knee; k = 1 hyperbolic, gentle; same slope at 0). Fixed at 3 since E079 (27 km base; 79a k = 1
and 79c relative never written up); never swept on the 54 km ladder or the absolute-Allee yearling base.
Design: [cur], frozen readouts, K real; k in {0.5, 1, 3 (base), Poisson} at the base's gamma (n50 location NOT refit;
Fmax scale free); per k: (i) sigma 0 (no corrections, Fmax only), (ii) per-cell arrival corrections sigma 1,
--tight, maxiter 1500. k = 3 (ii) is the R24(a) continuation.
Predictions: a gentler knee (small k) leaves more of the dispersing trickle partly mated, so with no corrections fronts
get EARLIER (less pinning), and the corrections needed to hold arrivals get larger. A sharper knee (Poisson) holds
fronts harder with no corrections, but makes the fit stiffer. If timing is insensitive to k at fixed gamma, the
knee shape is not where the timing problem lives (the slope at 0, i.e. n50 / gamma, is).
- (b) cell-year rerun --tight, maxiter 300 (cap): -log-lik 133,265 (gap .024), prior 284, offsets -25 / -32 / -42 / -30,
  never .10. The flat surface is REAL, not a tolerance artefact: year-by-year settlement corrections barely move the
  likelihood or the fronts (each year's suppression is refilled by next year's trickle -- user's "weak" side), while
  one persistent correction per cell (a) moves fronts by 14-31 yr. Established for this base: what holds a front is a
  persistent per-place establishment difference, not transient per-year luck.

### R23 RESULT, K-real arm (2026-11-03; [cur]; gate = clock x N/K, i.e. MIS-SPECIFIED for absolute Allee, see J19)
One fecundity correction per cell, active only while the cell is newly colonised (E < ~10 yr) and sparse; frozen
readouts, K real; maxiter 200 (CAP hit, not converged; 12.6 min CPU float64).
-log-lik 114,023 (gap closed .83; start 133,831; fit 110,045; R21 Z-establishment readout 111,682), prior 704,
Fmax x1.10, offsets +3 / 0 / -4 / -9 (fit-like; K-real no correction -27/-34/-44/-32), never .096.
u: East mean -.31 (sd .83, 11% |u| > 1), Plains -.31 (sd .91), West -.04 (sd .10).
East 1966-79 log-lik -804 (R24a arrivals: -5,266).
Read: a STATIC per-place establishment pattern -- one number per cell, no calendar time, acting only while a cell is
new and sparse -- recovers fit-like timing and 83% of the likelihood gap, for a prior cost of ~700 (likely; not
converged). Corrections are mostly NEGATIVE in the East and Plains: early establishment held back, not boosted. This is
the establishment / low-density decomposition in its most reduced form, and it says the timing job of the fitted
habitat change can be done by a fixed spatial establishment field. Caveats: 4,318 free numbers (check spatial
structure: does u track Z, the fitted habitat change, distance from release?); N/K gate; Fmax constant.
Compared with R24a (arrival corrections, absolute gate, 200 it): R23 is far stronger (.83 vs .22; offsets ~0 vs -13..-16).
Likely why: acting on the reproduction of settled-but-sparse birds compounds over the cell's first decade, whereas
arrival corrections are refilled each year by new arrivals.

### R23b PRE-REGISTRATION (2026-11-03): R23 with the ABSOLUTE Allee gate, run to convergence
Same as R23 K-real, but the low-density factor is the base's Allee mate-finding deficit D (absolute density) instead of
exp(-(N/K)/.3); colonisation clock kept (tau 10). Arms (GPU float64, --tight, maxiter 1500, K real): (i) gate allee;
(ii) gate ratio (R23 converged, for comparison). Prediction: the absolute gate gives similar timing and gap (the
establishment pattern, not the gate's exact form, does the work); if it is much weaker, R23's effect leaned on N/K
(i.e. on K's fitted change), which would weaken the "static establishment field" reading. Then: spatial structure of u
(vs Z, fitted habitat change, distance from release).

### u_structure: where the per-cell corrections sit (2026-11-03; u_structure.py; Spearman within region)
| map | region | dlambda_fit | lambda_frozen | logK | dlogK | dist release | first detect | OLS R2 |
|---|---|---|---|---|---|---|---|---|
| R23 K-real | East | -.53 | -.60 | -.60 | -.52 | .12 | .32 | .57 |
| R23 K-real | Plains | -.51 | -.86 | -.80 | -.53 | .58 | .13 | .42 |
| R23 K-real | West | -.34 | -.55 | -.54 | -.34 | .40 | -.09 | .29 |
| R24a arrivals | East | -.65 | -.67 | -.67 | -.65 | .11 | .28 | .23 |
| R24a arrivals | Plains | -.41 | -.55 | -.54 | -.42 | .45 | .07 | .20 |
(dlambda_fit = the fit's potential-growth change 2006-25 minus 1951-65; covariates are collinear, so OLS betas are not
read.) Read (likely): establishment is held back MOST in the best habitat (high K, high frozen lambda) and where the fit's
habitat change is largest; not by distance from the release. The static establishment field is largely the EARLY half
of the fitted habitat change re-expressed as "the best places were slow to be colonised" -- R2 .57 East. So it is not
an independent mechanism yet: it relocates the habitat change's timing job into a fixed spatial pattern that tracks
habitat quality. Open: is "good habitat colonised late" ecological (e.g. urban / high-K habitat needs a denser donor
pool, or is occupied late by competitors) or an artefact (frozen Z's time-mean overstates early quality where Z
improved)? The second is the simpler reading, because dlambda_fit and lambda_frozen are themselves correlated.

### R26 PRE-REGISTRATION (2026-11-03): does a per-cell TIMING term change what the frozen model uses to time fronts?
Background: K-timing plan row; the NB zero costs ~mu, so fractional-bird early fronts are nearly free (R22). Any term
built on mu linearly (Poisson hazard 1 - exp(-effort mu)) inherits that weakness, so R26 uses a LOG-density link
(forms 1 + 2 combined): first-detection term, one event per cell (BBS real routes, 25-49N):
  p_ct = sigmoid(a + b log mu_ct), mu_ct = the model's expected count summed over that cell's surveyed routes in year t;
  L_c = prod_{surveyed t < t1} (1 - p_ct) * p_{c,t1}  (detected first in t1), or prod over all surveyed t (never
  detected; right-censored). Cells detected in their first 3 surveyed years are left out (left-censored).
  a, b fitted (b >= 0 via softplus). Added to the BBS objective with weight w = 1 (this partly re-uses BBS zeros:
  a diagnostic, not a likelihood proposal).
Arms ([cur], frozen readouts, K real; GPU float64; --tight):
  T0 / T0w: no corrections; Fmax scale and Allee gamma scale (n50) free; without / with the timing term.
  T1 / T1w: + R23b-type per-cell establishment correction (absolute Allee gate, clock tau 10); without / with.
Predictions: (1) without corrections, the timing term pulls toward a STRONGER Allee (higher n50) and later fronts at
some BBS cost -- if so, the weak incentive was hiding a mechanism the counts alone do not reward; if gamma barely
moves and fronts stay early, a single Allee scale cannot time the fronts regardless of incentive. (2) With
corrections, the timing term fixes timing at least as well as R23b, and the u map's tie to habitat quality
(R23 rho -.5 to -.6) weakens if part of that tie was the counts' abundance job leaking into establishment.
R26 smoke (6 evals, T1w-like): timing data = 1,201 detected cells, 427 never detected, 624 left-censored (dropped);
33,326 cell-years. Already: n50 x1.72, Fmax x.74, a -1.12, b .67, offsets -10/-12/-12/-13 but never .16 -- the timing
term pulls toward a STRONGER Allee, as predicted (1). bbs term == readout exactly (consistency check).

### R23 RESULT, K-frozen arm (N/K gate; 200-it cap, not converged; 14.6 min CPU)
-log-lik 132,758 from 178,022 (gap .67), prior 1,952, Fmax x.73, offsets +2 / -1 / -7 / -11, never .165 (high).
u: East -.41 (sd 1.27), Plains -.36, West -.40 (sd .74). With K frozen too, the static establishment field must also
hold back the West, and still leaves a third of the gap (abundance: K's change does that job, as in mix_replay).

### R26 RESULT T0w (no corrections; Fmax + Allee n50 free; + per-cell log-link timing term; K real)
Stopped at 111 evaluations (not converged; L-BFGS line-search stop, see whether T0 does the same).
BBS -log-lik 118,764 (from 133,831: gap .63 with TWO numbers), timing -log-lik 3,641 (from 15,892), n50 x13.4,
Fmax x1.93, a -1.33, b .80, offsets -3 / -4 / -3 / -6.5, never .156 (vs .09 uncorrected).
Read (likely; pending T0): a much STRONGER absolute Allee effect (13x the mate-finding scale) with doubled fecundity --
a pushed wave -- times the fronts with no time-varying growth rates and no per-cell corrections, and also improves
the BBS term by ~15,000. Prediction (1) MET in direction and far beyond the size expected. Key question for T0: does
the BBS likelihood ALONE reach this region (then the fit's real optimum was simply never found from its start: a
basin / optimisation problem), or does it need the timing term (the weak-incentive reading)? Caveats: never-reached
.156 (16% of front cells never reach 10% of the reference -- fit is .047); frozen readouts (only K real); the Allee knee
at this n50 will be steep (stiff fits, cf. E079).
Prior context (established, overlay grid54_step43_factor_p2_veit_s41.json allee_prior): n50_raw_loc -3.4915, scale .25
(log scale), i.e. the production fit pins n50 to within ~x1.3; x13.4 is log 2.6 = ~10 prior SDs away. The sandbox
objective has NO n50 prior. So if T0 also finds this region, the production prior on n50 is what keeps the fit off the
pushed-wave solution, and the time-varying habitat change is then doing the work a stronger Allee effect would do.
(n50 is in the model's BBS density units; no conversion to birds is implied.)

### R26 RESULT T0 (no timing term, no corrections; Fmax + n50 free; K real) -- converged, 88 evals
BBS 118,763 (= T0w's 118,764), n50 x13.3, Fmax x1.92, offsets -3 / -4 / -3 / -6.5, never .156. IDENTICAL to T0w.
Read (established for this sandbox): the BBS counts ALONE choose the strong-Allee / high-fecundity solution once
growth rates cannot change in time; the timing term is not needed to reach it (so the weak-incentive reading is NOT
what hides it here). Records first: this is E002 again in a cleaner setting -- E002 (2026-10-16) found free
n50 = .28 route counts (9.5 prior sds above the prior), demanded by the EARLY EASTERN FRONT (weak Allee cost 757
nats within +-5 yr of arrival); E055 (user decision, 2026-10-18) pinned n50 to the literature because with time-varying
habitat the Allee strength was unidentified (n50 x1/3..x5/3 moved U < 75 nats) and free strong Allee + K drift made
Allee-dead cells (the absence-switch loop). Here: with growth rates frozen, n50 is strongly identified (+15,000).
Likely mechanism: a strong Allee effect and the fitted habitat change in growth rates are SUBSTITUTES for the timing
job (a pushed front vs a habitat that opens late). The user's prior (Veit 1-D vs 2-D: a weaker Allee is more likely)
and the literature pin weigh against the strong solution; never-reached .156 is the Allee-dead risk again.
R25 k .5 no-correction arm: 129,174 (gap .13), Fmax x1.30, offsets -24/-30/-37/-24, never .106: my R25 direction
prediction was WRONG -- at equal slope at 0, smaller k means MORE mate limitation at intermediate density (k = 1 is
x/(1+x) < 1 - e^-x), i.e. effectively a stronger Allee; consistent with T0.

### R27 PRE-REGISTRATION: the substitution curve (alpha*) -- n50 and Fmax free at partial habitat change
timing.py --alpha a (Z_eff = Zref + a (Z_t - Zref) for the readouts; K real), w 0, no corrections, a in
{.25, .5, .75, 1}. Plus a = 0 (= T0). Read: n50*(a), Fmax*(a), BBS(a), offsets, never reached.
Predictions: n50* falls toward the pin as a -> 1 (at a = 1 it should sit near the pin, E055: unidentified), and BBS(a)
is FLAT-ish if the two are substitutes; the share a where n50* reaches the pin is the share of the habitat change the
Allee effect can replace. If BBS at a = 1 with free n50 beats the fit by a lot, the pin binds in production too.

### allee_dead.py on T0 (strong-Allee solution) vs fit vs uncorrected (K real) -- share of OBSERVED-occupied cells the
model never occupies (density never >= 1% of reference median) / share of all cells never occupied
| arm | East missed / never | Plains | West | offsets | never reached (10%) |
|---|---|---|---|---|---|
| fitted | .007 / .101 | 0 / .005 | .004 / .017 | +3/-1/-3/-6 | .068 |
| frozen, K real | .006 / .092 | 0 / 0 | .004 / .013 | -27/-34/-44/-32 | .093 |
| T0 (n50 x13.3, F x1.92) | .009 / .107 | .030 / .039 | .024 / .062 | -3/-4/-3/-6.5 | .156 |
Read (established for this replay): the strong-Allee solution is NOT the E055 Allee-dead pathology (.16-.18 there):
it misses 3% of observed-occupied Plains cells and 2.4% in the West, ~1% East. Its higher never-reached share is mostly
cells that are reached but stay below 10% of the reference (thin populations under a strong Allee), not dead cells.

### R28 PRE-REGISTRATION: production-structure refits with the Allee n50 FREED (full log posterior)
refit_fields.py --free_n50 3: n50_raw joins the refitted sites; its prior N(-3.49, .25) is swapped exactly for
N(-3.49, 3) (same location). Arms ([cur], CPU float64, maxiter 400):
  (a) standard single-field refit (habitat field reads real Z; all rates time-varying) + free n50;
  (b) A9 decoupled structure (K real Z, Sa/Sj/Fmax frozen Z, + Z establishment readout), warm start from A9 cont, + free n50.
Read: n50 end, -log posterior and BBS vs the pinned versions (fit 135,591 / A9 136,395), lambda_split start/end (potential
lambda rise), offsets, never reached.
Predictions: (a) n50 moves only modestly (E055: unidentified when habitat change is free) and the lambda rise barely
changes -- with the habitat change available the fit has no need for a strong Allee. (b) Growth rates are frozen here,
so n50 should rise substantially (toward T0's x13) and the establishment readout's East x4.6 should SHRINK: the
strong Allee replaces part of the establishment field's time change. If (b) closes its ~800 gap to the fit, the
pin -- not the data -- is what makes the fit need time-varying growth for timing.

### R25 RESULT, no-correction arms (Fmax only, gamma fixed, K real)
| k (aggregation) | BBS -log-lik | Fmax x | offsets | never |
|---|---|---|---|---|
| .5 (most mate limitation at intermediate N) | 129,174 | 1.30 | -24/-30/-37/-24 | .106 |
| 1 (hyperbolic) | 132,010 | 1.06 | -23/-30/-39/-26.5 | .104 |
| 3 (base) | 133,830 | 1.00 | -26/-34/-44/-32 | .093 |
| Poisson (sharpest knee) | 134,336 | .96 | -27/-35/-46/-35 | .093 |
Read (established for this replay): monotone -- more mate limitation through the intermediate-density range helps
the likelihood and slightly delays fronts; same direction as T0's larger n50, much smaller effect (n50 is the bigger
lever). The knee's sharpness per se is not what matters; the breadth of mate limitation above the front density is.
R28 correction (records): [cur] does not sample n50 -- it is PINNED (age_priors.PINNED, numpyro condition) at
n50_raw = -2.797, n50 = .059 route counts (the E055 Veit & Lewis value); the overlay prior loc -3.49 never applies.
So T0's x13.3 is n50 ~ .79 route counts (E002 free fit: .28). R28 now unpins the site, starts at the pin, and uses a
N(pinned value, 3) prior on n50_raw.

### R27 a = 1 (real habitat change in all rates; n50 + Fmax free, BBS only): n50 x.984, Fmax x.997, BBS 110,044.3
(fit 110,044.7), offsets = the fit's. Prediction MET: with the fitted habitat change present, the BBS term has no use
for a different n50 (E055 again). Together with T0 (a = 0: n50 x13.3, +15,000): the Allee strength is identified ONLY
when growth rates cannot change in time -- substitutes. a = .5 / .25 / .75 running.
GPU note: R25 k .5 + corrections died at eval 1,097 (obj 118,696 incl. prior -- promising) with OOM under a .3 cap;
OOMs recur near ~1,000 evals even with 2 jobs, so per-job caps were too small (allocator growth), not only
over-subscription. Lost arms now run strictly serially at .7 (q_gpu_serial.sh).
R27 a = .5: n50 x6.4, Fmax x1.33, BBS 113,407, offsets +1/-2/-3/-6, never .133. n50* falls smoothly with a
(13.3 at 0, 6.4 at .5, .98 at 1). Controls queued: same a with n50 held at the pin (q_r27c.sh), to read the free-n50
gain at each a.
R27 a = .25: n50 x10.4, Fmax x1.64, BBS 116,204, offsets -1/-3/-3/-6, never .152 (line-search stop, 64 evals).
Curve so far (a: n50 x, Fmax x, BBS): 0: 13.3, 1.92, 118,763 | .25: 10.4, 1.64, 116,204 | .5: 6.4, 1.33, 113,407 |
1: .98, 1.00, 110,044. Fronts fit-like at every a once n50 is free.

### R28 RESULT (a): production single-field refit with n50 UNPINNED (prior N(pin, 3)), from the fit (125 evals, converged)
-log posterior 135,587 -> 135,579 (-8), BBS 110,044.7 -> 110,040.9, n50 x.81 of the pin (.048 route counts; slightly
WEAKER), offsets +3/0/-3/-6, potential lambda 1940 -> 2010: .207 -> 1.081 (start) vs .202 -> 1.076 (end) -- unchanged.
Prediction (a) MET: inside the fit's basin, the Allee strength is free but unwanted; the lambda rise does not move.
Combined read with R26/R27 (likely): the time-varying growth rates do TWO jobs, and the Allee strength can replace only
one. (1) Front timing: fully substitutable -- with n50 free, fronts are fit-like at every habitat share a (R27).
(2) Abundance / fill-in: NOT substitutable -- BBS still rises 110,044 (a=1) -> 113,407 (.5) -> 116,204 (.25) ->
118,763 (0) with identical fronts, so the remaining ~8,700 is the growth-rate change's abundance job (core growth
and fill-in after arrival), which a stronger Allee cannot do. A local refit from the fit never visits the strong-Allee
region because, by the BBS, the full habitat change is better there (by ~8,700) -- the strong-Allee solution is a
worse optimum for the counts, not a hidden better one. Caveat: R27 readouts shrink ALL growth rates' change together
(Sa / Sj / Fmax); R28 (b) frees n50 in the decoupled structure, where an establishment readout can take timing.
R27 a = .75: n50 x2.9, Fmax x1.09, BBS 111,006, offsets +2/-1/-3/-6, never .109.
Full curve (a: n50 x / Fmax x / BBS): 0: 13.3 / 1.92 / 118,763; .25: 10.4 / 1.64 / 116,204; .5: 6.4 / 1.33 / 113,407;
.75: 2.9 / 1.09 / 111,006; 1: .98 / 1.00 / 110,044. Smooth, monotone substitution; fronts fit-like throughout.

### R23b RESULT (i): per-cell establishment correction with the ABSOLUTE Allee gate (K real; converged at a stall,
same value as the OOM-killed run at eval 510 -> a real optimum)
BBS 122,947 (gap .46; R23 N/K gate: .83 at a 200-it cap), prior 1,661, Fmax x.89, offsets -5 / -7 / -8 / -11, never .116.
u: East mean -.77 (sd 1.35, 24% |u| > 1), Plains -.17, West ~0.
Read (likely): the N/K-gated R23 was about twice as strong mainly because its gate (exp(-(N/K)/.3)) stays open up to
~30% of K -- well into FILLING -- so it also did abundance work, and leaned on K's fitted (real-Z) change. The
absolute gate closes once density passes the Allee scale (n50 = .059 route counts), i.e. it acts only at true
establishment. Pure establishment corrections recover about half the gap and leave fronts 5-11 yr early; the
pre-registered weakening of the "static establishment field" reading is CONFIRMED. Consistent with R26-R28: timing is
cheaply fixable (by a stronger Allee or by establishment), the large remaining value of the habitat change is fill-in
/ abundance.
R25 k = 3 + arrival corrections (serial GPU rerun): STOPPED at 41 evals reporting success, BBS 129,541 (+ prior 898),
offsets -13/-16/-16/-18 -- NOT an optimum: the OOM-killed run of the same arm reached objective 126,006 by eval 997.
L-BFGS stops early on this rough (Allee knife-edge) surface; GPU float64 reductions are non-deterministic, so paths
differ run to run. Treat arrival-correction numbers as lower bounds on what they can do; CPU rerun queued.
R27 control a = .5, n50 PINNED: BBS 120,054, Fmax x.81, offsets -8/-10/-8/-10, never .130. Free n50 at the same a:
113,407 and fit-like fronts -> freeing n50 is worth ~6,650 at half the habitat change (vs ~0 at a = 1, ~15,000 at 0).
R27 control a = .25, n50 PINNED: BBS 126,684, offsets -11/-14/-13/-14 (free n50: 116,204, fit-like) -> n50 worth ~10,480.
R27 control a = .75 pinned: BBS 113,647, offsets -3/-5/-5/-7 (free: 111,006) -> n50 worth ~2,640.

### R27 SUMMARY: substitution between Allee strength and the habitat change in growth rates (K real throughout)
| a (share of fitted habitat change in Sa/Sj/Fmax) | BBS, n50 pinned | offsets pinned | BBS, n50 free | n50* | value of free n50 |
|---|---|---|---|---|---|
| 0 | 133,830 | -26/-34/-44/-32 | 118,763 | x13.3 | 15,070 |
| .25 | 126,684 | -11/-14/-13/-14 | 116,204 | x10.4 | 10,480 |
| .5 | 120,054 | -8/-10/-8/-10 | 113,407 | x6.4 | 6,650 |
| .75 | 113,647 | -3/-5/-5/-7 | 111,006 | x2.9 | 2,640 |
| 1 (fit) | 110,045 | +3/-1/-3/-6 | 110,044 | x.98 | 0 |
Read (established for this sandbox): with the Allee pinned, each quarter of the habitat change is worth ~6,000-7,000 and
moves fronts by years; with n50 free, fronts are fit-like at EVERY a and the habitat change's remaining value is
~2,400-2,800 per quarter (abundance / fill-in). So roughly 55-60% of the habitat change's likelihood value at a = 0
(15,070 of 23,785) is the timing job a stronger Allee can do instead; the rest (~8,700) it cannot.
R25 k = .5 + per-cell arrival corrections (serial GPU, converged, 1,071 evals): BBS 116,694, prior 2,002, Fmax x1.32,
offsets -12/-12/-6/-11, never .087 (fit-like). vs k = .5 without corrections 129,174: corrections worth ~12,500 here, vs
~4,300+ at k = 3 (R24a/R25 k3, not converged; best seen obj 126,006). Read (likely): arrival corrections are far more
useful under a broad mate-limitation curve -- with more of the density range mate-limited, an arrival shortfall keeps
a cell below the knee longer, so per-place settlement differences compound. Interaction of shape x establishment.
R23b (ii) N/K gate run long (1,500 iterations cap, 1,926 evals, GPU .7): BBS 113,672 (gap .85), prior 685, Fmax x1.11;
u East -.23, Plains -.40, West -.07. Barely better than the 200-it R23 (114,023): R23's number was near its optimum.
Absolute gate (i): 122,947. So the gate form matters a lot (.85 vs .46 of the gap).
R25 k = 1 + arrival corrections (754 evals, line-search stop): BBS 119,094, prior 2,652, offsets -12/-11/-6/-11,
never .086 (no corrections at k = 1: 132,010) -> corrections worth ~12,900. Similar to k = .5 (116,694 / ~12,500);
both far above k = 3 (best seen ~126,000 objective).
R25 Poisson + arrival corrections: STOPPED at 45 evals (reports success), 132,205 -- like k = 3 (41 evals): with a
sharp knee the optimizer stalls early (E079's reason for k = 3). Shape matters for fitting as well as for the answer.

### R29 PRE-REGISTRATION: Allee shape x strength (no corrections, growth rates frozen, K real, n50 + Fmax free, BBS only)
timing.py --w 0 --agg_k {.5, 1} (k = 3 is T0: 118,763, n50 x13.3). GPU serial, frac .7.
Prediction: a broad curve with free n50 does at least as well as T0 with a smaller n50 multiplier (part of the
"strength" T0 found is breadth of mate limitation); if it beats T0 by thousands, part of the ~8,700 abundance residual
is Allee shape, not habitat change. Never-reached / Allee-dead to be checked as for T0.

### R28 RESULT (b): A9 decoupled structure (K real Z; Sa/Sj/Fmax frozen; Z establishment readout G) + n50 UNPINNED
(warm start A9 cont; 439 evals, not converged)
-log posterior 136,391 -> 136,267 (A9 cont 136,395; fit 135,591: still ~676 worse), BBS 110,619 (A9 110,754; fit
110,045), offsets +2/-1/-3/-6, never .043 (fit .047).
n50 -> .274 route counts = x4.6 the pin -- essentially E002's free value (.28).
Establishment readout G (multiplies low-density fecundity): East 1940 1.94 -> 2020 3.67 (x1.9; A9 pinned: .53 -> 2.45,
x4.6); West 1.71 -> 4.29 (x2.5; A9 x3.5). Sa .397 / Sj .148 / Fmax 2.73 constant (frozen readouts).
Low-density potential lambda without the Allee factor (Sa + Sj Fmax G), East: 1.18 -> 1.88 (+.70) vs A9 .62 -> 1.37
(+.75): the additive rise barely changes, because G's LEVEL rose to offset the stronger Allee at low density.
Prediction (b) MET for the multiplicative time change (x4.6 -> x1.9 East) -- a stronger Allee takes over part of the
establishment field's timing job -- but NOT for the additive potential-lambda rise (+.75 -> +.70). Read (likely): the
time change that remains is the fill-in / abundance job (R27), expressed through the same low-density boost; a
stronger Allee re-levels it but does not remove it. Caveat: potential lambda ignores the Allee factor, so the two
arms' "low-density growth" are not on the same footing; a realized-growth comparison (realized_r.py) is needed.

### R29 RESULT: shape x strength (no corrections, frozen growth rates, K real, n50 + Fmax free)
| k | BBS | n50 x | Fmax x | offsets | never |
|---|---|---|---|---|---|
| 3 (T0) | 118,763 | 13.3 | 1.92 | -3/-4/-3/-6.5 | .156 |
| 1 | 118,314 | 13.7 | 2.63 | -4/-5/-3/-6 | .155 |
| .5 | 118,295 | 13.0 | 3.81 | -7/-7/-5/-7 | .145 |
Prediction WRONG in part: a broad curve does NOT reduce the n50 multiplier (~13 at every k) and gains only ~470 --
once n50 is free, shape is nearly irrelevant to the fit (k only trades against Fmax: broader curve -> more mate
limitation at moderate density -> higher Fmax to compensate). So the ~8,700 abundance residual (R27) is not Allee shape.
Shape mattered in R25 only because n50 was pinned (shape was then the only way to add mate limitation) and for
per-place corrections (fit stiffness). Established for this sandbox.

### R26 RESULT T1 (GPU run "_T1g"; per-cell establishment corrections, ABSOLUTE Allee gate, + n50 and Fmax free;
frozen growth rates; K real; no timing term; 1,925 evals, cap)
BBS 113,674 (fit 110,045; uncorrected frozen 133,831), correction prior ~208 (u rms .31, 2% |u| > 1), n50 x2.9,
Fmax x1.25, offsets +3 / 0 / -2 / -7, never .107.
u structure (East): mean -.08, rho with habitat quality -.30 to -.36, R2 .20 -- vs the same corrections with n50
PINNED (R23b i): East mean -.77, rho -.74 to -.84, R2 .84, BBS 122,947.
Prediction (2) MET: once the Allee strength is free, the per-place establishment field shrinks to small corrections
and loses most of its tie to habitat quality. Read (likely): what the pinned-Allee establishment field encoded as
"good habitat colonised late" is mostly a uniformly stronger mate-finding brake. With a moderately stronger Allee
(x2.9, ~.17 route counts) plus small place-specific establishment differences, a model with NO time change in growth
rates reaches fit-like timing and comes within ~3,600 of the fit's BBS -- the remaining value of the habitat change
in growth rates. (Full log posterior / held-out not assessed: sandbox BBS rows only, frozen readouts, K real.)
R25 k = 3 + arrival corrections, CPU rerun (618 evals, line-search stop): BBS 126,270, prior 1,111, offsets
-12/-13/-11/-14, never .093 -> corrections worth ~7,560 at k = 3 (vs ~12,500-12,900 at k = .5 / 1). Shape x arrival
interaction stands; stops remain early on this surface (GPU run stopped at 41 evals; this at 618).

## CORRECTIONS from the independent review of R22-R29 (subagent, 2026-11-04 overnight; numbers verified against raw JSON)
1. R27 label: timing.py calls edit_habitat with the default keys (Z_gathered AND Z_disp_gathered), so "a" scales the
   habitat change in Sa/Sj/Fmax AND the dispersal input Z_disp (worth ~1,800 per mix_replay). The "~8,700 abundance
   residual" includes Z_disp's share.
2. R27 arithmetic: 15,070 / 23,785 = 63% (not 55-60%). Per-quarter values are not uniform: free 2,559 / 2,797 / 2,401 /
   962; pinned 7,146 / 6,630 / 6,407 / 3,602. Pinned controls at .25 / .75 stopped unconverged after 38-39 evals.
3. "~8,700 needs growth-rate time change" is UNPROVEN: T0/R27 refit only 2 numbers (Fmax scale, n50) and keep habitat
   coefficients tuned for time-varying Z; A9/R28b, which refit them, reach BBS 110,619-110,754.
4. T0 regional split of its 8,719 BBS gap to the fit: East 1966-79 +625, 1980-93 +746, 1994-2009 +1,905, 2010-25 +2,131;
   Plains +829; WEST +2,483 -- T0's West is 1,699 WORSE than the uncorrected frozen start (native range, no invasion
   timing). The strong Allee buys ~16,400 in the East and pays in the West; front eras still lose ~1,370, so "timing
   fully substitutable" is too strong.
5. Fit's never-reached baseline is .068 (timing_a1, R28a), not .047; A9/R28b (.043-.051) beat the fit on it, and
   "fit-like fronts" in T0/R27 sit beside never .13-.156 (J18 partly recurring).
6. R23b(i) is NOT converged (converged false, 468 evals); "a real optimum" withdrawn. Its East u is strongly ordered by
   first-detection year (rho +.76) and distance from release (+.50): calendar time entering through colonisation order.
7. R28b's pre-registered decisive criterion (close the ~800 gap) was NOT met (676 remain); its posterior gain (124) is
   inside the ~1,000 optimizer-noise rule; "x4.6 -> x1.9" is mostly G's 1940 LEVEL rising (x3.7) along a flat n50-G
   ridge. Additive rise +.87 (fit) / +.75 (A9) / +.70 (R28b); log ratio 1.64 / .79 / .47; R28b's East starts ABOVE
   replacement (1.18). Verdict on inflation: UNKNOWN pending realized growth.
8. J19 recurs: A9/R28b's establishment readout G (lowdens_forward) fades with N/K (rho <= .3) -- the gate R23b showed
   does fill-in work. "Low-density potential lambda" for G is a mislabel.
9. Sandbox objective is BBS-only (25-49N): [cur]'s posterior also has the NestWatch fecundity likelihood and priors.
   T0's Fmax x1.92 / R29's x3.81 never scored against NestWatch.
10. T1 caveats: single GPU run, eval cap, no replicate; freeing n50 widens the correction gate 2.9-fold (gate uses the
   scaled gamma), so smaller u is partly mechanical; no dimension-matched control.
11. Pin provenance: .059 = V&L n50/K* .0197 x 51a's median K 3.0 (E055) -- a transferred calibration. V&L's own range
   .02-.098; T1 (.17) ~1.7x above its top, R28b (.274) ~2.8x, T0 (.79) ~8x. Interpolating R27 inside the literature
   range (n50 <= x1.67) gives a* ~ .9: under the user's weaker-Allee prior the Allee can replace only ~10%.
12. R25 net of prior: k .5 / 1 corrections worth ~10,500 / ~10,300 (not 12,500 / 12,900).

## ROUND 2 (2026-11-04, from the review's ranked list) -- PRE-REGISTRATIONS
R30 (review D1, decisive): can the FULL posterior do without any time change in growth rates once the habitat mapping
  and n50 are refitted? refit_fields --form single --decouple --estab --ridge_sd 1e-6 (b ~ 0, so G = exp(g0) is
  constant in time; K reads real Z), warm start A9 cont habitat sites with the establishment vector RESET; arms: n50
  free (--free_n50 3) and pinned. CPU. Prediction: if growth must change in time for abundance, ends >= 3,000 worse
  than R28b (136,267); falsified if within ~1,000. Caveat: G stays N/K-gated (J19) -- constant in time here, so it
  cannot carry a calendar signal.
R31 (review D2): full-posterior scoring (timing.py --full: all observation rows incl. other latitudes and pseudo-zeros,
  NestWatch fecundity likelihood, priors). Arms: T0 (a = 0, n50 + Fmax free), a = .5 free, a = .5 pinned, T1 (+ per-cell
  establishment). GPU serial. Smoke: fit NestWatch term -1,966 vs frozen start -2,359. Prediction: NestWatch + northern
  rows pull n50* down and cut the value of free n50 by > half; falsified if n50* stays > x8 at a = 0.
R32 (review D3b): region transfer -- n50 + Fmax fitted on West + Plains rows only, then East only (a = 0, BBS). CPU.
  Prediction: the native range alone picks n50 near the pin; falsified if West+Plains alone selects >= x5.
R33 (review D5-i): a = 0 with Z_disp kept REAL (--keys Z_gathered), n50 free -- isolates Z_disp's share of the R27
  residual. CPU. Prediction: BBS improves by ~1,800 over T0 with n50* about unchanged.
In flight: R26 T1w (timing term + corrections, CPU) then a CPU replicate of T1 (review: T1 had no replicate).

### R31 RESULT T0 under the FULL posterior (all rows, NestWatch, priors; GPU; 50 evals, line-search stop)
n50 x13.3, Fmax x1.96 -- the same optimum as BBS-only T0. Observed-data terms (log-lik):
| term | fit | frozen start (K real) | T0 full |
|---|---|---|---|
| all observation rows (incl. pseudo-zeros, other latitudes) | -112,058 | -136,009 | -120,894 |
| NestWatch fecundity | -1,966 | -2,359 | -3,538 |
NestWatch charges the doubled fecundity ~1,180 relative to the frozen start (~1,570 vs the fit), against a ~15,100 gain
in the counts; adding the overlay's own n50 prior N(-3.49, .25) (not charged here: [cur] pins n50, the sandbox moves
gamma directly) would cost ~110 more. Prediction (n50* pulled below x8, value of free n50 cut by half) FALSIFIED:
under the full data, freezing the growth rates still makes the counts demand a ~13x stronger Allee effect. Read
(established for this sandbox): the strong-Allee solution is not a BBS-only artefact; fecundity data push back but
weakly. It remains ~8,800 worse than the fit on all observation rows (incl. the West damage).
R31 a = .5 under the full posterior: free n50 -> x6.5, Fmax x1.35, BBS rows 113,417, full objective 139,766; pinned
control: full objective 145,970, BBS 120,076, offsets -8/-10/-9/-10. Value of free n50 at a = .5: ~6,200 (full) vs
~6,650 (BBS-only). The substitution curve survives the full posterior almost unchanged.

### R32 RESULT: region transfer (a = 0, K real, n50 + Fmax free, BBS)
| fitted on | n50 x | Fmax x | East -log-lik | Plains | West | offsets |
|---|---|---|---|---|---|---|
| West + Plains rows | 2.9 | 1.13 | 78,296 | 6,400 | 44,573 | -11/-15/-17/-17 |
| East rows | 14.6 | 2.01 | 66,042 | 6,597 | 46,272 | -2/-3/-2/-6 |
| all (T0) | 13.3 | 1.92 | 66,170 | 6,487 | 46,106 | -3/-4/-3/-6.5 |
| fit (a = 1, pinned) | 1 | 1 | 60,764 | 5,658 | 43,623 | +3/-1/-3/-6 |
(uncorrected frozen start, K real: East 82,604 / Plains 6,821 / West 44,407)
Prediction ("native range alone picks n50 near the pin; falsified if >= x5"): PARTLY MET -- the native West+Plains pick
x2.9 (~.17 route counts; above V&L's top .098 but far below x13), while the East alone demands x14.6. Read (likely):
the strong Allee is the EASTERN INVASION's demand (timing of the introduced front with frozen growth rates), not a
range-wide property; the native range mildly prefers a stronger brake than the pin, and is hurt by the East's value
(West 44,573 -> 46,272, +1,700). The two regions disagree, which is what a time-varying process absent from frozen
readouts would produce -- or a real East/West difference in establishment (lineage, feeders, migration).

### R33 (first run): Z_disp kept REAL, n50 free (a = 0 on Z_gathered only)
In-fit BBS 116,470 (T0, Z_disp frozen: 118,763 -> Z_disp's real change worth ~2,290 here), n50 x8.7 (T0 x13.3),
Fmax x1.47. BUG: the readout replay ignored --keys (froze Z_disp too), so its offsets / never (-5..-8 / .144) and
readout 120,017 are INVALID; fixed (habitat_keys passed to sb.run) and rerun as _zg_only_v2.
Read so far (likely): keeping Z_disp's real change removes ~a third of the "Allee strength" T0 needed (x13.3 -> x8.7)
-- part of what the strong Allee substituted for was the change in DISPERSAL habitat, not growth rates.
R33 (v2, readout fixed): in-fit = readout 116,470, n50 x8.7, Fmax x1.47, offsets -4/-5/-3/-5.5, never .129.

### R31 RESULT T1 under the full posterior (GPU, 1,885 evals, cap)
n50 x2.95, Fmax x1.27, BBS rows 113,675, correction prior 244, offsets +3/+1/-2/-7, never .102.
Observed-data terms: all obs rows -115,736 (fit -112,058; T0 -120,894; a = .5 free -115,500); NestWatch -2,555
(fit -1,966; T0 -3,538; a = .5 free -2,699). Matches its BBS-only twin (113,674 / x2.94): T1 is robust to the full
posterior, and its NestWatch cost (~590 vs the fit) is the smallest of the frozen-growth solutions.
Summary under the full data (obs rows; gap to the fit): T0 +8,836 | a = .5 free +3,442 | T1 +3,678 -- with NO time
change in growth rates, T1 is as close to the fit as keeping HALF the habitat change with a free Allee.

### R34 PRE-REGISTRATION (review D6): dimension-matched controls for T1 (memory rule match-dimensional-access, J10)
Same as T1 (frozen growth, K real, n50 + Fmax free, one u per cell, prior N(0,1), BBS), but:
 (i) --gate none --tau 1e9: a per-cell fecundity multiplier at EVERY density and year (no establishment targeting);
 (ii) --gate allee --tau 1e9: the absolute Allee gate without the colonisation clock.
Prediction: if (i) ~ T1 (113,674), T1's corrections fix the frozen readouts' spatial misfit, not establishment --
then "establishment" is not what they are. If (i) is much worse in timing but similar in BBS, the targeting matters for
fronts only. (ii) vs T1 isolates the clock.

### R35 PRE-REGISTRATION (review D3a): realised growth anchors (unit-free; NOT used in fitting) for frozen-growth solutions
realized_r.py (East total r per window; anchors 1941-51 r .12-.18 Elliott & Arbib; 1962-71 ~.22 CBC NE) on: fit,
frozen K-real pinned, T0 (n50 x13.3), a = .5 free, WP-fit (x2.9), R33 (Z_disp real, x8.7). harness gains arm key
k_real. Prediction: the strong-Allee arms deepen the 1950s stall near the release (1951-62 r well below the fit) and
fall short of the 1962-71 ~.22; falsified if their r rises toward .2 in 1962-71 with a 1940s r inside .12-.18.

### R35 RESULT: realised growth r (East total; release 150 km in brackets), unit-free
| arm | 1941-51 | 1951-62 | 1962-71 | 1971-85 | 1985-94 |
|---|---|---|---|---|---|
| fit | .104 (.67) | .026 (.011) | .189 (.243) | .336 | .151 |
| frozen, K real, pinned | .149 (.53) | .056 (.068) | .053 (.063) | .072 | .049 |
| T0 (n50 x13.3) | .054 (.49) | .087 (.139) | .276 (.262) | .224 | .140 |
| a = .5 free (x6.4) | .073 (.58) | .045 (.056) | .240 (.290) | .286 | .136 |
| West+Plains fit (x2.9) | .059 (.48) | .239 (.271) | .250 (.115) | .127 | .058 |
| R33 Z_disp real (x8.7) | .111 (.60) | .141 (.170) | .282 (.225) | .204 | .140 |
Anchors: 1941-51 r .12-.18 (Elliott & Arbib: 40-100 founders -> ~280 birds); 1962-71 ~.22 (CBC NE party-hour index).
Review prediction (strong-Allee arms deepen the 1950s stall, fall short of ~.22 in 1962-71) FALSIFIED: T0's 1951-62
r (.087) is ABOVE the fit's (.026) and its 1962-71 (.276) meets the anchor. The frozen pinned arm is the one that fails
(no acceleration: .05-.07 throughout). Every arm, the fit included, overshoots the 1940s founder growth near the
release (.48-.67 vs .12-.18) -- the 1940s anchor discriminates nothing here (F1 drain / founder spread already known).
The fit's own 1950s is a near-stall (.011-.026); R33 (.14-.17) and T0 (.09-.14) grow through the 1950s, which the CBC
spread (first beyond 54 km 1955; 14 circles 1960) arguably supports more -- not scored here (no 1951-62 anchor).
Read (likely): the realised-growth anchors do not reject the strong-Allee family; they reject frozen growth WITHOUT it.

### R36 PRE-REGISTRATION (review D7): T0 multi-start -- n50 start x.3 / x5 / x30 (T0 started at x1 -> x13.3)
Prediction: all reach n50 ~x13, BBS ~118,763 (one basin); a different optimum (e.g. a weak-Allee one near x.3) would
mean the strong-Allee result depends on the start.

### R34 RESULT (i) UNGATED control: one static fecundity multiplier per cell (all densities, all years; no clock), n50 +
Fmax free, frozen growth rates, K real, BBS (GPU, 1,033 evals, not converged)
BBS 110,608 (fit 110,045: +563), prior 182 (u rms .29, 1.6% |u| > 1, max 2.9), n50 x6.5, Fmax x1.71, offsets
+2/-1/-3/-7, never .104. By region vs fit: East 61,162 (+398), Plains 5,635 (-23), West 43,810 (+187) -- no West damage.
u structure: weak (East rho with habitat quality -.2 to -.3, with first detection +.35; R2 .21; Plains/West ~0).
Prediction "if (i) ~ T1, the corrections fix spatial misfit, not establishment": EXCEEDED -- (i) is ~3,070 BETTER than
T1 (113,674). Read (likely): what the frozen readouts miss is mostly a small STATIC spatial pattern in fecundity plus a
stronger Allee brake; establishment targeting (gate/clock) is a restriction, not the active ingredient. With these,
a model with NO time change in growth rates reproduces the fit's BBS within ~560 and its fronts.
Caveats: 4,318 free numbers (cheap in prior, but flexible); n50 x6.5 (~.39 route counts) is ~4x above V&L's top;
Fmax x1.71 unscored against NestWatch; BBS-only; not converged; the multiplier also shifts equilibrium abundance
(c is computed upstream), so it is a static growth+abundance map, not pure fecundity.
Queued: (iii) the same with n50 PINNED (is the Allee strength still needed once the static map is free?) and (iv) the
same under --full (NestWatch / all rows).
R26 T1w (T1 + timing term, CPU, 1,887 evals, cap): BBS 113,664, n50 x2.94, Fmax x1.25, offsets +3/0/-2/-7, never .106
-- identical to T1 (113,674 / x2.94). The timing term changes nothing once corrections and n50 are free (its NLL falls
only through its own link a, b), as with T0 vs T0w. Established for this sandbox: the counts alone determine these
solutions; the per-cell timing term adds no information here.

### R36 RESULT: T0 multi-start (a = 0, K real, n50 + Fmax free, BBS)
start x.3 -> x13.26, 118,763 (= T0 exactly); start x5 -> x12.75, 118,884 (50 evals, line-search stop, same basin);
start x30 -> JUMPED to the lower bound (n50 x.0067 = no Allee; gamma bound hit), 134,167, fronts -61..-78, a degenerate
corner ~15,400 worse (a too-strong start kills the population, gradients point to "remove the brake", the line
search overshoots). Prediction MET for the starts x.3-x5: one strong-Allee basin. Established for this sandbox.

### R37 PRE-REGISTRATION (review D4, J19 fix): A9 with the establishment readout gated by the ABSOLUTE Allee deficit
refit_fields --form single --decouple --estab --estab_gate allee (lowdens_forward fade = 1 - mated_fraction(gamma N)
instead of exp(-(N/K)/rho)); warm start A9 cont (habitat sites + establishment vector); n50 pinned; CPU, maxiter 400.
Prediction (review): G's time change shrinks and the posterior worsens by > 1,000 vs A9 (136,395) -- the N/K-gated G
was doing fill-in; falsified if both the posterior and G's rise hold.

### R30 (n50 free) first pass -- NOT CONVERGED (maxiter 400 cap, 461 evals, |grad| ~3e5, still gaining ~30/eval)
-log posterior 183,081 -> 145,560 (A9 136,395; R28b 136,267), BBS 119,641, n50 x1.61, G constant 1.78 (b ~ 0 as
designed), offsets -16/-20/-21/-17, never .090. No verdict: continuations queued on GPU -- (c1) from these params,
maxiter 800; (c2) same sites with n50 started in T0's basin (x13). For scale: the sandbox ungated control (R34 i,
static per-cell fecundity + n50 x6.5, K real) reaches BBS 110,608, so a static-growth solution near the fit exists
in the sandbox's richer parameterisation; whether the habitat-mapping refit finds one is what R30 asks.
R34 (ii) absolute Allee gate WITHOUT the colonisation clock (GPU, 1,959 evals, cap): BBS 111,611, prior 749, n50 x7.3,
Fmax x1.31, offsets +1/0/-2/-7, never .104. Ordering: ungated 110,608 < gate-no-clock 111,611 < T1 (gate + clock)
113,674. Read (likely): the clock is the main restriction (~2,060); the Allee gate costs ~1,000 vs ungated. Per-cell
corrections that act whenever a cell is sparse (any year) do almost as well as unrestricted static ones.
R34 (iii) UNGATED static per-cell map with n50 PINNED (GPU, 776 evals): BBS 122,889, Fmax x1.19, offsets
-48/-50/-33/-20 (EARLIER than uncorrected), never .098. vs n50 free: 110,608, fit-like fronts.
Read (established for this sandbox): the static map alone cannot time the fronts -- it buys abundance (~10,900 over
uncorrected) by pushing fronts EARLIER; the stronger Allee (x6.5) is what holds fronts back, and together they nearly
reproduce the fit (12,300 better than the map alone). Both ingredients are needed; neither alone suffices.
R34 (iv) UNGATED static per-cell map, n50 free, under the FULL posterior (GPU, 2,074 evals, cap): n50 x6.1, Fmax x1.72,
correction prior 174, BBS 110,742 (fit 110,045), offsets +3/-1/-3/-7, never .100. Observed-data terms: all obs rows
-112,760 (fit -112,058: +702); NestWatch -1,924 (fit -1,966: BETTER than the fit). By region: East 61,222 / Plains
5,663 / West 43,857 (fit 60,764 / 5,658 / 43,623).
Read (likely; sandbox, frozen growth readouts, K real): under ALL the data the fit's posterior uses -- counts at every
latitude and the NestWatch fecundity likelihood -- a model with NO time change in survival or fecundity, a ~6x
stronger absolute Allee brake (n50 ~.36 route counts) and a small static per-cell fecundity map comes within ~700
log-lik of the fit, with the fit's front timing and slightly better fecundity data fit. NestWatch does not veto
Fmax x1.72 once the static map redistributes it.
What it does NOT show: (1) that this is better than the fit (it is ~700 worse on data, before counting the per-cell map's
flexibility -- 4,318 numbers vs the fit's habitat mapping); (2) held-out performance; (3) that n50 ~6x the V&L-derived
pin is ecologically plausible (the user's prior favours weaker); (4) anything in bird units. Implication to REPORT:
lambda inflation in [cur] is not demanded by the data as such; it is one of two near-equivalent explanations of the
counts and fronts, the other being a stronger Allee + static spatial heterogeneity in growth. The n50 pin (E055)
selects between them.

### R38 PRE-REGISTRATION: can the static map be built ON Z? (user's middle-ground request; review D6 iii)
timing.py --corr --zmap --gate none --tau 1e9: one static fecundity multiplier per cell = exp(Zstd . (.1 b)), Zstd =
standardised 1966-2025 mean of Z (64 dims), b ~ N(0,1) (64 numbers instead of 4,318); frozen growth readouts, K real.
Arms: n50 free (BBS); n50 free (--full); n50 pinned (BBS). CPU 20-23.
Prediction: if the needed static heterogeneity is habitat (expressible by the encoder), the Z map with n50 free gets
most of R34 (i)'s gain (110,608): within ~2,000. If it stays near T0 (118,763), the static map is NOT a Z-readable habitat
pattern (place-specific idiosyncrasy or a misfit of the frozen readouts' Z mapping).
R30 c1 (GPU, from the capped params): stopped on scipy's relative-reduction test at 85 evals with |grad| ~12,000 --
premature. -log posterior 143,861 (A9 136,395), BBS 117,977, n50 x2.05, G const 1.92, offsets -13/-13/-11/-11, never .073.
Still no verdict; a --tight continuation queued (refit_fields gains --tight).
R30 c2 (GPU; c1's habitat sites with n50 STARTED at x13; 872 evals, iteration cap, still improving):
-log posterior 369,218 -> 138,939 (A9 136,395: +2,544; R28b 136,267: +2,672; fit 135,591: +3,348), BBS 112,811,
n50 x7.7, offsets 0/-2/-3/-7 (fit-like), G constant. vs c1 (n50 basin x2): 143,861 -- the strong-Allee basin is
~4,900 better in the FULL-posterior habitat refit too. Pre-registered rule (>= 3,000 worse than R28b = growth must change;
within ~1,000 = falsified): currently +2,672 and still falling -- in between; tight continuation c4 running.
R30 PINNED arm (CPU, 466 evals, cap, not converged): -log posterior 144,923, BBS 119,129, offsets -20/-25/-25/-17.
With n50 pinned and no time change in growth, fronts stay 17-25 yr early (cf. free-n50 c2: fit-like at 138,939).

### R39 PRE-REGISTRATION (diagnostic only; static habitat is not a model candidate -- premise memory): R38's static
Z map + free n50 with K ALSO frozen (no habitat change anywhere). Prediction: much worse than R38 (mix_replay: K's change
carries ~67% of the abundance gain), quantifying what K's time change alone contributes once Allee + static map are free.

### R37 RESULT: A9 with the establishment readout gated by the ABSOLUTE Allee deficit (n50 pinned; CPU, 500 evals, cap)
-log posterior 171,281 -> 138,309 (A9 N/K-gated 136,395: +1,914; still improving at the cap), BBS 112,707, offsets
+3/-1/-3/-9, never .027.
G (multiplier on fecundity of mate-limited birds): East 1940 .128 -> 1960 .126 -> 1980 .189 -> 2000 .475 -> 2020 .749
(x5.8); West .270 -> .941 (x3.5). Under the absolute gate G is BELOW 1 throughout: it SUPPRESSES reproduction in sparse
cells, strongly early and less later.
Prediction (G's time change shrinks AND posterior worsens > 1,000): posterior part MET (+1,914, unconverged); G part
FALSIFIED in the multiplicative sense (x5.8 vs x4.6) -- but its MEANING flipped: with the J19 fix the readout acts as an
extra, time-varying Allee brake -- strong early (fecundity of mate-limited birds x.13 in 1940-60) and relaxing over time.
Read (likely): with n50 pinned, the decoupled refit uses the establishment readout to make the Allee effect ~8x
stronger EARLY and ~1.3x later -- the same strong-early-brake the free-n50 runs find (R26/R30/R34), here implemented
as a declining brake. That a brake relaxing through calendar time is preferred to a constant one (R30 c2, constant
strong Allee: 138,939 vs 138,309 here, both unconverged) is a small difference; whether the relaxation is real (an
evolving / acclimatising eastern population, feeders) or another absorber is the open question.

### R40 PRE-REGISTRATION: R37 + n50 FREE (absolute-gated establishment readout, decoupled, warm start R37, --tight)
Prediction: if R37's declining brake only compensated the pinned n50, freeing n50 makes n50 rise (x5-8) and G's
time change flatten (G East 1940/2020 ratio << 5.8); if G still rises several-fold, the data prefer a brake that
RELAXES through time (or another absorber does it), beyond a constant stronger Allee.
R30 c3 (tight, from c1; 1,000-iteration cap, 1,115 evals, still improving ~0.1/eval): -log posterior 138,136,
BBS 112,250, n50 x4.8, G const 1.83, offsets -2/-3/-3/-8, never .041 (fit .068). Gap: A9 +1,741; R28b +1,869; fit +2,545.
The free-n50 constant-growth refit keeps closing (145,560 -> 143,861 -> 138,136) and has moved from the weak-Allee
basin toward x4.8. Pre-registered rule: >= 3,000 worse than R28b = growth must change; within ~1,000 = falsified --
now +1,869 and falling slowly: neither threshold reached; c4 (from the x13-started c2) running.
R30 c4 (tight, from the x13-started c2; 1,000-it cap, still improving): -log posterior 137,900, BBS 112,007, n50 x7.2,
G const 2.05, offsets +1/-1/-3/-7, never .049. Potential lambda CONSTANT at 1.081 (Sa .515, Sj .168, Fmax 3.37) -- the
fit's own 2010 level, held from 1940 on: no inflation by construction.
Gap: R28b +1,633; A9 +1,505; fit +2,309. R30 summary (likely; unconverged, two basins converging from both sides,
c3 138,136 at x4.8 and c4 137,900 at x7.2): the full posterior with refitted habitat mapping and a free, ~5-7x
stronger Allee effect can do without ANY time change in survival or fecundity at a cost of ~1,500-2,300 log posterior
(above the ~1,000 noise rule, well below the pre-registered 3,000 "growth must change" line). Fronts fit-like,
never-reached better than the fit. So the fitted lambda inflation buys ~2,000 log posterior over a constant-growth,
strong-Allee alternative -- real but modest, and contingent on the n50 pin.

### R39 RESULT (diagnostic): static Z map + free n50 with K ALSO frozen (no habitat change anywhere; 299 evals, stop)
BBS 144,397 (all-frozen start 178,022; R38 with K real ~112,500), Fmax x8.5 (!), n50 x.72, offsets -58/-67/-79/-85.
Read (established for this sandbox): without K's time change, no constant growth / Allee / static-map combination fits
-- the optimizer flees to a fast, weak-Allee corner with fronts 60-85 yr early. K's real change is indispensable here;
the substitution findings above concern only the growth-rate part of the habitat change, never K (consistent with
mix_replay: K carries most of the abundance gain). Static habitat remains a diagnostic only (premise memory).

### R41 PRE-REGISTRATION (review D9 / plan F3): zero share by region x era for the JSON-expressible arms
arms_readout.py on fit, frozen K-real pinned, T0, a = .5 free, R33 (R35's arms.json). Prediction (review): strong-Allee
arms over-predict zeros in the modern East (thin populations); falsified if their predicted modern zero shares match
the observed as well as the fit's.

### R41 RESULT: zero share (observed / predicted P(0)) by region x era
| arm | East 66-79 | East 80-93 | East 94-09 | East 10-25 | Plains 10-25 | West 10-25 |
|---|---|---|---|---|---|---|
| observed | .977 | .695 | .446 | .488 | .674 | .528 |
| fit | .980 | .750 | .536 | .709 | .794 | .701 |
| frozen K real, pinned | .635 | .505 | .558 | .792 | .883 | .722 |
| T0 (n50 x13.3) | .950 | .734 | .560 | .755 | .900 | .728 |
| a = .5 free | .968 | .737 | .552 | .738 | .863 | .720 |
| West+Plains fit (x2.9) | .771 | .530 | .547 | .781 | .888 | .721 |
| R33 (Z_disp real, x8.7) | .943 | .732 | .544 | .747 | .880 | .719 |
Read (established for these replays): EVERY arm, the fit included, over-predicts modern zeros badly (East 2010-25
.71-.79 vs .49; West .70-.73 vs .53) -- a shared defect of the base, not of the strong Allee. Strong-Allee arms add
~+.05 in the modern East and ~+.1 in the modern Plains over the fit (thinner populations), but match the 1966-79 East
(.95 vs .98) where the pinned frozen arm fails badly (.64: birds present decades early). Review prediction MET in
direction, small in size next to the shared defect.
R30 c5 (tight, from c4; 1,500-it cap; plateaued: last steps ~.003/eval, |grad| ~3,400): -log posterior 137,680, BBS
111,796, n50 x8.0 (~.47 route counts), offsets +1/-1/-3/-7, never .050. Constant rates Sa .489 / Sj .163 / Fmax 3.47
(potential lambda 1.054 from 1940 on).
R30 VERDICT (likely; plateau, single basin reached from two starts): under the FULL posterior with refitted habitat
mapping, a model with NO time change in survival or fecundity (K still real) and an ~8x stronger Allee effect is
1,285 worse than A9, 1,413 worse than R28b, 2,089 worse than the fit. Above the ~1,000 noise rule, below the
pre-registered 3,000 line: NEITHER outcome cleanly -- the growth-rate time change ("lambda inflation") is worth ~1,300-
2,100 log posterior over a constant-growth, strong-Allee alternative that reproduces the fronts and beats the fit on
never-reached. Under the n50 pin, the same comparison is ~7,200+ (R30 pinned arm 144,923, unconverged).
R30 pinned, tight continuation (GPU, 1,500-it cap, |grad| ~8e4, still improving ~2/eval): -log posterior 140,803,
BBS 114,801, offsets -13/-8/-7/-11, never (see summary). vs free-n50 c5 137,680: the pin costs >= ~3,100 under constant
growth (an upper bound on that cost shrinks as this keeps improving; fronts still 7-13 yr early pinned vs fit-like free).

### R38 RESULT (i): static Z-readout fecundity map (64 coefficients) + free n50, frozen growth, K real, BBS (CPU, converged)
BBS 112,503 (fit 110,045: +2,458; per-cell map R34 i 110,608; T0 no map 118,763), prior 13, n50 x5.7, Fmax x1.63,
offsets 0/-1/-2/-6, never .091. Regions: East 62,106 / Plains 5,847 / West 44,549 (fit 60,764 / 5,658 / 43,623).
Map: cell u rms .50 (East mean +.07, Plains +.40, West +.03).
Prediction MET (within ~2,000 of R34 i: +1,895): the static heterogeneity the alternative needs is mostly READABLE FROM Z
-- 64 encoder coefficients recover ~75% of what 4,318 free per-cell numbers do (gain over T0: 6,260 of 8,155).
Read (likely): a middle ground built on Z exists: STATIC habitat (time-mean Z) read into fecundity + a ~6x stronger
absolute Allee + K from the real (time-varying) Z reproduces fronts and comes within ~2,500 of the fit on BBS rows with
NO time change in survival or fecundity. Full-posterior (_zmap_full) and pinned (_zmap_pin) arms running.
R38 (iii) static Z map + free n50 under the FULL posterior (GPU, converged, 935 evals): n50 x5.2, Fmax x1.64, BBS 112,576,
offsets 0/-1/-2/-6, never .087. All obs rows -114,630 (fit -112,058: +2,572); NestWatch -2,217 (fit -1,966: +251).
Read (likely): the Z-built middle ground holds under all the data: ~2,800 log-lik from the fit (obs + NestWatch), with
the fit's fronts, NO time change in survival / fecundity, an ~5x stronger Allee and 64 static Z coefficients.
R38 (ii) static Z map with n50 PINNED (GPU, converged): BBS 124,275, Fmax x3.97, offsets -34/-41/-46/-29, never .091.
vs n50 free 112,503 (fit-like fronts). As with the per-cell map (R34 iii): the static Z map alone cannot time fronts
(pushes them as early as uncorrected); the stronger Allee is indispensable for the middle ground.

### R40 RESULT: decoupled refit, absolute-gated establishment readout + FREE n50 (CPU, 800-it cap, 884 evals)
-log posterior 136,151 -- BEST decoupled structure so far (A9 136,395: -244; R28b 136,267: -116; R37 138,309; fit
135,591: +560). BBS 110,542 (fit 110,045), offsets +3/0/-2/-6, never .068 (= fit). n50 x6.3 (~.37 route counts).
Sa .410 / Sj .146 / Fmax 2.67 CONSTANT (potential lambda .80 at all times: no inflation in the vital rates).
G (multiplier on fecundity of MATE-LIMITED birds only): East 1940 .92 -> 1960 .96 -> 1980 1.43 -> 2000 3.35 -> 2020 5.85
(x6.4); West 1.46 -> 5.69 (x3.9).
Prediction (G flattens once n50 is free) FALSIFIED: with a ~6x stronger constant Allee, the data still want the
fecundity of sparse, mate-limited birds to rise ~6x through calendar time -- i.e. an Allee / establishment brake that
RELAXES over time, East more than West, starting ~1970-80.
Read (likely): this is the establishment / low-density decomposition the user asked for. In this structure the whole
time change sits in ESTABLISHMENT (reproduction at sparse density, gated by the absolute Allee deficit), while
survival, fecundity at density, and the settled level carry none; K's real change carries abundance. It fits within
~560 of the fit (inside the 1,000 noise rule) with the fit's never-reached share. What could relax establishment
through time (unknown): feeders (winter food at low density), behavioural / genetic adaptation of the eastern lineage
(Badyaev 2009: new populations start as sinks), the 1960s-70s spread of partial migration (Able & Belthoff),
or habitat change the encoder places in establishment. G is a readout of REAL time-varying Z here (A9 design), so its
rise is Z-driven -- whether Z's eastern signal is regional trend vs habitat is the E1-E4 question (blocked on TACC files).

## CORRECTIONS from the independent review of ROUND 2 (subagent, 2026-11-04 ~03:00; numbers verified against raw JSON)
1. MAIN CONFOUND: every decoupled refit (A9, R28b, R30, R37, R40) and every timing.py arm with alpha 0 (T0, R34, R38)
   freezes Z_disp (dispersal habitat) as well as Sa/Sj/Fmax readouts (edit_habitat default keys); the fit keeps Z_disp's
   real change, worth ~2,290 (R33) -- about the size of the gaps credited to lambda inflation (R30 +2,089, R38 +2,800,
   R40 +560). J21 item 1 was NOT absorbed for the decoupled family. Inflation's worth is an UPPER BOUND until rerun.
2. R30: comparison is an upper bound (only 139 habitat sites refitted; other latents co-adapted with the fit; ridge SD
   1e-6 badly conditioned). "Plateau / single basin" NOT shown: c5 still falling (39 nats over its last 500 evals,
   |grad| 3,400); c3 (x4.8) never continued. Regional split of R30's 1,751 BBS gap: East +912, Plains +132, WEST +707 --
   freezing growth costs ~700 in the native range (points to some real habitat change in growth rates; G4, M6).
   R30's constant G 2.18 gives sparse-cell potential lambda ~1.72 braked by x8 Allee: the §0.3 knife edge.
3. R32 mislabel: "native West+Plains x2.9" -- the West got WORSE (44,573 vs 44,407); the whole gain was in the Plains
   (a contact zone invaded during the BBS era). Not a native-range value.
4. R34: within ~700 says nothing without held-out data or the same map added to the fit (4,318 weakly shrunk numbers;
   inherits the fit's K and coefficients); the ungated map is a generic static abundance+growth absorber. Never .100 vs
   fit .068 (J18). R34 (ii) no-clock: East u +.33, Plains +.36 -- sign FLIPPED vs T1, unreported; and at n50 x7.3 the
   Allee gate stays >= 10% open to ~1.9 route counts (J19 via gate width).
5. R35 anchors misused: the 1962-71 CBC NE party-hour index must be compared with the north-east box (fit .242, T0 .307,
   a=.5 .290, WP .232, R33 .293 -> FIT closest), not the East total; the 1951-66 release-area anchor ~.24 (plan §0.7, F1)
   exists (fit .011, T0 .139, R33 .170, WP .271 -> alternatives closer); 1941-51 is initialisation for [cur] (init 1951)
   and must not be read. Net: unknown -- 1950s favour alternatives, 1960s mildly the fit.
6. R37 "posterior part met (+1,914)": upper bound, unconverged (still ~1/eval). R39 "established" -> likely (line-search
   corner like R36 x30).
7. R40: "best" is within noise (-116 / -244 vs R28b / A9; c1 136,147.65); 67 more parameters than the fit and still
   560 below. It RESTATES R37 along the n50-G ridge: low-density reproduction ~ G/n50 -> R37 .128 -> .749 (x5.8), R40
   .145 -> .93 (x6.4): effective low-density Allee ~7x the pin around 1940 relaxing to ~the pin by 2020.
   NOT "establishment": fecundity = Fmax (1 + (G-1) D) m peaks at 1.76 Fmax at m ~ .6, N ~ .5 route counts in 2020 -- a
   hump-shaped density BONUS; gate >= 10% open to N ~ 1.66. Median eastern potential lambda without G is .80 -> crowding
   c = 0 -> K does NOT regulate there; equilibrium ~1.2 route counts set by G and n50. "K carries abundance, G carries
   establishment" is likely FALSE for median cells (same for A9 lambda .78, R37 .88). "No inflation in the vital rates"
   is a G13 mislabel: at N ~ .5, lambda rises .64 (1940) -> 1.10 (2020), crossing replacement -- inflation moved INSIDE
   the Allee zone.
8. R41 omitted the arms that matter (R30, R34, R38, R40).
9. Pin plausibility: .059 = V&L n50/K* .0197 x 51a's median K 3.0 (E055) -- [cur]'s K* never measured; under the
   overlay's soft prior, n50 x6-x8 costs only ~55-70 nats. Only a hard pin keeps n50 low.

## ROUND 3 (2026-11-04, from the round-2 review) -- PRE-REGISTRATIONS
R42 (review 1, decisive): the decoupled family with Z_disp REAL (refit_fields --decouple_keys Z_gathered: only
  Sa/Sj/Fmax readouts frozen; K real; dispersal habitat real). Warm starts: A9 cont; R30 c5 (constant growth, free n50,
  ridge 1e-6); R40 (absolute-gated G, free n50). --tight, maxiter 1000. edited_inputs now recorded in summaries.
  Prediction: R30's gap to the fit shrinks by >= 1,000 (R33: Z_disp worth ~2,290); falsified if < 300.
R43 (review 2, dimension-matched): the SAME static maps added to the FIT (timing.py --alpha 1 --k_real --corr --gate
  none --tau 1e9; per-cell and --zmap; n50 free and pinned). Prediction: the fit + map gains >= 1,000 (restoring a gap
  to R34 iv of >= 1,500); falsified if < 300 (the map adds information only when growth is frozen).
R45 (review 7): --fit_regions west and plains separately (n50 start x.3). Prediction: West alone picks <= x1.5;
  falsified if >= x2.5.
Readouts (review 5, 6, 10) follow: K* re-measure of the pin; lambda(N, t) curves, gate share and c = 0 share; NestWatch
  exposure of R40's gate.

### R44a RESULT (review 2 item 6): the pin re-measured on [cur] (kstar.py; model units, no conversion)
K* = median capacity (real Z, 2010-25) over BBS-detected cells: East .161, Plains .062, West .082 (model units);
n50 (pin) = .00296. n50 / K*: East .018, Plains .048, West .036 (V&L .0197, their range .0066-.033).
Fitted multipliers as n50 / K* (East / Plains / West): x2.9 -> .053 / .138 / .105; x5.2 -> .096 / .25 / .19;
x6.3 -> .116 / .30 / .23; x8 -> .147 / .38 / .29; x13.3 -> .245 / .63 / .48.
Read (established): the pin IS V&L-consistent in [cur]'s own East (.018 vs .0197) -- the transferred calibration
happens to hold there (and is already above V&L's top in the Plains / West). The review's alternative (K* >> 3 so
x6-8 might be in range) is REFUTED: every strong-Allee solution (x5-8) sits 3-4.5x above V&L's top in the East, and
x13 ~7x. The user's prior (weaker Allee) and the literature both weigh against the constant-growth alternatives.

### R44b RESULT (review 2 item 5): low-density growth lambda(N, t) in the MEDIAN eastern cell (analytic, from the
refits' saved median rates, G_east(t), n50, k = 3; N in route counts; East K* = 3.2 route counts -- E055's assumed 3.0
holds). lambda = Sa + Sj Fmax (1 + (G-1) D) m(N).
| arm | crowding c | 1940: N .05 / .5 / 2 | 2020: N .05 / .5 / 2 | gate <= 10% at |
|---|---|---|---|---|
| A9 (N/K gate, pinned) | 0 | .50 / .68 / .78 | .79 / 1.11 / .87 | 2.2 rc |
| R37 (Allee gate, pinned) | 0 | .44 / .84 / .88 | .52 / .85 / .88 | .30 rc |
| R40 (Allee gate, n50 x6.3) | 0 | .44 / .62 / .76 | .60 / 1.10 / .92 | 1.9 rc |
| R30 c5 (constant G, n50 x8) | > 0 | .57 / .95 / 1.05 | same (constant) | 2.2 rc |
Read (established for these readouts; review CONFIRMED): in A9, R37 and R40 the median eastern cell has c = 0 (no
crowding brake: Fmax Sj / (1 - Sa) < 1), is a sink at EVERY density before ~1980, and in 2020 has lambda > 1 only in a
density window around .2-1 route counts, falling below 1 by ~2 -- so equilibrium abundance is set by G and n50, not K.
"K carries abundance, G carries establishment" is FALSE for median cells in the decoupled family; G is a density-
dependent growth hump inside the Allee zone. Lambda inflation in these structures has moved INTO that hump (R40 at
N .5: .62 -> 1.10). R37's median cell is a sink at all densities in all years (lambda max .88): abundance there is
sustained by immigration. Only R30 (constant growth, strong Allee) keeps a normal crowding brake.
R26 T1 CPU replicate: BBS 113,678 vs GPU 113,674 (n50 x2.96 vs x2.94) -- replicates within 4.

### R46 PRE-REGISTRATION (review 2 item 3): time-block held-out test -- train on BBS rows <= 1993, score 1994-2025
refit_fields --train_until 1993 (held-out obs rows removed from the full log density; NestWatch kept), --tight:
 (i) the fit's structure (single field, all rates time-varying) from the fit;
 (ii) R30 structure (decoupled, constant G, free n50, Z_disp REAL) from R30 c5;
 (iii) R40 structure (decoupled, Allee-gated G, free n50, Z_disp REAL) from R40.
Held-out score = BBS log-lik of the 1994-2009 + 2010-2025 eras (25-49N, by region). Caveats: starts and the
non-habitat latents (route / observer / disease / init) have seen the held-out data -- equally for all arms.
Prediction: if inflation encodes real habitat-driven growth change, (i) beats (ii) on 1994-2025 by more than their
in-sample gap (~2,000); falsified if (ii) or (iii) is at or better than (i) on held-out rows.
R40 c1 (tight continuation): 136,143 at eval 193 (from 136,151), then STOPPED by me to free cores for R46 -- within noise.

### R42 RESULT (A9 arm): A9 decoupled structure with Z_disp REAL (edited_inputs Z_gathered; GPU, 1,000-it cap, |grad| 16)
-log posterior 136,531 (start) -> 136,169 (A9 with Z_disp frozen 136,395: -226; fit 135,591: +578), BBS 110,582,
offsets +2/-1/-3/-6.5, never .047. G (N/K gate) East .65 -> 2.06 (x3.2; A9 frozen-Z_disp x4.6), West .87 -> 2.02 (x2.3).
Read (likely): keeping dispersal habitat real takes ~a third of G's eastern rise (x4.6 -> x3.2) but improves the
posterior only ~226 -- the Z_disp share credited by R33 (~2,290, a 2-parameter sandbox) mostly does NOT carry over to
the refitted decoupled structure; A9's remaining gap to the fit (~580) is inside the noise rule. Review prediction for
R30 (gap shrinks >= 1,000) pending its arm.
R42 (R40 arm): Allee-gated G + free n50 with Z_disp REAL (GPU, 1,000-it cap, |grad| 132): -log posterior 135,982
(R40 frozen Z_disp 136,151: -169; fit 135,591: +391, inside the noise rule), BBS 110,414, n50 x5.1, Sa .38 / Sj .14 /
Fmax 2.69 constant, offsets +3/0/-2/-6, never .066. G East .69 -> 5.28 (x7.7), West 1.64 -> 5.82 (x3.5): the rise in
low-density reproduction does NOT shrink with Z_disp real (x6.4 -> x7.7) -- Z_disp was not what G stood in for.
Read (likely): with dispersal habitat real, the decoupled time-varying low-density structure is within ~390 of the fit
(noise); G's rise is robust to the Z_disp confound.

### R43 RESULT (i): the FIT + the same static per-cell map (alpha 1, n50 + Fmax free; GPU, 1,752 evals, cap)
BBS 109,436 (fit 110,045: gain 609), prior 129, n50 x1.15, Fmax x.99, offsets = fit's, never .061. Regions East 60,540 /
Plains 5,559 / West 43,337 (all better than the fit by 224 / 99 / 286).
Prediction (fit gains >= 1,000; falsified if < 300): IN BETWEEN (609). With matched flexibility the comparison is
fit + map 109,436 vs frozen-growth + map + strong Allee (R34 i) 110,608: the fit structure leads by ~1,170 BBS
(vs ~560 without the map). Read (likely): the static map is partly generic spatial-misfit absorption available to
either structure (~600 for the fit), and partly specific to the frozen model (it closes much more there); with equal
flexibility the time-varying-growth fit keeps a modest lead (~1,200 BBS) -- n50 stays near the pin (x1.15) when growth
can change.
R43 (ii) fit + map, n50 PINNED: BBS 109,439 (free 109,436) -- with growth free in time, freeing n50 is worth ~3. Confirms
R28a: the Allee strength is unidentified when growth can change; it matters only where growth is frozen.
R43 (iii) fit + static Z map (64 coef), n50 free: BBS 110,019 (fit 110,045: gain 26), n50 x.96 -- the fit already
uses everything a static Z readout of fecundity can offer. vs frozen growth + Z map + strong Allee (R38 i, 112,503):
with matched Z flexibility the fit leads by ~2,480 BBS. Read (likely): the Z-built middle ground (R38) is NOT
dimension-matched-equivalent to the fit: the static Z map closes 6,260 for the frozen model but nothing for the fit,
because it re-creates the spatial part of the fit's own habitat readout that freezing removed. The fit's ~2,500 lead
under equal Z access is what the TIME change in Z-driven growth rates buys on in-sample BBS.

### R46 RESULT (i): the FIT's structure trained on <= 1993 (CPU, 632 evals, ABNORMAL line-search stop)
Training eras (1966-93) BBS -log-lik 27,911 (full-data fit on the same rows: 28,028); HELD-OUT 1994-2025: 83,142
(full-data fit on the same rows: 82,016 -> held-out cost of not seeing them: +1,126). By era/region: East 94-09 30,634,
10-25 17,949; Plains 2,881 / 2,032; West 18,157 / 11,489. R30 / R40 arms running; the comparison is their held-out
1994-2025 totals vs 83,142.

### R44c RESULT (review 2 item 4a): what drives the rising low-density readout G (gdecomp.py; log G = g0 + Zstd . b)
| structure | period | continental clock dlogG | East total (minus clock) | Plains (minus clock) | West (minus clock) | East local sd |
|---|---|---|---|---|---|---|
| R40 | 1940-2020 | 1.60 | 1.76 (+.16) | 2.07 (+.47) | 1.34 (-.25) | 1.09 |
| R42-R40 (Z_disp real) | 1940-2020 | 1.56 | 1.87 (+.31) | 1.85 (+.29) | 1.22 (-.33) | .94 |
| A9 | 1940-2020 | 1.44 | 1.53 (+.09) | 2.17 (+.74) | 1.18 (-.26) | 1.14 |
East rise by Z dimension (R40): dims 14 (.54), 19 (.29), 13 (.23), 59 (.16), 10 (.11), 18 (.08); top six = 57% of
the absolute contribution; the same dims (14, 19, 13) lead in every structure.
Read (established for these readouts): ~85-95% of G's eastern rise is a CONTINENT-WIDE uniform trend in Z (the same
yearly shift everywhere in the 25-49N band); the East-specific part is small (+.1 to +.3 in log), the Plains get a bit
more, the West a bit less; within-region spatial variation of the change (local sd ~1) is large but averages out.
What it does NOT show (memory: regional trend is not dynamics; M26): whether that continental trend in the community
prior is real continent-wide habitat change (e.g. built-up / feeder-related change the community responds to) or a
time artefact of the encoder (survey effort, observer, a drift in the all-species prior) -- the E1-E4 attribution on
the native 54 km DESK field (blocked on the TACC files) is what decides it. Implication to REPORT: in every structure
tonight, the time change that the counts want rides on a few Z dimensions' continental clock.

### R47 PRE-REGISTRATION (review 2 item 4b): may the low-density readout G read Z's continental clock?
R42-R40 structure (decoupled, Allee-gated G, free n50, Z_disp real), warm start from it, --tight, maxiter 1000, GPU:
 (i) --estab_z detrended (Z minus its 25-49N yearly mean: G cannot see the uniform trend);
 (ii) --estab_z static (time-mean Z: G constant in time per cell; dimension-matched control for time variation).
Reference: R42-R40 135,982 (G real Z); R30 c5 137,680 (constant G, Z_disp frozen; R42-R30 Z_disp real pending).
Prediction (review): if G's rise is mostly the uniform clock, (i) loses > half of the time-varying gain (vs (ii));
falsified if (i) keeps >= 80% of it.

### R47 RESULT: what the low-density readout G may read (decoupled, Allee-gated G, free n50, Z_disp real; GPU)
| G reads | -log posterior | n50 x | G East 1940 -> 2020 | never | stop |
|---|---|---|---|---|---|
| real Z (R42-R40) | 135,982 | 5.1 | .69 -> 5.28 | .066 | cap |
| Z minus continental yearly mean | 136,482 | 4.8 | 1.84 -> 3.31 | .063 | line-search (ABNORMAL) |
| time-mean Z (static, constant in time) | 136,421 | 6.3 | 2.52 const | .065 | cap |
| fit | 135,591 | 1 (pin) | -- | .068 | |
Prediction MET in direction: removing the continental clock removes ALL of G's time-varying value (detrended is no
better than static). But the time-varying value itself is small: real vs static = ~440, inside the ~1,000 noise
rule. Read (likely): once Z_disp is real and n50 free, a decoupled model with CONSTANT vital rates AND a CONSTANT
(static-Z) low-density readout is within ~830 of the fit's posterior, with the fit's never-reached share and fronts;
what the time change adds rides entirely on Z's continental clock and is worth a few hundred log posterior.
Caveats: unconverged / line-search stops; the readout's gate width scales with n50 (J19 via width); n50 x5-6
(n50/K* ~.09-.12 East, 3-4x above V&L's top, R44a); non-habitat latents still at fit values.
R46 (iv) added: the R47-static structure (constant vital rates, static-Z Allee-gated G, free n50, Z_disp real) trained
<= 1993 and scored 1994-2025, from R47 static params (GPU).

### R46 RESULT (iv) vs (i): time-block held-out, trained on BBS rows <= 1993, scored on 1994-2025 (BBS -log-lik)
| structure | training eras | HELD-OUT 1994-2025 | East | Plains | West |
|---|---|---|---|---|---|
| (i) fit structure (all rates read time-varying Z) | 27,911 | 83,142 | 48,583 | 4,913 | 29,646 |
| (iv) constant vital rates + static-Z Allee-gated G, n50 free, Z_disp real | 28,022 | 86,020 | 50,236 | 5,357 | 30,426 |
| difference (iv) - (i) | +111 | +2,878 | +1,653 | +444 | +780 |
Prediction ("inflation encodes real habitat-driven growth change -> (i) beats (ii)/(iv) on held-out by more than the
in-sample gap") MET for (iv): in-sample full-data gap ~830, held-out gap ~2,880, worse in every region incl. the
native West (+780). Read (likely): the fit's time-varying, Z-driven growth rates FORECAST the 1994-2025 counts better
than the constant-growth / stronger-Allee alternative when both see only 1966-93 counts (Z(t) is a known covariate in
both). So the "lambda inflation" carries real out-of-sample information -- a point for real habitat change in growth
(G4, M6) over a pure absorber. What it does NOT show: that the whole inflation is real (the fit's rise could still be
too large -- the training fit's own held-out cost was +1,126); whether that information is habitat or a continental
clock in Z that happens to continue (R44c: G's rise is ~90% clock). Caveats: starts and non-habitat latents saw the
held-out rows (equally); (i) stopped ABNORMAL at 632 evals, (iv) at the cap; one time split only. R30 / R40 held-out
arms still running.

### R48 PRE-REGISTRATION: a second time split (train <= 2005, score 2006-2025) for (i) fit structure and (iv) static-G
alternative, same settings as R46. Prediction: (i) beats (iv) again on the held-out rows, by more than ~800; falsified if
(iv) is within ~300 or better.
R46 (ii) R30 structure (constant vital rates + constant N/K-gated G, free n50, Z_disp real) trained <= 1993: training
eras 28,453, HELD-OUT 86,618 (East 51,387 / Plains 4,952 / West 30,279) -- vs the fit structure 83,142: +3,476 (East
+2,804, Plains +39, West +633). Same direction as (iv) (+2,878). Two constant-growth alternatives both forecast
1994-2025 worse than the fit by ~2,900-3,500.

### R42 RESULT (R30 arm): constant growth + free n50 with Z_disp REAL (CPU, 1,000-it cap)
-log posterior 137,254 (R30 c5, Z_disp frozen: 137,680 -> -426), BBS 111,471, offsets +1/-1/-2/-6. Gap to the fit
2,089 -> 1,663. Review prediction (gap shrinks >= 1,000; falsified if < 300): NEITHER -- shrank 426. The Z_disp confound
was real but modest in the refitted structures (A9 -226, R40 -169, R30 -426), far below R33's 2-parameter 2,290.
R46 (iii) R40 structure (constant vital rates, Allee-gated G reading REAL time-varying Z, free n50, Z_disp real)
trained <= 1993: training eras 27,968, HELD-OUT 85,939 (East 49,207 / Plains 6,121 / West 30,612) -- vs fit 83,142:
+2,797 (East +624, Plains +1,208, West +966). G East extrapolates .98 -> 7.35.

### R46 SUMMARY: time-block held-out (train <= 1993, score 1994-2025 BBS rows; lower is better)
| structure | held-out | minus fit | East | Plains | West |
|---|---|---|---|---|---|
| (i) fit (all vital rates read time-varying Z) | 83,142 | 0 | 48,583 | 4,913 | 29,646 |
| (iii) R40: constant vital rates, time-varying Z low-density readout, n50 free | 85,939 | +2,797 | +624 | +1,208 | +966 |
| (iv) constant vital rates, static-Z readout, n50 free | 86,020 | +2,878 | +1,653 | +444 | +780 |
| (ii) R30: constant vital rates, constant N/K readout, n50 free | 86,618 | +3,476 | +2,804 | +39 | +633 |
Read (likely; one split, contaminated starts equally): the FIT -- time-varying, Z-driven survival and fecundity, n50 at
the literature pin -- forecasts 1994-2025 best, by ~2,800-3,500 over every stronger-Allee / constant-vital-rate
alternative, in every region incl. the native West. Putting the time change only in a low-density readout (iii)
recovers the East but loses in the Plains and West. In-sample these alternatives were within ~400-1,700 of the fit;
out of sample the gap widens 2-7x. So lambda inflation, as the fit implements it, is not just an absorber that a
stronger Allee could replace: it carries forward-predictive information (whether that information is habitat change
or a continuing continental clock in Z, R44c, is still the encoder question).

### R45 RESULT (West only; a = 0, K real, n50 + Fmax free, start x.3): n50 x.93, Fmax x1.00, West BBS 44,406 (uncorrected
44,407). Prediction (West alone picks <= x1.5; falsified >= x2.5) MET: the native range does NOT want a stronger Allee
(it is at the pin); R32's x2.9 came from the Plains contact zone. The strong-Allee demand is entirely the invasion
(East, and the Plains contact zone).
R45 (Plains only, start x.3): jumped to the gamma lower bound (n50 x.0067 = no Allee) in 17 evals, Plains 6,810 (start
6,821) -- a degenerate corner like R36 x30; uninformative. (R32 WP reached Plains 6,400 at x2.9, so a better
strong-Allee basin exists for the Plains; this start missed it.)

### R48 RESULT: second time split (train <= 2005; scored era 2010-2025 -- 2006-09 rows are out of training but sit in
the 1994-2009 era, so unscored)
| structure | training eras | HELD-OUT 2010-25 | East | Plains | West |
|---|---|---|---|---|---|
| (i) fit structure | 79,051 | 31,115 | 17,703 | 2,021 | 11,391 |
| (iv) constant vital rates + static-Z readout, n50 free | 79,623 | 32,245 | 18,528 | 2,141 | 11,576 |
| difference | +572 | +1,130 | +825 | +120 | +185 |
Prediction ((i) beats (iv) by > ~800 on held-out) MET. Two splits agree: the fit's time-varying, Z-driven vital rates
forecast later counts better than the constant-growth, stronger-Allee alternative (1994-2025: +2,878; 2010-25: +1,130).

## CORRECTIONS from the independent review of ROUND 3 (subagent, 2026-11-04 morning; numbers recomputed from raw)
1. [cur] was FITTED with an observation_mask (overlay grid54_step43_factor_p2_veit_s41.json): years 2021-25 and 31
   pinned 216 km blocks are held out. The harness's fn.log_prob gives those rows 0, so every sandbox "BBS -log-lik" omits
   them: R46's "1994-2025" is 1994-2020 outside the blocks; R48's "2010-25" is 2010-2020. These rows were never in any
   objective, incl. the production fit's -- a clean time + space held-out set, unused so far.
2. R41's zero shares are very likely WRONG: p0 = exp(0) = 1 on masked rows. Corrected arithmetic brings the fit's
   modern predicted zero shares to ~.52 East / ~.50 West / ~.66 Plains vs observed .49 / .53 / .67. "Shared modern zero
   over-prediction, a defect of the base" is withdrawn pending recomputation on unmasked rows.
3. Held-out caveats: NestWatch stays in training (year_range to 2020) and carries post-cutoff density information; the
   non-habitat latents (observer x route, disease from 1993-94, init, dispersal) are held at the full-data fit, tuned
   with the FIT's own state path -- this favours arm (i) (size unknown); "contaminated equally" was wrong. R46 (ii)/(iii)
   started from Z_disp-frozen optima; (ii) ended 713 worse on its own training objective (partly under-convergence).
   Clean comparisons: (iii) +44 and (iv) +104 on training. Own-score degradation when rows are held out: fit +1,126
   (82,016 -> 83,142), R47-static +3,582, R42-R40 +3,705; R48 fit +134, static +1,133. R48's "training eras 79,051"
   includes the held-out 2006-09 rows.
4. Missed: trained only on early years, the alternatives' Allee WEAKENS -- R47-static n50 x6.3 -> x3.0 (<= 1993), x4.2
   (<= 2005); R40 x5.1 -> x3.1. Under constant vital rates the strong Allee is mostly demanded by post-1993 data: a
   device for reconciling early timing with late abundance (likely).
5. R44a: K* = K_flat median over ever-detected cells; equilibrium in the decoupled family is ~1.2 rc (set by G, n50),
   so the alternatives' ratios are even higher; with k = 3 the mated fraction at n50 is .46. Pin V&L-consistent -> likely.
6. R44b: lambda_split medians are over ALL East land cells incl. never occupied (J22): "median cell is a sink / c = 0"
   is not about occupied cells -> unknown for occupied, bird-weighted cells.
7. R44c: clock share 83-94% (not 85-95%); the band mean includes the East itself; against the West as reference the
   East-West difference is .42 / .65 / .35 log -> the rise is ~65-77% shared with the West. "Rides entirely on the
   clock" not shown; front-cell local sd (~1) is as large as the clock.
8. R47: detrended ABNORMAL / static at cap from ~21,000 away -> the pre-registered contrast is uninformative; R46 out of
   sample: time-varying G (iii) 85,939 vs static (iv) 86,020 -> G's time variation worth ~81 (likely: adds ~nothing).
9. R43 is matched in the map but not in Z_disp (R34 i / R38 i ran with Z_disp frozen): the ~1,170 / ~2,480 leads are upper
   bounds (J21/J22 again). R42-R30 n50 x6.5 (unreported).
10. R45: the West curve is nearly flat between x.9 and x1; stronger costs the West -> "West does not want stronger"
   likely; "picks the pin" overstated.

## ROUND 4 (2026-11-04, from the round-3 review)
### R49 PRE-REGISTRATION (review 3 item 1): score the rows NO fit has seen
harness.observe now scores obs_weight == 0 rows (overlay observation_mask: 2021-25 + 31 pinned 216 km blocks) with the
raw distribution (fn.base_dist) -> per region x era loglik_heldout (blocks <= 2020 / 2021-25) and zero shares on
unmasked rows. refit_fields --rescore reads out saved parameters without optimising. Arms: fit; A9 (Z_disp frozen);
R42-A9 / R42-R30 / R42-R40 (Z_disp real); R47 static; R30 c5. Prediction: the fit beats the alternatives on 2021-25 by
more than their in-sample gaps, and on the blocks by > ~300; falsified if the alternatives are within ~300 on the blocks.
R41 zero shares recomputed on unmasked rows.
### R50 PRE-REGISTRATION (review 3 item 2): the fit with the alternatives' freedom (n50 free, prior SD 3) in the R46
held-out setting (train <= 1993). Prediction: the fit's held-out score moves by <= 300; falsified if it worsens by
> 1,000 (then flexibility, not structure, explains the R46 gap).
R49 first readout (fit): zero shares on UNMASKED rows, observed / predicted -- East 2010-25 .528 / .517, 1994-09 .473 /
.467; Plains 2010-25 .681 / .655; West 2010-25 .531 / .501. R41's "shared modern zero over-prediction" WAS an artefact
(p0 = 1 on masked rows): the fit matches modern zero shares within ~.01-.03.

### R51 PRE-REGISTRATION (review 3 item 3): NestWatch leak -- rerun R46 (i) fit and (iv) static with NestWatch truncated
to <= 1993 (SBX_OVERLAY copy, data_loading respects nest_likelihood.year_range). Prediction: the held-out gap changes by
< 500; falsified if it shrinks by > half.

### R49 RESULT: scoring the rows NO fit has seen (the base overlay's observation_mask; -log-lik, difference vs the fit)
| structure (saved full-data params) | in-sample posterior | 31 spatial blocks 1966-2020 (fit 32,105) | E / P / W | 2021-25 (fit 21,557) | E / P / W |
|---|---|---|---|---|---|
| A9 (Z_disp frozen) | +804 | +172 | +322 / -33 / -116 | -75 | -56 / +16 / -36 |
| R42-A9 (Z_disp real) | +578 | +176 | +323 / -33 / -113 | -72 | -34 / +11 / -49 |
| R42-R40 (time-varying low-density G, n50 x5) | +391 | +222 | +259 / +17 / -54 | -38 | +52 / +3 / -92 |
| R47 static G (constant growth + static readout, n50 x6) | +830 | +464 | +383 / +18 / +63 | +186 | +215 / +36 / -66 |
| R42-R30 (constant growth + const G, n50 x6.5) | +1,663 | +392 | +276 / -20 / +136 | +136 | +146 / +32 / -42 |
| R30 c5 (constant growth, Z_disp frozen) | +2,090 | +498 | +398 / -18 / +118 | +150 | +92 / +55 / +3 |
Prediction (fit beats alternatives on 2021-25 by more than their in-sample gaps; on the blocks by > 300): PARTLY MET.
Blocks: the CONSTANT-GROWTH alternatives (R47 static, R30, R42-R30) lose +392 to +498, mostly in the East; the
decoupled structures that keep a time-varying readout (A9, R40) lose only +172 to +222. 2021-25 (a 5-yr forecast after
data the fits saw): all within +-190; the A9 family is marginally BETTER than the fit; constant-growth arms +136 to +186.
Read (likely): on data no model saw, the differences are a few hundred -- far smaller than the R46 time-block gaps
(2,800-3,500, a 27-yr forecast from 1993). Spatial transfer modestly favours time-varying structures over constant
growth (~+400-500, East); short-horizon forecasting does not discriminate. So the evidence against the constant-growth,
stronger-Allee alternative is mainly LONG-HORIZON forecasting (R46/R48) plus spatial blocks, not near-term fit.

### R50 RESULT: the fit with n50 free (prior SD 3) in the R46 held-out setting (train <= 1993; GPU, 1,062 evals)
HELD-OUT 1994-2020 (unmasked) 83,621 (fit pinned 83,142: +479; East +217 / Plains +43 / West +219); n50 -> x.44 (a
WEAKER Allee when the fit sees only 1966-93). Prediction (moves <= 300; falsified if worsens > 1,000): in between
(+479). Flexibility does not explain the R46 gap: given the alternatives' freedom, the fit still beats them by
2,300-3,000 on held-out rows -- and on early data alone it prefers a weaker Allee than the pin (the user's prior).

### R52 PRE-REGISTRATION (review 3 item 4b): is the fit's forecasting edge Z's continental clock?
refit_fields --form single --field_z detrended: the habitat field (all rates AND K) reads Z minus its 25-49N yearly
mean (time-mean level kept); habitat sites refitted. (a) full data (in-sample cost of losing the clock); (b) train
<= 1993, scored 1994-2020 unmasked (compare fit 83,142; static alternative 86,020). GPU.
Prediction if the edge is the clock: (b) loses > half of the fit's 2,878 advantage over (iv) (held-out > ~84,580);
falsified if it keeps >= 80% (held-out <= ~83,720).

### R52 RESULT: the fit with its habitat field reading DETRENDED Z (Z minus its 25-49N yearly mean; GPU)
(a) full data (converged, 906 evals): -log posterior 137,079 (fit 135,591: +1,488), BBS 111,368 (+1,323); potential
lambda 1940 -> 2010 .314 -> .526 (+.21; fit .207 -> 1.081, +.87). In-sample, the fit can drop Z's continental clock at a
cost of ~1,500 and most of its lambda inflation disappears with it.
(b) trained <= 1993 (1,211 evals, ABNORMAL stop): HELD-OUT 1994-2020 87,812 -- fit 83,142: +4,670 (East +875 /
Plains +239 / WEST +3,556); WORSE than every constant-growth alternative (static 86,020; R30 86,618).
Prediction (loses > half of the fit's 2,878 edge) MET, and then some. Read (likely): the fit's long-horizon
forecasting edge IS Z's continent-wide trend: without it the fit forecasts worst of all, above all in the native West.
Caveat: detrending the field removes the clock from K as well as from Sa / Sj / Fmax (one shared habitat field), so (b)
cannot say which channel carries the forecast; the West's loss suggests K / abundance trends there. So: the forecasting
information that "lambda inflation" carries is a continental trend in the community prior -- real continent-wide
habitat change, or an encoder time artefact; deciding that is the E1-E4 question (TACC files).

### R53 PRE-REGISTRATION: which channel carries the forecast -- vital rates' clock or K's?
Decoupled: K reads REAL Z (with its clock); Sa / Sj / Fmax read DETRENDED real Z (spatial change kept, continental clock
removed); constant establishment level (--estab --ridge_sd 1e-6); n50 pinned; Z_disp real; habitat sites from the fit.
(a) full data; (b) train <= 1993, scored 1994-2020 unmasked. GPU.
Prediction: if K's clock carries the forecast, (b) is close to the fit (83,142, within ~1,000); if the vital rates'
clock does, (b) is close to R52b (87,812).

### R53 RESULT: vital rates read DETRENDED Z, K reads REAL Z (decoupled, constant G, n50 pinned, Z_disp real; GPU, caps)
(a) full data: -log posterior 136,809 (fit +1,218), BBS 111,150, never .043; potential lambda .377 -> .762 (+.39; fit
+.87) -- the vital rates still rise through time from Z's REGIONAL / local change (East-specific), but by less than half.
(b) trained <= 1993: HELD-OUT 1994-2020 84,365 (fit 83,142: +1,223 -- East +65 / Plains +95 / West +1,063).
| HELD-OUT 1994-2020 | total vs fit | East | Plains | West |
|---|---|---|---|---|
| fit (clock in K and vital rates) | 0 | 0 | 0 | 0 |
| R53: clock in K only | +1,223 | +65 | +95 | +1,063 |
| R52: no clock anywhere | +4,670 | +875 | +239 | +3,556 |
| static-G alternative (constant vital rates, stronger Allee) | +2,878 | +1,653 | +444 | +780 |
Prediction: (b) close to the fit if K's clock carries the forecast -> MOSTLY MET (+1,223, ~26% of R52's loss).
Read (likely): ~3/4 of the fit's long-horizon forecasting edge is K's continental clock (abundance trend), ~1/4 the
vital rates' clock -- and that quarter is almost entirely in the native WEST (+1,063); in the EAST, vital rates without
the continental clock forecast as well as the fit (+65). The East's invasion-era lambda inflation is therefore NOT what
forecasts the East; what the East needs from vital rates is their regional / local (non-clock) time change. Caveats:
unconverged caps; one split; latent asymmetry (review 3 B3).

### R51 RESULT: NestWatch leak removed (NestWatch truncated to <= 1993; train <= 1993; CPU, caps)
| HELD-OUT 1994-2020 | NestWatch to 2020 (R46) | NestWatch to 1993 (R51) |
|---|---|---|
| fit structure | 83,142 | 83,655 (+513) |
| static-G alternative (constant vital rates, stronger Allee) | 86,020 | 92,198 (+6,178; East 54,963 / Plains 6,350 / West 30,885) |
| gap alternative - fit | +2,878 | +8,543 |
Prediction (gap changes < 500; falsified if it shrinks by > half): neither -- the gap TRIPLED. The post-1993 NestWatch
records were propping up the alternative (they pin its fecundity, which its constant-vital-rate structure otherwise
mis-sets when it sees only early counts), not the fit. Read (likely): with no post-cutoff information of any kind,
the constant-growth, stronger-Allee alternative forecasts much worse than the fit; the NestWatch leak had been
favouring the alternative. Caveats: caps; one split; latent asymmetry (review 3 B3) remains.

## CORRECTIONS from the independent review of ROUND 4 (subagent, 2026-11-04 morning; recomputed from raw)
1. R49 verdict mislabel: the falsifier (alternatives within ~300 on the blocks) is MET for the time-varying decoupled
   alternatives (A9 / R42-A9 / R40: +172 to +222) -> "falsified for them; met only against constant growth". What
   separates the groups on the blocks is the native WEST (A9 -116, R40 -54, static +63, R30c5 +118), not the East (all
   +259 to +398). Rescorings of capped parameter sets: differences of 100-300 are endpoint noise (R53b trained to 1993
   scores the blocks 31,735 -- better than the full-data fit 32,105). The West under-predicts zeros in every era
   (~.03), unreported.
2. R50: n50 x.44 is unidentified when growth can change (costs nothing at prior SD 3) -- "prefers a weaker Allee"
   over-reaches. Use ~500 as the held-out noise floor (one free parameter moved held-out by ~480).
3. R51 is mainly an "unconstrained fecundity" test, not a leak test: nests by period 1966-79 172 / 1980-93 121 /
   1994-2005 58 / 2006-20 738 -> it removes 73% of nests (mostly post-2006 East). The static alternative then over-fits
   early counts (East Fmax 3.09 -> 1.94, lambda .69 -> .47, never .102; 2021-25 +4,951). Post-1993 nests legitimately set
   a constant-rate model's Fmax. "Leak propped up the alternative" and "no post-cutoff information of any kind" are
   overstated (fixed latents, disease, Z(t) still carry post-1993 information). Fit arm never .107.
4. R52: Z_disp keeps its clock (journey survival untouched) -> "no clock anywhere" wrong; summary records edited_inputs
   'none' (J22). R53: Z_disp real though edited_inputs says both keys (J22); its 'start' readouts (bbs_nll_start 126,682,
   lambda_split_start) are R52's, not the decoupled start; the native seed sits at detrended K; the "constant G" is
   N/K-gated (J19, G ~.9-1.03). R53b is the least-trained arm (+400 vs the fit on training).
5. "The forecasting edge IS the continental clock" is NOT warranted for the invasion: the R46 edge was East-weighted
   (East +1,653 of +2,878) while R52's loss is 76% WEST; in the East the clockless fit still beats the static (-778) and
   R30 alternatives. Cleanest unreported comparison: R53 vs R46 (iv) -- both K-real with clock, differing only in the
   vital rates (regional non-clock Z change + pinned Allee vs constant rates + n50 x3): held-out East 48,648 vs 50,236
   (-1,588), Plains -349, West +283. The forecast value of time-varying vital rates in the East is their EAST-SPECIFIC
   change (the regional part of lambda inflation), not the clock (likely; memory: regional trend is predictability,
   not cause, may be habitat -- M6, M17).
6. The West result may be a detrending-reference artefact (the band mean includes the fast-changing East; West
   detrended trajectories are forced down). Unknown until West-only / per-region references.
7. The 1993 split coincides with disease onset (disease_start_year 1993): disease parameters are identified only by
   post-1993 rows, held at the full-data fit tuned on the FIT's abundance path; mortality scales with local N -> favours
   arm (i) systematically. R48 (onset inside training) shows a smaller edge (1,130).
8. Z(t) after 1993 is not a pure covariate: the native field's regional component R(s,t) is fitted on bird sources from
   1996 (E1) -- possibly continent-wide BBS observer / effort trends shared by all species (M3, M18): a concrete
   non-habitat candidate for "the clock forecasts the West".
9. "3/4 is K's clock" assumes additivity; the complement (vital rates real, K detrended) and an identity check were not
   run; R53a vs R52a differ by 270 in-sample but 3,447 held-out between unconverged runs -> needs a replicate.
10. Unreported never-reached: R50 .077, R51 .107 / .102, R52b .065, R53b .055.

## ROUND 5 (2026-11-04, from the round-4 review) -- PRE-REGISTRATIONS (fit structure, train <= 1993, held-out 1994-2020)
R55 detrending reference: --field_z detrended_west (subtract the WEST yearly mean) and detrended_region (each region's own
  yearly mean). Prediction (review): if R52's West loss is the band-mean mirror artefact, detrended_west brings the West
  within ~500 of the fit and the loss moves East; falsified if the West still loses > 1,000.
R56 linear-trend placebo (M11): --field_z linclock (detrended + a straight-line band clock per Z dim). Prediction: if Z's
  clock carries specific information, linclock stays > 1,000 worse than the fit (83,142); falsified if within ~500
  (then "a trend continues" is all the clock does).
R57 time-scrambled Z placebo (A1, M19): --field_z scrambled (years permuted in 5-yr blocks, spatial pattern kept).
  Prediction: worse than R52b (87,812); if it matches the fit, any Z time path helps.
R54 clean-row long forecasts: --rescore --train_until on the R46 / R48 / R52b / R53b parameter files -> blocks and
  2021-25 scores for models trained to 1993 / 2005 (28-32-yr forecasts on rows never in any objective).

### R54 RESULT: clean-row scores for held-out-trained parameter sets (rows never in ANY objective; -log-lik)
| trained to | structure | 31 blocks 1966-2020 (E / P / W) | 2021-25 (E / P / W) |
|---|---|---|---|
| 1993 | fit | 32,225 (18,542 / 977 / 12,706) | 21,767 (12,612 / 1,314 / 7,841) |
| 1993 | static-G alternative | 31,660 (-565: -360 / +53 / -258) | 23,170 (+1,403: +1,101 / +255 / +48) |
| 1993 | R52b no clock in Z_gathered | 33,599 (+1,374) | 24,162 (+2,395: +1,069 / +105 / +1,221) |
| 1993 | R53b clock in K only | 31,735 (-490) | 22,373 (+606: +345 / +138 / +122) |
| 2005 | fit | 32,125 | 21,585 (12,515 / 1,298 / 7,772) |
| 2005 | static-G alternative | 32,313 (+188) | 22,988 (+1,403: +1,190 / +189 / +24) |
Read (likely): on the cleanest test available -- 2021-25 rows that no model or production fit ever saw, forecast
16-28 years past training -- the fit beats the constant-growth, stronger-Allee alternative by ~1,400 at BOTH splits,
almost all in the EAST (+1,100-1,190); keeping the clock only in K (R53b) loses ~600; removing the clock (R52b) loses
~2,400 (East and West alike). On the spatial blocks (rows never fitted, but within the training years for most of the
period) the alternative is EQUAL or slightly better (-565 / +188; the noise floor is ~500): spatial transfer does not
favour the fit. So the fit's advantage is TEMPORAL extrapolation, concentrated in the East, robust to disease-onset
timing (the 2005 split carries the same +1,403 on 2021-25).

### R55 RESULT (a): fit field detrended against the WEST yearly mean (train <= 1993; converged, 768 evals)
HELD-OUT 1994-2020 84,541 (fit 83,142: +1,399 -- East +118 / Plains -11 / West +1,292); 2021-25 clean rows 22,453
(fit93 21,767: +686); blocks 32,672 (+447); never .064.
vs band-referenced R52b (+4,670; East +875 / West +3,556). Prediction (West within ~500 -> mirror artefact; falsified
if the West still loses > 1,000): FALSIFIED in the strict sense (West +1,292) but ~2/3 of R52's West loss (2,264 of
3,556) WAS the band-reference artefact. Read (likely): (1) the East forecasts as well as the fit once it keeps its
change RELATIVE to the West (+118) -- the East's forecasting information is its East-minus-West Z change; (2) the native
West's own yearly trend in Z is worth ~1,300 held-out -- a genuine continental / western trend the counts use.
R55 (b): each region detrended against its OWN yearly mean (train <= 1993; cap, 1,243 evals): HELD-OUT 86,029 (fit
+2,887 -- East +1,131 / Plains +181 / West +1,575); 2021-25 23,920 (fit93 +2,153: East +1,300 / Plains +150 / West +702);
blocks 33,686 (+1,461); never .050.
Read (likely): removing each region's OWN trend costs both regions (East +1,131 held-out, +1,300 on 2021-25), unlike
the West-referenced arm where the East kept its relative change (+118). So the East's forecasting information is the
East's own regional trend in Z (the part beyond the West's), and the West's is the West's own trend -- each region's
time path in the community prior carries forward information about House Finch counts there; it is not one uniform
continental clock (R44c's "~90% clock" was a band-mean artefact for the purpose of forecasting).

### R56 RESULT: linear-trend placebo (band clock replaced by a straight line per Z dim; regional deviations kept;
train <= 1993; ABNORMAL stop, 556 evals; training 51,962 = the fit's)
HELD-OUT 83,707 (fit +565 -- East -67 / Plains +14 / West +618); 2021-25 clean rows 22,008 (fit93 +241); blocks 32,522
(+297); never .072.
Prediction (if the clock carries specific information, linclock stays > 1,000 worse; falsified if within ~500):
FALSIFIED at the noise floor (+565 total, East better than the fit). Read (likely): the year-to-year SHAPE of Z's
continental clock adds nothing beyond a straight-line trend -- for forecasting, the clock behaves as "a trend
continues" (M11). What does carry information is each region's own change relative to that trend (R55 b: removing it
costs +2,887). So: the continental part of Z's time signal is exchangeable with a linear time trend (could be habitat,
effort, or encoder drift -- not separable here); the regional part is the substantive, place-specific signal.

### R57 RESULT: time-scrambled Z placebo (years permuted in 5-yr blocks; spatial pattern kept; train <= 1993; cap)
Training objective 53,069 (fit 51,961: +1,108), training eras 28,953 (fit 27,911); HELD-OUT 89,084 (fit +5,942 -- East
+3,057 / Plains +368 / West +2,516); 2021-25 23,341 (+1,574); blocks 34,136 (+1,911); never .027.
Prediction (worse than R52b 87,812) MET: a wrong time path in Z is worse than no clock at all. Read (likely): the fit is
not merely exploiting "any time variation in Z": the real ORDER of Z's change matters, and the training data alone
already penalise the scrambled path (+1,108), so Z's time path is informative within 1966-93 too.

## ROUND 5 SYNTHESIS (likely unless stated; one base, route-count units, latent asymmetry caveat stands)
- The fit's long-horizon forecasting edge over constant-growth / stronger-Allee alternatives is real on rows no model
  ever saw (2021-25: ~1,400 at both splits, East-concentrated; R54).
- What carries it: each REGION's own time change in Z (East-specific in the East, West's own trend in the West; R55);
  the continental (band-wide) component behaves like a straight-line trend (R56); the real order of Z's change matters
  (R57); ~3/4 of the edge runs through K, the rest through vital rates (R53, approximate, unconverged).
- Lambda inflation in the EAST therefore encodes the East's regional Z trajectory -- which may be habitat change (M6) or a
  regional observation / encoder trend (M3, M18). Deciding which is E1-E4 (TACC files) or an independent driver test
  (BUI / HYDE vs observer effort; review 4 items 5-6).

## ROUND 6
### R58 PRE-REGISTRATION (review 4 item 5): are the fit's place-specific rate changes habitat-like or effort-like?
r58_drivers.py (readout): per cell, change 1966-75 -> 2010-20 in the fit's log K and potential lambda vs changes in BUI q90,
HYDE population (encoder yearly states), BBS effort (rows per cell-year), first-year-observer share, observer-route
turnover; Spearman within region; and East-minus-West yearly log K vs the same drivers (raw and linearly detrended).
Prediction (driver reading, M26): K / lambda change tracks BUI or HYDE change more than effort / observer composition;
falsified if effort / observer terms correlate as strongly or more (observation artefact, M3 / M18).

### R58 RESULT: driver readout for the fit's place-specific change (1966-75 -> 2010-20; Spearman across cells)
| region | dlogK ~ dBUI | ~ dHYDE | ~ dEffort | ~ dFirstYearShare | ~ dTurnover | (dlambda: same pattern) |
|---|---|---|---|---|---|---|
| East (n 1,308; effort n 756) | .35 | .24 | .04 | .03 | -.03 | .35 / .25 / .03 / .03 / -.03 |
| Plains (380; 171) | .35 | .29 | -.10 | .00 | -.06 | .36 / .31 / -.10 / .00 / -.06 |
| West (1,566; 366) | .12 | .24 | .03 | -.01 | .04 | .14 / .27 / .00 / -.01 / .02 |
East-minus-West yearly mean log K vs: BUI raw r .98, linearly detrended .59; HYDE .15 / -.65; log effort .20 / -.16;
first-year-observer share .58 / -.10.
Prediction (tracks BUI / HYDE more than effort / observer composition) MET. Read (likely): where and when the fit's
capacity and growth rates rise, built-up intensity (and human population) rose; survey effort, observer turnover and
first-year-observer share show no relation (|rho| <= .1) -- the fit's place-specific Z signal looks HABITAT-like (built
environment), not like an observation artefact, in all three regions. BIG CAVEAT (circularity): BUI is itself an input
to the habitat encoder (user, earlier), so Z-driven change tracking BUI change is partly by construction; this shows
the encoder's time signal is BUI-like rather than effort-like, not that settlement CAUSES the finch's change (memory:
regional trend is predictability, not cause; M26). Effort metrics are from the House Finch BBS rows (same routes the
encoder sees). Decisive attribution still needs E1-E4 (encoder decomposition on the native field; TACC files).

## CORRECTIONS from the independent review of ROUNDS 5-6 (subagent, 2026-11-04; recomputed from raw)
1. R54 mislabel: ~2/3 of the fit93 block score (21,536 of 32,225) is from 1994-2020 rows, not "training years". On blocks
   BEFORE 1994: fit 10,688 vs alternative 10,676 (-12); 1994-2020: 21,536 vs 20,983 -> the ALTERNATIVE is 553 better
   (East -432, West -176); R53b also 611 better. On rows held out in BOTH time and place, the alternative equals or beats
   the fit. "The fit's advantage is TEMPORAL extrapolation" is NOT warranted; supported: the fit forecasts better at
   places whose earlier counts it saw. Likely mechanism: observer x route effects (sigma .97) are untuned (prior) in
   block cells but tuned on the full-data fit's path elsewhere -- the latent asymmetry (review 3 B3), now sized.
   Block differences < ~900 are weak (R47-static scores blocks 909 better trained to 1993 than with full data).
2. R55: the West's own Z is identical in both arms yet differs by 283 -> ~300 is the floor for regional splits
   (cross-region coupling through shared coefficients). "East +1,131" robust (stalled: 0.04 over 400 evals; training
   +127; reappears on post-1993 blocks East +949 / West +272). UNREPORTED: per-region detrending leaves potential lambda
   .455 -> .509 -- lambda inflation almost GONE at a training (<= 1993) cost of only +127 (medians over all East land
   cells; unknown for occupied). Cells outside the three regions keep their real clock in detrended_region.
3. R56: the line is fitted over ALL years incl. 1994-2025 -> it shows the band clock's year-to-year shape is not needed,
   NOT that extrapolating the training trend works. Lambda inflation survives intact (.201 -> 1.028).
4. R57: Z_gathered permuted for K and vital rates (Z_disp real); full-year permutation puts modern patterns in the
   spin-up (never .027); only 72 coefficients refitted, degenerate solution (Sa .04-.05, Fmax 3.5); one draw (M20) and the
   wrong kind of placebo (M11 / M19 want position x year or wrong-region) -> "real order matters" likely but weak.
5. R58: "looks habitat-like" -> UNKNOWN (not likely). The cross-cell Spearman tests the within-region SPATIAL pattern of
   change (which R55b kept), not the regional yearly-mean trajectory that carries the forecast; that is tested only by
   East-minus-West series: detrended BUI .59 but HYDE -.65 (contradicts "population rose"); dlambda duplicates dlogK in a
   single shared field; effort metrics are general BBS effort / crude observer shares and do NOT test the encoder's own
   post-1996 sources (e.g. eBird). Possible disease-era K confound.
6. J22 persists: R53b rescore summary still mis-labels edited inputs (Z_disp real). Plan row A18 understated R54.
   Placebos (M11) so far only for the continental component; the regional component has none.
7. Alternatives still open: regional observer / effort drift in the all-species encoder data (M3/M18); native Z's
   R(s,t) fitted to bird sources only from 1996 (pre-1996 extrapolated; held-out-period Z partly contemporaneous
   community information; possible 1996 seam); BUI's own construction (HISDAC survivorship bias growing back in time,
   CONUS-only); latent asymmetry (observer effects, disease).

## ROUND 7 (2026-11-04, from the rounds-5-6 review) -- PRE-REGISTRATIONS (train <= 1993 unless stated)
R59 latent asymmetry: refit fit93 and static93 with the observer latents (observer_route_eps, observer_first_year_eta)
  added to the refitted sites, <= 1993 rows only (warm start from the R46 parameters). Prediction (review): the 2,878
  gap shrinks by > 1,000; falsified if it stays within 500 of 2,878. CPU.
R60 regional detrending in the VITAL RATES only (K real Z, Z_disp real): --decouple --estab --ridge_sd 1e-6 --field_z
  detrended_region and detrended_west. Prediction: if K carries the regional information, East within ~300 of the fit;
  falsified if East loses > 700 (then the regional part of lambda inflation carries it). GPU.
R61 per-region linear placebo (M11 / M17): --field_z linregion (each region's own straight line, fitted on all years) and
  linregion93 (fitted on <= 1993, extrapolated). Prediction: if the trajectory's SHAPE matters, linregion stays > 1,000
  worse than the fit; falsified if within 500 ("each region's trend continues" is the information). GPU.
R64 wrong-region placebo: --field_z swap_ew (East cells get the West's yearly-mean trajectory and vice versa).
  Prediction: loses at least as much as R55b (+2,887); falsified if within 500 of the fit (any smooth rise works). GPU.
All edited inputs are now written by one helper (J22 fix).

### Round 7 reference values (train <= 1993): fit93 held-out 83,142 (E 48,583 / P 4,913 / W 29,646); 2021-25 21,767 (E 12,612 /
P 1,314 / W 7,841); post-1993 blocks 21,536 (E 13,640 / P 850 / W 7,046). static93: held 86,020; 2021-25 23,170; post-1993
blocks 20,983 (E 13,208 / P 905 / W 6,870).

### R60 RESULT (a): vital rates read per-region-detrended Z, K real Z (decoupled, constant G, n50 pinned; cap)
Training 52,431 (fit 51,961: +470); HELD-OUT 84,271 (+1,129 -- East +957 / Plains +74 / West +98); 2021-25 22,666 (+899 --
East +929 / Plains +93 / West -123); post-1993 blocks 20,875 (-661 -- East -429 / Plains -27 / West -205); potential
lambda .469 -> .525 (lambda inflation essentially gone).
Prediction (East within ~300 if K carries the regional information; falsified if East loses > 700): FALSIFIED on the
unmasked and 2021-25 rows (East +957 / +929) -- the East's regional trend in the VITAL RATES carries forecast
information at places with count history; but on post-1993 BLOCKS (no count history, observer effects untuned) the
no-inflation structure is BETTER than the fit (East -429), like the static alternative. Read (likely): the regional
part of lambda inflation helps forecast where the model has seen a place's counts and observers, not at new places --
consistent with the latent-asymmetry explanation that R59 tests.
R60 (b): vital rates read WEST-referenced detrended Z (East keeps its change relative to the West), K real (cap):
training 52,121 (+160); HELD-OUT 83,291 (+149 -- East -151 / Plains +59 / West +241); 2021-25 21,791 (+24); post-1993
blocks 21,191 (-345); potential lambda .228 -> .829 (most of the inflation kept: the East-minus-West part).
Read (likely): removing only the WEST's yearly trend from the vital rates costs nothing anywhere (all within noise); the
East's forecast information in the vital rates is its change RELATIVE TO THE WEST, and keeping it keeps most of the
eastern lambda inflation (+.60 of the fit's +.87). So the part of lambda inflation the counts use for forecasting is the
East-specific (relative) Z trend.

### R61 RESULT (a): each region's yearly-mean Z replaced by its own straight line (fitted on all years; all rates incl. K)
Training 52,002 (+41); HELD-OUT 83,428 (+286 -- East +102 / Plains +66 / West +118); 2021-25 21,993 (+226); post-1993
blocks 21,847 (+311); potential lambda .235 -> .976 (inflation kept).
Prediction (if the SHAPE matters, > 1,000 worse; falsified if within 500): FALSIFIED -- a straight line per region
forecasts like the fit everywhere (all within noise). Read (likely): the forecast information in each region's Z
trajectory is "this region's trend continues" -- its direction and rate, not its year-to-year shape. Any regional drift
with the right slope (habitat change, or a regional observation / encoder trend) could supply that; the line was fitted
over all years (incl. held-out), so linregion93 (fitted <= 1993, extrapolated) is the stricter test.
R61 (b) linregion93: each region's straight line fitted on <= 1993 Z ONLY and extrapolated (all rates incl. K):
training 51,993 (+32); HELD-OUT 83,663 (+521 -- East +280 / Plains +89 / West +152); 2021-25 22,329 (+562 -- East +371);
post-1993 blocks 22,061 (+525); potential lambda .26 -> .82.
Read (likely): extrapolating each region's PRE-1993 Z trend in a straight line forecasts 1994-2025 almost as well as the
real Z path (+~500, the noise floor) -- and much better than constant-growth alternatives (+2,878) or no clock (+4,670).
So the fit's forecasting information is available from the training period itself: each region's Z trend through 1993,
continued. The real post-1993 Z (partly fitted to contemporaneous bird sources, R(s,t) from 1996) adds only ~500.

## REFRAMING (user, 2026-11-04 midday): lambda INCREASE is expected; "inflation" names implausibly large absolute swings
(hostile -> paradisical). Goal for the day: separate the genuinely robust, NECESSARY lambda change from the change that
exists because it is a convenient lever. Experiments are now necessity tests / regularisation paths, not removal tests.

### R65 PRE-REGISTRATION: necessity path for the vital rates' time change
Decoupled, K real Z, Z_disp real, constant G (ridge 1e-6), n50 pinned; Sa / Sj / Fmax read Zref + a (Z - Zref) with
a in {0, .25, .5, .75} (a = 1 ~ the fit's structure); trained <= 1993; scored on held-out unmasked 1994-2020, 2021-25
clean rows and post-1993 blocks; report the eastern potential-lambda swing (1940 -> 2010) at each a. GPU.
Read-out rule: the NECESSARY share a* = the smallest a whose held-out score is within ~500 (noise floor) of the best; the
lambda swing at a* is the robust part; the remainder (fit's swing minus swing at a*) is the convenient-lever part.
Prediction: a* < 1 (part of the swing is convenience) but > 0 (R55 / R60: the East's relative trend forecasts).

### R64 RESULT: wrong-region placebo (East cells get the West's yearly-mean Z trajectory and vice versa; ABNORMAL stop)
Training 52,149 (+188); HELD-OUT 85,065 (+1,923 -- East +1,726 / Plains +95 / West +102); 2021-25 23,015 (+1,248 -- East
+1,292 / West -76); post-1993 blocks 22,089 (+553 -- East +672 / West -102); potential lambda .476 -> .903.
Prediction (loses at least as much as R55b's +2,887; falsified if within 500 of the fit): IN BETWEEN (+1,923). Read
(likely): giving the East the West's trajectory costs the East ~1,300-1,700 (about the same as removing the East's own
trend, R55b +1,131 / R60a +957) while the West barely cares which trajectory it gets. So the East's forecast information
is SPECIFIC to the East's own trajectory -- not "any smooth rise works" -- and the West's is generic (a rise is a rise).

### R59 RESULT: observer effects (observer_route_eps ~19,000 + first-year eta) refitted on <= 1993 rows too (CPU, caps)
| arm | training (<= 1993) | HELD-OUT 1994-2020 | 2021-25 clean | post-1993 blocks |
|---|---|---|---|---|
| fit93 (observers at full-data values, R46) | 51,961 | 83,142 | 21,767 | 21,536 |
| fit + observers refit | 48,829 (-3,132) | 111,430 (+28,288) | 24,640 | 21,302 |
| static93 (observers at full-data values) | 52,065 | 86,020 | 23,170 | 20,983 |
| static + observers refit | 50,532 (-1,533) | 95,578 (+9,558) | 23,699 | 20,932 |
Prediction (gap shrinks by > 1,000; falsified if within 500 of 2,878): MET and far beyond -- with observers refitted for
BOTH arms on training rows, the RANKING REVERSES: the static alternative beats the fit on held-out by ~15,850, on 2021-25
by ~940 and on post-1993 blocks by ~370.
Read (likely, with a big caveat): refitting ~19,000 observer-route effects on early rows OVER-FITS both arms badly
(held-out +9,500 / +28,300), and the fit over-fits far more -- its time-varying habitat field and the observer effects
co-adapt to the training years. So (1) the R46 / R48 / R50 / R51 / R54 held-out advantage of the fit was conditioned on
observer effects tuned to the FULL data along the fit's own path (the latent asymmetry is large, not marginal); (2) this
refit regime is itself pathological, so neither ranking is clean. A fair test holds observer effects COMMON to both arms
at training-only values (R66, queued). Until then the "fit forecasts better" conclusion is DOWNGRADED to unknown.

### R66 PRE-REGISTRATION: held-out comparison with COMMON training-only observer effects (both orders)
refit_fields --fixed_npz (observer_route_eps, observer_first_year_eta held fixed), train <= 1993, habitat sites refitted:
 A: fit structure with R59-STATIC's training-only observers  -> compare with R59 static_obs (95,578) -- same observers;
 B: static structure with R59-FIT's training-only observers  -> compare with R59 fit_obs (111,430) -- same observers.
Prediction: if the fit's forecasting edge is structural, the fit wins in BOTH pairings; if it lives in observer
co-adaptation, the static structure wins in both (or the winner flips with the observer set). CPU, parallel.

### R65 path so far (trained <= 1993; observers at full-data values -- R59 caveat applies to the whole path)
| vital rates read share a of Z's time change (K real) | training | HELD-OUT 1994-2020 | 2021-25 clean | post-1993 blocks | East potential lambda 1940 -> 2010 (all land cells) |
|---|---|---|---|---|---|
| 0 | 53,239 | 86,552 (+3,410) | 23,625 (+1,858) | 20,651 (-885) | .80 -> .80 |
| .25 | 52,644 | 85,071 (+1,929) | 22,564 (+797) | 20,721 (-815) | .57 -> .92 |
| .5 | 52,262 | 83,166 (+24) | 21,761 (-6) | 20,941 (-595) | .37 -> .94 |
| fit (a = 1, shared field) | 51,961 | 83,142 | 21,767 | 21,536 | .21 -> 1.08 |
At a = .5, OCCUPIED eastern cells (n 863): potential lambda median .57 (1940) -> 1.41 (2010); q10-q90 1940 .11-1.35,
2010 .34-2.80; 45% deep sinks (< .5) in 1940, 63% sources (> 1) in 2010.
Read (likely, conditional on observers): HALF the vital rates' time change already gives the fit's full held-out and
2021-25 skill and is better on new places; the rest of the swing adds no forecast skill -- a* <= .5. Pending .75 and the
fit's own occupied-cell readout for the comparison of swing sizes.
R65 a = .75: training 52,078; HELD-OUT 82,694 (fit -448: best of the path; East -269 / Plains -32 / West -147); 2021-25
21,599 (-168); post-1993 blocks 21,177 (-359); potential lambda (all land) .25 -> .96; OCCUPIED East median .39 (1940) ->
1.46 (2010), q10-q90 1940 .07-1.03, 2010 .32-2.91; 61% deep sinks in 1940, 64% sources in 2010.
R65 READ-OUT (rule: a* = smallest a within ~500 of the best held-out): best a = .75 (82,694); a = .5 is +472 -> a* ~ .5
(between .5 and .75; the fit, a = 1, is +448 vs the best). So the full time change in the vital rates is NOT needed for
forecasting: ~half to three-quarters is, and a = .75 even forecasts slightly better than a = 1 (within noise). What
moves between a = .5 and .75 is mainly the 1940 starting point (occupied median .57 vs .39; deep-sink share 45% vs 61%);
the 2010 level is the same (~1.4-1.5) -- the data pin the modern source strength, not the 1940 depth. Fit's own
occupied-cell readout pending (rescore). Caveat: all conditional on observers at full-data values (R59 / R66).

### R65 comparison: the swing in OCCUPIED eastern cells (n 863; potential lambda Sa + Sj Fmax, no Allee / crowding)
| structure | 1940 median (q10-q90) | deep sinks < .5 in 1940 | 2010 median (q10-q90) | sources > 1 in 2010 | held-out vs fit |
|---|---|---|---|---|---|
| fit, full data | .31 (.06-.78) | 74% | 1.53 (.36-2.81) | 67% | -- |
| fit, trained <= 1993 | .31 (.05-.80) | 74% | 1.56 (.32-2.83) | 69% | 0 |
| a = .75 | .39 (.07-1.03) | 61% | 1.46 (.32-2.91) | 64% | -448 |
| a = .5 | .57 (.11-1.35) | 45% | 1.41 (.34-2.80) | 63% | +24 |
| a = 0 (constant) | .80 | -- | .80 | -- | +3,410 |
(fit occupied WEST: 1940 median .55-.62, 2010 1.05-1.11; 43-46% deep sinks 1940, 53-54% sources 2010.)
Read (likely; conditional on full-data observers): the robust, forecast-NECESSARY part of the eastern change is a rise to a
modern median source strength of ~1.4-1.5 (all arms agree; 63-69% of occupied cells sources) from a 1940 state somewhere
between median .57 (a = .5) and .31 (fit). What the data do NOT require is the depth of the 1940 sink: going from a
median of .57 (45% deep sinks) to .31 (74% deep sinks) buys no held-out skill. In the user's words, the "paradise" end is
pinned; the "hostile" end is partly a convenient lever -- the fit makes 1940 more hostile than forecasting requires
(the required swing in median occupied-cell lambda is .84-1.07 of the fit's 1.22, i.e. ~70-88%; the optional ~12-30%
is almost all extra 1940 depth, which turns 45-61% deep-sink cells into 74%).

### R66 arm B result (static structure with R59-FIT's training-only observers): training 49,153; HELD-OUT 112,496 vs the fit
structure with the same observers (R59 fit_obs) 111,430 -> FIT better by 1,066 (East +563 / West +471 for static); 2021-25
24,846 vs 24,639 (fit better by 207); post-1993 blocks 20,763 vs 21,303 (static better by 540). Arm A pending.

### R67 PRE-REGISTRATION: the R65 necessity path with COMMON training-only observers (R59-static observers fixed)
a in {0, .5, .75} (decoupled, K real, Z_disp real, constant G, n50 pinned, train <= 1993); a = 1 is R66 arm A (same
observers). Prediction: the a* <= .75 conclusion and the "1940 depth is optional" reading survive (best a < 1 within ~500).

### R66 RESULT: structures compared with COMMON training-only observer effects (train <= 1993)
| observer set (fitted <= 1993) | structure | training | HELD-OUT 1994-2020 | 2021-25 clean | post-1993 blocks |
|---|---|---|---|---|---|
| static's (R59 static_obs) | static (R59) | 50,532 | 95,578 | 23,698 | 20,933 |
| static's | FIT structure (R66 A) | 50,846 | 91,855 (-3,723) | 22,182 (-1,516) | 21,364 (+431) |
| fit's (R59 fit_obs) | FIT structure (R59) | 48,829 | 111,430 | 24,639 | 21,303 |
| fit's | static (R66 B) | 49,153 | 112,496 (+1,066 vs fit) | 24,846 (+207) | 20,763 (-540) |
Prediction ("structural edge -> the fit wins in BOTH pairings") MET on the held-out unmasked and 2021-25 rows: with the
static arm's observers the fit structure wins by 3,723 / 1,516; with the fit's own observers by 1,066 / 207. On the
post-1993 blocks (no count history at those places) the static structure is better in both pairings (by 431 / 540).
Read (likely): with observer effects estimated from training rows only and held COMMON, the time-varying Z-driven vital
rates still forecast later counts better at places with count history (by ~1,000-3,700) -- the R46 advantage was not an
artefact of observer co-adaptation, though its size depends on the observer set. At NEW places (blocks) less time change
does slightly better, consistently. Restores "likely" to the fit's long-horizon forecasting edge (R59 downgrade lifted),
with the spatial-transfer caveat (blocks) standing.

### R67 RESULT: the necessity path with COMMON training-only observers (R59-static observers; train <= 1993)
| vital rates read share a (K real) | HELD-OUT 1994-2020 | 2021-25 clean | post-1993 blocks | OCCUPIED East 1940 median (q10-q90) | deep sinks 1940 | 2010 median | sources 2010 |
|---|---|---|---|---|---|---|---|
| 0 | 98,009 (+6,696) | 23,754 (+2,137) | 20,629 | 1.08 (const) | 18% | 1.08 | 53% |
| .5 | 92,883 (+1,570) | 21,816 (+199) | 20,634 | .48 (.11-1.23) | 51% | 1.35 | 61% |
| .75 | 91,313 (best) | 21,617 (best) | 20,850 | .36 (.08-.91) | 66% | 1.43 | 64% |
| 1 (fit structure, R66 A) | 91,855 (+542) | 22,182 (+565) | 21,364 | .40 (.06-.92) | 61% | 1.46 | 66% |
(+ = worse than the best; -log-lik.) a = .5 is now 1,570 worse (beyond the ~500 noise floor); a = .75 and 1 are within ~550.
Read (likely), CORRECTING the R65 read: with fair observers, a* ~ .75-1 -- nearly the FULL time change in the vital rates
is needed for forecasting, INCLUDING a deep 1940 sink (occupied East median .36-.40, 61-66% of occupied cells below .5)
rising to a modern median ~1.43-1.46 (64-66% sources). R65's "1940 depth is optional" was an artefact of full-data
observer effects. The necessary swing in median occupied-cell lambda is ~1.06-1.07; the full-data fit's 1.22 (1940 .31)
exceeds it by ~.15 -- a small "convenient" excess, mostly in the 1940 depth (74% vs 61-66% deep sinks).
At NEW places (blocks), every amount of time change does about equally (20,629-21,364; within ~750, near noise), with
less change slightly better.

### R68 PRE-REGISTRATION: R67 repeated with the OTHER common observer set (R59-fit's training-only observers), a = .5 / .75
(a = 1 is R59 fit_obs itself). Prediction: same ordering (a = .5 > 500 worse than the best; a = .75 ~ a = 1).
R67 Shapley shares of the eastern potential-lambda rise 1940 -> 2010 (median over all East land cells; rates median 1940 / 2010):
a = .75: Sa .28 / Sj .45 / Fmax .27 (Sa .13 -> .39, Sj .05 -> .15, Fmax 2.23 -> 4.01); fit structure (R66 A): Sa .32 / Sj
.48 / Fmax .20 (Sa .14 -> .42, Sj .05 -> .16, Fmax 2.82 -> 4.01); a = .5: Sa .24 / Sj .50 / Fmax .27. Juvenile survival
carries about half of the necessary rise in every arm; adult survival ~a quarter to a third. (Earlier "70% of [cur]'s
inflation is survival" holds: Sa + Sj ~ .73-.80 here.)

### R69 PRE-REGISTRATION: WHICH vital rate's time change is necessary? (common training-only observers, train <= 1993)
Decoupled (K real Z, Z_disp real, constant G, n50 pinned; R59-static observers fixed); --real_rates makes the named rates
read the REAL-Z pass (crowding c re-derived from the mixed rates); the others stay frozen. Arms: Sa only; Sj only; Fmax
only; Sa + Sj (survival). References: all frozen = R67 a = 0 (98,009); all real ~ R66 A (91,855). Caveat: lambda_split in
these summaries reports the FROZEN readouts for swapped rates (use held-out scores, not lambda_split, here).
Prediction (from the Shapley split): survival (Sa + Sj) recovers most of the held-out skill (within ~1,000 of all-real);
Fmax alone recovers little.

### R68 RESULT and NECESSITY SYNTHESIS across both common observer sets (train <= 1993; -log-lik; + = worse than the best a)
| a | static-set observers: held-out / 2021-25 | fit-set observers: held-out / 2021-25 | OCCUPIED East median 1940 -> 2010 (static set; fit set) |
|---|---|---|---|
| 0 | +6,696 / +2,137 | -- | 1.08 -> 1.08 |
| .5 | +1,570 / +199 | best (110,135) / +146 | .48 -> 1.35; .47 -> 1.29 |
| .75 | best (91,313) / best (21,617) | +45 / best (23,846) | .36 -> 1.43; .27 -> 1.40 |
| 1 (fit structure) | +542 / +565 | +1,295 / +793 | .40 -> 1.46; (fit_obs: n/a) |
Read (likely): a = .75 is best or tied-best under BOTH observer sets; a = 1 (the fit's full time change) is worse under both
(+542 / +1,295 held-out; +565 / +793 on 2021-25); a = .5 depends on the observer set. So, robustly:
- NECESSARY: about three-quarters of the fit's time change in the vital rates -- in occupied eastern cells a rise from a
  1940 median of ~.27-.36 (a deep sink; most occupied cells < .5) to a 2010 median of ~1.40-1.43 (about two-thirds sources).
- CONVENIENT EXCESS: the last ~quarter of the time change. It forecasts consistently WORSE out of sample, though its effect
  on the occupied-cell medians is small (the fit, full data: .31 -> 1.53) -- the excess sits mostly at the modern
  ("paradise") end (~+.1 in median, more in the upper tail) rather than in the 1940 depth.
- Juvenile survival carries ~half of the necessary rise, adult survival ~a quarter to a third, Fmax ~a quarter (R67 Shapley).
Caveats: one time split (<= 1993); unconverged caps; observer sets are training-only refits from two structures; median
over occupied cells of potential lambda (no Allee, crowding or movement).

### R70 PRE-REGISTRATION: necessity path at a SECOND split (train <= 2005; scored 2010-20 unmasked + 2021-25 + post-2005 blocks)
Step 1: static structure + observers refitted on <= 2005 (R59-like) -> common observer set. Step 2: a in {.5, .75} (decoupled)
and the fit structure (a = 1), each with those observers fixed, train <= 2005. GPU serial.
Prediction: a = .75 again best or tied (within ~500) and a = 1 worse; if a = 1 is best here, the "convenient last quarter"
is split-specific.

### R70 RESULT: necessity path at the 2005 split (common training-only observers from the static structure refit <= 2005)
| structure (train <= 2005) | held-out 2010-20 unmasked | 2021-25 clean | post-2005 blocks | OCCUPIED East median 1940 -> 2010 |
|---|---|---|---|---|
| constant vital rates (static) | 37,147 (+835) | 23,800 (+1,400) | 7,544 (+208) | .91 -> .91 |
| a = .5 | 36,312 (best) | 22,464 (+64) | 7,336 (best) | .53 -> 1.32 |
| a = .75 | 36,352 (+40) | 22,400 (best) | 7,392 (+56) | .39 -> 1.46 |
| a = 1 (fit structure) | 36,550 (+238) | 22,656 (+256) | 7,468 (+132) | .38 -> 1.55 |
Read (likely): at the second split every time-varying arm beats constant vital rates (by ~800-1,400), and a = .5-1 are
within the noise floor of each other, with a = 1 again the worst of the three on every score (+132 to +256). Across all
three necessity comparisons (two observer sets at 1993, one at 2005) the full time change (a = 1) is NEVER the best and
a = .75 is best or tied every time. ROBUST CONCLUSION: a substantial rise in the eastern vital rates is necessary (constant
rates always lose by ~800-6,700); about 3/4 of the fit's change suffices; the last quarter is never needed and is
consistently (if, at 2005, weakly) worse out of sample. It mainly lifts the modern end (occupied 2010 median 1.55 at a = 1
vs 1.46 at .75 and 1.32 at .5); the deep 1940 sink (median ~.4) is common to a = .75 and 1.

### R69 RESULT: WHICH vital rate's time change is necessary (common static-set observers; train <= 1993; K real)
| rates reading the REAL (time-varying) Z | training | HELD-OUT 1994-2020 | 2021-25 | post-1993 blocks | share of the frozen->best held-out gain |
|---|---|---|---|---|---|
| none (all frozen; R67 a = 0) | 51,551 | 98,009 | 23,754 | 20,629 | 0% |
| Fmax only | 51,317 | 99,630 | 23,509 | 21,357 | -24% (worse than none) |
| Sj only | 51,047 | 96,879 | 22,899 | 21,057 | 17% |
| Sa only | 51,038 | 92,227 | 22,367 | 20,981 | 86% |
| Sa + Sj | 51,374 | 92,652 | 22,523 | 21,579 | 80% |
| all (fit structure, R66 A) | 50,846 | 91,855 | 22,182 | 21,364 | 92% |
| best of the a path (a = .75, R67) | 50,985 | 91,313 | 21,617 | 20,850 | 100% |
Read (likely): the forecast-NECESSARY part of the vital rates' time change is ADULT SURVIVAL's. Letting only Sa follow Z's
time change recovers ~86% of the out-of-sample skill; juvenile survival alone recovers ~17% and adds nothing on top of
Sa; fecundity's time change alone is WORSE than none. Yet by the Shapley split (R67) juvenile survival carries ~half of the
eastern potential-lambda rise and Fmax ~a quarter -- so most of the lambda swing that juvenile survival and fecundity
provide is not supported by forecasting: a candidate for the "convenient lever". Caveats: rates mixed from two passes
with crowding c re-derived; one split; adult survival also governs how long settled birds persist (abundance), so its
forecast value may be abundance-trend information more than establishment.

### R71 PRE-REGISTRATION: structure-matched a = 1 (decoupled machinery, vital rates read ALL of Z's time change)
Same as R67 (static-set observers, train <= 1993, constant G, K real, Z_disp real) with --vr_alpha 1. Compare with R67
a = .75 (91,313) and the single-field fit structure (R66 A, 91,855). Prediction: if the extra quarter is genuinely
unnecessary, decoupled a = 1 is also worse than a = .75 (by > ~300); if decoupled a = 1 ~ a = .75, "a = 1 is worse" was the
structure difference.
### R72 PRE-REGISTRATION: R69 replicated with the OTHER observer set (R59-fit's training-only observers)
Arms: Sa only real; Sj + Fmax real (the complement). Reference with these observers: a = .5 / .75 (R68: 110,135 / 110,180)
and the fit (111,430). Prediction: Sa alone again recovers most of the skill (held-out within ~1,000 of the R68 best) and
the complement recovers much less.

### R71 RESULT: structure-matched a = 1 (decoupled machinery; static-set observers; train <= 1993)
Training 51,002; HELD-OUT 90,479 -- BEST of the path (a = .75: 91,313 -> a = 1 better by 834; single-field fit structure
91,855); 2021-25 21,739 (a = .75 21,617: +122); post-1993 blocks 21,304 (a = .75 20,850: +454); occupied East median .29
(1940) -> 1.51 (2010), q10-q90 .06-.75 / .33-2.94 -- essentially the fit's own swing (.31 -> 1.53).
Prediction (decoupled a = 1 also worse than .75 by > ~300 if the last quarter is unnecessary): FALSIFIED on the unmasked
held-out rows (a = 1 is 834 BETTER); on never-seen rows a = .75 is marginally better (2021-25 +122, blocks +454; both at or
below the noise floors). CORRECTION to the R68 / R70 synthesis: "a = 1 is worse" was largely the STRUCTURE difference
(single shared field vs decoupled K + vital rates), not the extra quarter of change. With matched machinery the data
support a = .75-1 about equally -- i.e. essentially the FULL eastern swing in occupied-cell potential lambda (median ~.3
in 1940 -> ~1.5 in 2010) is consistent with out-of-sample forecasting; no robust "convenient excess" in the SIZE of the
swing is detectable here. What remains as candidate convenience is WHICH RATES carry it (R69: adult survival's change
carries the forecast skill; juvenile survival's and fecundity's do not) -- replication R72 running.

## CORRECTIONS from the independent review of ROUND 7 + NECESSITY (subagent, 2026-11-04 afternoon; recomputed from raw)
1. vr_alpha is NOT "the share of the vital rates' time change" (mislabel, established): w_scale_raw is refitted, and the
   reproduction field's raw scale grows 2.65-3.24 (a = 1) -> 4.07-4.11 (a = .5 / .75), winning the time change back. Realised
   occupied-East swing, a = .75 vs a = 1 under the same observers: R67 1940->2010 1.07 vs 1.06, 1966->2010 .93 vs .90; R70 .92
   vs 1.00 -> a = .75 removes 0-9% of the realised swing, not 25%, and WIDENS the spread (2010 q90 3.02 vs 2.78 R67; 2.92 vs
   2.71 R70). The R68 claim "the excess sits at the modern end / upper tail" is REVERSED under matched observers. The path
   cannot answer "how large a swing is necessary". 1940 values are DESK backcast extrapolations (G6): report 1966.
2. Decoupled arms carry a fitted constant low-density multiplier G: a = .75 G .90-1.02; a = .5 .69-.81; a = 0 .68 -> a <= .5
   arms hide a 20-30% low-density fecundity penalty; G-inclusive effective lambda (median rates, approx.) East 1940 ~.27 / .24 /
   .28 and 2010 ~.79 / .99 / 1.05 at a = .5 / .75 / 1 -- in effective terms a = .5 trims the MODERN end (J22 again).
3. R68's a = 1 comparator (R59 fit_obs) had observers fitted JOINTLY (co-adaptation penalty); clean a = .75 vs 1 gaps are R67
   +542 and R70 +238 -- at or below noise. "a = 1 is never best" is not evidence of a convenient excess. (R71 then made a = 1
   decoupled the best held-out arm.)
4. a = .5 adequacy is observer-set dependent (R67 +1,570; R68 / R70 best) -> "the deep 1940 sink is necessary" is NOT robust;
   two of three fair comparisons accept a shallower start (occupied-East 1966 median .58-.66 vs .50-.54).
5. All necessity arms stopped at caps / ABNORMAL with |grad| in the thousands (only R66 B converged); the ~500 noise floor
   was never measured with restarts for these arms.
6. R69 rate mixing is consistent (c re-derived via c_of, checked against the model's c); no Sj <= Sa clamp across passes;
   lambda_split reports FROZEN rates for swapped ones -> the Sa-only arm's swing is UNKNOWN.
7. The two common observer sets are not neutral: static-set observer effects SD .54 vs fit-set .38, correlation .67 (the
   static structure loads more onto route effects).
8. Only habitat sites are refitted; disease (on K from 1993), the continental K trend, dispersal, init, overdispersion stay
   at the full-data fit; post-split NestWatch stays in -> "a = 0 always loses" margins are partly conditional on latents
   tuned to the time-varying fit; a < 1 wins are conservative.
9. Potential lambda ignores G, Allee, immigration, disease; in sinks abundance ~ immigration / (1 - lambda) -> report 1966, include
   G, add realised growth. R60 / R61 / R64 were scored with full-data observers (conditional). R70 "post-2005 blocks" are 2010-20.
10. Plan row A19 mislabels ("~25% larger than forecasting needs"); "ROBUST CONCLUSION" given on gaps of 40-542 (J21/J22).
Synthesis per review: robust = some eastern rise needed (constant rates always lose, conditional on the latent leak); the
modern end is pinned (occupied-East 2010 median 1.29-1.56, 59-69% sources); adult survival's change carries most forecast
value. NOT established: the size of the necessary swing (1966 -> 2010 somewhere between ~.66-.75 and ~.9-1.0); convenient part
0-35%, unresolved; juvenile survival + fecundity (~70% of the potential-lambda rise) carry little separate forecast value.

### R73 PRE-REGISTRATION (review item 2): a GENUINE swing path -- habitat weights held at the fit's values
Decoupled (K real, Z_disp real, constant G, n50 pinned, static-set observers fixed, train <= 1993); --exclude_targets
manifold_factor,w_scale_raw (the fit's weights fixed; only the four rate intercepts alpha_a/j/f/k + G refitted), so
vr_alpha a really scales the fit's OWN time change in the vital rates (spatial pattern fixed); a in {0, .25, .5, .75, 1}.
Report held-out, 2021-25, blocks and the realised occupied-East swing (1966 -> 2010 and 1940 -> 2010).
Prediction (review): the realised swing scales with a; a = .5 costs > 1,000 vs the best; falsified if a = .5 is within 500
(then up to ~half the swing is convenient).

### R74 (review item 1): readout-only rescoring with a swap-aware, G-inclusive lambda (lambda_split now takes real-pass
rates for --real_rates and reports low-density lambda Sa + Sj Fmax G for occupied cells, 1940 and 1966 -> 2010) on the R69
per-rate arms and the R67 path. Prediction (review): the Sa-only arm's occupied 1966 -> 2010 swing <= .45 while keeping 86%
of the skill; falsified if >= .8 (then Sa-only is not a smaller swing).

### R74 RESULT: swing vs forecast skill (occupied East, n 863; static-set observers; train <= 1993; HELD-OUT 1994-2020)
Low-density lambda = Sa + Sj Fmax G (G = fitted constant low-density multiplier), median over occupied cells, swap-aware.
| arm | 1966 -> 2010 low-density lambda (swing) | potential lambda 1966 -> 2010 | 2010 q90 (potential) | held-out | vs best |
|---|---|---|---|---|---|
| all frozen (a = 0) | .85 -> .85 (0) | 1.08 -> 1.08 | 2.54 | 98,009 | +7,530 |
| Fmax only real | .76 -> 1.09 (.33) | .89 -> 1.36 | 3.05 | 99,630 | +9,151 |
| Sj only real | .55 -> 1.37 (.82) | .56 -> 1.44 | 3.01 | 96,879 | +6,400 |
| Sa only real | .70 -> 1.17 (.47) | .77 -> 1.25 | 2.72 | 92,227 | +1,748 |
| Sa + Sj real | .43 -> 1.46 (1.03) | .45 -> 1.53 | 2.87 | 92,652 | +2,173 |
| a = .5 (weights refit) | .53 -> 1.16 (.63) | .60 -> 1.35 | 3.10 | 92,883 | +2,404 |
| a = .75 (weights refit) | .51 -> 1.45 (.94) | .50 -> 1.43 | 3.02 | 91,313 | +834 |
| a = 1 decoupled (R71) | .45 -> 1.58 (1.13) | .44 -> 1.51 | 2.94 | 90,479 | best |
Read (likely; one split, one observer set, caps): forecast skill is NOT a function of swing size alone -- it depends on
WHICH rate carries the change. Adult survival's change alone yields a modest swing (.47; 2010 low-density median 1.17)
and recovers ~77% of the frozen->best gain; juvenile survival's change alone yields a LARGER swing (.82) but only ~15% of
the gain; fecundity's alone (.33) is worse than none. The best arm has the full swing (1.13; 2010 median ~1.5-1.6), but
under the ~500 noise rule nothing short of it (a = .75 aside, +834) is within noise. Pre-registered test (Sa-only swing
<= .45 with ~86% of skill): ESSENTIALLY MET (.47; 77% vs the R71 best). So: the part of the eastern rise that is robustly
supported is adult survival's (~.5 in low-density lambda, 1966 -> 2010); the further ~.6 carried mainly by juvenile survival
buys the last ~quarter of forecast skill -- real but less robust, and juvenile survival's change on its own is a poor
forecaster (a candidate "convenient lever" that helps only in combination).

### R73 path so far (weights FIXED at the fit's: a now truly scales the fit's own time change; static-set observers; <= 1993)
| a | training | held-out 1994-2020 unmasked | 2021-25 clean | post-1993 blocks | occupied East potential lambda 1966 -> 2010 (q90 2010) |
|---|---|---|---|---|---|
| 1 | 51,601 | 89,792 (best) | 21,731 | 21,424 | .48 -> 1.59 (3.00) |
| .75 | 51,573 | 90,345 (+553) | 21,782 (+139) | 21,101 (+428) | .56 -> 1.47 (3.26) |
| .5 | 51,857 | 91,690 (+1,898) | 21,643 (best) | 20,673 (best) | .65 -> 1.26 (3.25) |
Read so far (likely): with the weights fixed the realised swing scales with a (1.11 / .91 / .61). On the unmasked 1994-2020
rows the full swing forecasts best (monotone); on the CLEAN rows no fit ever saw, the larger swing does NOT help: 2021-25 is
flat (within ~140) and the spatial blocks favour the SMALLER swing (by ~750). Likely reason for the disagreement: the
unmasked 1994-2020 rows are scored with non-habitat latents (disease from 1993, continental K trend, dispersal, init) tuned
to the FULL-data fit's path (review item 8) -- the clean rows were never in any objective. So the clean-row evidence does
not require more than ~half the fit's time change (swing ~.6 in occupied-East potential lambda, 1966 .65 -> 2010 1.26);
the unmasked-row preference for the full swing may be the latent leak. Pending a = .25 / 0.

### R73 RESULT (complete): genuine swing path (fit's weights fixed; static-set observers; train <= 1993; caps)
| a | swing: occupied East potential lambda 1966 -> 2010 | held-out 1994-2020 (latent-leaky) | 2021-25 CLEAN | blocks CLEAN |
|---|---|---|---|---|
| 0 | 1.26 -> 1.26 (0) | 95,015 | 25,308 (+3,665) | 21,391 (+718) |
| .25 | .79 -> 1.13 (.34) | 96,849 | 22,230 (+587) | 20,714 (+41) |
| .5 | .65 -> 1.26 (.61) | 91,690 (+1,898) | 21,643 (best) | 20,673 (best) |
| .75 | .56 -> 1.47 (.91) | 90,345 (+553) | 21,782 (+139) | 21,101 (+428) |
| 1 | .48 -> 1.59 (1.11) | 89,792 (best) | 21,731 (+88) | 21,424 (+751) |
(low-density G-inclusive lambda falls with smaller a because the fitted low-density multiplier G drops -- e.g. a = .5
.30 -> .53: with less time change the model adds an Allee-like low-density penalty; potential lambda is the comparable
column.)
NECESSITY READ-OUT on the CLEAN rows (never in any objective; free of the latent leak): a swing of at least ~.6 (a ~ .5) is
NECESSARY -- a = .25 / 0 lose clearly on 2021-25 (+587 / +3,665); beyond that, a = .5 / .75 / 1 are equivalent on 2021-25
(within ~140), and the spatial blocks PREFER the smaller swings (a = .25-.5 best; a = 1 +751). The unmasked 1994-2020 rows
prefer the full swing, but they are scored with disease / trend / dispersal latents tuned to the full-data fit.
Likely conclusion (in the user's terms): roughly HALF of the fit's eastern rise in the vital rates is robust and necessary
-- occupied-East median potential lambda from ~.65 (1966) to ~1.26 (2010): from a mild sink to a modest source. The other
half (to the fit's ~.48 -> ~1.59) is not required by any clean held-out test and is disfavoured at new places: the
"convenient lever" part, about half the swing, consistent with R74 (adult survival's ~.5 carries most of the skill).
Caveats: one split, one observer set, caps (a = 0 / .25 non-monotone on the leaky rows); weights held at the FIT's values
(spatial pattern not re-optimised); potential lambda ignores Allee / G / movement.

### R75 PRE-REGISTRATION: R73 replicated at the 2005 split (weights fixed; observers = R70 static_obs05 training-only set)
a in {.25, .5, .75, 1}; train <= 2005; clean rows = 2021-25 and the 2010-20 blocks. Prediction: on the clean rows a ~ .5
again suffices (a = .5 within ~300 of the best) and a = .25 loses > 500.
### R76 PRE-REGISTRATION: R73 with the OTHER observer set (R59 fit_obs), a = .5 and 1, train <= 1993 (CPU). Prediction:
a = .5 within ~300 of a = 1 on the clean rows (2021-25, blocks).

### R75 RESULT: genuine swing path at the 2005 split (fit's weights fixed; R70 training-only observers; caps)
| a | occupied East potential lambda 1966 -> 2010 | held-out 2010-20 | 2021-25 CLEAN (E / P / W) | 2010-20 blocks CLEAN |
|---|---|---|---|---|
| .25 | .90 -> 1.22 | 36,999 (+759) | 24,453 (+1,916; 14,595 / 1,531 / 8,327) | 7,468 (+175) |
| .5 | .68 -> 1.27 | 36,392 (+152) | 22,611 (+74; 13,102 / 1,374 / 8,135) | 7,293 (best) |
| .75 | .57 -> 1.46 | 36,240 (best) | 22,687 (+150) | 7,424 (+131) |
| 1 | .47 -> 1.59 | 36,256 (+16) | 22,537 (best; 12,965 / 1,321 / 8,251) | 7,471 (+178) |
Prediction (a = .5 within ~300 of the best on clean rows; a = .25 loses > 500) MET at the second split: a = .5-1 are
equivalent on every clean score (within ~180), and a = .25 loses badly on 2021-25 (+1,916, mostly East). At this split even
the leaky 2010-20 rows do not separate a = .5 from 1 (+152).
REPLICATED CONCLUSION (likely; two splits, two observer sets so far for a = .5 vs 1): the necessary eastern rise is about
half the fit's time change -- occupied-East median potential lambda ~.65-.68 (1966) -> ~1.26-1.27 (2010), a swing of ~.6;
the remaining ~.5 of the fit's swing (to ~.47 -> ~1.59) is neither required nor clearly harmful by any clean test -- it is
a free direction the data cannot pin (the "convenient lever" in the sense of unconstrained, not wrong).

### R73 / R75 CAVEAT (found on readout): the half-swing arms compensate with a strong LOW-DENSITY brake
Fitted constant low-density multiplier G (N/K-gated, fades over rho ~.11-.28 of K): a = 1 G .88-.93; a = .75 .52-.55;
a = .5 .16-.19; a = .25 .06-.10. Front offsets are fit-like for a >= .25 (e.g. a = .5: +2/-1/-2/-6, never .082-.083; fit .068).
So "half the swing suffices" holds only TOGETHER WITH an ~80% cut in low-density fecundity (G-inclusive low-density lambda
a = .5: .30 -> .53, i.e. sparse cells stay deep sinks throughout) -- the strong-Allee <-> habitat-change substitution (R26-R30)
again, here through an N/K-gated readout (J19). The clean-row equivalence of a = .5-1 is therefore an equivalence between
(i) the full rise with no low-density penalty and (ii) half the rise plus a strong low-density penalty. Forecasting alone
cannot separate them. Next: R77, the same path with G pinned at 1 (no low-density compensation).

### R77 PRE-REGISTRATION: genuine swing path with NO low-density compensation (G pinned at 1)
As R73 (fit's weights fixed, static-set observers, train <= 1993) plus --fix_g0 (with ridge b ~ 0, G = 1 everywhere).
a in {.5, .75, 1}. Prediction: without the low-density brake, a = .5 now loses clearly on the clean rows (2021-25 > 500 worse)
and/or puts fronts early; if a = .5 still matches a = 1 on clean rows with fit-like fronts, half the swing is genuinely
sufficient.

### R77 RESULT: swing path with G PINNED at 1 (no low-density compensation; fit's weights fixed; static-set observers; <= 1993)
| a | training | held-out | 2021-25 CLEAN | blocks CLEAN | potential lambda 1966 -> 2010 (occ. East) | fronts / never |
|---|---|---|---|---|---|---|
| .5 | 53,291 | 95,970 | 25,248 (+3,443 vs a = 1) | 22,799 (+1,286) | .67 -> 1.10 | -7/-7/-4/-7.5 / .078 |
| 1 | 51,611 | 89,808 | 21,805 | 21,513 | .49 -> 1.55 | +3/0/-3/-6 / .069 |
(both ABNORMAL line-search stops at ~200 evals; a = .75 running.) With G free (R73): a = .5 2021-25 21,643, a = 1 21,731.
Read (likely): WITHOUT the low-density brake, half the swing FAILS on every clean score (2021-25 +3,443, blocks +1,286) and
puts fronts 4-7 yr early; the full swing with G = 1 is as good as the full swing with G free (21,805 vs 21,731). So the
R73 / R75 "half suffices" equivalence was entirely bought by the low-density brake. RESOLVED: the data need EITHER (i) the
fit's full rise in the vital rates (occupied-East potential lambda ~.5 -> ~1.55, 1966 -> 2010) with no extra low-density
penalty, OR (ii) about half that rise plus an ~80% cut in fecundity at low density (the Allee-strength substitution). Which
one is right is not decided by forecasting; it is the same E002 / E055 / R26-R30 trade-off, and the user's prior plus the
literature-consistent pin (R44a) favour (i) -- i.e. the fit's swing -- over a strong low-density brake.
R77 a = .75 (G = 1): training 51,898; held-out 91,854 (+2,046 vs a = 1, leaky rows); 2021-25 21,941 (+136, within noise);
blocks 20,942 (-571, better); fronts +2/-1/-3/-6, never .098; occupied East potential lambda .44 -> 1.21 (swing .77; a = 1:
.49 -> 1.55, swing 1.06).
REFINED (likely; one split, one observer set, ABNORMAL stops): with no low-density brake, about THREE-QUARTERS of the fit's
swing in the vital rates suffices on the clean rows (2021-25 within noise; blocks better) while half does not. The necessary
rise is from a 1966 occupied-East median of ~.45 (deep sink) to a 2010 median of ~1.2 (moderate source); the fit's further
rise to ~1.55 is supported only by the latent-leaky 1994-2020 rows -- the convenient excess sits at the MODERN end
(~.3 of median lambda, ~25-30% of the swing) when no low-density brake is allowed.
### R78 PRE-REGISTRATION: R77 (G pinned at 1) replicated at the 2005 split, a in {.75, 1} (+ .5), R70 observers. Prediction:
a = .75 within ~300 of a = 1 on 2021-25 and not worse on blocks; a = .5 loses > 500.

### R72 RESULT: per-rate necessity with the OTHER observer set (R59 fit_obs; train <= 1993)
| rates reading real Z | held-out 1994-2020 | 2021-25 clean | blocks clean |
|---|---|---|---|
| Sa only | 109,160 (best of this set) | 24,847 | 20,820 |
| Sj + Fmax (complement) | 118,548 | 26,374 | 22,311 |
| refs: a = .5 / .75 (R68) | 110,135 / 110,180 | 23,992 / 23,846 | 20,491 / 20,658 |
| ref: fit structure (R59 fit_obs) | 111,430 | 24,639 | 21,303 |
Read (likely): REPLICATED -- adult survival's time change alone does as well as or better than any arm on the 1994-2020 rows,
while juvenile survival + fecundity without it are poor everywhere (held-out +9,388 vs Sa-only). On the clean 2021-25 rows
Sa-only is ~1,000 behind the a = .75 arm in this set (in the static set it was +750 vs a = .75): adult survival's change is
the backbone of the forecast, but the clean future rows also want some of the other rates' change.

### R78 RESULT: no-brake swing path at the 2005 split (G = 1; fit's weights fixed; R70 observers; short line-search stops)
| a | held-out 2010-20 | 2021-25 CLEAN | 2010-20 blocks CLEAN | occupied East potential lambda 1966 -> 2010 | fronts / never |
|---|---|---|---|---|---|
| .5 | 38,678 (+2,438) | 25,722 (+3,143) | 8,269 (+940) | .74 -> 1.16 | -8/-8/-5/-8 / .060 |
| .75 | 36,456 (+216) | 22,690 (+111) | 7,329 (-153) | .45 -> 1.22 | +2/-2/-3/-6.5 / .097 |
| 1 | 36,240 | 22,579 | 7,482 | .46 -> 1.53 | +3/0/-3/-6 / .075 |
Prediction (a = .75 within ~300 of a = 1 on clean rows, not worse on blocks; a = .5 loses > 500) MET at the second split.
REPLICATED (likely; two splits; one observer set per split; short ABNORMAL stops, 120-180 evals): with no low-density brake,
~3/4 of the fit's swing suffices on every clean score while half fails badly (and puts fronts 5-8 yr early). In occupied
eastern cells the necessary rise is median potential lambda ~.45 (1966) -> ~1.2 (2010); the fit's extra rise to ~1.53 at the
modern end is not required by any clean test (equivalent within noise; blocks slightly prefer less).
### R79 PRE-REGISTRATION: finer no-brake swing grid (G = 1, weights fixed, static-set observers, train <= 1993), a in
{.6, .7, .8, .9}. Prediction: clean-row scores reach the a = 1 level (within ~300 on 2021-25) somewhere in .6-.75.

## CORRECTIONS from the independent review of R71-R78 (subagent, 2026-11-04 evening; recomputed from raw + npz + traces)
1. a scales the time change only in Z-space: intercepts move a lot, INCLUDING K's -- R77 alpha_k -1.72 (a = 1) / -1.84 (.75) /
   -2.00 (.5): K ~12% / 24% lower (R78 K x.81 / x.69); adult survival drops a level (a = .75 Sa .14 -> .30 vs a = 1 .20 -> .48).
   a < 1 arms differ in time change AND survival level AND K; the realised swing is not proportional to a (.77 vs 1.06 at .75).
   The 1966 level is held ~.44-.49 for every a >= .7; the arms take their cut at the MODERN end by lowering levels.
2. --fix_g0 pins G = 1 (to 1e-8); rho carries no gradient at G = 1 (stays log rho -1.309). Ridge 1e-6 crushes the 64 G
   slopes (b_norm 1e-9 to 1e-7) in all R73/75/77/78 runs.
3. CONVERGENCE CAVEAT REVERSED: R77/R78 "ABNORMAL" stops are effectively 4-parameter fits at |grad| 0-5 (stationary points),
   BETTER converged than the G-free R73/R75 arms (600-it caps, |grad| 40-32,000). Unknown: multiple optima (alpha_a jumps
   with a: .63 / .16 / -.16 / .003 / .63 for a = .5/.6/.7/.75/1 -> an Sa-Sj ridge or several basins); no restarts run.
4. Clean-row noise never measured; proxy from the three a = 1 near-twins (R71/R73/R77): 2021-25 spread ~75, blocks ~210. The
   a = .75 vs 1 gap on 2021-25 (+136 R77, +111 R78) is 1.5-2x that spread, same sign at both splits, from the EAST (+146 /
   +165; West -79 / -112). R79: 2021-25 keeps improving with a (.6 23,510; .7 22,042; .75 21,941; 1 21,805) -- no plateau;
   "equivalent" is a threshold call (M13), not an equivalence test.
5. UNREPORTED clean evidence: PRE-1994 blocks (invasion era, never fitted). East pre-1994 blocks: a = .5 5,424 / .6 5,212 / .7
   4,959 / .75 4,953 / 1 5,008 (R77); R78 5,434 / 4,953 / 4,968 -> independently supports a >= .7.
6. Occupied-East medians: "occupied" = ever detected, so the 1966 median is mostly not-yet-colonised cells (pinned via front
   timing). The upper tail does not move with a (2010 q90 2.86-2.91 for a = .7 / .75 / 1): the "paradise" tail is SPATIAL
   (fixed weights), not the time change. The deep 1966 sink rests on adult survival ~.14-.20 -- likely far below published
   House Finch adult survival (to check against the literature).
7. R72 replicates R69 only on the LEAKY rows: on clean rows Sa-only is worse than every reference in that set (2021-25 24,847
   vs a = .75 23,846, fit 24,639; blocks 20,820 vs a = .5 20,491); in the static set (R74) Sa-only recovers 65% of the
   frozen->best 2021-25 gain and on blocks is WORSE than all-frozen. The Sj+Fmax complement is poor everywhere (holds). R72
   summaries' lambda block is the stale frozen readout -- do not quote. "Adult survival is the forecast backbone" rests mostly
   on leaky rows (inconsistent evidence standard).
8. J18: never-reached .096-.102 for a = .6-.75 vs .069 at a = 1 (R78 .097 vs .075); the PLAINS pay at a = .75 (2021-25 +68 /
   +57, post-1993 blocks +79) (M15). J22: summaries do not record exclude_targets / fix_g0; edited_inputs omits the refitted K
   intercept. J21: "RESOLVED" / "REPLICATED" on gaps of ~110-150 with unmeasured noise.
9. Alternatives: dispersal is FIXED (front speed trades growth vs dispersal; a = .5's early fronts might be fixed by refitted
   dispersal) -> the "necessary deep start" is conditional on the fit's dispersal, Allee pin, init. K carries the abundance
   trend, so the modern potential lambda barely affects established-cell abundance -> "the extra to 1.53 is unconstrained" is
   partly structural non-identifiability, not evidence of convenience. Mixed evidence on the modern level: in-sample at 2005
   favours a = 1 (+1,149); clean post-1993 EAST blocks favour LOWER modern lambda (a = .7 13,113 / .75 13,206 vs 1 13,733).
   The low-density-brake substitution remains open (R73 a = .5 with G .19 is best or tied on every clean score incl.
   pre-1994 blocks); only the prior / pin argue against it.
SYNTHESIS (review): established -- some rise needed; without the brake a = .5 fails everywhere. Likely -- occupied eastern
cells had to start DEEP (1966 median ~.45, invariant for a >= .7, backed by pre-1994 clean blocks and timing); the minimum
swing is ~.7 (a = .7: .43 -> 1.14). UNKNOWN -- the modern level between ~1.1 and ~1.55 (weakly identified, not demonstrated
convenience). All conditional on no brake, the fit's dispersal / disease, one observer set per split, no restarts.

### R80 PRE-REGISTRATION (review item 1): is the deep start a growth-vs-dispersal (or disease) trade?
R78 settings (G = 1, fit's weights fixed, R70 observers, train <= 2005) + --extra_targets dispersal_logit_intercept,
dispersal_logit_slope and the 11 disease sites (disease_b_late, _b_native_raw, _hill_n_raw, _lag0, _log_k_half_counts,
_log_tau_rec, _mu_sev, _rec_raw, _tau_raw, _w_lag, _w_sev), refitted per arm on <= 2005 rows; a in {.5, .75, 1}. GPU.
Prediction (review): a = .5's fronts move toward the fit's but it stays > 500 worse on 2021-25; falsified if a = .5 lands
within ~300 with fit-like fronts and never-reached share (then the deep start is a growth-dispersal trade, not necessary).

### Literature / data anchor for the deep start (records: SURVIVAL_BBL_SCOPING_2026-10-31.md s.2.1)
Bird Banding Lab dead recoveries (crude geometric MLE, L): adult survival East of 80W 1960-93 .56 (n 414, SE .02); 100-80W
.53 (.03); West .51 (.02); all adults .53; post-1994 East .49 (n 41, SE .06). Literature: Milby & Wright .72 (small n);
Sacramento CJS .59 / .47 (before / after West Nile); MAPS .48-.49.
Against the necessity arms: the deep 1966 eastern start is carried by MEDIAN adult survival ~.14-.20 (R77 / R78) and the fit's
own East median Sa in 1940 is ~.13-.14 (R28a / R67 rates) -- about a QUARTER of the banding estimate for the East in
1960-93 (.56), when birds were present and banded. Read (likely): the fitted deep sink is carried by an adult survival
far outside the empirical range for the very region and era it describes; whatever makes early eastern cells hostile, it is
not plausibly adult survival at ~.15. This is a strong plausibility argument that part of the fit's early-era "hostility"
sits in the wrong rate. Caveats: banding birds are in occupied, often feeder-associated places (selection); the crude MLE
assumes constant reporting; median over all occupied cells vs banding locations. Next: R81 -- anchor adult survival to the
BBL level and see where the necessary early hostility goes (juvenile survival? fecundity? the Allee brake?).

### R79 RESULT: fine no-brake grid (G = 1, fit's weights fixed, static-set observers, train <= 1993; stationary stops)
| a | 1966 -> 2010 occ.-East potential lambda | 2021-25 CLEAN | post-1993 blocks CLEAN | PRE-1994 blocks CLEAN (E / W) | never | fronts |
|---|---|---|---|---|---|---|
| .5 | .67 -> 1.10 | 25,248 | 22,799 | 11,438 (5,424 / 5,890) | .078 | -7/-7/-4/-7.5 |
| .6 | .52 -> 1.08 | 23,510 | 21,648 | 11,069 (5,212 / 5,727) | .096 | -3/-4/-3/-7 |
| .7 | .43 -> 1.14 | 22,042 | 20,852 | 10,727 (4,959 / 5,639) | .102 | +2/-1/-3/-6 |
| .75 | .44 -> 1.21 | 21,941 | 20,942 | 10,733 (4,953 / 5,653) | .098 | +2/-1/-3/-6 |
| .8 | .45 -> 1.29 | 21,855 | 21,036 | 10,750 | .093 | +3/-1/-3/-6 |
| .9 | .47 -> 1.42 | 21,776 | 21,264 | 10,808 | .078 | +3/-1/-3/-6 |
| 1 | .49 -> 1.55 | 21,805 | 21,513 | 10,865 (5,008 / 5,736) | .069 | +3/0/-3/-6 |
Read (likely): the clean rows pull in DIFFERENT directions on the modern end: 2021-25 improves until a ~ .8-.9 and is flat
after (21,855 / 21,776 / 21,805); post-1993 blocks are best at a = .7 (20,852) and worsen steadily to a = 1 (+661); pre-1994
blocks are best at a = .7-.8 (10,727-10,750; a = 1 +138). The 1966 start is invariant (~.43-.49) for a >= .7 and is what the
clean rows need (a = .5-.6 lose on everything, incl. fronts). The never-reached share is HIGHEST at a = .7-.75 (~.10) and
falls to the fit's .069 at a = 1 (J18 trade: the fringe needs the modern rise). Synthesis: the necessary deep start is
robust; the modern level is a three-way compromise -- future known places (2021-25) want ~1.3-1.4, new places (blocks)
want ~1.1-1.2, fringe colonisation (never-reached) wants ~1.55. Not a single "convenient lever": a genuinely weakly
identified, data-conflicted direction.

### R81 PRE-REGISTRATION: anchor adult survival to the banding level -- where does the necessary early hostility go?
Decoupled, a = 1, G pinned at 1, habitat weights FREE (so the time change can move between rates), static-set observers,
train <= 1993; --sa_anchor .55,.03 (mean Sa over occupied East cells 1966-93 ~ BBL East 1960-93 .56). Twin without anchor.
Prediction: with Sa held near .55, the deep early eastern start moves into juvenile survival and / or fecundity; clean-row
scores within ~500 of the twin (the anchor is cheap, as C2's fixed anchor cost 51-502); if clean rows lose > 1,000, the
counts genuinely need low early adult survival (or something it stands for).

### R80 RESULT (a = .5 and 1): dispersal + disease latents REFITTED per arm (G = 1, weights fixed, R70 observers, <= 2005)
| a | training (<= 2005) | held-out 2010-20 | 2021-25 CLEAN | 2010-20 blocks CLEAN | PRE-1994 blocks CLEAN | fronts / never |
|---|---|---|---|---|---|---|
| .5 | 94,536 (+2,322) | 36,553 (-132) | 22,638 (-65) | 7,611 (+31) | 11,567 (+818) | -7/-7/-3/-5 / .100 |
| 1 | 92,214 | 36,685 | 22,703 | 7,580 | 10,749 | +3/0/-2/-6 / .066 |
(R78, latents fixed: a = .5 2021-25 25,722 -> refitting dispersal + disease rescues it by ~3,100.)
Prediction (a = .5 fronts move toward the fit but stays > 500 worse on 2021-25) PARTLY FALSIFIED: on the MODERN clean rows
a = .5 is now equivalent to a = 1 (2021-25 -65, blocks +31, 2010-20 -132) once dispersal and disease can adjust; but the
INVASION ERA still rejects it -- fronts stay 5-7 yr early, the training fit (1966-2005) is 2,322 worse, and the clean
PRE-1994 blocks are 818 worse.
Read (likely): the necessity of the eastern swing comes from the INVASION ERA (1966-93: front timing, early fill, pre-1994
blocks), not from modern forecasting -- modern counts can be matched with half the swing if disease / dispersal re-tune.
The robust, necessary part is the deep early start (occupied-East potential lambda ~.45 in 1966) that times the invasion;
the size of the modern end is supported by neither clean modern test once nuisance latents are free.
R80 a = .75 (dispersal + disease refitted): training 93,171 (+957 vs a = 1); held-out 2010-20 37,107 (+422); 2021-25 22,521
(-182, best); blocks 7,494 (-86, best); PRE-1994 blocks 10,733 (-16, best); fronts +2/-1/-2/-4, never .090; occupied-East
potential lambda .46 -> 1.21.
Read (likely): with nuisance latents free, a = .75 is best or tied on EVERY clean score incl. the invasion-era pre-1994
blocks (10,733 vs a = 1 10,749), while a = .5 fails the invasion era. The NECESSARY eastern change: the deep start (~.45 in
1966) with a rise to ~1.2 by 2010 (swing ~.75); the fit's extra modern rise to ~1.5 is preferred only by the TRAINING fit
(+957) -- i.e. by in-sample fitting, not by any clean test. That extra ~.3 at the modern end is the best current candidate
for "convenient lever" (in-sample-only support), conditional on G = 1, the fit's weights, and one split.

### R76 RESULT: G-free swing path (weights fixed) with the OTHER observer set (R59 fit_obs; train <= 1993)
| a | G | held-out | 2021-25 | post-1993 blocks | pre-1994 blocks | occ.-East potential lambda 1966 -> 2010 | fronts / never |
|---|---|---|---|---|---|---|---|
| .5 | .19 | 108,504 (+1,776) | 23,278 (-71) | 20,530 (-533) | 10,654 (+24) | .65 -> 1.25 | +2/-1/-2/-6 / .078 |
| 1 | .90 | 106,728 | 23,349 | 21,063 | 10,630 | .46 -> 1.59 | +3/0/-3/-6 / .064 |
Read (likely): REPLICATES R73 with the other observer set -- with the low-density brake free (G .19 at a = .5), half the swing
matches the full swing on all clean rows (2021-25 -71, pre-1994 blocks +24, post-1993 blocks -533 better). The brake
substitution is robust across observer sets; without the brake (R77-R80) the invasion era needs ~3/4.

### R82 PRE-REGISTRATION (review item 3): is the POST-1993 part of the rise needed for forecasting? (rescore only)
The R77 a = 1 (G = 1) model trained on <= 1993 is rescored with the vital rates reading only a share a_late of Z's change
after 1993 (--vr_alpha_late {0, .5, 1}; K still real Z; training is unaffected by construction). Prediction: if the modern
end is needed for forecasts, a_late = 0 loses > 500 on 2021-25 and post-1993 blocks; if within ~300, the post-1993 rise
in the vital rates carries no forecast information beyond K's.

### R82 RESULT: scaling only the POST-1993 part of the vital rates' change (rescore of R77 a = 1, G = 1, trained <= 1993)
| a_late | occ.-East potential lambda 1966 -> 2010 | held-out 1994-2020 | 2021-25 CLEAN (East) | post-1993 blocks CLEAN |
|---|---|---|---|---|
| 1 | .49 -> 1.55 | 89,808 | 21,806 (12,639) | 21,514 |
| .5 | .49 -> 1.29 | 90,343 (+535) | 22,162 (+356; East +352) | 21,336 (-178) |
| 0 | .49 -> 1.04 | 92,485 (+2,677) | 23,748 (+1,942; East +1,626) | 21,350 (-164) |
Prediction (a_late = 0 loses > 500 on 2021-25 and blocks) MET for 2021-25 (+1,942), NOT for blocks (-164, slightly better).
Read (likely, with latents FIXED at the full-data fit): the vital rates' post-1993 rise carries forecast information at
places with count history (mostly East) -- halving it costs ~350 on 2021-25 (near noise), removing it ~1,900 -- but none at
new places. This conflicts with R80 (latents refitted per arm), where modern clean rows accepted half the overall swing:
the modern-end requirement is sensitive to whether disease / dispersal can re-tune. Net: the post-1993 rise is at least
partly supported, but by place-specific and latent-conditioned evidence -- the weakest-identified part of the swing.

### R81 RESULT: soft BBL anchor on early eastern adult survival (G = 1, a = 1, weights FREE, static-set observers, <= 1993)
| arm | training (no penalty) | mean Sa, occupied East 1966-93 (backed out of the penalty) | held-out | 2021-25 | post-1993 blocks | pre-1994 blocks |
|---|---|---|---|---|---|---|
| twin (no anchor) | 50,957 | ~.315 | 90,786 | 21,792 | 21,272 | 10,723 |
| anchor .55 +- .03 | 50,953 (+penalty 26) | ~.334 | 90,727 | 21,800 | 21,280 | 10,724 |
Read (likely): the soft anchor barely moved early eastern adult survival (.315 -> .334; the data accepted a 26-nat penalty
rather than move toward .55) and changed nothing on any clean score -- so the counts strongly prefer early eastern Sa far
below the banding estimate (.56), and the soft constraint was too weak to test where the hostility would go. (Shapley split
unchanged: Sa .28 / Sj .44-.45 / Fmax .28.) Next: R81b, a HARD anchor (SD .005), to measure what forcing Sa ~ .55 costs and
which rate takes over the early hostility.

### R81b RESULT: HARD BBL anchor (SD .005) on early eastern adult survival (G = 1, a = 1, weights free, <= 1993; cap)
Achieved mean Sa over occupied East cells 1966-93 ~.51 (penalty 32.5; twin ~.315). Cost vs twin: training +161 (51,118 vs
50,957, unpenalised); held-out 91,361 (+575); 2021-25 22,375 (+583); post-1993 blocks 21,686 (+414); PRE-1994 blocks 10,645
(-78, better); fronts +2/-1/-3/-7; never .046 (twin .074 -- better). East medians (all land): Sa .385 -> .646, Sj .089 ->
.154, Fmax 2.54 -> 3.95; Shapley Sa .34 / Sj .33 / Fmax .33. Occupied-East potential lambda 1966 -> 2010: .76 -> 1.55 (twin
.47 -> 1.50).
Read (likely): forcing early eastern adult survival up to the banding level costs only ~160 in training and NOTHING in the
invasion era (pre-1994 clean blocks slightly better; fronts on time; never-reached better), but ~400-600 on the modern clean
rows. The early hostility is NOT transferred to the other rates (Sj rises a little, Fmax unchanged): the 1966 occupied-East
median potential lambda becomes ~.76 instead of ~.47. So the "necessary deep start" was mostly an artefact of implausibly
low adult survival: with plausible adult survival, early eastern cells are mild sinks (~.75), not deep ones (~.45) -- and the
invasion is still timed correctly. The swing that remains (.76 -> 1.55, ~.8) is then carried evenly by the three rates.
What the modern clean rows lose (~500) is the forecast value that low early Sa provided through its large later rise.

### R81c / R81d PRE-REGISTRATION: replicate the hard adult-survival anchor
R81c: 2005 split (R70 observers), hard anchor (.55, .005) vs twin, G = 1, a = 1, weights free. R81d: 1993 split with dispersal +
disease latents ALSO refitted (anchor and twin). Prediction: as R81b -- the anchor costs < ~300 in training, nothing in the
invasion era (pre-1994 blocks, fronts, never-reached), < ~700 on modern clean rows; 1966 occupied-East potential lambda ~.7-.8.

### R83 PRE-REGISTRATION (Plan L, C1): BBL adult dead-recovery LIKELIHOOD instead of an anchor
bbl_prep.py: 8,778 cohorts (cell x banding year), 273,982 adults released (AHY+, normal wild; East of 100W Apr-Aug only),
543 dead recoveries at k >= 1 (West 328 / central 58 / East 157); Seber likelihood on the model's own Sa[cell, year] with 12
reporting-rate logits (3 regions x 4 eras; prior N(logit .005, 1.5)). Smoke at R71's parameters: BBL log-lik -5,225; model mean
Sa over occupied East cells 1966-93 .318 (BBL crude .56), 1994-2020 .516; West .399 / .445.
Arms: decoupled, a = 1, G = 1, habitat weights FREE, static-set observers, train <= 1993, --bbl; twin = R81 twin. GPU.
Prediction: the BBL term raises early eastern Sa substantially (toward ~.5) at small cost in the counts (< ~300 training),
no cost in the invasion era, and shallows the 1966 occupied-East start (cf. R81b ~.76); if Sa barely moves, the counts'
demand for low early Sa outweighs 543 recoveries (then the recoveries' information, not a fixed value, decides).

### R84 RESULT (user: "you haven't explored the disease model at all"): what the fit's disease process does
[cur] disease acts on SURVIVAL: added mortality m = amplitude(cell, t) x hill(N; n 2.27, k_half .041), ceiling .8.
| region (occupied cells) | 1966-93 | 1994-2000 | 2001-2010 | 2011-2025 |
|---|---|---|---|---|
| East: amplitude / realised mortality (q90) | .004 / .003 | .228 / .173 (.345) | .411 / .282 (.489) | .470 / .310 (.526) |
| Plains | .002 / .000 | .139 / .044 | .361 / .138 | .479 / .210 |
| West | .001 / .000 | .050 / .026 | .190 / .109 | .314 / .184 |
Disease OFF replay: delta -log-lik +43,984 (East 1994-2009 +13,951, 2010-25 +23,854; West 2010-25 +3,151); modern eastern
obs/expected falls 1.14 -> .26 (fourfold over-prediction without disease).
Read (established for the fit; likely for interpretation): disease removes ~28-31% of eastern survival from 2001 on (and a
rising ~11-18% in the West), so the fit's large modern rise in POTENTIAL lambda (habitat-driven, ~1.5) is offset by an
equally large disease toll -- the two are opposing levers; "potential lambda" without disease overstates the modern
"paradise". Data check: BBL eastern adult survival .56 (1960-93) -> .49 (1994-2025; n 41, SE .06): a ~12% drop, not ~30%
(crude, weak; caveat: banded birds near feeders may be the most exposed or best-fed). So a plausible-swing model must be
judged on EFFECTIVE (disease-inclusive) modern lambda, and the disease magnitude itself is a candidate convenient lever
paired with the vital-rate rise (more rise + more disease ~ same abundance).
BUG FOUND: the R83 BBL term used pre-disease Sa; fixed by censoring BBL intervals after 1993 (--bbl_until 1993; also
leak-free with train <= 1993). The first R83 launch (old code) was stopped before finishing.

### R85 PRE-REGISTRATION (L-DISEASE ii): which latent rescued the half swing in R80 -- disease or dispersal?
R80 settings (2005 split, G = 1, weights fixed, R70 observers) with ONLY the 11 disease sites refitted, or ONLY the 2 dispersal
sites, at a in {.5, 1}. Prediction: disease alone rescues a = .5 on 2021-25 (the disease toll trades against the modern
rise), dispersal alone does not; if dispersal does it, the trade is growth vs spread instead.

### R84b RESULT: EFFECTIVE (disease-inclusive) lambda in the fit's occupied cells (median; potential / effective)
| region | 1940 | 1966 | 1990 | 2000 | 2010 | 2020 |
|---|---|---|---|---|---|---|
| East | .31 / .31 | .48 / .48 | .94 / .94 | 1.27 / .89 | 1.53 / .95 | 1.56 / .96 |
| Plains | .34 / .34 | .48 / .48 | .51 / .51 | .66 / .62 | .92 / .78 | 1.06 / .86 |
| West | .55 / .55 | .75 / .75 | .90 / .90 | .97 / .91 | 1.11 / .88 | 1.19 / .94 |
(effective = Sa (1 - m) + Sj (1 - m) Fmax with m the realised disease mortality at the fit's own densities)
Read (established for the fit; likely for meaning): the fit's entire post-1990 rise in eastern potential lambda (.94 -> 1.56) is
CANCELLED by disease -- effective lambda stays ~.89-.96 after 1990. So:
- the fit's "paradise" end is a ridge between habitat-driven rates and disease magnitude (more rise + more disease ~ same
  abundance); only the effective modern level (~.95, near replacement for a saturated population) is pinned. This is very
  likely the R79 / R80 / R82 "weakly identified modern end": disease re-tuning let half the swing match on modern rows (R80).
- the identifiable eastern change is the EFFECTIVE rise from a deep sink (.31 in 1940; .48 in 1966) to near replacement by
  1990 -- a large swing but ending at ~.95, not ~1.55.
- the West shows the same pattern, weaker (potential 1.19 vs effective .94 in 2020).
Production implication to report: report and constrain EFFECTIVE lambda; the potential-lambda "paradise" is unidentified
without an independent handle on disease mortality (post-1994 BBL with disease-reduced survival; Mycoplasma survival literature).

### R81c RESULT: hard adult-survival anchor at the 2005 split (latents fixed; R70 observers)
| arm | training (incl. penalty) | held-out 2010-20 | 2021-25 | 2010-20 blocks | pre-1994 blocks | occ.-East potential lambda 1966 -> 2010 | East Sa median 1966 -> 2010 | never |
|---|---|---|---|---|---|---|---|---|
| twin | 91,335 | 36,346 | 22,486 | 7,456 | 10,723 | .47 -> 1.55 | .18 -> .44 | .064 |
| hard anchor | 91,682 (~+310) | 36,534 (+188) | 23,165 (+679) | 7,676 (+220) | 10,636 (-87) | .68 -> 1.55 | .34 -> .60 | .050 |
Replicates R81b (1993 split): plausible early adult survival costs nothing in the invasion era (pre-1994 blocks better,
fronts on time, never-reached better) and ~200-700 on the modern rows -- with disease FIXED at the full-data fit. Given R84b
(the modern end is a rates <-> disease ridge), the modern cost may be re-absorbed when disease can re-tune: R81d (1993, disease +
dispersal refitted) is the test.

### R86 PRE-REGISTRATION (Plan L A5 / S2): density-dependent juvenile survival -- does it bound the rates <-> disease ridge?
ddsurv_forward.py: juvenile stayers + arrivers x 1 / (1 + cs (N/K)), c re-solved with Sj / (1 + cs) so the equilibrium stays at
K; disease (on survival, before this step) lowers N and so RELAXES the juvenile density dependence (compensation).
Identity: cs -> 0 reproduces the fit exactly (135,590.798). Full data, fit structure (single field), habitat sites + the 11
disease sites refitted, observers at full-data values; scored in-sample and on the never-fitted rows (blocks, 2021-25).
Arms: (i) baseline (disease refit, no DD); (ii) + DD juvenile survival (cs free, start .5). CPU.
Prediction: with compensation available, (ii) fits at least as well with a SMALLER disease toll (realised eastern mortality
< ~.2 vs ~.3) and a smaller modern potential-lambda rise; if cs -> 0 or the toll does not shrink, fecundity-only crowding is
not what drives the ridge.
R86 note (user: "why relative density?"): for CROWDING, N/K is only a parameterisation of the resource scale (N/K = N x 1/K with
K free and spatial) -- unlike mate-finding / disease transmission, which are absolute-density encounters. The real flaw in
R86 (ii): re-solving c keeps the equilibrium IMPOSED at K. Added arms: (iii) --dd_emergent (DD juvenile survival, c NOT
re-solved: equilibrium falls below K and responds to rates / disease); (iv) --dd_only (no fecundity crowding at all; all
density dependence via juvenile survival; equilibrium N* = K/cs (Sj Fmax/(1 - Sa) - 1) EMERGES from the rates; K = resource
scale). Prediction: (iv) bounds the rates <-> disease ridge best -- higher rates now RAISE abundance, so they cannot be
inflated and cancelled by disease for free.

### R87 RESULT (user: "is the disease functional form another lever?"): how the fit uses each part of the disease form
m = ceiling x modifier(x) x gate(t - onset) x (1 - rec (1 - e^(-age / tau_rec))) x hill(N)   (age_fields.disease_mortality_amplitude)
- Arrival: fitted half-max onset East 1998 (q10-q90 1996-2001), Plains 2002, West 2005 vs the documented arrival map
  ~1996.6 / 2000.4 / 2004.4 -> follows the reconstructed spread within ~1-1.5 yr (lag0 .82, smooth lag field): NOT a lever.
- RESISTANCE / RECOVERY: prior recovery_logit N(-.5, 1) (median ~38% of the hit eventually recovered, tau ~12 yr); fitted
  rec_raw -3.26 (~4% recovered; ~2.8 prior SDs below), tau_rec ~16 yr. Mortality amplitude keeps RISING with years since
  onset (East .25 at onset -> .45 at 10 yr -> .48 at 25 yr): NO recovery. The data push resistance to ~0, so disease becomes a
  PERMANENT ~30% eastern survival tax -- exactly what cancels the modern potential-lambda rise (R84b). LIKELY a lever.
- SPATIAL SEVERITY FIELD (24 coefficients, budget .5): peak amplitude q10-q90 East .33-.57, West .18-.53 -- a flexible regional
  field that can absorb regional abundance misfit (cf. the static per-cell map, R34). LIKELY a lever.
- Ceiling .8 with median modifier .8: tolls > 50% are permitted by the prior.
Next: R88 diagnostic arms (full data, fit structure; habitat + remaining disease sites refit): (i) resistance HELD at the
prior median (rec_raw -.5) -- what does a recovering epizootic cost, and does the modern potential-lambda rise shrink?; (ii)
spatial severity field OFF (w_sev = 0). Diagnostics only (production: priors / likelihood, not fixed values).

### R81d RESULT: hard adult-survival anchor at the 1993 split with dispersal + disease latents REFITTED (anchor vs twin)
Neither arm converged (cap, ~900 evaluations). -log-lik sums over the held-out rows (post-1993 / pre-1994 clean blocks).
| arm | training (incl. penalty) | held-out BBS 1994-2025 | post-1993 blocks | 2021-25 | pre-1994 blocks | never | occ.-East potential lambda 1966 -> 2010 | East Sa median 1966 -> 2010 |
|---|---|---|---|---|---|---|---|---|
| twin | 50,940 | 91,537 | 21,258 | 22,060 | 10,756 | .079 | .47 -> 1.51 | .18 -> .43 |
| hard anchor | 51,227 (+287) | 91,570 (+33) | 21,319 (+61) | 22,071 (+12) | 10,761 (+5) | .054 | .72 -> 1.59 | .34 -> .60 |
By region (held-out BBS, anchor minus twin): East 1994-2009 +42, East 2010-25 -326 (better), West 1994-2009 +102, West
2010-25 +228, Plains -13.
Pre-registered prediction met: training cost < 300; nothing lost in the invasion era (pre-1994 blocks +5, never-reached
better); 1966 occupied-East potential lambda .72 (predicted .7-.8). The NEW finding is the modern rows: with disease and
dispersal free to re-tune, the anchor's modern cost falls from +200-700 (R81b/c, latents fixed) to ~+10-60 -- i.e. the
forecast value that low early Sa seemed to carry was mostly the latents' adaptation to the twin's rate path, not
information in the low early survival (likely; the mechanism -- which latent re-tuned -- is being read out: r84-style
disease readout on both arms' params, next). The residual cost moved WEST (+330), where the anchor does not apply directly
but shares the eastern-fitted weights (likely: the shared habitat-to-rate weights). Production implication (report, not a
decision): a BBL-level adult survival prior / likelihood looks affordable once disease and dispersal are estimated jointly;
the remaining swing (.72 -> 1.59) is the one to explain.

### CORRECTION (2026-11-04): the "dispersal" latents are DEAD sites on [cur] -- R80 / R81d / R85 refitted no dispersal
[cur] uses dispersal_departure.form = veit: age_forward.dispersal_step_age_structured overwrites the logistic departure with
Veit & Lewis eq. 18, p = N/K (smoothed, "no parameters"), so dispersal_logit_intercept / dispersal_logit_slope never reach
the likelihood. Fitted values sit exactly at their prior means (2.0 / 4.0) in every arm (ESTABLISHED: code + values). So
R80's "dispersal + disease refit" was a disease-only refit; R85's "dispersal-only" arm is a no-latent control (habitat sites
only), and must be read as such. Note for Plan L: departure on [cur] is a function of RELATIVE density N/K with no fitted
parameter -- the same relative-density coupling the user questioned for crowding (CONSIDERATIONS; cf. A5).

### R81d disease readout (refit_fields --disease_readout, rescore at the saved params; reproduces the fits exactly)
| arm | East toll 1994-2000 / 2001-10 / 2011-25 | West toll 2011-25 | East potential / effective lambda 1990, 2000, 2010 | East 1966 |
|---|---|---|---|---|
| twin | .16 / .28 / .31 | .19 | .88/.88, 1.22/.90, 1.51/.96 | .47 |
| hard anchor | .17 / .28 / .31 | .16 | 1.10/1.10, 1.37/1.00, 1.59/1.00 | .72 |
Disease latents moved < .05 between arms (mu_sev .45 vs .49, b_late -.24 vs -.20; rec, tau, k_half unchanged); disease-off
still costs ~52,000 in the anchor arm. So the R81d interpretation above is REVISED: the anchor's smaller modern cost was NOT
disease re-tuning (and dispersal is dead). What differs is the juvenile intercept -- alpha_j -1.06 (anchor) vs -.58 (twin),
alpha_a .91 vs .38, alpha_f 4.78 vs 4.91, alpha_k -1.59 vs -1.75 -- i.e. an Sa <-> Sj trade plus habitat weights (likely), and
a different warm start from R81b (R81b started from the soft-anchor fit, R81d from R67: start-point dependence, unknown
size). With plausible adult survival the modern East still sits on the rates <-> disease ridge (potential 1.59, effective
1.00): the anchor removes the deep early sink but not the modern paradise.

### R87 addendum: the epizootic is a ~17-year RAMP, not an epizootic -- the onset sharpness is the hidden disease lever
amplitude = ceiling x modifier x gate x (1 - resistance), gate = sigmoid((t - onset) / tau), tau = softplus(tau_raw) + .25.
Prior tau_raw N(.5, .5) -> median tau ~1.2 yr (a 10-90% rise in ~5 yr). [cur] tau_raw 3.61 (~6 prior SDs above) -> tau ~3.9
yr, a 10-90% rise over ~17 yr. That (not missing resistance) is why the amplitude climbs .25 -> .45 -> .48 at 0 / 10 / 25 yr
after onset (R87's "keeps rising" corrected: the ramp is the gate; no-resistance only keeps it from falling).
Literature: Hochachka & Dhondt 2000 (vault, p. 1, q2): "Three years after the epizootic arrived, house finch abundance
stabilized at similar levels" -- a fast epizootic with a plateau. No-recovery (rec ~4%) is CONSISTENT with that plateau;
the 17-year ramp is NOT (likely a lever). Mechanism (likely): a slow ramp + no recovery turns disease into a smooth
post-1994 calendar trend that cancels the vital rates' continued 1994-2010 rise (East potential 1.10 -> 1.59 while effective
stays 1.00). Test: R88 _tau_prior (tau_raw held .5; CPU 16-19, running) and _tau_rec_prior (tau and resistance both at prior
medians; GPU after R88). Prediction if the ramp is the lever: the modern potential-lambda rise shrinks (2010 East potential
< ~1.35) at a cost concentrated in 1994-2009 East rows; if instead the data demand the slow decline, the cost is large
(> ~2,000) and the potential rise is unchanged. Production implication (report): the gate-sharpness prior is the cheapest
place to stop disease acting as a trend absorber; the vault holds only one disease paper (acquisition: Hosseini et al.
2006, Dhondt et al. prevalence series, Hawley et al. on virulence / tolerance -- user via the UT browser).

### R83 RESULT: BBL adult dead-recovery LIKELIHOOD (Seber, <= 1993 censored; G = 1, a = 1, weights free, <= 1993)
| arm | training (incl. BBL) | held-out BBS 1994-2025 | post-1993 blocks | 2021-25 | pre-1994 blocks | never | occ.-East lambda 1966 -> 2010 | East occupied Sa 1966-93 mean |
|---|---|---|---|---|---|---|---|---|
| R81 twin (no BBL) | 50,957 | 90,786 | 21,272 | 21,792 | 10,723 | .074 | .47 -> 1.50 | ~.315 |
| R83 BBL likelihood | 54,318 (BBL -3,341) | 91,091 (+305) | 21,169 (-103) | 21,816 (+24) | 10,593 (-130) | .079 | .51 -> 1.46 | .32 |
| R81b hard anchor | 51,151 | 91,361 | 21,686 | 22,375 | 10,645 | .046 | .76 -> 1.55 | ~.51 |
Not converged (cap). Reporting rates fitted .002-.005 where there are recoveries; the 6+ region-era cells without
recoveries stay at the prior centre .005.
The likelihood barely moved adult survival (.315 -> .32). Offline profile of the SAME m-array (constant S per region,
reporting rate profiled; bbl_prep data, <= 1993): East MLE S ~.60 (132 recoveries; S = .30 is 43.6 log-lik units worse),
central ~.60 (22; weak), West ~.50 (227; S = .30 is 44 worse). So the banding data ARE informative and DO say ~.5-.6
(established, given the Seber model and the bbl_prep cohort rules) -- but the BBS training rows pay ~160-290 to raise early
eastern Sa to that level (R81b-d), several times the ~45-unit banding pull. In a joint fit the BBS preference wins
(established for this structure). Held-out: post-1993 and pre-1994 clean blocks slightly better than the twin, 2010-25
all-rows worse (+305, mostly unblocked rows).
Read (likely): the low early eastern adult survival is a BBS-driven choice that the banding data contradict by ~45 units;
it is the "convenient lever" (cheap low start feeding the large later rise), not a necessity -- R81b-d show raising it costs
nothing in the invasion era. Production implication (report): a BBL likelihood alone will not hold Sa at the banding level;
it needs either the data weighted as an independent survival study (it is ~273k birds, but only ~540 recoveries), or a
structure in which the BBS rows cannot buy fit with early Sa (e.g. adult survival with a smaller habitat/time swing than
juvenile survival -- the convenient-lever candidate per R72/R81) -- both are Plan L (b)/(a) items. Caveat: per-cell model Sa
at the banding cells (feeder-dense) vs the occupied mean is not separated here (unknown).

### NB-CAL (user: "is the NB dispersion becoming Poisson-like and the model over-fitting?") -- [cur] calibration readout
nb_calibration.py / nb_calib_split.py / nb_calib_eps.py; outputs E:/.../sandbox/NB_calib/. No sandbox refit frees the NB
sites: every arm uses [cur]'s phi0 = .44, kappa = .86 (phi(mu) = phi0 mu^kappa), Mexico multiplier .41.
- Shape: Var/mean = 1 + mu^.14 / .44 = ~2 at mu .02, ~3.5 at mu 2, ~4.4 at 17, ~5.7 at 140. Not Poisson, but the
  density-dependent concentration makes it nearly QUASI-POISSON (variance ~ mean^1.14) -- far tighter at high counts than a
  constant-phi NB2 (which at phi .44 would give Var/mean ~300 at mu 140). (established)
- Training rows by expected-count bin: mean Pearson r^2 .48-.84 below mu 3 (NB too LOOSE; 90% coverage .97-.99), ~1.05 at
  3-30, 1.65 at 30-100 and 2.7 above 100 (NB too TIGHT; 90% coverage .86 / .73). Observed var/mean at mu > 100 is 15 vs the
  model's 5.7. So the densest routes (~2,800 rows) are scored as ~2-3x more precise than they are (established) -- they carry
  outsized pull in every refit (likely one reason ~160-290 BBS units outvote ~45 banding units, R83).
- The larger over-fit is NOT the dispersion: observer-route effects (19,127 levels, sd .63) have count-WEIGHTED mean +.98
  on training rows. With each row's own route effect removed, training rows are under-predicted 1.65x (1.35x after the
  lognormal mean correction); the 31 held-out blocks are under-predicted ~2x in EVERY era (1966-93 2.14, 1994-2009 1.93,
  2010-20 1.82; dense rows 2.0-2.3x) and 2021-25 1.37x. In-sample mean calibration (obs/exp 1.01) is bought by positive
  route effects where birds are (established). Mechanism (likely): the 54 km process surface is systematically low on the
  routes that hold birds (route placement along towns / feeders vs the cell mean, or K / habitat level too low), and route
  effects absorb the gap; held-out space cannot borrow them.
- Consequence for this record (established): held-out-block scores in every R-row are dominated by a ~2x level offset common
  to all arms; arm DIFFERENCES remain comparable, but absolute held-out skill is poor and an arm can gain by raising the level
  on unseen routes. Production implications (report, not decisions): (i) free / re-prior the NB density power (kappa) or use
  a variance form that does not collapse toward quasi-Poisson at high counts; (ii) route effects need a structure that
  transfers (route-level covariates such as developed area along the route, a spatially smooth route field) or a tighter
  prior; (iii) held-out scoring should report level and pattern separately.

### NB-CAL addendum: is the K link (softplus) behind the under-prediction? (user: "it could be the transform")
[cur] uses capacity_level_prior.link = softplus in density space (docs/PRIORS.md "Capacity link"); exp is a config switch.
nb_calib_klink.py on [cur]: softplus argument over occupied cell-years q10 / q50 / q90 / q99 = -4.35 / -3.64 / -.96 / 1.80,
so ~90% of occupied cell-years sit in the exponential regime (softplus / exp >= .85); the top decile is compressed, the
top 1% to ~.3 (established). Observed / expected with the route effect removed, by expected count: training 1.74 / 1.73 /
1.86 / 1.53 / 1.44 / 1.77 / 2.65 (bins <.1, .1-.5, .5-1, 1-3, 3-10, 10-30, > 30); held-out blocks 2.38 / 2.32 / 1.71 / 1.36 /
1.83 / 1.95 / 2.91. So the bias has TWO parts (likely): a roughly FLAT ~1.4-1.9x offset across 0-30 counts -- not the link
(the link is ~exp there) -- and an extra rise to ~2.7-2.9x above 30 counts, where softplus compression acts (consistent
with the link). Part of the flat offset is a scoring convention (established): unseen routes are predicted at route
effect ~0 (the median route), whereas the mean route is e^(sd^2/2) = 1.22-1.41x higher (sd .63 all levels / .83 training
rows); the remaining ~1.2x (training) / ~1.5x (held blocks) is unexplained (K level, route placement: unknown).
KLINK PRE-REGISTRATION: refit [cur] with link = exp (overlay copy, E:/.../sandbox/KLINK/overlay_exp.json; alpha_k location
re-solved by the code; warm start at [cur] latents -- nearly identical K except the top decile), full data, habitat + K
sites refit, maxiter 800, plus a softplus twin with identical settings. Predictions if the link matters: training gain
concentrated in rows with expected count > 10 (> ~100 units), the dense-row Pearson r^2 (2.7 at > 100) falls, the route
effects' count-weighted mean (+.98) falls, and held-out blocks improve at > 30 counts but not in the flat 0-30 offset.
Stability is the risk the user flagged (exp K was unstable in past versions; the recorded mechanism in docs/PRIORS.md is
a penalty acting multiplicatively without a floor in the exp regime): report any divergence, K at q99, and the share of
cells with K > 100 counts.

### R85 RESULT (dsp1 pending): disease-only vs "dispersal-only" refit (2005 split, G = 1, weights fixed, R70 observers)
| arm | training | held-out 2010-25 | 2021-25 | post-1993 blocks | pre-1994 blocks | never | occ.-East lambda 1966 -> 2010 | fronts |
|---|---|---|---|---|---|---|---|---|
| a = .5, disease only | 94,587 | 36,396 | 22,522 | 21,522 | 11,563 | .105 | .76 -> 1.15 | -7/-6/-3/-5 |
| a = .5, dispersal only | 95,393 (converged) | 38,678 | 25,722 | 22,803 | 11,486 | .060 | .74 -> 1.16 | -8/-8/-5/-8 |
| a = 1, disease only | 92,216 | 36,594 | 22,715 | 21,539 | 10,763 | .070 | .46 -> 1.53 | +3/0/-2/-6 |
Consistent with the dead-site correction (established): the "dispersal-only" arm reproduces the latents-FIXED R78 a = .5
2021-25 score exactly (25,722), and disease-only reproduces R80's "dispersal + disease" (94,587 vs 94,536; 2021-25 22,522 vs
22,638). Pre-registered prediction MET: disease ALONE rescues the half swing on the modern rows (2021-25 -3,200 vs no
refit; a = .5 then equals or beats a = 1 on 2021-25, -193). Mechanism (likely): the disease toll trades against the modern
part of the vital-rate rise -- the rates <-> disease ridge (R84b) -- and with the R87 addendum the trading variable is
probably the 17-yr onset ramp + no recovery (tested by R88 _tau_prior / _tau_rec_prior). The invasion era still rejects
a = .5 under disease-only refit (pre-1994 blocks +800, fronts 5-7 yr early), as in R80.
R85 dsp1 (a = 1, "dispersal only" = no effective latent refit): training 92,264, held-out 2010-25 36,240, 2021-25 22,579,
post-1993 blocks 21,353, pre-1994 blocks 10,761, never .075, occ.-East .46 -> 1.53, fronts +3/0/-3/-6. At a = 1, refitting
disease buys 48 in training but LOSES on every clean modern score (held-out +354, 2021-25 +136, blocks +186): at the full
swing the disease latents over-fit the training rows (likely); at a = .5 they are what rescues the modern rows. So disease
latents are useful exactly when the vital rates are constrained -- the signature of a trade-off ridge, not of disease
information (likely).

### R88 _rec_prior RESULT: resistance HELD at the prior median (rec_raw -.5, ~38% recovered), other disease sites + habitat refit (full data)
Objective 135,601.0 vs the same-settings base (R86 _base) 135,590.8: +10 (BBS +3.4); never .068 (same); fronts identical;
occupied-East potential lambda 1966 -> 2010 .48 -> 1.52 (base .48 -> 1.53); vital-rate ranges unchanged. Not converged
(1,033 evals) but the start cost (+1,540) was almost entirely recovered.
R87's "resistance is LIKELY a lever" is FALSIFIED (established for this structure): resistance is weakly identified -- the
data are nearly indifferent between ~4% and ~38% recovery once the other disease sites re-tune -- and holding it at the
prior does NOT shrink the modern potential-lambda rise. Which sites compensated (onset ramp, mu_sev, severity field) is read
by the queued rescore readout _rec_prior_dis. Production implication (report): the literature prior on resistance is
harmless but will not by itself remove the rates <-> disease ridge; the onset-ramp test (_tau_prior, running) is the
remaining disease-form candidate.

### R88 _wsev_off RESULT: spatial severity field OFF (w_sev = 0), other disease sites + habitat refit (full data)
Objective 135,857 vs base 135,591: +266 (BBS +288); not converged (891 evals; start +1,004). BBS cost by region-era (vs base):
East +2 / +33 / +56 / +72 (1966-79 / 80-93 / 94-09 / 10-25), Plains +3 / +3 / +12 / +1, West +10 / +20 / +18 / +57 -- spread
over pre-disease eras too, i.e. the habitat sites took on part of the lost spatial severity pattern (likely). Never .073
(.068); fronts identical; occupied-East potential lambda .43 -> 1.51 (base .48 -> 1.53); West .72 -> 1.10.
R87's "severity field LIKELY a lever" is only PARTLY supported: the field carries ~270 units of regional fit (a real but
modest absorber), and removing it does NOT shrink the modern potential-lambda rise (established for this structure). Both
named disease-form levers (resistance, severity field) are now ruled out as the source of the modern swing; the onset ramp
(_tau_prior, running) is the last disease-form candidate, and the rates <-> disease ridge (R84b, R85) may instead sit in the
overall severity level (mu_sev / ceiling) -- next if _tau_prior also leaves the swing intact.

### R88 _tau_rec_prior RESULT: onset sharpness AND resistance held at prior medians (tau_raw .5, rec_raw -.5), full data
Objective 135,900 vs base 135,591: +309 (BBS +295; start +1,350); not converged (935 evals). BBS cost by region-era: East
1994-2009 +183, East 2010-25 +69, West 2010-25 +10, all pre-1994 eras within +-9 -- concentrated where predicted.
Disease (readout): East realised toll .15 / .26 / .27 (1994-2000 / 2001-10 / 2011-25; base .17 / .28 / .31); mu_sev .16
(base .50), b_late -.34 (-.24), hill_n 2.15 (1.84).
ESCAPE: log_tau_rec went to 6.26 (tau_rec ~ 500 yr; prior N(log 12, .5) -> ~7.6 prior SDs). With rec held at 38% the fit
made recovery INFINITELY SLOW instead -- resistance is switched off through its timescale, paying ~29 prior units
(established). So the data reject any recovery within 1994-2025 (likely; consistent with Hochachka & Dhondt's stabilised,
not rebounding, abundance). The R88 _rec_prior arm probably escaped the same way (its readout is queued) -- which would
explain its tiny +10 cost.
Swing: occupied-East potential lambda 1966 -> 2010 .50 -> 1.42 (base .48 -> 1.53): the sharp epizootic removes ~.1 of the
~1.05 eastern swing (~10%) at ~300 units, nearly all in the 1994-2009 East rows. Effective East lambda 2000 / 2010 .83 / .97.
Against the R87 prediction (ramp is the lever if 2010 potential < ~1.35; data demand the ramp if cost > ~2,000 and
potential unchanged): INTERMEDIATE -- the onset ramp carries a small part of the modern rise (likely), and the 1994-2009
eastern rows mildly prefer a gradual disease build-up (~180 units). Most of the modern potential-lambda rise survives every
disease-form restriction so far (resistance, severity field, onset sharpness).
R88 _rec_prior readout (confirms the escape): log_tau_rec 4.60 (tau_rec ~100 yr; ~4.2 prior SDs above log 12) with rec held
at 38%, tau_raw 3.79 (ramp kept), mu_sev .59 (base .50); East toll .17 / .28 / .31 -- identical to the base. Holding the
recovery FRACTION at the prior simply moved the no-recovery solution into the recovery TIMESCALE (established). Any
production resistance prior must constrain the timescale and the fraction jointly (e.g. a prior on the fraction recovered
by 10-15 yr after arrival), or it is not a constraint.

### R89 PRE-REGISTRATION (L-RIDGE): profile the rates <-> disease ridge along overall disease severity
mu_sev FIXED at -1.0 / -.25 / 1.25 (base .50; modifier median sigmoid -> ~.27 / .44 / .78 of the ceiling), all other
disease sites + habitat refit, full data, --disease_readout (GPU, serial). Readouts per point: objective, East realised
toll 2011-25, occupied-East potential and effective lambda 2010. Predictions: if the ridge is real and long, 2010 East
potential lambda moves with the toll (lower severity -> smaller potential rise, effective ~1 kept) at a small cost
(< ~300 per step); if bounded by data, the cost grows steeply away from .50 (> ~1,000 at -1.0). Escape check: report
b_late, w_sev norm, tau_raw, hill / k_half -- other sites re-creating the same toll would show a flat profile with the
toll unchanged (the R88 pattern).

### R86 (ii) _dd RESULT: density-dependent juvenile survival, fecundity crowding re-solved so equilibrium = K (full data)
Fitted cs .133 (juvenile survival x .88 at N = K); objective 135,550 vs base 135,591 (-41; BBS only -8: East -3 / -2 in
1994-2009 / 2010-25, West -8 / -9 in 1966-79 / 2010-25 -- within noise of an extra site); not converged (743 evals). Never
.063 (.068); fronts identical. Occupied-East potential lambda 1966 -> 2010 .50 -> 1.56 (base .48 -> 1.53); West .81 -> 1.14.
Pre-registered prediction (smaller toll AND smaller modern rise) NOT MET for the rise: the rise is unchanged (established
for this arm). The toll is read by the queued rescore (R86_disease). As anticipated in the R86 note, re-solving c keeps the
equilibrium imposed at K, so juvenile DD here only re-partitions how K is enforced -- the data barely use it (cs small, BBS
-8). The informative arms are (iii) emergent and (iv) dd-only (running).

### R89 RESULT (one point; rest stopped): mu_sev fixed at -1.0 -> b_late ESCAPE
Objective 135,668 (+77 vs base), converged (863 evals). b_late went -.24 -> +.93 (onset_decades term) and rebuilt the toll:
East realised toll .17 / .27 / .30 (base .17 / .28 / .31). East potential lambda 2010 1.49 (1.53), effective .95. A parameter
profile cannot trace this ridge -- the disease block re-creates the same ~.3 eastern toll through whichever site is free
(R88 rec -> tau_rec, R89 mu_sev -> b_late; established). The remaining two mu_sev points were stopped as uninformative.
Read (likely): the ~30% eastern toll is strongly demanded by the post-1994 counts GIVEN the rates' modern rise; the question
is whether the rise is demanded given a smaller toll -- so the toll itself must be capped.

### R90 PRE-REGISTRATION (L-RIDGE, toll-capped): disease mortality CEILING .35 / .25 / .15 (base .8), full data
Overlay copies of [cur] with disease_prior.mortality_ceiling changed only; habitat + all 11 disease sites refit,
--disease_readout (GPU, serial). The ceiling caps the per-capita toll for every site combination (no escape). Predictions:
LONG ridge -> East realised toll 2011-25 falls with the cap, the 2010 East potential-lambda rise shrinks with it (effective
~1 kept), cost < ~500 at .25; BOUNDED -> cost grows steeply (> ~2,000 at .15) concentrated in East 1994-2025 and the
potential rise does NOT shrink (the counts need the decline itself, and the rates' rise is separately demanded).
R86 (ii) disease readout: East realised toll .17 / .29 / .31 (1994-2000 / 2001-10 / 2011-25; base .17 / .28 / .31); East
potential / effective lambda 2010 1.56 / .97 (base 1.53 / .95); disease-off costs ~43,400 (base ~44,000). Pre-registered
"smaller toll" prediction NOT MET (established for (ii)): juvenile DD with equilibrium re-solved to K leaves the toll and the
ridge unchanged. The compensation mechanism the arm was built to supply is unused when K stays imposed.

### R86 (iv) _ddonly RESULT: ALL density dependence via juvenile survival, no fecundity crowding (equilibrium emerges)
NOT CONVERGED: 893 evals from a far start (517,941 -> 136,246, still creeping; the last gradient norm 336) -- read the cost
as an upper bound. Fitted cs 4.80 (juvenile survival x .17 at N = K: strong). Objective +655 vs base (BBS +762): West +133 /
+143 / +170 / +68 (1966-79 / 80-93 / 94-09 / 10-25), East -2 / +138 / +67 / +35, Plains +2 to +8. Never .040 (base .068),
fronts +3/-1/-3/-7 (base +3/-1/-3/-6).
Vital rates move: Fmax range 2.71 -> 3.08 (base 2.41 -> 4.06, i.e. fecundity's change nearly vanishes); Sj 1966 / 2010 .154 /
.359 (base .060 / .147: low-density juvenile survival is now ~2.4x higher and still rises ~2.3x); Sa .18 -> .42 (.20 -> .49).
Occupied-East LOW-DENSITY potential lambda .76 -> 2.07 (base .48 -> 1.53); West 1.05 -> 1.42.
CAUTION (established): potential lambda is the N -> 0 rate; in (iv) density dependence acts on Sj instead of Fmax, so its
low-density value is not comparable to the base's -- the ratio (2.7x vs 3.2x) and the absolute swing (+1.31 vs +1.05) are.
Read (likely, pending convergence and the queued disease readout): moving crowding off fecundity does NOT bound the rise --
the swing relocates from fecundity into juvenile survival (the R72/R81 "convenient lever" candidate) and grows in absolute
terms, at a cost concentrated in the WEST (sedentary/native range, where the strong Sj density dependence fits worst). The
pre-registered prediction "(iv) bounds the ridge best" is NOT supported at this point. Next: (iii) emergent (still running),
the disease readout for (iv), and a converged continuation of (iv) from its end point if (iii) is also inconclusive.

### R90 c = .25 RESULT: disease mortality ceiling .25 (base .8), habitat + all disease sites refit, full data
Not converged (888 evals; last gradient norm 44, objective changing < .01 / eval -- near a plateau). Objective 136,702 vs
base 135,591: +1,111 (BBS +1,096). BBS cost by region-era: East +56 / +265 / +234 / +163 (1966-79 / 80-93 / 94-09 / 10-25),
West +66 / +68 / +69 / +133, Plains +5 to +22. Never .066; fronts +1/-1/-3/-6 (base +3/-1/-3/-6).
Disease: East realised toll .10 / .18 / .18 (1994-2000 / 2001-10 / 2011-25; base .17 / .28 / .31) -- the cap binds (no escape).
Swing: occupied-East potential lambda 1966 -> 2010 .64 -> 1.14 (base .48 -> 1.53): the eastern swing HALVES (+.50 vs +1.05);
effective East lambda 2000 / 2010 / 2020 .85 / .87 / .85 (base .89 / .95 / .96). West .70 -> 1.01 (base .75 -> 1.11). Rates:
Fmax 2.68 -> 3.62 (2.41 -> 4.06), Sj .070 -> .118 (.060 -> .147), Sa .25 -> .43 (.20 -> .49).
Against the pre-registration: neither pure case. The ridge is REAL and its direction is as stated -- capping the toll
shrinks the modern rise in step (toll -40%, swing -52%) -- but it is NOT flat: halving the swing costs ~1,100 (likely; near
plateau). Notably ~1/3 of the cost falls in the PRE-DISEASE East (1966-93: +321) and the West in every era: the shared
habitat -> rate weights that carried the big modern rise also time the eastern invasion, and the deep start eases with it
(1966 .48 -> .64). So the "deep start" and the "modern paradise" are partly ONE lever (the habitat-driven rate swing), traded
against disease at the modern end (likely). Effective eastern lambda ~.85 after 2000 under the cap = a population still
declining; the counts would then need K / habitat to carry the modern level (unknown: the readout of K change is next).
Remaining profile points (.15, .35) queued; together they give the ridge's slope.

### R86 (iii) _emergent RESULT: DD juvenile survival with fecundity crowding NOT re-solved (equilibrium below K, emergent)
cs .093 (juvenile survival x .92 at N = K: weak); objective 135,557 (-34 vs base; BBS -7, every region-era within +-8); not
converged (1,038 evals). Never .066, fronts identical. Occupied-East potential lambda .50 -> 1.52 (base .48 -> 1.53); rates
essentially the base's. Read (established for this arm): with fecundity crowding still present, the data choose only weak
juvenile density dependence and the swing is unchanged -- a small emergent shortfall below K is not used to relieve the
rates <-> disease ridge. Of the four R86 arms only (iv) dd-only changes the structure of the swing (relocates it into Sj,
continuation running); none shrinks it. Contrast R90: capping the disease toll halves the swing at ~1,100 -- the ridge is
bounded by the toll, not by the form of density regulation (likely).

### R88 _tau_prior RESULT: onset sharpness only held at prior (tau_raw .5), resistance free, full data
Objective 135,871 (+280; BBS +285: East 1994-2009 +177, East 2010-25 +67, all else within +-17); not converged (973 evals).
rec_raw -4.73 (~1% recovered: resistance pushed further to zero), log_tau_rec 3.38; East toll .15 / .26 / .27; East potential
lambda 1966 -> 2010 .50 -> 1.42, effective 2010 .97. Identical to _tau_rec_prior (+309, 1.42) within noise: holding the
fraction too adds nothing because it escapes via the timescale (established). So the onset-ramp test stands: a sharp
epizootic costs ~280-310, almost all in 1994-2009 East rows, and removes ~.1 of the ~1.05 eastern swing (likely).
DISEASE-FORM SUMMARY (R87-R90): resistance -- not identified, escapes (no lever); severity field -- ~270 of regional fit, no
swing change; onset ramp -- ~300, ~10% of the swing; overall toll (ceiling .25) -- ~1,100, ~50% of the swing. The lever is
the SIZE of the modern disease toll, not its form; the form choices only decide how cheaply that size is reached.

### R90 c = .15 RESULT (not converged, 925 evals, gradient norm 48; near plateau)
Objective 137,425: +1,834 vs base (BBS +1,830): East +19 / +393 / +368 / +215, West +148 / +147 / +190 / +311, Plains <= +21.
Never .056 (better), fronts identical. East toll .06 / .10 / .10. Rates: Sa .20 -> .28 (base .20 -> .49), Sj .091 -> .123,
Fmax 2.74 -> 3.23. Occupied-East potential lambda .64 -> .91 (+.27); effective 2000 / 2010 / 2020 .79 / .79 / .75.
RIDGE PROFILE so far (East toll 2011-25 / eastern swing 1966 -> 2010 / cost):
  ceiling .8 (base): .31 / +1.05 / 0;   .25: .18 / +.50 / +1,111;   .15: .10 / +.27 / +1,834;   (.35 queued)
The swing scales almost linearly with the permitted toll (swing ~ 3.4 x toll; established for these points): the vital
rates' modern rise exists to the extent that disease removes it again. The cost is bounded but moderate (~9,000 per unit
toll, i.e. ~1,100 per .1 of toll) and spread over the pre-disease East and all western eras -- the shared habitat-to-rate
weights again (likely). So the data do not pin the swing directly; they pin the PRODUCT of the rise and the toll at the
modern end (effective lambda ~.8-.95), plus a preference (~1,800 units) for the large-rise / large-toll corner.
Production implication (report): an independent constraint on the eastern toll -- e.g. the documented ~50% post-epizootic
abundance decline (Hochachka & Dhondt) expressed as a likelihood on the decline, or survival from post-1994 banding with
disease (Plan L C1 extension) -- would fix the position on this ridge; without one, the L-BFGS mode sits at the corner the
in-sample fit prefers.

### BBL-ERA screen (offline Seber profile, constant S per region x banding era; bbl_prep cohorts; reporting rate profiled)
| region | banded 1960-89: released / recoveries / S MLE [95%] | banded 1994-2019: released / recoveries / S MLE [95%] |
|---|---|---|
| West | 92,901 / 219 / .50 [.475, .55] | 50,552 / 82 / .425 [.375, .50] |
| Central | 13,017 / 17 / .60 [.425, .775] | 18,022 / 27 / .55 [.425, .65] |
| East | 52,950 / 124 / .575 [.525, .625] | 5,433 / 19 / .525 [.375, .65] |
Post-1994 eastern banding is thin (19 recoveries) but bounds DISEASE-INCLUSIVE adult survival at >= ~.375. Rough comparison
(East all-land median Sa 2010 x (1 - East realised toll 2001-10)): base .49 x .72 ~ .35; ceiling .25: .43 x .82 ~ .35;
ceiling .15: .28 x .90 ~ .25. Every point on the ridge puts modern eastern effective adult survival AT OR BELOW the banding
lower bound, the low-toll end most clearly (likely; crude -- all-land medians, not the banded cells; a proper per-cohort
readout of effective Sa at the banding cells is the next step). The ridge preserves the EFFECTIVE rates, so post-1994
banding cannot pick a point on it, but it says the effective adult survival the whole ridge uses is too low -- the same
low-adult-survival pattern as the pre-1994 era (R83), now in the modern era. Western post-1994 survival is lower than
pre-1994 (.425 vs .50; disease reached the West ~2005; likely).

### R91 PRE-REGISTRATION: full-data hard adult-survival anchor (.55, .005 on occupied East 1966-93), disease + habitat refit
[cur] structure (not decoupled), full data, all 11 disease sites + habitat sites refit, --disease_readout; CPU 16-19.
Question: with plausible early adult survival AND the disease block free, where on the rates <-> disease ridge does the fit
sit? Prediction: early East potential lambda ~.7-.8 (R81b-d); modern toll ~.3 and 2010 potential ~1.55 unchanged (the anchor
lifts the start, not the end), so the swing shrinks only by the raised start (~.25); cost ~150-300.

### R90 c = .35 RESULT (not converged, 984 evals) and the completed toll-capped ridge profile
c = .35: objective 136,096 (+505; BBS +479: East +33 / +108 / +104 / +78, West +29 / +22 / +21 / +64); never .072; East toll
.14 / .23 / .24; occupied-East potential .58 -> 1.34 (+.76), effective 2010 .93; rates Sa .24 -> .47, Sj .061 -> .118,
Fmax 2.77 -> 4.15.
| ceiling | East toll 2011-25 | East swing 1966 -> 2010 (potential) | East effective 2010 | cost (BBS) |
|---|---|---|---|---|
| .8 (base) | .31 | .48 -> 1.53 (+1.05) | .95 | 0 |
| .35 | .24 | .58 -> 1.34 (+.76) | .93 | +505 (+479) |
| .25 | .18 | .64 -> 1.14 (+.50) | .87 | +1,111 (+1,096) |
| .15 | .10 | .64 -> .91 (+.27) | .79 | +1,834 (+1,830) |
Established for these (near-plateau, unconverged) points: the swing tracks the permitted toll almost linearly (~3.7 x toll);
cost rises ~steadily (~7,000-9,000 per unit toll); the fronts and never-reached share barely move; and ~1/3 of every cost
is in PRE-disease eastern rows (1966-93) plus all western eras -- the toll and the swing are coupled through the shared
habitat-to-rate weights. The ridge is BOUNDED (each .1 of toll costs ~700-1,100) but SHALLOW relative to any independent
information on the toll in the model (none). Answer to the user's L-RIDGE question (likely): this is a long, data-bounded
ridge -- not endless, not multimodal -- whose position is set by the in-sample preference for the high-toll corner; an
independent toll / modern-survival constraint is the model revision it calls for. Shrinking the 1966 start (deep sink)
comes WITH a smaller toll (.48 -> .58-.64), consistent with R81/R83: the deep start and the modern paradise are one lever.

### R92 PRE-REGISTRATION (Plan L C1 extension): BBL likelihood on DISEASE-INCLUSIVE adult survival, all years
refit_fields --bbl_disease: the Seber term uses Sa (1 - m), m = disease amplitude x hill(K) (local density approximated by K;
the forward uses current N), recoveries through 2024 (--bbl_until 2024). Full data, [cur] structure (decoupled a = 1, G = 1),
habitat + all disease sites refit, --disease_readout (GPU after the R86 readouts; 2-iteration smoke first). Twin: R86 _base.
Prediction: post-1994 banding (East 19, central 27, West 82 recoveries) pushes modern effective adult survival up toward
~.45-.5, which on the R90 ridge means a SMALLER toll and a smaller swing; but the information is thin (East) -- expect a
modest move (toll .31 -> ~.25, swing shrinking by < ~.3) and a BBS cost of a few hundred; pre-1994 Sa stays low as in R83.
R86 (iii)/(iv) disease readouts: emergent -- East toll .17 / .28 / .31, effective 2010 .96 (= base). dd-only (first run,
unconverged) -- East toll .17 / .30 / .34 (HIGHER than base), low-density potential / effective 2010 2.07 / 1.24. With
crowding moved onto juvenile survival, disease is NOT relieved: the toll rises slightly and the low-density effective rate
stays > 1 (sources at low density, held down at K by strong Sj density dependence). Established for these arms: no form of
density regulation tried (R86 i-iv) shrinks the toll or the swing; R90's toll cap does. The rates <-> disease ridge is not a
by-product of fecundity-only crowding (likely; dd-only continuation pending).

### R92 RESULT: BBL likelihood on disease-inclusive adult survival, recoveries through 2024 (full data; twin R86 _base)
Not converged (878 evals). Objective 140,525 (incl. BBL -4,857); BBS 110,132 vs base 110,045 (+87: West +15 / +26 / +17 / -2,
East -1 / +12 / +8 / +6). BBL improved -5,215 -> -4,857 (+357), almost entirely through the 12 reporting rates (moved off
the .005 start to .0022-.0050). Effective adult survival (occupied mean) went DOWN, not up: East 1966-93 / 1994-2020
.36 / .36 at the start -> .33 / .32; West .44 / .42 -> .40 / .38. East toll .18 / .28 / .31 (unchanged); occupied-East
potential .45 -> 1.49, effective 2010 .93. Disease latents ~unchanged.
Pre-registered prediction NOT MET (established): post-1994 banding does not move the fit along the ridge. As in R83, the
banding information (~45 units per region on S, offline profile) is outvoted by the BBS rows, and with free reporting
rates the Seber term is satisfied mostly by r (likely: the per-cell heterogeneity of model Sa at the banded cells lets r
absorb the level; the decay with years since banding carries the S information and is weak at ~540 recoveries).
Production implication (report): the BBL data cannot pin adult survival -- or the toll -- through a likelihood of this
size against ~110k BBS rows whose dense routes are over-weighted (NB-CAL). Options: correct the BBS dense-row
over-confidence first (NB-CAL i), tie reporting rates across eras (fewer nuisance r), or treat the banding estimate as an
informative prior on the survival intercept.

### R93 PRE-REGISTRATION (NB-CAL i): constant-concentration NB2 (observation.conc_density_power off) -- does re-weighting
### the dense routes move the swing / toll / adult survival?
Overlay copy of [cur] with conc_density_power false only (one global phi, refit; the Mexico multiplier kept). Full data,
decoupled a = 1 / G = 1 ([cur]-equivalent), habitat + disease sites + concentration refit, --disease_readout; GPU. Arms:
(a) NB-const; (b) NB-const + BBL disease-inclusive likelihood (R92 settings). Diagnostic only (the user's question about
the dispersion): the objective is NOT comparable to [cur]'s (different likelihood), so read swing, toll, effective Sa,
held-out-block level, and the calibration (nb_calibration-style) instead. Predictions: phi falls well below .44 at high
counts' expense... i.e. dense rows lose weight; if the dense rows drive the high-toll corner, the toll and the swing
shrink (2010 East potential < ~1.4) and in (b) banding moves effective Sa up (> ~.4 East 1994-2020); if not, the ridge
position is not a dispersion artefact.

### R93 (a) RESULT: constant-concentration NB2 (conc_density_power off), full data, habitat + disease + phi refit
Not converged (876 evals). Fitted phi 2.50 (one global value; Var/mean = 1 + mu / 2.5: 1.4 at mu 1, ~57 at mu 140 -- looser
than [cur] at high counts, TIGHTER at low counts). Swing and toll UNCHANGED: occupied-East potential .50 -> 1.52, effective
2010 .95; East toll .18 / .29 / .31; rates as base; never .066; fronts identical.
Held-out predictive log score (each model's own NB; a proper comparison of predictive distributions): pre-1994 blocks
10,070 vs [cur] 10,731 (-661, better); post-1993 blocks 22,380 vs 21,374 (+1,006, worse); 2021-25 23,370 vs 21,557 (+1,813,
worse).
Read: the ridge position (swing, toll) is NOT a dispersion artefact (established for this pair of variance forms). The
dense-row over-confidence found in NB-CAL is not what holds the fit in the high-toll corner. The variance form matters for
prediction, era-dependently: constant phi scores the sparse invasion-era blocks better and the dense modern rows worse --
consistent with the variance growing faster than the mean at high counts but not as fast as phi = 2.5 implies (likely; an
intermediate kappa or a two-parameter variance would be the production candidate). Arm (b) (+ BBL) running.

## REVIEW 8 (2026-11-04, subagent, read-only; checked by me against code before recording) -- corrections to R81d-R93
1. "EFFECTIVE lambda" (r84_disease / R84b / R85 / R90 / R92) is MISLABELLED (certain, by construction): it is Sa (1 - m) +
   Sj (1 - m) Fmax with m = amplitude x hill(N) at the fit's OWN densities and NO crowding. Disease mortality is density-
   dependent (hill_n ~2, k_half ~.04 counts), so at N -> 0 m -> 0 and the low-density rate equals the potential rate: the
   modern eastern LOW-DENSITY lambda is ~1.5 in [cur], not ~.95. "Disease cancels the paradise" holds only at fitted
   densities; in the potential-growth (G13) sense disease cancels nothing -- it is a SECOND DENSITY REGULATOR alongside
   fecundity crowding (likely consequence). The rates <-> disease ridge is therefore an A5 (regulation) problem: after 1994,
   disease's hill(N) is the model's only survival density dependence. Readouts are being corrected (low-density lambda,
   realised per-capita growth at actual N, Sa at the banded cells).
2. R81d's disease refit had NO DATA (certain): trained <= 1993, there are no post-arrival rows (East onset ~1998), so the
   disease sites saw only priors; the "disease re-tunes" premise was untestable at that split.
3. Start-point noise: the R81 and R81d twins (near-identical settings) differ by ~750 on held-out (90,786 vs 91,537) with
   training flat (certain). Several reported gaps sit inside that band (R81d anchor +12-61; R85 dsp1 +136-354; R81b vs R81d
   "575 -> 33"): read them as within start-noise until calibrated (likely). Full-data arms warm-started at [cur]'s optimum
   (R86, R88, R92, R93) are cleaner.
4. R90 costs are UPPER BOUNDS (likely overstated as "established"): far starts (+15-33k), iteration-capped, small gradient
   while crawling a ridge; the ridge may be flatter than 700-1,100 per .1 toll; the pre-1994 share of cost may be residual
   start shock. NB dense-row over-confidence (NB-CAL) inflates every cost scale.
5. "Decoupled" at a = 1 is the [cur] forward (certain): it only adds G (crushed) / rho parameters and changes the optimizer
   path. R71's "single vs decoupled STRUCTURE" attribution should read "free constant G + start point". Plan A1 (separate K
   and rate weights) remains untested.
6. --bbl_disease approximates local density by K; hill(K) >= hill(N) where N < K, so the BBL term sees MORE mortality than
   the forward (up to amplitude/realised ~ .41/.28): R92's effective Sa is biased low against its own prediction (likely).
7. R83 / R92 report OCCUPIED-mean Sa; the Seber term reads Sa at the banded cells (West recoveries 328 > East 157). R92's
   +357 BBL gain may be spatial reallocation of Sa toward banded cells, not only reporting rates (unresolved).
8. BBL-ERA "Sa x (1 - toll)" mixes an all-land median with an occupied-mean toll (certain; direction unknown) -- crude.
9. R86 "established" overstated: (iv) unconverged; density dependence on ADULT survival (S1) and disease-as-regulator were
   not tested.
10. Alternatives now on the list: equilibrium non-identifiability (counts pin N* only; the decomposition needs TRANSIENTS --
   the 1994-98 decline rate, fronts); low Sa buying SPEED (short time constant tracks yearly K(Z) wiggles) rather than depth;
   banding says Sa roughly FLAT (East .575 -> .525) while the fit's Sa rises 2.4x -- the banding contradicts the Sa RISE, not
   only the early level; route-effect offset partly mechanical (count-weighted mean of a random effect is positive even
   under a correct model) and partly zero-route anchoring of the K level.
Plan changes (section L): parameter profiles obsolete (escape) -> profile on output caps, converged; disease work re-centred
on its DENSITY DEPENDENCE (hill), not its form; NB form + route-effect structure promoted to the top (they set every cost
scale); C2 (dispersal trade) void on [cur]; BBL demoted behind the NB / reporting-rate fixes; add a banding-informed prior on
Sa's TIME CHANGE.

### R94 / R95 PRE-REGISTRATION (review 8 items C2, C3)
R94 DENSITY-INDEPENDENT DISEASE: disease_log_k_half_counts fixed at -9 (k_half ~1e-4 counts: hill(N) ~1 wherever birds
exist), hill_n and k_half excluded from refit; other 9 disease sites + habitat refit; full data, GPU. Disease can then no
longer act as a density regulator: it must hit low-density and front cells too. Prediction (review): toll < ~.2 and the
(low-density) swing shrinks, cost < ~500; FALSIFIED if toll and swing are unchanged (then the ridge is not a regulator
equivalence).
R95 ADULT SURVIVAL TIME-STATIC: --decouple --vr_alpha 0 --real_rates Sj,Fmax (Sa reads the habitat field frozen at its
1966-2025 mean; Sj, Fmax read the real field; c re-derived), full data, habitat + disease refit; (i) alone, (ii) + BBL
likelihood on pre-disease recoveries (--bbl --bbl_until 1993; no K-approximation bias). Prediction (review): Sa settles
>= ~.45, cost < ~300 vs [cur], the swing moves into Sj / Fmax; FALSIFIED if cost > ~1,000 (-> low Sa buys SPEED / tracking,
not depth). Banding says Sa roughly flat (BBL-ERA): a time-static Sa is the banding-consistent structure.

### RD (corrected readouts, review 8 item C1) -- [cur]: adult survival AT THE BANDED CELLS -- MAJOR CORRECTION
Release-weighted model Sa over the banded cells, years by .. by + 4 (bbl_prep cohorts), vs the BBL-ERA Seber estimates:
| region, banding era | model Sa (no disease) | model Sa x (1 - m) | BBL estimate [95%] |
|---|---|---|---|
| West 1960-89 | .746 | .746 | .50 [.475, .55] |
| West 1994-2019 | .671 | .578 | .425 [.375, .50] |
| Central 1960-89 | .681 | .675 | .60 [.425, .775] |
| Central 1994-2019 | .738 | .469 | .55 [.425, .65] |
| East 1960-89 | .724 | .722 | .575 [.525, .625] |
| East 1994-2019 | .757 | .454 | .525 [.375, .65] |
At the cells where birds were actually banded (dense, high-habitat), [cur]'s adult survival is HIGHER than banding says
(pre-1994: .72-.75 vs .50-.60), not lower (established for [cur]). The "implausibly low adult survival" reading (R81, R81b-d,
R83 text, BBL-ERA, review 8 item 10) compared banding with the OCCUPIED-CELL MEAN / all-land median (.18-.36), which is
dominated by marginal cells banding never samples -- the J22 lesson again (compare on the anchor's own spatial support).
Consequences (likely): (1) the R81 "BBL anchor" (occupied East mean Sa 1966-93 = .55) was NOT a banding anchor; it forced
survival up in marginal cells -- its findings stand as "raising marginal-cell Sa costs ~150-300 and nothing in the invasion
era", not as "plausible survival is affordable"; (2) R83/R92's BBL likelihood did not raise Sa because at the banded cells
Sa is already above the banding level -- consistent, not "outvoted"; (3) what banding contradicts is the SPATIAL CONTRAST
(.75 in the core vs ~.2 in marginal occupied cells) and, post-1994 West, the level (.58 vs .425); the deep early sink is a
statement about marginal cells (deep-sink share of occupied East cells ~.7 at 1966), which banding cannot test; (4) the
disease-inclusive post-1994 model Sa at banded East cells (.454) is inside the banding interval -- the modern toll is
CONSISTENT with banding there, and the West post-1994 model is too high (.578 vs .425 [.375, .50]).
Realised per-capita growth (occupied median N_{t+1}/N_t): East 1.08 / 1.38 / .98 / .98 / 1.00 (1966-79 / 80-93 / 94-2000 /
2001-10 / 2011-24), Plains 1.05 / 1.10 / 1.10 / 1.01 / 1.01, West 1.03 / 1.02 / 1.01 / .98 / .99.
R91 (running) uses the occupied-mean anchor: read it as a marginal-cell survival test only.

### R93 (b) RESULT: NB-const + BBL disease-inclusive likelihood
BBS 114,590 (arm a 114,569: +21); BBL -5,149; reporting rates stayed near the .005 start (.0046-.0050). Swing, toll and
rates identical to (a) (East .49 -> 1.52; toll .18 / .29 / .31). The variance form does not change how much the banding term
moves the fit: nothing (established for this pair). Consistent with the RD correction: at the banded cells the model's Sa is
already >= banding, so the Seber term has little to pull on. (R92's --bbl_disease K-approximation bias also applies here.)
RD c = .35 banded-cell Sa: East .73 / .51 with disease post-1994 (base .76 / .45), West .68 / .60 (base .67 / .58): along the
ridge the banded-cell post-1994 effective survival moves only ~.05 -- banding cannot locate the ridge position (likely).

### RD across the R90 ridge (corrected readouts; rescores at each arm's saved params)
| arm | East realised growth 66-79 / 80-93 / 94-00 / 01-10 / 11-24 | banded-cell Sa East pre-1994 / post-1994 with disease | West pre / post |
|---|---|---|---|
| base (.8) | 1.084 / 1.377 / .980 / .978 / .999 | .72 / .45 | .75 / .58 |
| c .35 | 1.095 / 1.362 / .976 / .983 / .997 | .73 / .51 | .76 / .60 |
| c .25 | 1.106 / 1.360 / .968 / .982 / .987 | .71 / .51 | .74 / .59 |
| c .15 | 1.098 / 1.369 / .973 / .979 / .967 | .56 / .41 | .63 / .49 |
| BBL (Seber) | -- | .575 [.525, .625] / .525 [.375, .65] | .50 [.475, .55] / .425 [.375, .50] |
The REALISED dynamics are the same at every point of the ridge (growth within ~.01-.03 per era; established): the ridge
re-decomposes one population trajectory into different potential rates x disease x crowding -- the equilibrium /
transient non-identifiability of review 8 B, made concrete. The swing in POTENTIAL (low-density) lambda that the ridge
trades is therefore not seen by the counts except through the ~1,800-unit preference. Banded-cell adult survival is
closest to the banding estimates at the LOW-toll end (c .15: East .56 vs .575; West .63 vs .50; post-1994 .41 / .49 inside
or near the intervals) and too high in the core at the base end (.72-.76) -- the one external datum that discriminates
along the ridge points (mildly) toward the low-toll / small-swing end (likely; crude cohort weighting, unconverged arms).

### R94 RESULT: DENSITY-INDEPENDENT disease (k_half fixed ~1e-4 counts; hill ~1 wherever birds exist), full data
Not converged (949 evals). Objective 136,009 (+418; BBS +302: East +0 / +48 / +72 / +48, West +22 / +8 / +4 / +76, Plains
<= +13). Never .066; fronts +3/-1/-3/-7. East toll .18 / .31 / .35 (base .17 / .28 / .31), Plains .29, West .22 (2011-25).
Occupied-East potential .54 -> 1.49 (base .48 -> 1.53); realised East growth unchanged (1.08 / 1.37 / .99 / .97 / 1.00).
Review-8 prediction (toll < .2, swing shrinks) FALSIFIED (established for this arm): forced to hit low-density and front
cells, disease gets LARGER, not smaller, and the swing is unchanged. The ridge is not an equivalence between two density
regulators. With density-independent disease the low-density effective rate IS cancelled (~1.49 x (1 - .35) ~ 1.0 at the
front too), and the fit accepts that for ~400 -- so the high-rate / high-toll decomposition survives even when disease must
also act where birds are sparse (likely). The ridge is between the vital rates' RISE and an equally large post-1994 mortality
term, whatever its density form; only a cap on the toll (R90) or an outside constraint moves it.

### R95 (i) RESULT: ADULT SURVIVAL TIME-STATIC (Sa on the 1966-2025 mean field; Sj, Fmax on the real field), full data
(summary label "Sa/Sj/Fmax readouts frozen" is edited_label not knowing --real_rates; the rates confirm Sa static.) Not
converged (875 evals). Objective 136,297 (+707; BBS +629: East +136 / +199 / +75 / +38, West +70 / +19 / +56 / +21, Plains
<= +6). Never .060 (base .068); fronts -1/-2/-3/-6 (base +3/-1/-3/-6: the first cohort now on time / slightly early).
Rates: Sa .355 flat (all-land East median), Sj .057 -> .151, Fmax 2.36 -> 4.16; Shapley Sj .59 / Fmax .41. East toll .18 /
.29 / .31 (= base). Occupied-East potential .43 -> 1.56 (base .48 -> 1.53).
Review-8 prediction (cost < ~300, Sa >= .45): cost FALSIFIED (+707, ~2/3 in the invasion era East 1966-93 and West 1966-79);
not > 1,000, so the "Sa buys speed" alternative is not supported either (likely). The swing does NOT shrink when Sa cannot
change: it RELOCATES into juvenile survival and fecundity at the same size, and the disease toll is unchanged (established
for this arm). With R86 (iv) (crowding on Sj -> swing into Sj) this is the third time the swing has moved to whichever rate
is free: the potential-lambda swing is a property of the habitat-driven rate system + the toll, not of any one rate. The
banding-consistent structure (flat Sa) costs ~700, mostly in fitting the early eastern invasion.
R86 (iv) continuation (_ddonly_c2, +895 evals, still not converged): objective 136,163 (+572 vs base; was +655), BBS +672
(West +121 / +129 / +164 / +70, East -4 / +106 / +52 / +34); cs 5.0 (juvenile survival x .17 at K); Sj .078 -> .366 (4.7x),
Fmax ~flat 2.86 -> 3.16, Sa .30 -> .50; East toll .17 / .31 / .36; occupied-East low-density potential .69 -> 2.25 (+1.56).
The earlier reading holds and strengthens: with all crowding on juvenile survival the swing moves into Sj and GROWS, the
toll grows, and the West pays ~480 (likely; still unconverged).

### R96 PRE-REGISTRATION (multistart, user's mode / ridge criterion): [cur] structure (ceiling .8), full data, habitat + all
### disease sites refit, STARTED from the R90 c = .25 end point (low toll .18, swing +.50); CPU 24-31, --disease_readout.
Prediction (one ridge, data-bounded): L-BFGS climbs back toward the high-toll corner (toll -> ~.3, swing -> ~1.0) and the
objective approaches [cur]'s 135,591; if instead it stalls near the start (toll < ~.22) with an objective > ~500 above
[cur] and a small gradient, the start is in a separate basin or on a very flat ridge (distinguish by continuing).

### R90-VIZ: the STANDARD map suite (map_diagnostics.py, the step's own frozen code = 31_model_viz) on the ridge arms
Spliced MAP checkpoints ([cur]'s map_checkpoint.pkl with each arm's 17 refit sites; arm overlay). Outputs:
E:/Datasets/houfin/grid54/sandbox/R90_viz/<base|c0.35|c0.25|c0.15>/map_diagnostics/ (barrier figure skipped: path meta not
local; animations as GIF). Splice check: base metrics.json = [cur] TACC viz_ref/metrics.json to 3-4 digits (established).
| metric | base | c .35 | c .25 | c .15 |
|---|---|---|---|---|
| holdout temporal (2021-25) mean lpd | -1.792 | -1.785 | -1.796 | -1.840 |
| holdout spatial (blocks) mean lpd | -1.433 | -1.436 | -1.436 | -1.442 |
| fitted mean lpd | -.713 | -.717 | -.721 | -.725 |
| modern mean lambda (fundamental) | .98 | .88 | .80 | .72 |
| mean |delta lambda| since invasion | .57 | .45 | .37 | .32 |
| realised modern source fraction | .40 | .38 | .34 | .27 |
| disease hill_n / onset tau (yr) | 2.3 / 3.9 | 4.2 / 2.4 | 8.0 / 1.4 | 14.6 / .6 |
Validation is nearly flat along the ridge (c .35 best on 2021-25; spatial within .01 lpd) until the .15 end. Figure 01b:
at the base end the whole East is sink in 1940-55 and almost all of it "gained lambda > 1" by 2010-25 (hostile ->
paradise); at c .15 the Northeast / Great Lakes belt is ALREADY a source in 1940-55 ("persistently lambda > 1") and gains
are mostly Southeast / Mid-Atlantic interior. Under the cap the disease block turns into a sharp-onset (tau .6-2.4 yr),
steep density-threshold (hill_n 4-15) epizootic -- closer to the documented one (Hochachka & Dhondt). So the hostile ->
paradise eastern story is the base end of a ridge that validation cannot discriminate (established for these metrics;
unconverged arms).

### R95 (ii) RESULT: time-static Sa + BBL likelihood (pre-disease recoveries, <= 1993)
BBS 110,579 (static alone 110,674: -95; base 110,045: +534); BBL -3,472; reporting rates mostly at the .005 start. Sa flat
(all-land East median .33; occupied mean East .44, West .44); Sj .056 -> .155, Fmax 2.28 -> 4.11; East toll .19 / .29 / .31;
occupied-East potential .40 -> 1.55. Never .064, fronts 0/-2/-3/-6. Same picture as (i): banding adds little, the swing sits
in Sj / Fmax at full size, the toll is unchanged (established for this arm).

### DECLINE readout (review 8 C5; decline_readout.py): the post-arrival transient along the ridge
log(mean count onset+2..+4 / onset-3..-1), same cells, onset = documented arrival map (data disease_onset); route rows:
| region (cells, rows pre/post) | observed | base (.8) | c .35 | c .25 | c .15 |
|---|---|---|---|---|---|
| East (831; 3,786 / 3,805) | -.42 | -.33 | -.28 | -.25 | -.21 |
| Plains (190; 562 / 594) | -.25 | +.14 | +.14 | +.10 | +.08 |
| West (633; 2,164 / 2,097) | +.10 | +.01 | -.01 | -.03 | -.03 |
The eastern post-arrival decline (observed -.42, ~34%) is UNDER-predicted at every ridge point, least at the high-toll base
end (-.33) and most at c .15 (-.21): the transient discriminates along the ridge by ~.12 (below the pre-stated .2 bar, but
systematic) and points the OPPOSITE way to banding -- toward a substantial toll (established for these arms). The Plains
decline (-.25) is missed by every arm (all predict growth): a model failure independent of the ridge (likely: disease
arrival / severity there, or the Plains abundance level). Read (likely): the counts' transient wants a large, SHARP drop
(none of the arms reproduces its size -- the base drop is a 17-yr ramp, the capped arms' drops are sharp but small);
the banding wants modest modern core survival. A sharp, large epizootic with a plateau (Hochachka & Dhondt) and no rising
potential lambda behind it is the combination none of the arms has tried: sharp onset prior + toll free + a constraint on
the potential-lambda swing (e.g. a prior on the rates' time change, or K carrying the modern level). That is the next
structural test.

### R97 PRE-REGISTRATION: sharp epizootic + NO post-1993 vital-rate rise (the combination the decline + banding point to)
--decouple --vr_alpha 1 --vr_alpha_late 0 --late_from 1993 (vital rates read the habitat field frozen at its 1993 state
after 1993; K keeps the real field, so modern abundance change must come through K or disease), full data, habitat + disease
refit, G = 1. Arms: (a) onset sharpness fixed at the prior (tau_raw .5) -- sharp epizootic; (b) tau free. GPU after the R90
continuations; --disease_readout. Predictions: if the post-1993 rise exists only to be cancelled by the toll, (a) keeps the
eastern decline (observed -.42) at least as well as base (-.33), the toll stays large (>= ~.25) but no longer cancels a
paradise (post-1993 potential lambda flat), cost < ~500 vs base; if instead the post-1993 rise carries real fill-in, cost
> ~1,000 concentrated in 1994-2025 East rows and K / the habitat weights move to compensate.

### R91 RESULT: full-data occupied-mean Sa anchor (.55 East 1966-93; a MARGINAL-cell survival test per the RD correction)
Not converged (865 evals). Objective 135,905 (+314 incl. anchor penalty; BBS +196: East +39 / +50 / +43 / +14, Plains <= +19,
West <= +16). Never .056 (base .068); fronts +1/-1/-3/-6. Rates: Sa .34 -> .61 (all-land median; base .20 -> .49), Sj .073 ->
.134, Fmax 2.40 -> 3.92. East toll .16 / .27 / .28 (base .17 / .28 / .31). Occupied-East potential .67 -> 1.47 (base .48 ->
1.53): raising marginal-cell adult survival lifts the 1966 start by ~.2 and trims the 2010 end slightly, at ~200 BBS units --
consistent with R81b-d (established for this arm). The swing (+.80) remains; the toll falls only ~.03.

### KLINK RESULT: exp vs softplus capacity link (habitat + K sites refit; disease, route effects, NB at [cur] values)
softplus twin: converged at [cur] (135,590.8; identity check). exp: CONVERGED (1,016 evals) from a far start (210,861 -- the
warm start was NOT near-identical as pre-registered: the exp link's re-solved alpha_k location and the top decile moved K a
lot) to 136,120 (+529; BBS +513: West +73 / +90 / +158 / +105, East -3 / +15 / +58 / +13, Plains < +6). Never .070; fronts
+3/0/-3/-6. Held-out: pre-1994 blocks 10,834 (+103), post-1993 blocks 21,181 (-193, better), 2021-25 21,456 (-101, better).
Occupied-East potential .42 -> 1.49 (softplus .48 -> 1.53).
Against the pre-registration: training did NOT improve (worse, mostly West); held-out modern rows improved modestly. Not a
clean test (likely): the route effects that absorbed the softplus compression (count-weighted +.98, NB-CAL) and the NB
power were held at softplus-fit values, so the exp link could not re-allocate level between K and the routes; and the far
start may have landed in a different basin. Next: KLINK2 with observer_route_eps (19,127) + disease sites free under both
links (CPU 8-15). The link does not touch the lambda swing (.42 -> 1.49).
R90 c .35 continuation: CONVERGED (1,180 more evals) at 136,095.3 -- only .45 below the capped run's 136,095.7. The c .35
cost (+505) is a converged value, not an upper bound (established); review 8's "upper bound" caveat is void at this point.
R90 c .15 continuation: 1,716 more evals, 137,425.0 -> 137,422.2 (-2.8; not flagged converged but flat); toll and swing
unchanged (.10; .64 -> .91). The c .15 cost (+1,831) is effectively converged too. The ridge profile costs (+505 / ~+1,100 /
+1,831 at ceilings .35 / .25 / .15) stand as near-converged values (established for .35 and .15; .25 continuation on CPU).

### R97 RESULT: NO post-1993 vital-rate rise (rates frozen at their 1993 habitat state; K on the real field), full data
| arm | objective (vs base) | BBS (vs base) | East toll 94-00 / 01-10 / 11-25 | occ.-East potential 1966 / 1990 / 2000 / 2010 | held-out post-1993 blocks / 2021-25 (base 21,374 / 21,557) | never |
|---|---|---|---|---|---|---|
| (a) sharp onset (tau prior) | 136,293 (+702) | 110,775 (+731) | .15 / .24 / .23 | .48 / .95 / 1.06 / 1.06 | 21,532 (+158) / 21,492 (-65) | .086 |
| (b) onset free (ramp kept, tau_raw 3.6) | 136,039 (+448) | 110,481 (+436) | .16 / .24 / .23 | .49 / .95 / 1.05 / 1.05 | 21,314 (-60) / 21,780 (+223) | .073 |
Neither converged (~1,090 evals). BBS cost (b): East +3 / +20 / +100 / +98, West +37 / -5 / +58 / +74, Plains +4 to +23.
Realised East growth unchanged (1.09 / 1.38 / 1.00 / .96 / .99). Rates (East all-land 1966 -> 2010): Sa .21 -> .35, Sj .061 ->
.107, Fmax 2.56 -> 3.47 (base .20 -> .49 / .060 -> .147 / 2.41 -> 4.06).
Pre-registration: (b) met "cost < ~500 with a large toll" (+448; toll .23 -- just under the .25 bar); (a) cost +702 (above).
Removing the post-1993 rise HALVES the eastern potential-lambda swing (+.56 vs +1.05; 2010 1.05 vs 1.53), drops the toll .31
-> .23, keeps the realised trajectory, and moves held-out scores by < ~1% in mixed directions (established for these arms).
Per unit of swing removed this is the CHEAPEST lever found: ~450 for -.49 of 2010 potential lambda (the toll cap needed ~1,100
at c .25 for -.53 -- R90). The post-1993 eastern "paradise" (potential 1.05 -> 1.53 after 1993) is the convenient-lever part
of the inflation: its counts' value is ~450 units, mostly the 1994-2025 East and West rows, and no clean held-out test
supports it (likely; unconverged). The pre-1993 rise (.48 -> .95 by 1990, the invasion-era fill-in) is kept in both arms --
consistent with R80/R82: the invasion era needs its rise; the modern era's extra rise is cheap to drop.
R97 follow-ups queued (GPU): (b) continued to convergence (_free_c2); _half: post-1993 rate change at 50% (vr_alpha_late .5),
onset free -- the post-1993 profile point between full (base) and frozen. Prediction: cost ~150-250, 2010 potential ~1.3.
R90 c .25 continuation (CPU, 1,039 evals): 136,701.6 -> 136,701.0 (-.6). All three ridge points are at their optima within
~3 units: +505 / +1,110 / +1,831 at ceilings .35 / .25 / .15 (established).
R97 (b) continuation (+1,633 evals): 136,039 -> 135,951 (+360 vs base; BBS +357); still not converged (gradient norm ~4,000
at a stall -- line-search limited; cost is an upper bound). East toll .16 / .23 / .23; potential .49 / .94 (1990) / 1.03
(2010). Held-out: pre-1994 blocks 10,733 (base 10,731), post-1993 blocks 21,332 (-42), 2021-25 21,661 (+104). The
post-1993 eastern potential-lambda rise (+.5) is worth <= ~360 in-sample and ~0 on held-out blocks (likely).

### R97 _half RESULT: post-1993 vital-rate change at 50% (onset free), full data
Objective 135,667 (+76; BBS +78), not converged (1,648 evals). East toll .17 / .27 / .28; potential .48 / .94 (1990) / 1.31
(2010). Held-out: pre-1994 blocks 10,744 (+13), post-1993 blocks 21,315 (-59), 2021-25 21,543 (-14) -- equal or better than
base on every held-out score. POST-1993 PROFILE (2010 East potential / cost): full rise 1.53 / 0; half 1.31 / +76; frozen
1.03 / <= +360. Prediction (cost 150-250, ~1.3) met on lambda, cost lower than predicted (established, upper bounds).
Read (likely): HALF of the post-1993 eastern potential-lambda rise (-.22) is free -- no in-sample cost worth the name and no
held-out loss; the rest costs ~280 in-sample and nothing on held-out blocks (2021-25 +104 at full freeze). The modern
"paradise" beyond ~1.3 has no support from never-fitted data. Production implication (report): a prior that shrinks the
POST-1993 vital-rate change (or a smaller time-change budget for the rates after the invasion era) would remove most of the
modern inflation at ~0-360 units, leaving the invasion-era rise (.48 -> .94) that the fronts need.
R97 _quarter (post-1993 change at 25%): +201 (iteration cap, gradient large -- upper bound); East toll .17 / .25 / .26;
potential 2010 1.17; held-out pre / post blocks / 2021-25 10,748 (+17) / 21,297 (-77) / 21,600 (+43).
POST-1993 PROFILE: 1.53 / 0 (full); 1.31 / +76 (half); 1.17 / <= +201 (quarter); 1.03 / <= +360 (frozen). Roughly linear in
the retained rise; held-out post-1993 blocks best at quarter-to-half, 2021-25 best at half (-14) (established, upper bounds).

### R96 interim (multistart): both starts (R90 c .25 and c .15 end points, uncapped structure) have reached 135,590.9 /
135,591.1 -- [cur]'s optimum (135,590.8) within ~.3 -- after ~1,000-1,200 evals; still running (readouts at the end). The
uncapped model returns to the same objective from far along the ridge: one optimum, no separate basin at this scale
(established for these starts; where along the ridge it settled is pending).
R97 _half_sharp PRE-REGISTRATION: half post-1993 rise + onset sharpness at its prior (tau_raw .5). Prediction: the sharp
epizootic costs ~300 (R88) on top of half (+76) unless the smaller modern rise makes a sharp drop cheaper; East decline
readout closer to observed (-.42).
R97 _half_sharp RESULT: +396 (BBS +440; East 1994-2009 +237, 2010-25 +83, West 1994-2009 +67); not converged (1,646 evals).
East toll .16 / .26 / .26; potential 2010 1.28. Held-out: pre-1994 blocks 10,822 (+91), post-1993 blocks 21,490 (+116),
2021-25 21,355 (-202, best of any arm today). Prediction met (sharp onset adds ~320 on top of half's +76; the smaller modern
rise does not make the sharp drop cheaper). The in-sample 1994-2009 East rows keep preferring the gradual ramp, while the
2021-25 forecast prefers the sharp epizootic + half rise (likely; one split, unconverged).
R98 PRE-REGISTRATION: R97 _half and full-rise twin with observer_route_eps (19,127) free -- does the post-1993 profile
survive when route effects can re-absorb level? Prediction: the half-rise cost stays < ~150 (route effects cannot create a
temporal rise at fixed routes).
R98 RESULT (route effects free): full rise -- converged at [cur] in 183 evals (135,590.8; identity: [cur] is already the
joint optimum incl. route effects); half rise -- 135,660.4 (+70; R97 half with route effects fixed +76), East potential 2010
1.30, toll .28; held-out pre / post blocks / 2021-25 10,755 (+25) / 21,315 (-59) / 21,534 (-23). Prediction met: freeing
19,127 route effects changes the half-rise cost by 6 units (established). The post-1993 profile is not a route-effect artefact.

### R99 PRE-REGISTRATION: the missed Plains post-arrival decline -- is native-lineage disease suppression the cause?
Decline readout: Plains observed -.25 vs +.08 to +.14 predicted by every ridge arm. [cur] b_native_raw .89 (suppression
-softplus(.89) = -1.24 logit in the native core, applied through native_shape = the 1902 seed map). Arm: b_native_raw fixed
-6 (suppression ~0), habitat + other disease sites refit, full data (GPU); then ridge_maps compute + decline_readout on it.
Prediction: if native suppression hides the Plains epizootic, the Plains decline appears (< ~-.1) at small cost (< ~200) and
the West changes little; if not, the Plains miss lies in abundance level / arrival timing there.
R99 RESULT: native suppression OFF -> +94 (BBS +45, almost all West 2010-25 +42); tolls unchanged by region (Plains .04 /
.14 / .21, West .03 / .11 / .19 -- the severity field and other sites re-create the same pattern); decline readout Plains
+.16 predicted vs observed -.25 (base +.14), West +.09 (observed +.10; base +.01), East unchanged. Prediction FALSIFIED: the
missed Plains decline is not native-lineage suppression (established for this arm). The Plains toll is ~.14-.21 in every arm,
yet the model predicts GROWTH across the arrival window: the Plains vital-rate rise over 1999-2008 (Plains potential .59 ->
.94 by 2010, R90/RD) outruns the disease there (likely). The same post-1993 rise that is cheap to drop in the East (R97) is
what makes the Plains miss the documented decline -- a further argument against the modern rise (likely; the R97 half/frozen
arms' Plains decline readout is the direct check, not yet run).

### KLINK2 exp arm (exp link; route effects + disease + habitat free), from the KLINK exp end point
135,573.1 (-18 vs [cur]; BBS -3 -- in-sample tie), not converged (1,632 evals). Held-out: pre-1994 blocks 10,384 (-347 vs
[cur] 10,731), post-1993 blocks 21,072 (-302), 2021-25 21,322 (-235): ~-880 in total on never-fitted rows, the largest
held-out gain of any arm in this sandbox. East toll .32; potential .43 -> 1.57 (the link does not touch the swing). The
softplus twin with the same free sites is at 135,590.1 (still running): the comparison is pending its held-out scores, but
the in-sample tie + held-out gain says the exp link generalises better once route effects can re-allocate level (likely).

### R96 RESULT (start R90 c = .25 end point, uncapped [cur] structure): returned to [cur]
135,590.6 after 1,841 evals ([cur] 135,590.8). End point = [cur]: East toll .17 / .28 / .31, potential .48 / .94 / 1.53, disease
latents within ~.05-.2 of [cur] (tau_raw 3.62, rec_raw -3.37, mu_sev .50), habitat-weight distance ~3% of their norm.
Established for this start: from far along the ridge (toll .18, swing +.50, +1,110) L-BFGS climbs back to the SAME high-toll
corner with the same objective -- one optimum at the ridge's end, not a separate basin; the ridge is long but its top is a
single point under [cur]'s structure. Against the user's criterion (bounded ridge vs distinct modes): BOUNDED, unimodal along
this direction -- the problem is not multimodality but that the optimum sits at the large-swing / large-toll end because
nothing in the model penalises the swing, while banding, the 1940s source belt, the sharp documented epizootic and the
post-1993 profile (R97: half the modern rise is free; held-out prefers it) all point down-ridge. _from_c15 pending.

### R100 PRE-REGISTRATION: first ASSEMBLED candidate -- exp K link + route effects free + half post-1993 rate rise
Overlay exp (KLINK), --decouple a = 1 / G = 1 --vr_alpha_late .5 --late_from 1993, habitat + disease + observer_route_eps
refit, start = KLINK2 exp end point, full data, GPU, --disease_readout. Compare with KLINK2 exp (full rise). Prediction:
in-sample cost ~+70 (R97 half / R98), held-out at least as good as KLINK2 exp (blocks pre 10,384 / post 21,072; 2021-25
21,322), East 2010 potential ~1.3, toll ~.28.
R96 _from_c15 (start c = .15 end point: toll .10, swing +.27, +1,831): 135,590.7 after 1,862 evals; end point = [cur]
(toll .17 / .28 / .31, potential .48 / .94 / 1.53). Both multistarts return to the same optimum (established).

## REVIEW 9 (2026-11-04, subagent, read-only; key claims checked by me) -- corrections to R93-R100
1. R97's "cheap post-1993 freeze" is UNVERIFIED SUBSTITUTION until K and Q are read (certain that nobody looked): K reads
   the real field (swap_real in the estab branch), the shared habitat weights re-tilted (manifold_factor 8.7% / alpha_k
   -1.615 -> -1.676 in _free_c2), and --decouple_keys Z_gathered leaves Z_disp real, so journey survival Q keeps its post-1993
   rise. R82 (latents fixed) found dropping the post-1993 rise costs +1,942 on 2021-25; R97 is cheap only because K / weights
   / disease re-tune. "Consistent with R82" was wrong. The freeze date also coincides with disease arrival (toll .31 -> .23):
   possibly the same ridge by another lever. "Cheapest lever" / "held-out prefers it" are overstated: R97 held-out gaps
   (13-104) sit inside the ~750 start-noise band; _half_sharp's 2021-25 -202 is offset by +207 on the blocks.
2. KLINK2: my "route effects re-allocate level" explanation is WRONG (checked: exp_eps route effects vs [cur] correlation
   .995, sd .629 vs .630); a habitat weight moved instead (w_scale_raw[2] 5.49 -> 4.04, likely K's slope), and KLINK2 also
   freed disease -- KLINK vs KLINK2 confounds the two. The ~880 held-out gain is unattributed (exp tail, disease refit, or a
   different basin from the far start); the held-out set is now heavily reused (M13) -- gaps < ~200 should not rank structures.
3. R90-VIZ "validation flat" fails at c .15 (per-row lpd x rows: ~+580 on 2021-25, ~+200 on blocks); flat within ~+-100 for
   .35 / .25 only. The 1940-55 eastern source belt is a BACKCAST (model output, not evidence); hill_n 14.6 / tau .6 under the
   cap is a forced density switch, not a match to Hochachka & Dhondt (likely).
4. Decline readout: pairs CELLS not routes (docstring wrong); pooled means dominated by over-confident dense rows; no CI;
   single +2..+4 window cannot tell sharp-then-plateau from a 17-yr ramp; pre-window catches the 1980s-90s growth; observer
   turnover (M3) and regional co-species change (M6) uncontrolled.
5. R96 "one optimum" is narrow: habitat + disease free only (routes, observers, NB, Allee, process fixed), all starts within
   ~3% of [cur]'s habitat weights -- not dispersed starts.
6. Alternatives: equilibrium non-identifiability (N* = K: the potential-lambda rise only sets return speed in settled cells;
   plan A1 -- separate K and rate weights -- still untested); the swing inherited from DESK's temporal over-movement (~2.2x,
   G3/G4); decline/toll partly observer turnover or post-1994 K change; trade with the pinned Allee n50 (P5).
Next round (review 9 C, ranked): (1) K / Q readout on base, R97 frozen / quarter / half, R98 half, KLINK2, R100 [running
next]; (2) freeze placebos (Z_disp frozen too; late_from 1980 / 2005; West-only); (3) decline v2 (per-year profile,
route-paired, observer-continuous, bootstrap, disease-off counterfactual); (4) KLINK attribution (count bins, level vs
pattern, exp with disease fixed, second exp start); (5) plan A1 separate K / rate weights; (6) production-like assembly with
a soft B1 prior on realised post-invasion rate change + local full L-BFGS multistart from dispersed starts.

### R101 / KQ PRE-REGISTRATION (review 9 C1 + user, 2026-11-04: "rates vs K ... how much should these be coupled?")
KQ: rescores of base / R97 frozen / half / quarter with the K the forward uses (real-Z pass) and journey survival Q:
occupied median K 1966 / 1993 / 2010 / 2020, dlogK and dQ 1993 -> 2010 (CPU 16-19). Prediction (review): if the freeze
RELOCATES the rise, frozen dlogK(East) exceeds base by >= .1 and/or Q's rise persists; within +-.03 = the rise was redundant.
R101: the exchangeable (separately-correlated fields) form (plan A8 overlay; private parts eps (M, 4) per rate incl. K,
coupling rho ~ logit-normal(.85, .15)), full data, habitat + disease refit, start = exact single-field reproduction (CPU
24-31). The user's question: can K carry abundance change while the low-density rates stay put, if the fields may decouple?
Prediction: if the counts only constrain abundance, K's private part takes the modern change, rho falls below ~.8, East 2010
potential < ~1.3 at a cost <= 0 (more parameters); if rho stays ~.85+ and the swing persists, the coupling prior (or the
data) holds rates and K together. Known failure mode (age_priors._exchangeable_w_env docstring): a decoupled K field became an
absence switch historically -- check K's private part for that.

### R100 RESULT (assembled: exp link + route effects free + half post-1993 rise), vs KLINK2 exp (full rise)
135,672.0 (+99 vs KLINK2 exp 135,573.1; +81 vs [cur]); BBS +96; not converged (1,621 evals). Held-out: pre-1994 blocks 10,410
(+26), post-1993 blocks 21,041 (-31), 2021-25 21,299 (-23) -- equal to KLINK2 exp within noise and ~-320 / -330 / -260 vs
[cur]. East toll .17 / .27 / .29; potential .43 / .91 / 1.33 (KLINK2 exp 1.57). K (forward K, exp link) East median 1966 /
1993 / 2010 .038 / .090 / .129, dlogK 1993-2010 +.30 (softplus base +.69 -- different link, not directly comparable; the
KQ softplus arms are the like-for-like check); Q +.041. Prediction met (cost ~+70-100, held-out no worse, potential ~1.3)
(established for this start; held-out gaps within the reuse / start-noise caveats of review 9).
KLINK3 PRE-REGISTRATION (review 9 C4): exp link + route effects + habitat free, disease FIXED at [cur], start KLINK exp end
point (GPU). If the ~880 held-out gain of KLINK2 is the link, KLINK3 keeps most of it (> ~600); if it was the disease refit,
KLINK3 loses it.

### KQ RESULT (base vs R97 frozen; forward K from the real-Z pass; occupied medians, density units)
| arm / region | K 1966 / 1993 / 2010 / 2020 | dlogK 1993-2010 | dQ 1993-2010 |
|---|---|---|---|
| base East | .023 / .063 / .154 / .161 | +.686 | +.036 |
| frozen East | .022 / .058 / .138 / .147 | +.680 | +.036 |
| base Plains / West | -- | +.586 / +.219 | +.036 / +.015 |
| frozen Plains / West | -- | +.575 / +.206 | +.036 / +.014 |
Review 9's relocation hypothesis is REFUTED for K (established for this pair): the frozen arm's eastern dlogK is within .006 of
base (pre-registered "redundant" band +-.03); K's modern rise (~2x in the East) is the SAME with or without the rates' rise --
K was already carrying the abundance change in [cur], and the rates' post-1993 rise was on top of it. Q's rise (+.036) is
identical in both because the freeze never touched Z_disp (the freeze is incomplete for Q, as review 9 said, but nothing
moved into Q either). So the R97 cheapness is not substitution into K or Q: the post-1993 vital-rate rise is largely
REDUNDANT with K's rise for the counts (likely; R82's +1,942 with latents fixed is the cost of removing it WITHOUT letting
K's weights / disease re-tune, i.e. the counts need the abundance change, which K supplies). This is the user's point made
concrete: in settled cells the counts see abundance (K), not low-density growth.
KQ half arm: dlogK East / Plains / West +.718 / +.616 / +.222, dQ +.038 / +.038 / +.016 -- within ~.03-.04 of base (+.686 /
+.586 / +.219). Same conclusion: K's modern rise is common to all post-1993 profile points (established for these arms).
R102 PRE-REGISTRATION (review 9 C2, freeze placebo): vital rates frozen from 1980 (vr_alpha_late 0, late_from 1980), K real,
disease + habitat refit, full data (CPU 16-19). Prediction (review): disease-specific cheapness -> East cost > ~1,000 (the
1980s fill-in needs the rates' rise); equilibrium non-identifiability -> cheap wherever cells were already at K.
KLINK3 RESULT (exp link, route effects + habitat free, disease FIXED at [cur]): 135,602.5 (+12 vs [cur]); held-out pre-1994
blocks 10,381 (-350 vs [cur]), post-1993 blocks 21,055 (-319), 2021-25 21,300 (-257): -926 in total -- the WHOLE KLINK2 gain
survives with disease fixed (established): the disease refit is not the source. Remaining confound: KLINK (exp, route
effects fixed, 1,016 evals) scored worse on pre-1994 blocks (10,834); KLINK2 / KLINK3 continued from its end point with route
effects free. The like-for-like control is the softplus twin with route effects free (still running, 135,589.9 at 1,656
evals): if its held-out stays near [cur], the gain is the exp link (+ its re-tilted K slope); if it gains too, it is the extra
optimisation of route effects.
R100 start 2 (review 9 C6, start dependence): same candidate structure started from [cur]'s softplus values (not the KLINK
end point), GPU, maxiter 2,000. Prediction: within ~100 of R100 (135,672) with the same toll / swing; a gap > ~500 with a
different toll = distinct basins (the user's criterion).
R100 start 2 interim: 136,122 after 2,165 evals (start 1 135,672: +450), gradient norm ~30,000 and still descending (NOT
converged); habitat weights 17.6% apart, toll .27, potential 1.26, pre-1994 blocks 10,767 (start 1 10,410). Same signature
as KLINK exp from a far start (136,120, "converged" with route effects fixed, then -547 when continued with them free).
Continuing (start2_c2, GPU) before any basin claim.
KLINK softplus twin (route effects + disease + habitat free, from [cur]): 135,589.8 (-1); held-out pre / post blocks /
2021-25 10,736 / 21,381 / 21,557 = [cur] (+5 / +7 / 0). Like-for-like (same free sites): exp link -926 (disease fixed) /
-880 (disease free) on never-fitted rows at an in-sample tie (+12 / -18). ATTRIBUTION SETTLED (established for this pair of
fits): the held-out gain is the EXP CAPACITY LINK (with its re-tilted K slope), not route-effect optimisation and not the
disease refit. Production implication (report): the softplus link's compression of the dense upper tail costs ~900 units of
out-of-sample skill; the exp link (or a less compressive link) is the better observation-level structure -- the user's
"transform" hypothesis (NB-CAL addendum) holds on held-out data even though it leaves in-sample fit unchanged. Caveat: the
exp fits started from a far point (KLINK exp end); start dependence of the exp structure is being checked (R100 start 2).

### R103 PRE-REGISTRATION (review 9 C6; toward the end goal): SOFT prior on post-invasion vital-rate change, production-like
refit_fields --late_prior SD: N(0, SD) on the occupied-mean 1993 -> 2010 change of logit Sa, logit Sj, log Fmax for each of
East / Plains / West (9 terms). Structure: exp link + route effects free + disease free + full rise (no hard freeze), decoupled
a = 1 / G = 1, start KLINK2 exp end point. Arms SD .15 (CPU 0-7) and SD .30 (CPU 8-15). Readouts: cost vs KLINK2 exp (135,573),
East 2010 potential, toll, K / Q change, held-out. Predictions: SD .30 -> East 2010 potential ~1.35-1.45, cost < ~50;
SD .15 -> ~1.2-1.3, cost ~100-200, K's rise unchanged (KQ), held-out within ~100 of KLINK2 exp; FALSIFIED if the prior pushes
the change into K (dlogK +.1 over KLINK2's) or the toll rises.
R103 first attempt VOID (optimizer stall, not a result): started at the KLINK2 exp optimum (gradient norm ~61), the first
L-BFGS step overshot to 4e9, the line search collapsed and it stopped on relative reduction within .0003 of the start (SD .30;
SD .15 killed). Restarted both SDs from the R100 end point (exp link, half post-1993 rise; the prior now acts against the
data's pull back toward the full rise), maxiter 2,000 (CPU 0-7 / 8-15).
R100 start 2 progression: 136,122 -> 135,926 -> 135,788 (+450 / +254 / +116 above start 1) over ~7,600 evals, gradient norm
~1,000 and still descending; habitat weights 14.2% from start 1 (was 17.6%); toll .27 (start 1 .29); potential 2010 1.26
(1.33); pre-1994 blocks 10,724 (start 1 10,410). Read (likely): slow convergence toward the same region, not a separate
basin -- but the pre-1994 held-out gap (~300) has NOT closed, and the in-sample gap is shrinking geometrically (~-45%
per leg). Continuing (c4); a basin claim waits for it.
R100 start 2 leg 4 (iteration cap): 135,713.6 (+42 above start 1; legs +450 / +254 / +116 / +42); toll .28 (start 1 .29),
potential 2010 1.27 (1.33), habitat weights 13% apart; held-out pre / post blocks / 2021-25 10,715 / 21,061 / 21,357 (start
1 10,410 / 21,041 / 21,299). In-sample: converging to the same objective (likely within ~0-30 eventually) with the same toll /
swing -- no distinct basin in-sample (likely). BUT the pre-1994 held-out score differs by ~300 at near-equal in-sample fit:
the two starts sit at different points of a FLAT in-sample direction that the never-fitted invasion-era blocks distinguish
(likely) -- a residual ridge in the habitat weights (13% apart) that the training data do not pin. Not "distinct modes" by
the user's criterion (in-sample gap < ~50, same toll), but a weakly identified direction worth a prior / HMC look.
R103 SD .08 arm added (GPU, from the R100 end point) -- the strong end of the prior profile.

### R101 RESULT: exchangeable (separately correlated) fields, full data, habitat + disease refit
Not converged (1,607 evals). Objective 135,588.8 (-2 vs [cur]; BBS -239 -- the private parts buy fit, paid by their prior);
coupling rho .85 -> .73; private-part norms (columns) 4.9 / 6.9 / 4.7 / 6.1 (all four fields used, none collapsed to an
absence switch). Swing NOT reduced: occupied-East potential .49 / .95 / 1.62 (1966 / 1990 / 2010; base 1.53); rates Sa .25 ->
.60, Sj .049 -> .157, Fmax 1.99 -> 3.56; toll .16 / .28 / .32; K East dlogK 1993-2010 +.71 (base +.69), dQ +.053. Held-out:
pre / post blocks / 2021-25 10,647 (-84) / 21,285 (-89) / 21,549 (-8). Never .065; fronts as base.
Pre-registration (K takes the modern change, rho < .8, potential < ~1.3): rho fell to .73 but the swing did NOT shrink and K
did not take more (established for this arm). Answer to the user's coupling question (likely): letting the rate fields
decouple from K does not, by itself, stop the rates' modern rise -- with freedom, the fit keeps (slightly enlarges) the
rates' rise AND K's. So the rise is not only forced by the shared field; the counts (or the disease ridge) still prefer it
when nothing penalises it. Combined with R97 / KQ (the rates' post-1993 rise is redundant with K for the counts, ~0-360)
the picture is: the rise is weakly preferred, cheap to remove, and not removed by structure alone -- it needs a prior
(R103) or the re-weighting nudge to be removed.

### R102 RESULT (freeze placebo: vital rates frozen from 1980), full data
Not converged (1,617 evals, gradient ~4,800; upper bound). Objective 137,027 (+1,436; BBS +1,373): East +14 / +169 / +315 /
+207, West +112 / +55 / +230 / +193, Plains +8 to +29. East toll .14 / .19 / .17 (smaller); potential .62 / .84 / .84 (the
1966 start lifts and the rise stops at 1980); K East dlogK 1993-2010 +.60 (base +.69). Never .055; fronts +2/-1/-3/-7.
Prediction (review 9: disease-specific cheapness -> East 1980 freeze > ~1,000) MET: freezing from 1980 costs ~4x the 1993 freeze
(+1,436 vs <= +360), with East 1980-93 +169 (the fill-in era) and large post-1993 costs in BOTH East and West (established
for this arm, upper bound). So the 1993 freeze's cheapness is specific to the post-1993 era (settled East + disease): the
1980s rate rise is needed (fill-in, R80/R82), the post-1993 one is not. Not general equilibrium non-identifiability (likely).

### R103 SD .08 RESULT (soft prior on the post-invasion change; exp link + route effects + disease free; from R100 end point)
Not converged (3,314 evals). Objective 135,669.6 (+96 vs KLINK2 exp incl. the prior penalty; BBS +11). East toll .17 / .28 /
.31; potential .45 / .92 / 1.53 (KLINK2 exp 1.57; R100 hard half-freeze 1.33); K East dlogK +.56 (R100 +.30); held-out
10,371 / 21,064 / 21,342 (KLINK2 exp 10,384 / 21,072 / 21,322). The tightest prior did NOT shrink the swing: the
OCCUPIED-MEAN 1993 -> 2010 change of each rate is constrained, yet the occupied-MEDIAN potential lambda still rises .92 ->
1.53 (likely ESCAPE by spatial redistribution -- rates rise in the cells that set the median / sources and fall elsewhere,
mean unchanged; to be confirmed from the per-cell change). Lesson (likely, J-log candidate): a prior on regional MEAN
changes is not a constraint on the swing; the production prior must act per cell (e.g. on each cell's 1993 -> 2010 change,
or on the change in potential lambda itself), as the hard freeze does. SD .15 / .30 arms still running.
R103b PRE-REGISTRATION: PER-CELL prior (--late_prior_cell SD: independent N(0, SD) on every occupied cell's 1993 -> 2010
change of logit Sa, logit Sj, log Fmax), same structure and start as R103. Arms SD 1.0 (GPU) and SD .5 (CPU 16-19).
Prediction: unlike the regional-mean prior it cannot be escaped by redistribution -- SD 1.0 -> East 2010 potential ~1.35-1.45,
SD .5 -> ~1.2-1.3; in-sample cost ~50-300; K's rise unchanged; held-out within ~100 of KLINK2 exp.

### R103b per-cell prior SD 1.0 RESULT (exp link + route effects + disease free; from R100 end point)
Not converged (3,249 evals, gradient ~1,200). Objective 135,968 (incl. ~340 prior penalty); BBS +58 vs KLINK2 exp. East
potential .50 / .94 / 1.25 / 1.46 (1966 / 1990 / 2000 / 2010; KLINK2 exp 1.57; regional-mean prior SD .08 1.53); toll .17 /
.26 / .29; K East dlogK +.49; Plains .54 -> .89, West .89 -> 1.08. Held-out vs KLINK2 exp: pre-1994 blocks 10,339 (-45),
post-1993 blocks 21,043 (-29), 2021-25 21,392 (+70). Prediction (1.35-1.45 at a BBS cost of 50-300) MET at the edge
(established for this arm, unconverged): the per-cell prior bites where the regional-mean prior was escaped, for ~60 BBS
units and no held-out loss beyond noise. SD .5 running (CPU).
R103b SD 1.0 continuation (+4,345 evals; still not converged): objective 135,911.7 (-56); BBS +113 vs KLINK2 exp (the fit
moved INTO the prior: penalty fell, BBS rose); East potential .55 / .95 / 1.35 (1966 / 1990 / 2010); toll .16 / .25 / .26;
K East dlogK +.41 (falling, not rising: the prior is NOT pushing change into K -- review 9 falsification criterion not met);
held-out vs KLINK2 exp pre / post blocks / 2021-25 10,329 (-55) / 21,108 (+36) / 21,438 (+116). The per-cell prior at SD 1.0
settles the eastern 2010 potential at ~1.35 (swing .55 -> 1.35, +.80 vs +1.14 unpriced) with the toll down to .26, for ~110
BBS units and held-out within ~120 either way (likely; unconverged). This is the first PRIOR (not a hard edit) that removes
the convenient part of the modern rise without moving it into K, the toll or held-out skill.
R103 regional-mean SD .30 (from R100 end point): 135,582.7, BBS +.6 vs KLINK2 exp, toll .32, East potential .43 -> 1.56 -- returned to the unpriced fit (the weak mean prior is fully escaped; consistent with SD .08).
R103b SD 1.0 leg 3: 135,890.3 (-21), BBS +102 vs KLINK2 exp; potential .55 -> 1.35, toll .26, dlogK +.40; held-out 10,325 / 21,077 / 21,439 -- stable across legs 2-3 (the answer has settled; objective still creeping).
R103b SD 1.0 start 2 (from the unpriced KLINK2 exp end point, full rise): prediction -- converges to East 2010 potential ~1.35 within ~50 of the from-half arm (prior posterior unimodal along the swing).
R103 regional-mean SD .15: 135,603.8, BBS +.5 vs KLINK2 exp, toll .32, potential .43 -> 1.56 -- fully escaped like SD .08 / .30 (established: the regional-mean prior form does not constrain the swing at any strength tried).

### R103b two-sided result (per-cell prior SD 1.0; exp link + route effects + disease free)
| start | objective | BBS vs KLINK2 exp | East potential 1966 / 1990 / 2010 | East toll 11-25 | dlogK East | held-out pre / post blocks / 2021-25 |
|---|---|---|---|---|---|---|
| from half-rise (3 legs, ~12k evals) | 135,890.3 | +102 | .55 / .95 / 1.35 | .26 | +.40 | 10,325 / 21,077 / 21,439 |
| from full rise (1 leg, 4.3k evals) | 135,935.5 | +83 | .52 / .94 / 1.40 | .27 | +.45 | 10,335 / 21,085 / 21,400 |
Both starts converge toward the SAME region from opposite sides of the swing (gap 45 in objective and closing; 2010 potential
1.35 vs 1.40; toll .26 / .27; held-out within ~40) (likely; the full-rise start is one leg in). Against the user's criterion
for the posterior: under the per-cell prior the swing direction is bounded from both sides with one optimum -- the first
configuration that is (a) prior-controlled rather than hard-edited, (b) unimodal along the ridge from opposite starts, (c)
held-out no worse than the best unpriced fit, (d) not relocating the change into K / Q / the toll. This is the candidate
structure for the local full L-BFGS multistart (all latents) -> HMC (TACC via the user).
R101 continuation (+2,277 evals, not converged): 135,539.5 (-51 vs [cur]; BBS -286), rho .68, East potential .48 -> 1.58, toll .30, dlogK +.72; held-out 10,637 / 21,362 / 21,556 (-94 / -12 / -1 vs [cur]). Same reading: decoupled fields fit better in-sample but keep the swing (established for this arm).

### R104 PRE-REGISTRATION (toward the local full L-BFGS): candidate structure with the OBSERVATION and ALLEE sites free
Per-cell prior SD 1.0 + exp link + route effects + disease + habitat, PLUS concentration, obs_conc_kappa, quality_conc_mult and
the Allee n50 (--free_n50 .5), start = R103b SD 1.0 leg-3 end point (CPU 24-31). Prediction: kappa falls (NB-CAL: [cur] too
tight at high counts) with an in-sample gain of a few hundred, n50 moves < ~.5 prior SDs, and the swing / toll stay within
~.05 of R103b (1.35 / .26) -- i.e. the candidate is robust to freeing the remaining nuisance sites; if the swing reopens, the
fixed observation model was holding it.
R103b per-cell SD .5 (CPU, 2,167 evals, not converged; upper bound): objective 136,613; BBS +343 vs KLINK2 exp; East potential
.59 / .96 / 1.24; toll .15 / .23 / .23; dlogK +.32; held-out pre / post blocks / 2021-25 10,299 (-85) / 21,133 (+61) / 21,657
(+335). PRIOR-STRENGTH PROFILE (per cell; 2010 East potential / BBS cost vs unpriced exp / 2021-25 held-out change):
none 1.57 / 0 / 0; SD 1.0 1.35 / +102 / +117; SD .5 1.24 / +343 / +335 (likely, unconverged). At SD .5 the forecast rows
start to pay (+335 on 2021-25) -- SD ~1.0 is where the swing shrinks without held-out cost; SD .5 is past the knee.

## TAKEOVER BACK FROM CODEX (2026-11-05) -- what Codex did, my reading, and the plan
User's direction to Codex (Codex session, 2026-11-05): prior changes on the historical rates are INDICATIVE, not the final
answer -- a prior that sets the swing hides whether another structural change would give a better likelihood shape; prior
tweaks, if any, belong upstream (single vs multiple capacity fields, whether an amplitude term should exist at all); sharp
disease timing may be fixed / strongly constrained (externally grounded); lambda change stays LEARNED. So R103/R103b (my
direct change priors) are DIAGNOSTIC COMPARATORS only.
Codex's work (records in .worktrees/*/research/experiments/TAKEOVER_*; artifacts .research_runtime/takeover/):
- Audits: frozen step43 replay reproduced (135,590.798); exact factor x amplitude rescaling invariance of the likelihood
  (so fixing a redundant amplitude cannot improve the likelihood family); veit logistic departure sites have exactly zero
  data gradient (confirms my dead-site correction); exact-stage disease readout (toll at post-dispersal density: [cur]
  East 2011-25 .264, my end-of-year approximation gave .31 -- my tolls since R84 are ~.05 high).
- Joint fits with EVERY active site free (observers, NestWatch, NB, disease, habitat; no output priors):
  | arm | objective | data NLL | East potential 1966 -> 2010 | toll | held-out pre / post / 2021-25 |
  |---|---|---|---|---|---|
  | F0 original learned onset (refined) | 135,589.8 | 114,023.9 | .48 -> 1.53 | .264 | 10,735 / 21,380 / 21,557 |
  | F0 COMPACT 3-yr onset (3 starts, refined) | 135,761-135,772 | 114,122-114,145 | .61-.62 -> 1.19-1.23 | .150 | 10,715-10,772 / 21,413-21,534 / 21,543-21,604 |
  | compact 2-yr / 6-yr, midpoint, matched sigmoid | 135,759-135,772 | ~114,135-114,145 | .60-.62 -> 1.20-1.23 | .15 | same band |
  | F2 independent K field, original onset | 135,083 | 113,569 | .57 -> 1.84 | .279 | 10,208 / 20,892 / 21,726 |
  | F2 independent K, compact onset | 135,199 | 113,668 | .73 -> 1.73 | .185 | 10,160 / 20,835 / 21,736 |
  | maxcor-200 control (compact) | 135,743 | 114,098 | .58 -> 1.13 | .152 | 10,834 / 21,571 / 21,626 |
  My reading: the compact onset is a STRUCTURAL change that shrinks the swing by ~1/3 (1.53 -> ~1.2) and the toll ~.26 -> .15
  at ~+110 data NLL with held-out flat -- robust across three starts, onset duration (2-6 yr) and date convention (sharpness,
  not timing, matters). This is what the user asked for (a better likelihood shape from structure, no output prior). It
  replaces my R88 reading (+280, swing 1.42) which had observers / NB / NestWatch fixed. F2 (separate K direction) buys a
  lot of spatial skill (~-500 to -700 on blocks) and in-sample fit but ENLARGES the swing and worsens 2021-25 (+170):
  decoupling K does not remove the inflation (consistent with my R101).
- Stiffness: finite differences agree with the gradient, but curvature along the normalised gradient is ~1.6e6 at the
  compact endpoint (a 1e-4 step already increases the objective); the larger-curvature-memory control moved -18 and
  shrank the swing further (1.13) -- the compact fits are NOT converged; where the stiffness sits was never localised.
- Conditional likelihood profiles (data-only objective, observers fixed, RMS per-cell log-lambda change target via
  augmented Lagrangian): BROKEN -- every stage >= 1 fails Codex's own gauge-identity assertion (stage 1 re-gauges an
  already-gauged checkpoint); stage-0 points stalled short of target (RMS .357 vs .30; .291 vs .15); and with priors removed
  the fits move to prior-implausible states (prior cost 150-180k vs 135k; free fit RMS .42). Supervisor stopped by me
  (user permission, 2026-11-05) after the queue proved to be failing; its running workers left to finish.
PLAN (mine): (1) localise the stiffness by Hessian-vector products (stiffness_hvp.py: Lanczos top curvature directions, data
vs prior split, per-site shares) at the compact and original endpoints -- hypothesis: the compact gate makes the objective
stiff in the onset-timing sites (disease_lag0 / w_lag) and the density-Hill sites, which a 3-yr gate couples to habitat;
(2) if localised, test a reparameterisation / preconditioning that removes it and re-run the compact fits to stationarity
(is the converged compact swing ~1.1-1.2 or lower?); (3) upstream structural question the user named: amplitude terms
(drop / pin amplitudes so the single field cannot scale its temporal change) under compact onset; (4) profile likelihood
redone properly (posterior objective, fixed gauge bug) only after the fits are converged.
CONV PRE-REGISTRATION: compact 3-yr onset F0, posterior objective (Codex runner, all sites), L-BFGS maxcor 200, maxiter 4000, from the history_control endpoint (135,743; swing 1.13), CPU 16-31. Question: what does the compact swing converge to? Prediction: objective falls a further ~20-60, East 2010 potential settles 1.05-1.2, toll ~.15; if the swing keeps shrinking as the objective falls, the compact fits' 1.2 was an optimizer artefact.
