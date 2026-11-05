# B6 virtual-ecologist benchmark with an outside-our-code truth (rangr) -- pre-registered 2026-10-31, before any simulation

Parent: SDM_BASELINE_PLAN_2026-10-30.md sec. 4 B6. Framing (user, 2026-10-30): a correlative fit to invasion-era counts
can encode "not yet colonised" as a Z / habitat effect (equilibrium confound). A synthetic truth whose habitat is
KNOWN to be static lets us measure that directly; the truth must not come from our own model structure.

## Truths (rangr 1.0.10, R 4.5.1, project-local library; research/r/b6_rangr_truth.R)
Grid: the GRID54 54 km grid as a planar raster (67 x 112, 54 km cells, km units), ocean NA, years 1902-2025.
Dynamics (rangr's, not ours): Ricker growth with a strong Allee term, x' = x exp(r (1 - x/K)(x/K - A/K)), Poisson
demographic noise, individual dispersal from an exponential kernel, settlement "K2N" (proportional to K / N of the
target), border "reprising".
Parameters, chosen for a House-Finch-like invasion before seeing any result (documented, not tuned to BBS):
- r = log 2 (.69; the age model's plausible lambda0 range is 1.5-2);
- A = 2 individuals (strong Allee: lone or paired arrivals rarely establish);
- kernel: exponential, mean 30 km (rate 1/30), max_dist 540 km (10 cells); with 54 km cells ~40% of individuals
  leave their cell each year;
- K = round(10 x S): S = the B1 SDM's expected route count at u = 0 (B1-static for the STATIC truth; B1-dyn year by
  year for the TIME-VARYING truth); 10 birds per unit of expected route count;
- start 1902: native seed cells (initpop_shape > 0, 1,110 cells) at N = K; 1940: 50 founders added to the 4 release
  cells (13/13/12/12) -- implemented as two rangr runs (1902-39, then 1940-2025 started from the 1939 state plus
  founders; rangr has no mid-run introduction);
- one realisation per truth (seed 1), a second (seed 2) for the static truth to gauge realisation noise.
STATIC truth: habitat constant in time; any front lag is dispersal + Allee. TIME-VARYING truth: K follows B1-dyn
(which itself absorbed the real arrival confound), i.e. a world where habitat DOES gate the front.
Observations: at every row of the GRID54 design (all real BBS route-years 1966-2025 and the 1902-39 pseudo-zero
site-years, same cells, years, observer-route ids and held-out masks): y ~ Binomial(N(cell, year), p), with p set so
the 2010-20 mean synthetic count over real rows equals the real BBS mean over the same rows (an observation scale, not
dynamics). No observer effects in the truth (stated limitation).

## What is fitted to the synthetic counts (same code, same masks; only y replaced)
B1-static, B1-dyn (sdm_b1_sdm.py), B2-static, B2-dyn (sdm_b2_migsdm.py), B2-Allee (sdm_b2_allee.py), B0 (sdm_b0_naive.py).
The AGE MODEL needs a GPU fit with HOUFIN_SYNTHETIC_OBS (research/age_model/exp/simulate_observations.py convention);
the synthetic files are written in that format but the fit is NOT launched here (CPU-only instruction; a GPU slot is
the coordinator's call).

## Readouts
- CONFOUND TEST (primary): B1-dyn's EARLY-EAST RATIO (expected count east of 80W, 1966-75 / 2010-20) fitted on the
  STATIC truth. The truth's habitat ratio is exactly 1 there. A ratio well below 1 (< .5) means a correlative Z
  readout invents early-East habitat deficit from arrival timing alone. ESTABLISHED by construction if it happens.
- FRONT recovery: each model's first-occupied year (10% of the 2010-25 median, the M-F code) minus the TRUTH's
  first year with N >= 10% of its own 2010-25 median, same eastern cell set (cells surveyed >= 3 yr before their
  first SYNTHETIC positive); median and cohorts. Also the truth's own lag vs first synthetic positive (detection lag).
- Held-out: mean NB lpd on the same temporal / S-real masks (synthetic y), paired across models.
- On the TIME-VARYING truth: B2-static should fail on the front and B2-dyn succeed; on the STATIC truth: B2-static /
  B2-Allee should succeed and B2-dyn should still fit (it can absorb timing into Z) -- the point is whether held-out
  scores or the front can tell the truths apart.

## Predictions (mine)
- P1: on the STATIC truth, B1-dyn's early-East ratio < .5 (the confound appears without any habitat change). LIKELY.
- P2: B2-static / B2-Allee recover the STATIC truth's front within +-5 yr median. LIKELY (it is their model class plus
  Allee) -- if not, B2's failure on real data is less informative than it looked.
- P3: held-out S-real does not clearly separate B2-static from B2-dyn on the STATIC truth (both fit counts). UNKNOWN.

## Rules (implications reported, not decided)
- If P1 holds: the early-East lambda / suitability deficit in any Z-readout model fitted to real data cannot by itself be
  read as habitat; the age model must be shown to recover ratio ~1 on this STATIC truth before its real-data early-East
  sink is interpretable (that GPU fit is the decisive follow-up).
- If B2-static recovers the static truth's front but failed on real data (SDM_BASELINE_B2/B2-Allee): the real front is
  not a pure dispersal + Allee front on static suitability (in this model class).
- If B2-static also fails on the static truth: the B2 class is too crude to read front timing; drop B2's front verdicts.

## RESULTS (see below)


## REVISION 1 (2026-10-31, after the first STATIC truth run; before any model was fitted to it)
The pre-registered STATIC truth is degenerate for the question: with constant K the native population spreads east on
its own and the whole continent is at capacity by 1939 (total N 59,035 in 1902 -> 93,168 in 1939 -> ~94,000 from 1940
on), before the release matters. There is no eastern invasion front to recover (the same thing B2-static and the age
model's frozen-habitat diagnostic do: the West crosses the Plains early). Added truths (same parameters):
- STATIC-REL: static K, NO native seed, release only (50 founders in 1940) -- an eastern invasion on fixed habitat.
- DYN (as pre-registered, native + release) and DYN-REL (release only).
Models are fitted to STATIC-REL and DYN-REL first (eastern-only species; western rows are true zeros); the degenerate
STATIC run is kept (N_static_s1, y_static_s1) but not used. The pre-registered readouts and rules apply unchanged.

## REVISION 2 (2026-10-31, after the release-only runs; still before any model fit)
With the pre-registered strong Allee (A = 2 individuals) the 50-founder release goes extinct in both release-only
truths (STATIC-REL: total N 50 -> 35 (1966) -> 14 (1990) -> 29 (2025), a few birds lingering; DYN-REL: extinct). The
reason is rangr's Ricker-Allee form, x' = x exp(r (1 - x/K)(x/K - A/K)): at low density the per-capita growth rate is
~ r (x - A) / K, which is near zero wherever K is large, while ~40% of birds leave the cell each year -- founders
thin out below A. Revision: release-only truths use A = NA (no Allee; Ricker growth with Poisson demographic noise,
so low-density stochastic loss remains). All other parameters unchanged. The DYN (native + release, A = 2) truth ran
and is kept: total N 42,873 (1902) -> 44,737 (1939) -> 72,561 (1966) -> 82,400 (2025); synthetic p .447.

## RESULTS (2026-10-31; E:/Datasets/houfin/grid54/baselines/b0/b6/; R research/r/b6_rangr_truth.R; Python research/exp/sdm_b6_prep.py, sdm_b6_make_design.py; fits with the unchanged B1 / B2 scripts on design_<truth>/)
rangr runtime: ~25-60 s per truth (R, Idle priority, cores 0-3).

### The truths actually usable
| truth | seeding / Allee | outcome | usable? |
|---|---|---|---|
| STATIC (pre-registered) | native + release, A = 2 | continent at K by 1939 from the native seed (N 59k -> 93k) | no (no eastern front) |
| STATIC-REL | release only, A = 2 | release fails (50 -> 35 -> 14 birds) | no |
| DYN-REL | release only, A = 2 / A = NA | extinct (B1-dyn's K near the release site in 1940 is ~0) | no |
| DYN (pre-registered) | native + release, A = 2 | release extinct; the native population reaches only 10 eastern front cells by 2009 | degenerate for the front |
| STATIC-REL-noAllee | release only, A = NA, static K | 50 founders (1940) -> 9.5k (1966) -> 11.8k (1990) -> 70k (2025); east of 80W occupied by ~1950-66, central-east 2% (1950) -> 40% (1966) -> 71% (1980), the WEST empty until ~2000 then 40% by 2020 | YES (primary) |
STATIC-REL-noAllee synthetic counts: binomial p .457; 2010-20 mean 4.14 (real 4.13); zero share 51% (real 63%); 218
eastern front cells (surveyed >= 3 yr before first synthetic positive). The truth's own occupancy leads first
synthetic detection by a median 1 yr (cohorts 0 / -2 / -2 / -8.5): the reference any model's front should match.

### CONFOUND TEST (primary) on the STATIC truth (true habitat ratio = 1 everywhere, by construction)
| band | truth N ratio 1966-75 / 2010-20 | B1-dyn (Z-only SDM fitted to the synthetic counts) expected-count ratio | B1-static |
|---|---:|---:|---:|
| east of 80W (occupied before 1966) | 1.00 | 1.04 | 1.00 |
| central-east 95-80W (arrival 1950-80) | .84 | .71 | 1.00 |
| west of 105W (arrival ~2000-2020) | .00 | .26 | 1.00 |
| east of 80W, 1940-55 (extrapolation) | -- | .66 | 1.00 |
- P1 HELD where arrival happened during the survey era: in the west, with habitat constant by construction, the Z-only
  SDM assigns 1966-75 expected counts at .26 of 2010-20 -- a ~4x "early deficit" invented from arrival timing alone.
  In the central-east it exaggerates the true change (.71 vs .84); where the truth was already established before 1966
  (east of 80W) it correctly finds no change (1.04). ESTABLISHED (synthetic, one realisation): Z carries enough temporal
  structure that a correlative readout turns not-yet-arrived into a Z effect on a truth whose habitat never changed --
  the equilibrium confound reproduced outside our model structure. It is partial (.26, not ~0): Z cannot fully encode
  arrival.
- Rule implication (pre-registered): a Z-readout model's early-vs-modern contrast in a region invaded during the survey
  era cannot by itself be read as habitat. The decisive follow-up is the AGE MODEL fitted to this static truth: does it
  recover a flat habitat (fundamental lambda ratio ~1 in the west and central-east) or invent an early sink? Synthetic
  observation file ready in the HOUFIN_SYNTHETIC_OBS format: b6/age_synthetic_obs_static_rel_noallee_s1.npz (y, mu in
  load_data order). Caveats for that fit: this truth has no native western population (the age model's native seed
  would be wrong -- set initpop to zero or accept a seeding mismatch) and no observer effects. NOT launched (GPU).

### Held-out on synthetic counts (STATIC-REL-noAllee; observer effect integrated with the pinned sigma .97, which the
truth does not have -- marginal scores are therefore penalised for every model alike)
B1-dyn S-real -1.880, T marginal -2.286; B1-static S-real -1.828, T marginal -2.421; B2-static (native-seeded, see
below) S-real -1.825; B2-dyn -1.848. On the static truth B1-static beats B1-dyn on S-real by .05 (the true habitat is
static) but loses on T marginal by .14 (B1-dyn's Z change carries the 2021-25 western expansion as "habitat").

### FRONT recovery on STATIC-REL-noAllee (10% density front minus first synthetic positive; truth's own: -1, cohorts 0 / -2 / -2 / -8.5)
| model (fitted to the synthetic counts) | median | cohorts 66-74 / 75-84 / 85-94 / 95-09 | occupancy p >= .5 median | S-real | T marginal |
|---|---:|---|---:|---:|---:|
| TRUTH | -1 | 0 / -2 / -2 / -8.5 | -- | -- | -- |
| B2-static, native + release seeding (MISMATCHED: the truth has no native population) | -62 | -57 / -64 / -68.5 / -75 | -53 | -1.825 | -- |
| B2-dyn, native + release (mismatched) | -47 | -47 / -44 / -50 / -52 | -29 | -1.848 | -- |
| B2-static, release-only seeding (matched; added --seed-mode release) | -3 | -1 / -4 / -9 / -19.5 | 0 | -1.779 | -3.152 |
| B2-dyn, release-only seeding | -6 | -5 / -7 / -12 / -22.5 | -3 | -1.815 | -2.515 |
| B1-static / B1-dyn (no dispersal) | -75 | -70 / -76 / -86 / -98 | -- | -1.828 / -1.880 | -2.421 / -2.286 |
(The seeding option was added after the mismatched fits; the mismatch is a property of my test set-up, not of B2.)
- P2 HELD for the median: with matching seeds, B2-static recovers the static truth's front within 2 yr (-3 vs -1).
  But its late cohorts are still early relative to the truth (1985-94: -9 vs -2; 1995-2009: -19.5 vs -8.5): B2's
  continuous expected-occupancy spread runs ahead in the thinly surveyed late-invaded cells. So part (~10 yr) of the
  "late cohorts too early" signature on real data is a property of the B2 class / readout itself (ESTABLISHED on this
  truth), not of the House Finch.
- P3: S-real DOES separate the truths' structure here: the static-suitability B2 beats the Z-gated B2 by .036 on the
  static truth (and B1-static beats B1-dyn by .05); T marginal points the WRONG way (B2-dyn better by .64, B1-dyn better
  by .14), because Z's temporal change can carry the 2021-25 western expansion. ESTABLISHED (one realisation): on a
  static-habitat truth, T marginal rewards the confounded readout; S-real does not.
- Rule reading: B2-static recovers the static truth's median front but failed badly on real data (-33, cohorts down to
  -44, well beyond the ~10 yr class bias). With the caveat that the real-data B2 used native + release seeding and the
  native population's eastward spread dominated it (SDM_BASELINE_KISSMIG_B2ALLEE caveat), the reading is: the real front
  is not reproduced by dispersal-limited spread on static suitability FROM THE NATIVE + RELEASE SEEDS in this class.
  Whether a release-only eastern front on static suitability would fit the real data is untested (next check, cheap now
  that --seed-mode exists).

### Not done
- The age model fitted to the static truth (the decisive test of whether it invents an early sink): needs a GPU slot;
  synthetic file ready (see above). Recommended next.
- Second realisation (seed 2) of the usable truth; the DYN truth needs a design where the release survives (its K at the
  release site in 1940 is ~0 because it inherits B1-dyn's confounded early-East surface) -- as a habitat-gated truth it
  should use a time-varying K that is NOT derived from a fit to House Finch counts.
- B4 (stripped dynamic range model): not started in this session.

## SLOW-FRONT truth for GRID54 step 28b (pre-registered 2026-10-31, before any candidate was simulated)
Why: step 28 (age model on STATIC-REL-noAllee) kept lambda flat, but that truth fills the East by 1966, so the
survey-era arrival was only in the West. A sharper test needs a static-habitat truth whose EASTERN front arrives during
the BBS era, like the real one.
Fixed: static K (B1-static x 10), release only (50 founders, 1940), Ricker, Poisson noise, K2N settlement, exponential
kernel, max_dist 540 km, no Allee (rangr's Allee form extinguished the release earlier).
Calibration on the TRUTH ONLY (no model is fitted during selection): grid kernel mean m in {10, 15, 20} km x
lambda = e^r in {1.4, 1.6, 2.0}, seed 1. Targets: the release survives; median first-arrival year (N >= 1) of land
cells east of 80W (25-49N) in 1955-1975 and of central-east cells (95-80W) in 1975-1995; >= 80% of central-east cells
reached by 2010. Selected = the surviving candidate minimising |median east - 1965| + |median central-east - 1985|
among those meeting the targets (if none meets them, the closest, reported as such). The selected truth is then
observed binomially on the real BBS design (p matched to the real 2010-20 mean) and frozen; only then are B1-dyn and the
age model (step 28b) fitted.

### SLOW-FRONT truth selection (run 2026-10-31; slow_candidates.json; first-arrival = first year N >= 1)
| m (km) | lambda | east80 median first (p10-p90) | central-east median first (p10-p90) | central-east reached by 2010 | total N 2025 | score |
|---:|---:|---|---|---:|---:|---:|
| 10 | 1.4 | 1970 (1948-1997) | 2005 (1987-2023) | .12 | 7,248 | 25 |
| 10 | 1.6 | 1964 (1946-1993) | 2004 (1979-2020) | .24 | 9,690 | 20 |
| 10 | 2.0 | 1960 (1945-1981) | 1999 (1976-2019) | .38 | 10,773 | 19 |
| 15 | 1.4 | 1959 (1945-1981) | 1999 (1974-2019) | .41 | 10,916 | 20 |
| 15 | 1.6 | 1957 (1944-1975) | 1994 (1972-2016) | .55 | 11,263 | 17 |
| 15 | 2.0 | 1954 (1944-1967) | 1987 (1966-2009) | .70 | 11,598 | 13 |
| 20 | 1.4 | 1955 (1944-1972) | 1994.5 (1969-2014) | .63 | 11,180 | 19.5 |
| **20** | **1.6** | **1954 (1944-1969)** | **1986 (1965-2007)** | **.74** | **11,573** | **12** |
| 20 | 2.0 | 1951 (1943-1963) | 1977 (1961-1995) | .80 | 11,977 | 22 |
| (step 28 truth) 30 | 2.0 | 1948 (1942-1955) | 1967 (1954-1981) | .84 | 70,263 | 35 |
No candidate met every target (the east-of-80W median sits at 1951-1970 but ~40% of east80 cells have K = 0 and are
never reached, so "reached by 2010" tops out ~.58 there; central-east reach <= .80). By the pre-registered fallback
the closest is SELECTED: m = 20 km, lambda = 1.6 (r = .47), tag static_rel_noallee_k20_l1.6_s1 -- east-of-80W arrivals
centred on 1954 (10-90%: 1944-1969), central-east on 1986 (1965-2007): the central/eastern survey-era front the step
28b test needs. Frozen now; observed binomially on the real BBS design in the same R run (p matched to the real 2010-20
mean). Note the slow truths plateau near 11-12k birds by 1990-2025 (the West is never reached by 2025 from the release),
so their western rows are true zeros.

### Step 28b / 28c (the age model on the SLOW-FRONT truth; full record in GRID54_LADDER_2026-10-29.md)
- B1-dyn on the slow truth (Z-only SDM): central-east expected-count ratio 1966-75 / 2010-20 = .33 (truth N ratio .24,
  truth habitat ratio 1), east of 80W .84 (truth N .98), east of 80W 1940-55 extrapolated .47. B1-static: 1.0 by
  construction. ESTABLISHED (one realisation): the correlative Z readout invents a ~3x early central-East deficit on a
  static-habitat truth whose central-East front arrives 1965-2007.
- Step 28b (age model, pueue 956): MAP valid (loss 117,070); the float64 refine diverged (|grad| 3e46) because the NB
  concentration (1.7 -> 12,037) and its density power kappa (.22 -> 31) ran off: the synthetic counts are binomial
  thinning of Poisson populations with no observer heterogeneity -- under-dispersed relative to the model's NB -- and
  the density-dependent concentration lets the likelihood chase that without bound (mu^31 overflows). Observation-model
  mismatch specific to the synthetic counts.
- MAP-ONLY provisional answer (band_lambda): all land .918 / .895 / .813 (1902-15 / 1940-55 / 2010-25); central-east
  1.322 / 1.302 / 1.292; east of 80W 1.314 / 1.333 / 1.149; west .783 / .744 / .642. No early-East sink, no inflation
  (all changes negative) -> RESISTS at the MAP point, while B1-dyn invents the deficit on the same counts.
- Step 28c (pueue 957, running): 28b with population_model.observation.conc_density_power = false, full refit.
