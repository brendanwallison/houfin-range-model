# SDM baseline B1 -- correlative count SDM on the age model's own Z (pre-registered 2026-10-30, before fitting)

Parent: research/experiments/SDM_BASELINE_PLAN_2026-10-30.md sec. 4 B1. Uses the B0 design table and scorer unchanged
(research/experiments/SDM_BASELINE_B0_2026-10-30.md; /mnt/e/Datasets/houfin/grid54/baselines/b0/design_step3), so
every number is on the same rows, masks and likelihood as GRID54 step 3. Code: research/exp/sdm_b1_sdm.py.
Run: WSL venv, CPU, 2 threads, taskset 0-3, nice 19.

Question (plan): is the age model's held-out skill just correlative niche on the same habitat field? Where does an
equilibrium SDM fail? Does an SDM fitted to invasion-era BBS data, through Z alone, also learn "the early east is
poor habitat" (the age model's early-east fundamental sink, GRID54 steps 1-4)?

## Model (all arms)
- Counts: NB2, log link, one concentration phi (fitted). Linear predictor = intercept + ridge-penalised linear and
  squared terms of all 64 Z dimensions the age model reads (Z_gathered; standardised on training rows) + an
  observer x route random intercept u ~ N(0, sigma_u^2) with sigma_u PINNED at .97 (step 3's value, so the
  observer treatment is identical). Joint penalised maximum likelihood (L-BFGS); obs_weight as in the age model.
- Scoring: the D2 scorer exactly as for step 3: plug-in (fitted u for seen observer-routes, 0 otherwise), MARGINAL
  (u integrated, Gauss-Hermite 24, for every real held-out row), CONDITIONAL (integrated only for unseen
  observer-routes). Compared with the age model's marginal (and conditional for T).
- Training rows: real BBS rows outside both held-out sets (1966-2020, outside the 31 pinned blocks).
- Ridge strength lambda in {.1, 1, 10, 100, 1000}: chosen by an inner split of TRAINING 216 km blocks (20% of training
  blocks withheld, seed 0), scored by the marginal NB log density there; then refitted on all training rows.

## Arms
- B1-dyn: year-matched Z (habitat change enters through Z; no dynamics). PRIMARY.
- B1-static: each cell's mean Z over 1966-2025 (a classic SDM on a static habitat map).
- B1-native: B1-dyn trained only on WEST training rows (lon < -105), projected everywhere (classic invasive-species
  SDM: fit the native range, project the invaded one).
- B1-dyn+yr: B1-dyn plus a piecewise-linear year trend per region (west / plains / midwest / east; knots 1980, 1995,
  2010; linear continuation into 2021-25). Statistical trend, NOT habitat (M6).
- B1-dyn+pz: B1-dyn also trained on the training pseudo-zeros (1902-39), as the age model is.

## Readouts (same strata as B0)
T plug-in / marginal / conditional; S-all, S-real marginal; region; region x era for S-real; log1p RMSE / corr;
paired differences vs step 3 with the 216 km block bootstrap (90%); observed vs predicted P(y = 0) by region on
S-real and T (M-E). Two structural readouts:
- EARLY-EAST RATIO: B1-dyn expected count at u = 0, mean over land cells east of 80W, 25-49N, for 1966-75 divided by
  2010-20 (Z change only, coefficients fixed). The age model's analogue is its early-vs-modern fundamental lambda in
  the same band (step 3: east of 80W .283 -> 1.373).
- FRONT (M-F, front_speed.py definitions re-implemented on the design table): eastern cells east of 95W, 25-49N,
  surveyed >= 3 years before their first BBS positive; model first year with expected count >= 1% (10%) of the
  2010-25 median over those cells, minus the first BBS positive; median and by observed-arrival cohort. Validated by
  reproducing step 3's front_speed.json "fitted" numbers (-8 / -4) from its exported density before reading B1.

## Predictions (mine, before fitting)
- P1 T: B1-dyn marginal within .03 of the age model's marginal (both near cell persistence, B0). LIKELY.
- P2 S-real: B1-dyn falls between S-IDW (-2.050) and the age model (-1.918); the age model's lead over B1-dyn is
  largest in the east and in the expansion eras (1966-93). LIKELY.
- P3 B1-native is clearly worst in the east and midwest (niche transfer fails where the species is not at
  equilibrium). LIKELY.
- P4 EARLY-EAST RATIO < .7 for B1-dyn: the SDM, fitted to 1966-2020 counts, assigns Z dims that track the invasion
  so that the early east looks poorer through Z alone. UNKNOWN (a real test).
- P5 FRONT: B1-dyn's thresholded expected count "arrives" in the east within +-10 yr of the BBS median only if P4 holds
  strongly; otherwise it puts the east as occupied from the start of the series (cells are suitable). LIKELY the
  latter for B1-static by construction.

## Rules (implications reported, not decided)
- R1 If B1-dyn's S-real marginal >= the age model's - .01: the spatial held-out skill is correlative niche on Z; the
  process structure adds none on S, so process claims must rest on front timing, source/sink and synthetic recovery.
- R2 If the age model beats B1-dyn by > .05 on S-real in the east (interval excluding 0): the process structure earns
  held-out skill at the invasion region.
- R3 If the EARLY-EAST RATIO < .5: Z itself carries an invasion-shaped temporal signal that a purely correlative fit
  converts into "early east poor"; the age model's early-east sink is then reachable through the habitat readout alone
  (consistent with G3's worry that a front can be fitted as "habitat improved"). If ~1 (>= .8): Z's change in the east
  is mild and the age model's early-east sink is created by the coefficients the dynamics demand, not by Z's own
  temporal structure.
- R4 If B1-dyn+yr beats B1-dyn by > .03 on T marginal, the temporal skill available is regional trend (M6: not
  necessarily dynamics), and T marginal comparisons should be read against B1-dyn+yr.

## RESULTS (2026-10-30; /mnt/e/Datasets/houfin/grid54/baselines/b0/design_step3/b1_results.json, log ../b1.log)

Front code check: the re-implemented M-F reproduces step 3's front_speed.json exactly (782 cells; 1% median -8, 10%
median -4; cohorts 0 / -4 / -7 / -13). All final fits converged (L-BFGS relative-reduction stop, 1,000-1,900 it).

### Scores (mean NB log density per observation; age = GRID54 step 3 refined; paired = age minus arm, 90% 216 km block bootstrap)
| model | lambda / phi | T marg | T cond | S-all marg | S-real marg | paired T marg | paired S-real | paired S-real east | paired S-real east 1966-93 |
|---|---|---:|---:|---:|---:|---|---|---|---|
| age step 3 | -- | -1.844 | -1.731 | -1.219 | -1.918 | -- | -- | -- | -- |
| B1-dyn (primary) | 1000* / 1.77 | -1.936 | -1.808 | -1.330 | -2.083 | +.092 [.073, .114] | +.165 [.092, .217] | +.233 [.071, .298] | +.390 [.143, .485] |
| B1-static | 100 / 1.45 | -1.972 | -1.834 | -1.552 | -2.202 | +.128 [.106, .150] | +.283 [.122, .399] | +.475 [.112, .640] | +.770 [.207, .978] |
| B1-native (west-trained) | 1000* / 2.86 | -2.197 | -2.725 | -1.792 | -2.597 | +.353 [.292, .425] | +.679 [.290, 1.07] | +1.20 [.28, 1.94] | +1.13 [.38, 1.60] |
| B1-dyn+yr (year x region trend) | 0.1* / 2.34 | -1.925 | -1.808 | -1.297 | -1.976 | +.081 [.060, .105] | +.058 [.027, .093] | +.033 [.002, .061] | +.066 [.029, .086] |
| B1-dyn+pz (+ pseudo-zeros) | 100 / 1.78 | -1.949 | -1.817 | -1.315 | -2.067 | +.105 [.083, .128] | +.149 [.096, .188] | +.188 [.071, .226] | +.324 [.108, .396] |
| (B0) cell persistence / S-IDW | -- | -1.836 | -- | -1.303 | -2.050 | | | | |
*edge of the pre-registered lambda grid (B1-dyn and B1-native preferred the strongest ridge, B1-dyn+yr the weakest);
the inner-validation gains at the edge were .01 (dyn) and .001 (yr), far smaller than the gaps to the age model.

S-real by region (marginal): age -2.384 / -.695 / -1.494 / -2.306 (west / plains / midwest / east); B1-dyn -2.447 /
-.778 / -1.747 / -2.539; B1-dyn+yr -2.433 / -.784 / -1.592 / -2.339. Paired age minus B1-dyn: west +.063 (-.005,
.151), plains +.084 (.017, .164), midwest +.252 (.162, .357), east +.233 (.071, .298); east 1994-2020 +.109.

### Structural readouts
| model | expected count east of 80W, 1940-55 / 1966-75 / 2010-20 (u = 0) | EARLY-EAST RATIO 1966-75 / 2010-20 | central-east ratio | front 1% median (cohorts) | front 10% median (cohorts) | never reach 10% |
|---|---|---:|---:|---|---|---:|
| age step 3 (fundamental lambda east of 80W: .283 -> 1.373) | -- | -- | -- | -8 (0, -4, -7, -13) | -4 (+2, -1, -3, -6.5) | 4% |
| B1-dyn | .029 / .166 / 2.086 | .080 | .049 | -67 (-68, -72, -36, -50) | -16 (-10, -13, -16, -19) | 6% |
| B1-static | 1.21 constant | 1.00 | 1.00 | -90 | -89 | 8% |
| B1-native | .78 / .85 / .93 | .92 | .83 | -90 | -90 | 0% |
| B1-dyn+yr (ratio includes the year term) | .000 / .015 / 1.73 | .009 | .001 | -11 (-5, -9, -10, -16) | -6 (0, -4, -5, -10) | 4% |
| B1-dyn+pz | .006 / .130 / 2.26 | .057 | .025 | -24 | -13 | 8% |
Zero fraction (M-E; observed vs predicted P(0), S-real): age west .52/.55, plains .86/.89, midwest .62/.65, east
.42/.43; B1-dyn .52/.47, .86/.82, .62/.61, .42/.40 (the SDM under-predicts zeros slightly, the age model over-predicts
them slightly; neither is far off). On T the age model over-predicts midwest zeros (.48 observed vs .56).

### Predictions vs outcome
- P1 FAILED: B1-dyn is .092 worse than the age model on T marginal (interval .073-.114), not within .03. On a 5-year
  forecast a correlative SDM on the same Z is also .10 worse than cell persistence.
- P2 PARTLY: the age model's lead is largest in the east during the expansion (+.39 in 1966-93) and in the midwest,
  as predicted; but B1-dyn is NOT between IDW and the age model -- it is slightly worse than kernel interpolation on
  S-real (-2.083 vs -2.050).
- P3 HELD: the west-trained SDM collapses east (+1.2 for the age model), the classic niche-transfer failure.
- P4 HELD STRONGLY: early-east ratio .080 (1940-55: .014).
- P5 PARTLY: B1-static is "occupied" from the start (-89/-90), by construction. B1-dyn, despite P4, arrives 16 years
  early at the 10% threshold (all cohorts -10 to -19) and decades early at 1%.

### Rules applied (implications reported, not decided)
- R1 NOT triggered: the age model's spatial held-out skill is NOT just correlative niche on Z; it beats the best
  pure-Z SDM by +.165 on S-real (and +.15 with pseudo-zeros in training).
- R2 TRIGGERED: the process structure earns held-out skill at the invasion region: east +.233 (interval .07-.30),
  east 1966-93 +.39, midwest +.25. ESTABLISHED for step 3 vs these SDMs.
- R3 TRIGGERED (see the user's interpretation correction below, which governs this reading): Z carries an
  invasion-shaped temporal contrast. Fitted to 1966-2020 BBS counts with nothing but Z, a correlative model's expected
  counts east of 80W are ~12x lower in 1966-75 than in 2010-20 (and ~70x lower in 1940-55 -- an extrapolation: no BBS
  before 1966, pre-1966 Z is a backcast). This is the equilibrium confound: counts were low because finches had not
  arrived, and the SDM attributes that to Z. It is NOT evidence about location quality. It shows the confound is
  available to any readout tying abundance to Z, including the age model's (G3's worry: a front fitted as "habitat
  improved"). What it does NOT say: whether that Z change is
  genuine habitat change for the finch (suburbanisation, feeders, the co-occurring community that tracks them -- M6,
  premise: habitat change must stay learnable) or a community signal that merely co-varies with the invasion. Mechanism
  of the signal: UNKNOWN. Reference-community note: production Z excludes House Finch itself, so the signal is not the
  finch's own counts leaking in (established by construction), but species that spread or increased alongside it can
  carry the same timing (likely; untested).
- R4 NOT triggered on T (dyn+yr gains .011 on T marginal), but the year x region trend gains .107 on S-real over
  B1-dyn and comes within .058 of the age model. A smooth regional time trend explains most of what Z-only SDMs miss
  spatially; per M6 it is a statistical trend, not proof of dynamics. Its front (-6 at 10%) is close to the age
  model's (-4) because the trend term encodes the regional timing directly.

### What this means for the ladder (reported)
- The age model clears both naive (B0) and correlative (B1) baselines on S-real, with its margin concentrated where
  dynamics matter (east and midwest, expansion eras). T marginal remains a non-discriminating score (B0, R1).
- The lambda-inflation question gains a concrete lead: a correlative fit can turn "not yet colonised" into a ~12x
  early-vs-modern contrast through Z alone (equilibrium confound). The age model's refined fits put the east of 80W at
  fundamental lambda .28 early vs 1.37 modern -- the same direction, LIKELY via the same confound. A useful next diagnostic (not run): which Z dimensions carry the east's 1940-2020
  change, and whether they are shared by sedentary common proxy species that did NOT invade (if yes: habitat; if they
  move only with invaders/suburban specialists: possibly a co-invasion signal). Pre-register before running.
- B2 (dispersal-limited SDM) is now well posed: B1-static gives a suitability field with no temporal gating; B1-dyn
  gives one with strong Z-driven gating. Running the colonisation automaton on both asks directly whether the front
  needs the Z gating at all.

## Interpretation correction (2026-10-30, user)
R3's "the early-east sink is reachable through the habitat readout alone" must NOT be read as evidence about location
quality. B1-dyn is fitted to 1966-2025 BBS counts that are low in the early East because finches had not ARRIVED; a
correlative model has no arrival term, so it attributes that dispersal limitation to whatever in Z separates early-East
from modern-East cells (the equilibrium confound, Zurell et al. 2016). The 1940-55 ratio (.014) is extrapolation: no BBS
rows exist before 1966, and pre-1966 Z is DESK's covariate/winter-driven backcast. Established: Z carries enough
eastern temporal contrast that any model tying abundance to Z can encode "not yet colonised" as "poor habitat".
Likely: the age model's early-East sink / lambda inflation exploits the same confound. Unknown: whether Z's eastern
change reflects real habitat change -- tested by the Z-attribution diagnostic (non-invading sedentary common species)
and indirectly by whether dispersal levers (GRID54 step 20) remove the fit's need for it.
