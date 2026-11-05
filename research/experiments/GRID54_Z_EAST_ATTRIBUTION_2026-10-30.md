# Which Z dimensions carry the east's 1940-2020 change, and is that change shared by the community? (pre-registered 2026-10-30, before any number below exists)

Origin: SDM baseline B1 (research/experiments/SDM_BASELINE_B1_2026-10-30.md, rule R3): a Z-only correlative count SDM
(B1-dyn: NB2 on 64 Z dims + squares, ridge lambda 1000, observer x route effect) puts expected House Finch counts east
of 80W at .080 of the 2010-20 level in 1966-75 and .014 in 1940-55 -- Z carries an invasion-shaped temporal signal.
The age model's refined fits show the same direction (step 3 fundamental lambda east of 80W .28 -> 1.37).
FRAMING NOTE (added after the user's correction, 2026-10-30; the pre-registered text below is unchanged): that
contrast is the equilibrium confound (not-yet-arrived attributed to Z; 1940-55 is extrapolation), not evidence about
early-East location quality. Wording below such as "make the early east poor" describes what the readout does, not
the East; read the results section's framing.
Question (coordinator 2026-10-30): which Z dims carry it, and is their change shared by the bird community in general
-- in particular sedentary common species that did NOT invade the east -- or does it move only with species that
spread with the finch?

Records read first: DIGEST (E039: DESK ~ good nonlinear model of covariates; temporal skill ~ regional-trend
placebo; M6 regional trend can be habitat); CONSIDERATIONS G3 (a front can be fitted as "habitat improved"), G4
(too little habitat change is also a bias), M1 (scaling vs direction), M3 (observer turnover inflates shared
change), M4 (centred vs uncentred), M6; ASSIM_FIELD_2026-10-27 (covariate-only Z lands near the same modest centred
change correlation; the observed teacher reaches .76); E085/E086 (readout-block attribution on the DESK side, linear
heads add no information); the age model's own z_attribution (step 3 viz_ref metrics.json, figure 05d): all-land mean
|delta fundamental lambda| top dims Z_19 (.106), Z_17 (.066), Z_18 (.044) for 1902-15 -> 2010-25 and Z_19, Z_17, Z_7
for 1940-55 -> 2010-25; ~53% of the total is NOT attributable to single dims (interaction residual). Memory: premise
(habitat change must stay learnable; static habitat diagnostic only), sedentary primary, reference community.

## Data
- Z: the GRID54 step-3 input (design_step3/z_cells.npy: 124 years x 4,318 54 km land cells x 64 dims).
- House Finch: the B0/B1 design table (real BBS rows).
- Community: the DESK research cache /work/07980/bwa386/ls6/houfin/research_cache/desk_tempho_1995_s1 (27 km
  cell-years 1966-2025, X_comm 96 + X_dev 273 = 369 non-sealed species, route-mean counts; sealed species are NOT
  read; House Finch is not in it). 27 km cells map to their 54 km parent (row // 2, col // 2); parents that are not
  54 km land are dropped (count reported). A 54 km cell-year value = mean over its 27 km children's cell-year values.
- Residency: species_similarity.csv (E059; AVONET migration 1 sedentary / 2 partial / 3 migratory). Prevalence =
  presence share over all cache cell-years 1966-2020. SEDENTARY COMMON (migration 1, prevalence >= .08) is the
  primary stratum; sedentary, residents (1-2), migrants reported.
- Eastern colonisers (data-defined, before looking at Z): species whose presence share in east-of-80W cell-years is
  < .05 in 1966-75 and >= .20 in 2010-20 ("spread into the east with or like the finch"). Non-colonisers = the rest.

## Analyses
A. ATTRIBUTION (House Finch, B1-dyn refit at lambda 1000, coefficients saved): the change in the band-mean log
   expected count between eras splits exactly into per-dim terms beta_d (z~_d,late - z~_d,early) + gamma_d (z~_d,late^2
   - z~_d,early^2) (standardised z). Bands: east of 80W, 25-49N; central-east 95-80W, 25-49N; all land. Eras
   1966-75 -> 2010-20 (primary) and 1940-55 -> 2010-20. Report each dim's share, the top dims, and the number of dims
   needed for 50% / 80% of the total.
B. COMPARISON with the age model's own attribution (all-land, its top-3): rank of Z_19, Z_17, Z_18, Z_7 in B1-dyn's
   all-land attribution and in the eastern bands.
C. SHARED CHANGE: for every species s (and the House Finch, same method), a ridge readout of log1p(count) on the
   same standardised Z (+ squares), fitted on 1966-2020 54 km cell-years (ridge chosen per species by GCV-free fixed
   lambda = 1000 in the same scaling as B1, stated now). Then, per band:
   - predicted change P_s = band-mean readout(2010-20 Z) - readout(1966-75 Z);
   - predicted change from the House Finch's top dims only (the dims carrying 80% of A's eastern total), P_s^HF;
   - observed change O_s = band mean log1p count 2010-20 - 1966-75, over cells surveyed in both eras (uncentred,
     regional; M4/M7).
   Statistics by stratum: Pearson corr(O, P) and corr(O, P^HF) across species with a species bootstrap (2,000) 90% CI;
   slope k of O on P (M1); where the House Finch's P sits in each stratum's distribution (z-score, rank).

## Predictions (mine)
- PA: the eastern change is concentrated: <= 5 dims carry 50% of B1-dyn's eastern total. LIKELY.
- PB: B1-dyn's top eastern dims overlap the age model's top-3 (at least one of Z_19 / Z_17 / Z_18 / Z_7 in B1-dyn's
  eastern top 5). UNKNOWN.
- PC: across sedentary common non-colonisers, corr(O, P) > 0 (Z's eastern change tracks real community change; DESK's
  temporal skill is regional-trend-like, E039i) but modest (.2-.5). LIKELY.
- PD: the House Finch's P is an outlier (top 5% of all species' P in the east of 80W). LIKELY (12x).

## Rules (implications reported, not decided)
- HABITAT-LIKE: corr(O, P^HF) >= .3 among sedentary common NON-colonisers with the 90% CI above 0. Then the dims that
  make the early east poor for the finch also predict the observed eastern change of resident, non-invading species:
  the Z change is shared community change, which M6 says can be genuine habitat change. The finch-specific question
  becomes SCALING (M1, G4): is the finch readout's 12x response plausible, given how much the same dims move other
  species (compare the finch's P with residents' P and k)?
- CO-INVASION-LIKE: corr(O, P^HF) < .1 among sedentary common non-colonisers (CI including 0) while corr >= .3 among
  colonisers. The dims track species that spread into the east, not general resident community change; a correlative
  (and possibly the age model's) readout then turns a co-spread signal into "habitat improved" -- G3's failure.
- Otherwise: MIXED / unresolved; report both and the scaling comparison.
- Caveats fixed now: observer turnover inflates shared uncentred change (M3; this measure does not remove it);
  1966-75 BBS coverage in the east is thinner; 1940-55 has no BBS so O exists only for 1966-75 -> 2010-20.

## RESULTS
(below the addendum)

## ADDENDUM before C2 (written after seeing C's first numbers, before running C2)
C's species readouts were fitted on ALL 54 km cell-years, including the band's own cells in both eras, so its
corr(O, P) measures how well a 128-feature Z readout can FIT each species' regional change in-sample (it came out
~.96-.99 for every stratum), not whether Z's eastern change transfers. That is circular for the habitat question.
C2 (added now): the same readouts fitted WITHOUT the band's cells (all years), then used to predict the band's change
(spatial transfer: does the relation between Z and each species learned elsewhere predict that species' change in
the east?). Same statistics, same strata, same rules applied to C2; C is reported as the in-sample ceiling.
Also: the pre-registered coloniser definition (east-of-80W presence < .05 in 1966-75 and >= .20 in 2010-20) matched
NO species in either band, so the CO-INVASION rule cannot be evaluated as written; it is reported as not evaluable
rather than replaced by a looser definition chosen after the fact.

## RESULTS (2026-10-30; /mnt/e/Datasets/houfin/grid54/baselines/b0/zattr/: z_east_attribution.json (C, in-sample), z_east_attribution_c2.json (C2, readouts fitted without the band), b1dyn_coef.npz; code research/exp/z_east_attribution.py)

Data: 114,172 cache cell-years; 2,167 dropped (27 km parent not 54 km land); 100,848 kept for 1966-2020. 250 (east of
80W) and 495 (central-east) 54 km cells surveyed in both 1966-75 and 2010-20. Sedentary common stratum: 20 species.

### A. Which dims carry B1-dyn's eastern change (exact split of the band-mean log expected count)
| band, eras | total d log | ratio of mean counts | dims for 50% / 80% | top dims (share; standardised dz) | age-model top dims' rank (Z_19, Z_17, Z_18, Z_7) |
|---|---:|---:|---|---|---|
| east of 80W, 1966-75 -> 2010-20 | 1.39 | .080 | 2 / 4 | Z_15 (.31; +1.04), Z_6 (.25; +.35), Z_18 (.17; -.81), Z_21 (.12; -1.0), Z_28 (.11), Z_20 (.10) | 10, 11, 3, 64 |
| east of 80W, 1940-55 -> 2010-20 | 2.82 | .014 | 3 / 6 | Z_15 (.22), Z_18 (.17), Z_6 (.14), Z_19 (.12), Z_28 (.09), Z_17 (.09) | 4, 6, 2, 64 |
| central-east 95-80W, 1966-75 -> 2010-20 | 2.61 | .049 | 4 / 8 | Z_15 (.26), Z_17 (.10), Z_6 (.10), Z_19 (.10), Z_45 (.08), Z_18 (.08) | 4, 2, 6, 18 |
| central-east, 1940-55 -> 2010-20 | 4.82 | .0065 | 4 / 8 | Z_15 (.23), Z_19 (.16), Z_17 (.11), Z_18 (.09), Z_45 (.08), Z_6 (.07) | 2, 3, 4, 14 |
| all land, 1940-55 -> 2010-20 | 1.63 | .55 | 3 / 6 | Z_19 (.19), Z_15 (.19), Z_17 (.16), Z_33 (.12), Z_18 (.12) | 1, 3, 5, 9 |
- PA HELD: the eastern change is concentrated -- 2 dims carry half of it east of 80W, 4 in the central-east band.
  Z_15 leads every eastern band (22-31%).
- PB HELD: B1-dyn and the age model agree on the dims. All-land since 1940-55, B1-dyn ranks the age model's top dims
  Z_19 1st, Z_17 3rd, Z_18 5th (age model: Z_19, Z_17, Z_18 top-3 since 1902; Z_19, Z_17, Z_7 since 1940-55). In the
  central-east they rank 2-4. Z_7 carries nothing in B1-dyn's east. Z_15, B1-dyn's leading eastern dim, is not in the
  age model's reported top-3 (that readout lists 3 dims, all-land, on |d lambda|, with ~53% interaction residual), so
  its role in the age model is UNKNOWN. Different quantities (B1: log expected count; age model: |d fundamental lambda|).
- House Finch dims carrying 80% east of 80W: Z_15, Z_6, Z_18, Z_21.

### C / C2. Is the Z change shared by the community?
Observed (O) vs Z-predicted (P) 1966-75 -> 2010-20 change in band-mean log1p counts, across species. C = readouts on
all cells (in-sample ceiling); C2 = readouts fitted WITHOUT the band (spatial transfer; the honest test).
| band | stratum (n) | C corr(O, P) | C2 corr(O, P) [90%] / k | C2 corr(O, P_HFdims) [90%] / k | House Finch P (C2): rank, z |
|---|---|---|---|---|---|
| east of 80W | sedentary common non-colonisers (20) | .98 | .89 [.75, .95] / 1.15 | .25 [-.19, .77] / .66 | 2nd of 20, z 1.9 |
| east of 80W | sedentary (108) | .98 | .88 [.78, .94] / 1.14 | .29 [-.17, .77] | 2nd, z 4.7 |
| east of 80W | migrants (179) | .95 | .78 [.68, .86] / .88 | .26 [.02, .49] | 1st, z 4.8 |
| east of 80W | all (369) | .96 | .81 [.74, .86] / 1.01 | .26 [.02, .49] | 2nd, z 4.9 |
| central-east | sedentary common non-colonisers (20) | .99 | .93 [.69, .98] / 1.05 | .66 [.18, .88] / 1.88 | 1st of 20, z 1.8 |
| central-east | sedentary (108) | .99 | .93 [.66, .97] / 1.04 | .54 [.07, .80] | 1st, z 4.0 |
| central-east | all (369) | .98 | .91 [.82, .94] / 1.07 | .42 [.20, .61] | 1st, z 5.7 |
House Finch (C2): east of 80W P .755 vs observed O .822; central-east P .851 vs O .884 (log1p scale).
Colonisers (pre-registered definition): NO species qualifies in either band -> the CO-INVASION rule is NOT EVALUABLE.

Species most moved by the House Finch dims (P_HFdims, C2), east of 80W: rock pigeon, house sparrow, white-breasted
nuthatch, black-capped chickadee, American goldfinch, mourning dove, tufted titmouse, chipping sparrow, red-eyed vireo.
Central-east: horned lark, tufted titmouse, Carolina wren, barn swallow, white-breasted nuthatch, purple martin,
Carolina chickadee, rock pigeon, killdeer, downy woodpecker. Mostly human-associated and feeder-visiting residents.

### Reading
- ESTABLISHED (this measure, these data): Z's 1966-75 -> 2010-20 change in the east is shared community change. A Z
  readout learned outside the band predicts the band's observed change across species with corr .88-.93 for sedentary
  (common) species and slope ~1.0-1.15 (no over-scaling). The in-sample C numbers (.98-.99) were a fitting ceiling.
- ESTABLISHED: on the same log1p readout scale, the House Finch's Z-predicted eastern increase (.76 / .85) is close to
  its observed increase (.82 / .88) and ranks 2nd / 1st among the 20 sedentary common species -- large, but within the
  range of resident species' observed changes (mourning dove +.87, black-capped chickadee +.72, tufted titmouse +.71
  east of 80W). The finch is an outlier only against all 369 species (z ~5), most of which barely moved.
- RULE OUTCOME: MIXED. The House Finch dims alone predict sedentary common species' change in the central-east
  (corr .66, CI above 0: HABITAT-LIKE there) but not clearly east of 80W (.25, CI includes 0). Co-invasion: not
  evaluable.
- FRAMING (user correction, 2026-10-30; see SDM_BASELINE_B1 end): B1-dyn's early-East contrast is the equilibrium
  confound -- counts were low because finches had not arrived, and the SDM attributes that to Z; the 1940-55 values
  are extrapolation (no BBS before 1966; pre-1966 Z is a backcast). Nothing here is evidence about early-East location
  quality. The question this diagnostic answers is narrower: is Z's eastern change real community change shared by
  non-invading sedentary common species, or specific to the invasion window/region (a proxy the fit can exploit)?
- Answer, all Z dims (ESTABLISHED, this measure): shared. Readouts learned outside the band predict non-invading
  sedentary common species' eastern change with corr .89 / .93 and slope ~1.1 / 1.05. Z's 1966-2020 eastern change is
  real community change, not something only the invasion window carries.
- Answer, the specific dims the House Finch fit exploits (Z_15, Z_6, Z_18, Z_21): only partly shared. In the
  central-east they predict sedentary common species' change (corr .66 [.18, .88]); east of 80W they do not clearly
  (.25 [-.19, .77]). So east of 80W the finch fit leans on a part of Z's change that resident non-invaders do not
  clearly share -- consistent with a proxy the fit can exploit there (LIKELY, given the wide interval with n = 20).
- The House Finch's own Z-predicted change matching its observed change (.76 vs .82) does NOT validate Z as finch
  habitat: the out-of-band readout was learned partly on invasion-era data elsewhere (the central-east and midwest when
  predicting east of 80W), so the same confound can transfer (LIKELY).
- LIKELY (not tested): the dims the finch fit uses form a human-associated / feeder-community axis -- the species they
  move most are feeder residents that increased in the east over the same decades (titmouse, nuthatch, chickadees,
  Carolina wren, mourning dove, goldfinch, downy woodpecker) plus synanthropes (pigeon, house sparrow) whose net change
  runs through other dims. Part of that change is real (it is shared in the central-east); how much of the finch's
  early-vs-modern contrast it should carry, as opposed to arrival timing, is UNKNOWN -- residents already present early
  increased by ~.7-.9 log1p units with no invasion, so the shared axis supports change of that order, not an
  order-of-magnitude contrast (M1 scaling).
- Scale note: B1-dyn's 12x is a ratio of MEAN expected counts (dominated by high-count cells); on mean log expected
  count the east-of-80W change is 1.39 (x4.0); on the log1p readout .76. The size of the early-vs-modern contrast
  depends strongly on the scale it is read on.
- Caveats: observer turnover inflates uncentred shared change (M3) for every species alike; only 20 sedentary common
  species; Z is the covariate-driven DESK field (M31 LUH3 caveats for any irrigation-linked dims); the attribution is
  B1-dyn's -- only the age model's all-land top-3 were available for comparison.
- Suggested next (not run): the age model's own per-dim attribution in the eastern bands; test the feeder-axis reading
  against urban_tolerance (species_similarity.csv) and a feeder-use trait.
