# Constraining House Finch fecundity (and survival): what the prior does, what the literature says, and a joint nest-data design (scoping, 2026-10-31)

Scope: design and data reconnaissance only. No model edits and no fits. The one new computation is a read-only
re-evaluation of two existing refined points (step 3 and step3_native) on the CPU, plus a join of public NestWatch
records to those fields.

Tags: **established** = read from code, records or data in this session. **likely** = follows from those but not tested.
**unknown** = not verified.

Considerations respected: age-model C2 (fecundity ~10 per adult is absurd) and C4 (prefer structural statements to
penalties; a jointly fitted dataset is a structural statement, a tighter prior is a penalty); repo G3 (habitat carries
habitat, the population block carries dynamics); "premise: changing habitat" (the nest data are used as a
space-and-time-varying constraint, not one global number); sedentary-first (House Finch is a resident). The user's
view (2026-10-31): a single global prior is a weak tool, easily overwhelmed, so its harm and benefit are both small. A
jointly fitted dataset would behave differently. Section 1 measures how overwhelmed it is.

Code references are to the live ladder tree, `~/houfin/work/houfin/research/GRID54_LADDER_20261029/code` at
6fbbe81. The step-3 frozen snapshot has the same lines about 1-170 lines earlier. Records: that tree's
`research/age_model/{DIGEST,CONSIDERATIONS}.md`, experiments E001, E027 and E052/E053, and this repo's
GRID54_LADDER_2026-10-29.md.

---

## 1. What `fecundity_prior` constrains, and how far the fits sit from it

### 1.1 The parameter and its link (established)
- `alpha_f ~ Normal(fecundity_prior.loc, fecundity_prior.scale)` (`src/model/age_priors.py:986-988`). Every one of
  the 58 grid54 ladder runs resolves this to N(4.993, .75) (checked across `runs/*/out/resolved_config.json`).
- `Fmax = softplus(alpha_f + gamma_f*H_r + d_r)` with `gamma_f` fixed at 1 (`src/model/age_fields.py:302`,
  `age_priors.py:1040`). softplus(4.993) = 5.00, so the prior says **Fmax = 5.0 at typical habitat, sd about .75 in
  Fmax units**. At that point softplus is effectively the identity.
- "Typical habitat" means H_r = 0. Under `habitat_centering = occupied`, that is the mean habitat of occupied cells
  (`habitat_center_offsets` in the reconstruction).
- The prior touches only the intercept. It says nothing about:
  - the fecundity field's amplitude (`w_scale[1]`; fitted softplus(8.25) = 8.25 against a prior median of about .70,
    docs/PRIORS.md "GP amplitude"), which lets Fmax swing from about 0 to about 20 across cells;
  - its spatial pattern;
  - its change over time.

### 1.2 What Fmax means in the model's life cycle (established from code)
- None of the ladder runs sets `population_model.reproduction`. All use the standard form, in which births are
  `N_j_new = surviving_adults * F_actual` (`src/model/age_forward.py:1032`).
- `F_actual` is `F_eff` times the Allee factor, with `F_eff = Fmax / (1 + c N/K)` (Beverton-Holt, `:995`).
  `c` is solved so that lambda = 1 at N = K.
- So **Fmax is the number of juveniles (both sexes) per surviving ADULT individual per year**, at low density with no
  mate limitation, counted before juvenile survival Sj. Per pair it is 2·Fmax.
- **First breeding happens at about age 2.** A juvenile joins the adult class the next year and breeds only after
  surviving one more year (lambda0 = (Sa + sqrt(Sa² + 4·Fmax·Sa·Sj))/2, `age_forward.py:577`).
- E052 recorded this as a model bug, because House Finch yearlings breed (Veit & Lewis 1996). The switch
  `reproduction.yearling_breeding` exists (`age_priors.py:259-262, 387-389`) but is off throughout the ladder.
- Under the delayed form a given growth rate demands a much larger Fmax. The yearling-breeding fecundity F' that gives
  the same lambda0 is **F' = Fmax·Sa/lambda0** (algebra: lambda0² = Sa·lambda0 + Fmax·Sa·Sj, so
  lambda0 − Sa = Sj·Fmax·Sa/lambda0).
- Sj in the model is survival from the birth census to adulthood the next year. If births are counted at fledging,
  which is likely, Sj includes post-fledging mortality.
- `Sj = Sa·sigmoid(alpha_j + H_sj)` (`age_fields.py:297-299`, juvenile_survival.form = ratio).

### 1.3 Fitted values in observable units

The fields were re-evaluated at the refined MAP points with `reconstruct_map`, float64, on the CPU.
"Weighted" means weighted by simulated abundance, so it describes the habitat of a typical bird.

| quantity (2010-25) | step 3 (averaged Z) | step3_native |
|---|---:|---:|
| alpha_f; Fmax at typical habitat | 11.04 | 12.60 |
| prior SDs from N(4.993, .75) | **+8.1** | **+10.1** |
| weighted Fmax, q10 / median / q90 | 7.2 / 12.9 / 19.4 | 5.8 / 16.6 / 29.1 |
| median juveniles per pair per year (2·Fmax) | ~26 | ~33 |
| weighted Sa (median) | .61 | .65 |
| weighted Sj (median); Sj/Sa | .60; .98 | .65; 1.00 |
| weighted lambda0 (median) | 2.49 | 2.98 |
| yearling-equivalent F' = Fmax·Sa/lambda0 (median, q10-q90) | **3.3** (2.9-3.9) | **3.6** (2.1-4.9) |

Other fits, Fmax at typical habitat and prior SDs out:
- step 3 seed 42: 10.5, +7.3.
- J1 null: 9.40, +5.9.
- J1 recovery: 8.29, +4.4.
- step7 allee020: 24.5, +26.

East (>95 W) runs higher than West (<104 W): 14.1 vs 12.5 (step 3) and 20.3 vs 15.6 (native).

How to read these (likely):
- In the model's own units, the fitted fecundity is about 2.5-5 times anything biological. 26 juveniles per pair per
  year is not plausible; the literature is roughly 4-9 fledglings per pair per season (section 2).
- Converted to yearling-breeding units, the same fits are **F' about 3.3-3.6 per adult, or 6.5-7 fledglings per pair**.
  That is at the upper end of the literature but not absurd.
- So **most of the "Fmax about 11" is the delayed-first-breeding structure (E052), not a fecundity estimate.**
- What remains implausible:
  - lambda0 of about 2.5-3. E027 called about 1.5-1.6 defensible, and the realised eastern expansion was about
    1.28 per year.
  - Sj/Sa pinned at about 1. Its prior median is .60. A first-year survival from fledging of about .6 is high for a
    small passerine.

### 1.4 Survival priors (established)
- **Adult:** `alpha_a ~ N(.5, .5)` on the logit, a default because no `adult_survival_prior` is set
  (`age_priors.py:973-975`). Prior median Sa = .62.
  - Fitted alpha_a is .219 at step 3 (Sa .555 at typical habitat; −0.56 SD) and .429 at native (−0.14 SD).
  - **Adult survival is not in tension with its prior.**
- **Juvenile:** a ratio prior, `alpha_j ~ N(.405, .5)` (`age_priors.py:976-980`), so the prior median of Sj/Sa is .60.
  - Fitted 2.505 at step 3 (ratio .92 at typical habitat, **+4.2 SD**) and 4.514 at native (**+8.2 SD**).
  - Weighted Sj/Sa is .98-1.00, so the ratio sits at its ceiling.

### 1.5 How badly the prior is overruled (established arithmetic; the scale is likely)
- A +8 SD miss costs 0.5·8² = **32 nats** of prior penalty (+10 SD = 50 nats).
- The data's pull is far larger. E001 (27 km, older model) pinned typical-habitat Fmax: Fmax 5 cost +252 nats and
  Fmax 3 cost +630, spread across the whole eastern invasion. Compensation went through Sa and through the juvenile
  ratio held at its ceiling.
- E052/E053 showed that every structural change moved the distortion rather than removing it. The sequence was
  F → Sj → the west field → Sa .95 (veit form) → a flattened near-replacement landscape when the V&L demography was
  pinned.
- So the user's reading holds for this prior. **It is overwhelmed by roughly an order of magnitude, its benefit is
  about nil, and tightening it would only push the same pressure into Sj or Sa** (likely, from E001/E053).
- Step 21 shows the cost of leaving fecundity free. A planted journey hazard (h = .03) came back as h = .002, while
  fecundity fell 9.40 → 8.29 (−12%) and absorbed it.

---

## 2. Literature values

Tags: V = read in the source. S = search summary or secondary citation, not opened. U = not found or paywalled.
Sources were gathered by a research subagent on 2026-10-31; local PDFs of Veit & Lewis, Pellegrini and Wootton are
in the session tool-results folder.

### Fecundity

| quantity | value | where / when | source |
|---|---|---|---|
| clutch size | East 4.53 ± .86 (n 108); S. California 4.21 (128); Northwest 4.35 (195); East > West (P .002 / .037) | Nest Record Cards, before 1985 | Wootton 1986, Wilson Bull. 98:459 (V) https://digitalcommons.usf.edu/wilson_bulletin/vol98/iss3/11 |
| total nest failure | East 19%, West 27% | same | Wootton 1986 (V) |
| young fledged per clutch | 1.91 ± 2.01 | SE Michigan | Hill 1993, cited in Veit & Lewis (V via V&L) |
| young per pair | 1.8-2.9 (39 pairs) | Año Nuevo Island, CA | Hooge 1990 Condor 92:1066, via V&L (S) |
| c, surviving young per pair per summer | **1.5**: ~2 fledged per clutch × .75 summer survival. V&L note that double and triple broods would make it "substantially higher" | model parameter | Veit & Lewis 1996 Am. Nat. 148:255, doi:10.1086/285924 (V) |
| clutch | 4.4; heavy loss to cats | Sacramento 1950-54 | Evenden 1957 Condor 59:112 (S) |
| clutch; nest survival | 4.4; 73% / 46% | Kansas 1987-88 | Villaseñor 1988 thesis (S) |
| broods per season | 1-6; "double and even triple" in Michigan | general | All About Birds (S); Hill 1993 via V&L (V) |
| productivity index (young:adult in captures) | .83-.88, the highest of 6 finches | MAPS 1992-2006 | Vital Rates of N. American Landbirds (V) https://www.vitalratesofnorthamericanlandbirds.org/pages/bird.php?id=HOFI |
| Badyaev Montana/Alabama, Stangel 1985, Birds of the World demography | not obtained | | (U; BOW is paywalled) |

### Survival

| quantity | value | where / when | source |
|---|---|---|---|
| adult / immature annual survival | .719 (SE .098) / .640 (SE .179); small n | Kern Co., CA, 1963-73 | Milby & Wright 1976 J. Field Ornithol. 47:119 (S; values as used by V&L, V) |
| s used for both ages | **.67** | model | Veit & Lewis 1996 (V) |
| adult apparent survival (CJS) | .59 ± .15 before West Nile (2001-04); .47 ± .11 after (2005-08) | Sacramento Co., CA | Pellegrini et al. 2011 Condor 113:233, doi:10.1525/cond.2011.090233 (V) |
| adult apparent survival | .48-.49; recapture probability only .035-.052, so biased low by emigration | MAPS 1992-2006 | Vital Rates site (V) |
| Mycoplasma survival effect | seasonal survival in relation to MG | Ithaca, NY, 2000+ | Faustino et al. 2004 J. Anim. Ecol. 73:651 (U, numbers not obtained) |
| MG population effect | decline, larger where MG arrived early at high density; about 40% drop / "rapid halving" | East, 1994+ | Hochachka & Dhondt 2000 PNAS 97:5303 (V abstract); press release and Hochachka et al. 2021 (S) |
| natal dispersal | 333 recaptures; 21-28% recaptured where banded; mean 109 km | East | Belthoff & Gauthreaux 1991, via V&L (V) |

### Literature numbers in the model's units (likely)
- **Per season:** clutch about 4.2-4.5, 1.9-2.9 fledged per attempt including failures, 2-3 attempts → about 4-8.5
  fledglings per female per season.
- **Per adult:** F' about 2-4 per adult per year in yearling-breeding units, before post-fledging survival.
- **Converted to the model's delayed form** (Fmax = F'·lambda0/Sa, with Sa .6 and lambda0 1.3-1.6): about 4.5-11.
  - The current prior median of 5 sits inside that range only because the delayed structure inflates the conversion.
  - The fitted 11-13 is at or beyond its top, and only with lambda0 about 2.5.
- **Survival:** adult .5-.72 (apparent estimates biased low). First-year survival from fledging is not measured for
  House Finch in anything retrieved (unknown).
- **Region and era:** eastern clutches about .2-.3 eggs larger in the low-density invasion years (Wootton). West Nile
  (CA) and Mycoplasma (East, 1994+) cut survival, but no paired before/after survival numbers were retrieved for
  Mycoplasma (unknown).

---

## 3. Data reconnaissance

### 3.1 NestWatch, Cornell (established: downloaded and counted)
- **Access.** The *NestWatch Open Dataset* is on Mendeley Data, doi:10.17632/wjf794z7gc.6 (v6, 2026-01-20,
  updated about 31 January each year).
  - No registration was needed. Licence CC BY-NC 4.0 per the subagent (V). Terms ask that the data not be
    redistributed.
  - Field definitions are in Bailey, Larson & Bonter 2024, *Ecology* 105:e4230, doi:10.1002/ecy.4230. That page
    returned 403, so the outcome codes below are unverified.
- **Files.** Attempts with locations: 26.7 MB zip, sha256 6870ebe0…. Visit-level checks: 49.0 MB zip, sha256 0f1c847e….
  Both were fetched to the session scratchpad; the extracted CSVs were deleted after counting.
- **Coverage.** 860,226 attempts across 1,176 species. **House Finch: 4,869 attempts** (rank 21), at 4,153 locations,
  from 1,630 observers.

Attempts by decade and region:

| decade | East (>95 W) | Plains | West (<104 W) | all |
|---|---:|---:|---:|---:|
| 1950s | 0 | 0 | 4 | 4 |
| 1960s | 29 | 44 | 222 | 295 |
| 1970s | 71 | 7 | 557 | 635 |
| 1980s | 292 | 4 | 67 | 363 |
| 1990s | 281 | 14 | 55 | 350 |
| 2000s | 340 | 6 | 83 | 429 |
| 2010s | 962 | 95 | 264 | 1,321 |
| 2020s | 1,102 | 47 | 322 | 1,471 |
| all | 3,078 | 217 | 1,574 | 4,869 |

- **Top states:** CA 797, NY 528, PA 307, CO 276, NM 261, NC 240.
- **Mycoplasma split (East):** 530 attempts before 1994 and 2,548 after.

Fields:
- **Attempt level:** clutch size, young total, young fledged, unhatched eggs, first-lay / hatch / fledge dates
  (with "estimated" flags), outcome code, substrate, nest height, habitat within 1 m and 100 m, predator guard,
  observer ID, and entry technique (which identifies the historical source).
- **Visit level:** date, host eggs, live and dead young, brood parasites. This is what exposure-day (Mayfield or
  logistic-exposure) nest survival needs.

**Summary fields are sparse; visit counts are not.**
- Clutch is recorded on 1,422 HOFI attempts, young fledged on 1,332, and first-lay date on 1,399, almost all from
  2000 on.
- Visit-level egg counts exist for 4,001 attempts in total, including about 1,200 in 1960-99.
- Outcome codes: u1 1,577 (likely "unknown"), s1 1,348 (likely success), f* failures about 760. Meanings unverified.
- Attempts per location-year: 4,146 singles, 257 doubles, 59 with three or more. Broods per female are **not
  identifiable**: attempts are not linked to marked birds.

**Bias markers (established).**
- Substrate: buildings 2,031, "other" 953, live tree branch 733, shrub 379, nest box 287, pole 201.
- Habitat within 100 m: residential yard ("ry") is the most common class (984 of 2,217 recorded).
- Location precision: 6-7 decimal places for most records, but county or state centroids for 337 historical cards.

Raw rates where recorded:
- Clutch: East 4.35 (n 1,027), Plains 4.14 (65), West 4.20 (282).
- Fledged per attempt: East 2.81 (995; 26% zero), West 2.72 (274; 31% zero).
- These are apparent rates. They are likely biased high, because nests found late and nests with unknown outcomes
  drop out.

### 3.2 The historical North American Nest Record Card Program (established for HOFI)
- **About 1,646 House Finch cards (1950s-1990s) are already in NestWatch.** Entry technique reads
  "NANRCP bulk upload 2023-05-02"; transcription went through the Zooniverse "Nest Quest Go!" project (S).
- Location type: 1,309 "self", 296 county centroid, 41 state centroid. County centroids are adequate at 54 km;
  state centroids are not.
- The cards carry visit-level egg and young counts (median 3 visits) but no summarised fledged count. Clutch and
  fledging would have to be derived from the visits, with care over incomplete observation.
- Raw pre-2000 maximum counts: eggs East 3.85 / West 3.80; young East 3.19 / West 3.27. Maximum egg count
  under-reads clutch when nests are visited once.
- The cards are mostly from 1960-80 in the West and from the 1980s-90s in the East.
- What remains un-transcribed overall, and whether more HOFI cards remain: unknown. The program has about 291k
  cards; per a 2023 report more than a quarter were available (S).
- Wootton's 1986 sample (431 completed clutches) came from these cards.

### 3.3 Other sources (not downloaded)
- **MAPS (Institute for Bird Populations).** HOFI 1992-2006: 2,212 adults banded at 77 stations, but only 50
  between-year recaptures (V, Vital Rates site). Productivity and survival indices are published per region.
  - Raw data require a request to IBP (Danielle Kaschube, dkaschube@birdpop.org; V per the subagent).
  - Too few recaptures for a regional survival submodel (likely). Useful as a cross-check on productivity
    (young:adult ratio).
- **USGS Bird Banding Laboratory.** North American Bird Banding Program Dataset 1960-2025, doi:10.5066/P1KPZGAR,
  ScienceBase, CC0 (S).
  - Massachusetts alone banded more than 8,000 HOFI in 1960-90 (V&L, V). This is the source V&L used for their
    dispersal kernels.
  - The HOFI encounter count is unknown; it would need the dataset's species files. This is the only candidate for a
    continental, multi-decade survival submodel (dead recoveries plus live encounters).
- **Birds Canada Project NestWatch / Ontario Nest Records Scheme (NatureCounts).** Access levels vary per dataset;
  HOFI counts unknown.

---

## 4. The NestWatch records against the fitted fecundity field (an offline check, no fit)

Method: each HOFI nest's lon/lat was projected to the grid54 reference raster and joined to Fmax at its land cell and
year (step 3 and step3_native refined points). 4,635 nests fall on land cells within 1902-2025; 1,296 have a recorded
fledged count.

| step 3: Fmax quartile at the nest's cell-year | n | fledged / attempt | share with zero fledged | clutch | median Fmax |
|---|---:|---:|---:|---:|---:|
| Q1 | 324 | 2.97 | .24 | 4.13 | 9.4 |
| Q2 | 324 | 2.69 | .30 | 4.24 | 14.2 |
| Q3 | 324 | 2.59 | .32 | 4.14 | 17.6 |
| Q4 | 324 | 2.88 | .24 | 4.21 | 20.2 |

Spearman correlations with Fmax:
- **Step 3:** fledged −.02 (n 1,296), fledged given success −.03, clutch +.02 (n 1,339).
- **Native:** fledged −.06. Its Fmax spans 4.5× across nests (Q1 median 8.6, Q4 29.9), while fledged per attempt
  stays at 2.6-3.1.

Read:
- **Established:** the 2.4-4.5-fold spatio-temporal range of the fitted fecundity field, at places and years where
  House Finches actually nested, has no counterpart in per-attempt clutch size or fledging.
- **Likely:** the fitted fecundity field carries growth and colonisation signal from the BBS that is not fecundity.
  This is the same pressure E001 found "diffuse over the whole eastern invasion".
- **Unknown:** whether broods per season, which nest records cannot see, vary with the field. Whether suburban
  self-selection flattens the nest data. A real fecundity gradient would have to sit entirely in brood number to
  reconcile the two.

---

## 5. A joint nest submodel (design sketch)

### 5.1 The likelihood
For each House Finch nest attempt i at land cell c(i) and year t(i):

- **Expected fledglings per attempt.** `m_i = 2·F_real[c,t] / B_r(i) · exp(x_i·δ)`.
  - F_real is the model's realised per-adult fecundity at that cell-year (`F_actual`: F_eff × Allee, at the
    simulated N).
  - B_r is attempts per female per season in region r.
  - x_i are nuisance covariates: substrate or nest box, the historical-card vs app era, how the nest was found, and a
    Mycoplasma-era term in the East. δ has a weak prior and is centred so the reference class is "natural site, app
    era".
- **Outcome.** A hurdle: P(fledged > 0) = π_i, with logit π_i = a_π + covariates, estimated with exposure days from
  the visit file so late-found nests do not inflate success. Positive nests: fledged ~ truncated NB(mean
  m_i/π_i, φ_n), or Binomial(clutch_i, p) when clutch is known.
- **Clutch (optional).** clutch ~ Poisson(κ_r(i)) with a regional mean. It carries the Wootton east-west contrast but
  is not tied to F.
- **Why tie to F_real and not Fmax.**
  - Nests occur at real densities. In this model, density dependence acts only through fecundity, and c is solved so
    lambda = 1 at N = K.
  - So at carrying capacity, realised F = replacement F = (1 − Sa)/(Sa·Sj). At the fitted Sa .61 and Sj .60 that is
    about 1.1 juveniles per adult, or about 2 per pair per year.
  - The nest data say about 2.8 per attempt × 2-3 attempts, or roughly 6-8 per pair.
  - Tied to F_real, the nest block will therefore push hardest on the *at-K* demography. The likely result is lower
    Sj, or a need for density dependence in survival. That is a real biological question the BBS cannot answer, not
    a defect of the design.
  - The Fmax variant (ignore density) is cheaper but would mis-state what nests in saturated suburbs measure.

### 5.2 What it would and would not identify (arithmetic plus likely)
**Would identify:**
- the level of output per attempt (1,300+ fledged counts give a standard error of about 2-3% on the log mean);
- its spatial and temporal pattern, via F_real's habitat slope (w_scale[1]·H_r) and time change;
- the at-K relation F_real·Sj ≈ (1 − Sa) per adult, i.e. a three-way link between nests, first-year survival and
  adult survival.

**Would not identify:**
- **Broods per season B.** The nest records give about one attempt per location-year, with no marked birds. **The
  Fmax intercept is identified only up to B.**
  - With B free under a loose prior (e.g. LogNormal sd .3), the step-21 trade (fecundity −12% to absorb a planted path
    cost) would simply move into B for about 0.1 nat. The nests would not stop it.
  - So the nest block constrains the intercept only if B is anchored, by one of:
    - (a) fixing B per region from the literature, a structural statement;
    - (b) deriving season length per region from NestWatch first-lay-date spreads with an assumed renesting cycle
      (about 30-40 d, unverified);
    - (c) a cross-check against the MAPS young:adult productivity index.
- **Per-breeder vs per-adult scaling,** i.e. the sex ratio and the share of adults that pair: absorbed into B.
- **Age-specific fecundity** (yearlings vs older birds).
- **Post-fledging survival.** It sits inside Sj, and Sa/Sj remain identified only through the BBS and the at-K
  identity.

**Prerequisite:** yearling breeding.
- Without it, the per-attempt output the nests fix cannot produce the growth the BBS demands, because first breeding
  is delayed to age 2.
- The fit would then pay the nest block, or move to Sa .85-.95 as in E037/E053.
- The switch already exists (E052). Production implication: a fecundity constraint only means something on the
  yearling-breeding life cycle.

**Survival should come in jointly too.** E001/E053 show that constraining F moves the distortion to Sa or Sj.
- The only continental survival source is the BBL encounter data, CC0. A dead-recovery and live-encounter submodel
  is a separate, larger build.
- Interim: an informative adult survival prior from Milby & Wright, Pellegrini and MAPS (about .5-.7, a band that
  honestly reflects the apparent-survival bias) is weak in the user's sense. It still keeps the escape route visible.

### 5.3 Biases and guards
- **Suburban, building and nest-box selection.** Residential yards dominate, and observers choose the nests.
  - Likely only a modest problem, because House Finch is itself strongly human-associated.
  - Guard: substrate and habitat covariates; a sensitivity arm restricted to natural substrates.
- **Detection and reporting of failed nests.** Many outcomes are unknown, and nests found late inflate success.
  - Guard: exposure-day survival from visit dates; drop unknown outcomes from the hurdle, not as zeros.
- **Era and source.** Historical cards (1960s-90s) vs app records (2010s+). Mycoplasma from 1994 in the East; West
  Nile from 2004 in California.
  - Guard: a source-era term on π and on the positive mean. A Mycoplasma term on survival in the main model already
    exists (the disease block). Do not let a nest-era term stand in for it.
- **Thin coverage where it matters.** East 1960s-70s has only 100 attempts, so the invasion front is barely sampled.
  The nests constrain the established-range demography, not the front.
- **Location precision.** Drop state-centroid cards; county centroids are kept at 54 km.

**Synthetic guards, pre-registered before any real fit** (same truth and gate pattern as ladder step 21):
- **G0 (done here, section 4).** Do the nest data covary with the current fitted Fmax? No (ρ about 0).
- **G1 null.** Truth = the base refit (native Z, yearling breeding). Simulate nests at the real NestWatch cell-years:
  same n, missingness and zero share, drawn from the truth's F_real, B and nuisances. Refit with the nest block on.
  Gate: h ≤ .01, and fecundity, B and Sj inside the seed band.
- **G2 recovery, J1.** Planted h = .03 plus simulated nests. Three arms: B pinned at truth; B under a loose prior;
  B misspecified +25%.
  - Gate: h in [.015, .045] with the fecundity intercept inside the null band.
  - Expected: passes only with B pinned (likely, from 5.2). The misspecified-B arm shows where a wrong B goes
    (into h, Sj or Sa).
- **G3 bias.** Plant a +15% nest-site output bias at the nest locations, and separately drop 30% of failed nests.
  Report where each goes.
- **G4 real fit,** read against G1-G3.

### 5.4 Cost
- **Data preparation (about 1 day).** Turn NestWatch attempts and checks into one row per attempt: land index, year,
  clutch, fledged, exposure days, stage when found, substrate, source, precision flag. Inputs are public and already
  verified (two zips, 76 MB).
- **Model engineering (about 2 days with tests).**
  - Expose per-year realised fecundity at the nest cells from the forward scan. Today only `Fmax_flat` is returned;
    `F_actual` is internal to `reproduction_age_structured`.
  - Add the nest likelihood block with a config switch, and regional B (pinned, prior or season-derived).
  - Add a synthetic nest generator.
  - Compute per evaluation is negligible: about 1-5k gathers from the realised-fecundity field against a forward
    simulation over 124 years.
- **Fits.** About 7 ladder-sized fits (G1, G2 × 3, G3 × 2, G4), each a cold MAP plus float64 refine like any ladder
  step. Runs in parallel across the local 3090 and TACC.
- **Survival (BBL) submodel.** Not costed here. It needs a reconnaissance of HOFI encounter counts first (a
  ScienceBase species file).

---

## 6. Recommended next step
1. **Read step 3 / native on the yearling-breeding life cycle first** (one fit plus its seed). Most of "Fmax about 11"
   is the delayed-breeding structure: its yearling equivalent is about 3.3-3.6 per adult. Until this is done, any
   fecundity constraint is stated in the wrong units. Expected: fitted F falls toward 3-4 while lambda0 and Sj/Sa stay
   high (E052 saw F 8.4 at 27 km, an older model).
2. **Build the nest block tied to realised fecundity, with B handled three ways, and run guards G1-G3 before G4.**
   The decisive question is whether h recovers in the J1 design. If it recovers only with B pinned, the constraint is
   only as good as the brood-number statement. That statement should then be made explicit and regional (from
   NestWatch lay-date spreads), not hidden in a global prior.
3. **In parallel, scope BBL survival.** Count HOFI encounters by region and decade in the ScienceBase release. Section 4
   and E001/E053 say the pressure will move to Sj/Sa once fecundity is held.

Scripts used (session scratchpad; read-only):
- `recon.py`: `reconstruct_map` at a refined point; abundance-weighted quantiles; saves the fields.
- `nw_scripts/count.py`, `checks.py`, `join.py`: NestWatch counts, visit-level coverage, and the cell-year join.
