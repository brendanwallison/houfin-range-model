# House Finch survival and dispersal from the Bird Banding Lab: data reconnaissance and a submodel design (scoping, 2026-10-31)

Why: DISPERSAL_SYNTHESIS_2026-10-31.md s.2.1 / 2.5 -- every front mechanism tested at 54 km failed its recovery guard
because planted effects were absorbed by the free compensators (fecundity, founders, adult / juvenile survival).
Fecundity gets a NestWatch block (FECUNDITY_CONSTRAINT_SCOPING_2026-10-31.md, ladder step 31). Founders get a CBC anchor
(ladder step 32). This note covers the third compensator, survival, plus what the same records say about dispersal
distance and partial migration. DESIGN ONLY: no model code. Cites CONSIDERATIONS G1 / G3 / P1 / P3. Every claim is
marked established (E), likely (L) or unknown (U).

## 1. Data (E: downloaded and counted)
- Source: USGS / ECCC North American Bird Banding Program Dataset 1960-2025 (retrieved 2025-07-11), ScienceBase item
  68837a85d4be027deba86316, doi:10.5066/P1KPZGAR, CC0. House Finch (5190-HOFI) is alone in group 37.
- Local copy: /mnt/e/Datasets/houfin/bbl/raw_2025/ -- NABBP_2025_grp_37.csv (141 MB, 1,005,739 rows), lookups, README,
  FGDC xml. Only group 37 was kept (the 376 MB group 31-40 zip was deleted after extraction). Untrusted data, read only
  with `python -I`. Scripts: scratchpad bbl_tab.py (counts) and bbl_s.py (crude survival, distance tails).
- Record types: 1,000,051 bandings (B); 4,533 encounters from the encounter database (E); 1,155 recaptures (R).
  - 94% of bandings are 10-minute-block centroids (~18 x 14 km), 7% exact. Fine at 54 km.
  - Band numbers and permits are aliased but linkable. Every encounter joins to its banding record.
- Encounter condition: 2,608 dead, 2,876 alive (1,726 encounter DB + 1,150 recaptures), 204 unknown.
  - how_obtained: 66 "trapped and released by a bander" 2,710; 00 found dead 1,727; 12 cat 288; 13 window / structure 217.
  - Years since banding: 36% same year, 35% next, 15% / 7% / 4% at 2 / 3 / 4 years. Longevity records: 22 yr adult, 24 yr HY.
- What is NOT here (E, important): routine same-station recaptures. Only 1,155 R records, mostly 2006+ in the West. So
  there is no Cormack-Jolly-Seber (live-recapture) panel. Survival must come from DEAD RECOVERIES plus a small
  live-encounter stream by other banders.

### 1.1 Bandings by region x decade x age at banding (E)
Regions match the ladder's lambda bands. Ages: HY = hatching year (flying, banded in its first calendar year);
L = nestling / local; adult = AHY / SY / ASY / TY / ATY; unk = unknown (18%).

| region | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s | 2020-25 | HY / adult / L / unk (all decades) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| west of 100W | 57,670 | 40,673 | 79,623 | 68,156 | 49,769 | 29,130 | 9,554 | 102.5k / 172.4k / 5.0k / 54.6k |
| plains 100-95W | 54 | 651 | 6,109 | 14,289 | 8,153 | 1,362 | 169 | 11.4k / 15.1k / .3k / 4.1k |
| central-east 95-80W | 26 | 1,483 | 89,786 | 108,116 | 27,508 | 18,596 | 7,539 | 105.3k / 103.8k / 3.5k / 40.5k |
| east of 80W | 27,361 | 69,387 | 194,475 | 60,491 | 16,809 | 9,972 | 3,140 | 128.6k / 170.4k / 1.9k / 80.7k |

- Effort is very uneven (E). The 1980s East (194k) and central-East (90-108k in the 1980s-90s) dominate. Eastern
  banding collapses after 1994 (Mycoplasma arrival), to 17k / 10k per decade. Distinct banding blocks per region-decade:
  127-437 in the East, 295-661 in the West.
- Bandings are spread across all months (6-13% each). Winter bandings are common, so in the partly migratory East a
  winter banding location is often not the bird's breeding cell (E for the seasonal pattern, s.2.3).

### 1.2 Encounters by banding region x era (E; dead recoveries, years since banding >= 0)
| banding region | 1960-79 | 1980-93 | 1994-2006 | 2007-25 |
|---|---:|---:|---:|---:|
| west of 100W | 403 | 185 | 150 | 99 |
| plains | 0 | 29 | 29 | 8 |
| central-east | 5 | 327 | 96 | 116 |
| east of 80W | 445 | 611 | 60 | 45 |

- Dead recovery rate per banding: about .26% (2,608 / 1.0M). This is typical for a small passerine (L).
- Live recaptures by other banders, at least one year later: 27 / 37 / 262 in the West (1980-93 / 1994-2006 / 2007-25),
  and at most 94 in any other region-era. This stream is too thin to do more than tune reporting rates (E).

## 2. First look: what the counts already say (crude, no model)

### 2.1 Adult survival (L: the method is crude)
Method: the geometric MLE of S from dead recoveries of birds banded as adults, years since banding >= 1. It assumes S and
the reporting rate are constant over years. It ignores right-truncation at 2025, which biases late cohorts low. The
banding year runs Apr-Mar.

| banding region | 1960-93: n, S (SE) | 1994-2025: n, S (SE) |
|---|---|---|
| west of 100W | 230, .51 (.02) | 101, .42 (.04) |
| 100-80W | 123, .53 (.03) | 101, .51 (.04) |
| east of 80W | 414, .56 (.02) | 41, .49 (.06) |
| all adults | 1,014, .53 (.01) | |

- Birds banded young, from year 1 on: .55 overall (n 437). That is post-first-year survival, so it behaves like adult
  survival (E).
- Same-year share of dead recoveries: young .29-.59 vs adults .16-.25. This is the first-year mortality signal, but it
  is confounded with age-specific reporting and partial-year exposure (s.4).
- Literature check (FECUNDITY scoping s.2): Milby & Wright adult .72 / immature .64 (small n); Sacramento CJS .59 / .47
  before / after West Nile; MAPS .48-.49. The BBL crude .51-.56 sits inside that range (L).
- **Against the fits (L, the key number):** step29_yb_native has adult survival logit -.41, so Sa ~ .40 at typical
  habitat. step3_native has .43, so Sa ~ .61.
  - The BBL value sits between them.
  - On the yearling base a survival block would pull Sa UP by ~.1-.15. Fecundity would then have to fall to hold
    lambda: the compensation becomes visible and priced.
- Era contrasts (U as to cause):
  - West -.09 after 1994. Likely partly right-truncation, plus West Nile (2004+, Pellegrini et al.).
  - East -.07 +- .06 after 1994 (Mycoplasma). Too few recoveries to separate from zero.

### 2.2 Dispersal distances (E for the counts; L for the reading)
Natal = banded L / HY in Apr-Sep, re-encountered Apr-Jul of a later year (n 316).

| sample | n | at the banding block | mean km | > 155 km | > 483 km |
|---|---:|---:|---:|---:|---:|
| natal, breeding season | 316 | 40% | 62 | 9.8% | 2.8% |
| banded young, any season, >= 1 yr | 752 | 37% | 102 | 16.8% | 4.8% |
| Massachusetts banded young, <= 1990 (V&L's source) | 27 | 30% | 70 | 18.5% | 0% |
| adult breeding dispersal, Apr-Jul -> Apr-Jul | 334 | median 0 | 10 (West) - 39 (East) | | |
| model juvenile kernel prior (V&L Weibull, MDD 330) | | | ~330 | 63% | 21% |
| step 8a fitted band weights | | | ~625 | 95% | 60% |

- By region, natal medians are 0 km in the West, 55 km in the central-East and 14 km in the East (90th percentile 16 /
  215 / 142 km) (E).
- Reading (L): BBL natal distances put far less mass beyond 155 km than the V&L kernel. They put an order of magnitude
  less than 8a's fitted weights.
- This must NOT be read raw (s.4.3). Recovery is far likelier near people and banders, so long moves are
  under-reported. Still, a reporting bias of 6-20x at > 483 km would be needed to reconcile with the prior. That is a
  testable quantity (s.3.3).
- Caveat (E, step 26): the shorter, banding-consistent kernels (MDD 110 / 200) FIT the BBS counts WORSE. The tension is
  real, and the dispersal-distance block is the place to price it.

### 2.3 Partial migration (E for the counts)
Banded May-Jul, encountered Dec-Feb within a year (n 205):

| banding region | median winter displacement | 90th percentile |
|---|---:|---:|
| West | 0 km | 31 km |
| central-East | 194 km | 794 km |
| East | 147 km | 486 km |

- This is the Able & Belthoff eastern partial migration, and western sedentariness, in this release.
- It agrees with ladder step 25: the native-departure term was detected, with r = .65 (western sedentariness used) (L).
- The age model is annual, with no winter state. So these records cannot test the annual kernel directly. They can:
  1. flag that all-season encounter distances mix migration into "dispersal". V&L's kernel had no season restriction:
     that is the likely source of its fat tail.
  2. set which bandings may be assigned to a breeding cell. Winter bandings in the East cannot be.

## 3. Submodel design

### 3.1 Dead-recovery likelihood tied to the model's Sa / Sj (Brownie / Seber parameterisation)
- **Cohort unit:** bird i is banded in cell c_i, in banding year t_i (Apr-Mar), at age class a_i (young = L / HY,
  adult). Only breeding-season bandings (Apr-Aug) are used to assign c_i in the East and central-East. All months are
  used in the West, which is sedentary (s.2.3).
- **Survival path:** S_i(k) = the model's survival from year t_i + k - 1 to t_i + k, read at the bird's expected cell.
  - Adults: Sa[c_i, t] (breeding dispersal median 0 km).
  - Young, first interval: Sj_eff = Sj[c_i, t_i]^(f_i) x Q-bar. Here f_i is the fraction of the first year still ahead
    at banding (from the banding month), and Q-bar is the model's journey survival averaged over its own juvenile
    kernel from c_i.
  - Young, later intervals: Sa averaged over the kernel destinations.
  - Tying Sj through the model's own dispersal step (not a bare cell value) is what makes the block constrain the
    QUANTITY that absorbs front costs. Q and Sj are exactly what J1 / J2 / step 20 moved.
- **Seber recovery probability:** P(found dead and reported in interval k) = prod_{m<k} S_i(m) x (1 - S_i(k)) x
  r[g(i), t_i + k]. Here r is the probability that a dead bird is found and reported. The probability of never being
  recovered is 1 minus the sum over k up to 2025.
  - The likelihood is a product of categoricals over bands. It aggregates exactly to an m-array per (cell, year, age).
    That is small: about 1M bandings collapse to ~60k cohort cells, so it is cheap on the GPU.
- **Reporting rates r (nuisance):** one logit per region (4) x era (pre-1980, 1980-94, 1995-2005 after the BBL toll-free
  line, 2006+ after web reporting), plus a human-density covariate.
  - Priors centred on .3% (s.1.2), wide.
  - r is shared across ages within region-era. That is the standard identifying assumption: young and adult bands are
    found by the same people.
- **Live encounters by other banders (optional, phase 2):** add a Burnham live-dead stream with its own re-encounter
  rate R[region, era], scaled by banding blocks per region-year.
  - Its information is small (s.1.2), so leave it out of phase 1.
- **Exclusions:**
  - banding mortality (how_obtained 10 / 51) and bird status != 3, that is not normal wild birds;
  - band-only finds (98) with no date of death;
  - "unknown" age at banding (18%). These go to a sensitivity arm with a mixture over age.

### 3.2 What it would identify (L; to be confirmed by the guards)
- **Adult survival level by region x era (E for the counts).** Good where dead recoveries exceed ~100:
  - West all eras (SE about .02-.04);
  - East 1960-93 (.02);
  - central-East 1980+ (.03-.04).
- **Weak where recoveries are few:**
  - post-1994 East (41 adult recoveries, SE .06). The Mycoplasma survival hit (the model's disease target is survival,
    mortality ceiling .8) is only loosely tested;
  - the Plains (66 recoveries in total).
- **Habitat gradient in survival (U):** across ~4 regions x 4 eras, the block mostly pins the Sa INTERCEPT and broad
  regional contrasts, not the within-region Z . beta_s field. A cell-level test would need many more recoveries.
- **First-year survival (L, weakly):**
  - It is identified from the young / adult difference in year-0 vs later recoveries, under shared r.
  - It is sensitive to the banding-month exposure f_i and to the HY / adult reporting assumption.
  - It gives Sj x Q-bar jointly, not Sj and Q separately. That is still exactly the product a journey-cost mechanism
    has to change.
- **The compensator logic (L):**
  1. With Sa pinned near .5 and Sj x Q-bar bounded by data, a planted front cost (J1 h) can no longer be paid by
     raising Sa or Sj.
  2. With NestWatch on as well, fecundity cannot pay it either.
  3. With founders pinned (step 32), founders cannot pay it.
  4. What remains to absorb it is habitat (w_scale / beta), and the habitat field's centring and amplitude priors bound
     that.
  5. So the decisive test is the J1 recovery guard with all three blocks on (G2 below).

### 3.3 Dispersal-distance likelihood (checks the V&L kernel; phase 2)
- **Destination likelihood:** for each natal record (banded young Apr-Sep in c_i, encountered Apr-Jul of a later year
  in cell d), P(d | c_i) = K_j(c_i -> d) rho(d) / sum_d' K_j(c_i -> d') rho(d').
  - K_j is the model's juvenile kernel, including its band weights and the departure term.
  - rho(d) is the relative encounter opportunity at d. Model it as human population density^gamma (dead finds), plus a
    banding-blocks-per-year term (live finds by banders).
  - gamma is fitted, but is identified only through the adult breeding-dispersal records. Those should be near 0 km
    for a sedentary adult, which calibrates how fast rho falls with distance from the banding site. This is the
    weakest link (U).
- **What it checks:**
  1. the juvenile band masses (37 / 42 / 21% prior; 8a's 5 / 35 / 60%);
  2. whether the stayer fraction (~40% at the block) matches the model's departure probability at the banding
     cell-year's N/K;
  3. regional contrast in departure (West 0 km median vs central-East 55 km). This is the native-departure term of
     step 25, tested on direct movement data instead of BBS counts.
- **Partial migration:** use the summer -> winter pairs (n 205) only as a diagnostic. Show that all-season distances
  overstate natal dispersal by the migratory component; that justifies the breeding-season restriction. The annual
  model has no winter state, so no likelihood term.

## 4. Biases, and the guard for each
1. **Reporting rate varies in space and time (E that it varies; L for the size).**
   - Suburban and feeder areas report more. Cats and windows dominate known causes (288 / 217).
   - Reporting rose after the 1995 toll-free number and 2000s web reporting.
   - If r correlates with habitat (urban Z), S and r trade off in space.
   - Guard: r by region x era plus a human-density covariate.
   - Sensitivity: r by cell-level urban class.
   - Synthetic G3 plants an r-habitat correlation.
2. **Banding effort and site selection (E).**
   - Feeder banding concentrates on suburban, high-density sites. The 1980s East and central-East dominate. Eastern
     banding collapses after 1994.
   - Survival from dead recoveries is per-banded-bird, so effort sets precision, not bias. The exception is when site
     selection tracks local survival (feeder sites, L).
   - Guard: weight nothing by effort. Report per-region-era information. Never pool a headline across regions
     (memory: sedentary primary).
3. **Distance-biased recovery (E that it exists).** This affects the dispersal block only (s.3.3). Guard: the rho model,
   plus a synthetic test with a known kernel and a known rho.
4. **Migration (E for the East).**
   - Winter bandings misassign the breeding cell.
   - Guard: breeding-season bandings only for cell assignment in the East and central-East. The sensitivity arm uses
     all months with region-only assignment.
5. **Age misclassification (L).**
   - HY becomes AHY on Jan 1, so autumn HY are correctly young.
   - Unknown-age bandings (18%) are excluded in phase 1.
   - SY vs ASY is irrelevant at two age classes.
6. **Mycoplasma era (U for the size).**
   - Diseased birds at feeders are likelier to be found dead, so r rises with disease and conflates with S.
   - The post-1994 East sample is small.
   - Guard: an era-specific r in the East, and report the Mycoplasma survival effect only as a bound.
   - The model's disease term (target survival) is the thing tested, not a free era effect on Sa.
7. **Right truncation (E).** Cohorts banded after ~2015 have short windows. The Seber likelihood handles this exactly
   when the never-recovered term is summed only up to 2025. The crude s.2.1 numbers do not handle it.
8. **Band loss / wear (L, small for a ~20 g bird over 2-5 years).** Ignore it; it is absorbed into S (a slight low
   bias).
9. **Transients / handling effects (L).** Exclude same-day and banding-mortality records. Year-0 recoveries also enter
   only through partial exposure f_i.

## 5. Synthetic guards (to pre-register before any real fit; same pattern as ladder step 21 and the fecundity block)
- Truth: the base refit (step29_yb_native with founders pinned per step 32 if adopted).
- Simulated data: bandings at the REAL BBL cell-year-age counts, with recoveries drawn from the truth's Sa / Sj / Q-bar
  and r set to the real-data region-era estimates. For G1-G2, NestWatch nests are simulated too, as in the fecundity
  scoping.

| guard | truth | fit | gate |
|---|---|---|---|
| G0 (offline, no fit) | -- | -- | crude region-era S (s.2.1) vs the fitted Sa at the banding cell-years of each base: does the base's Sa covary with the BBL contrasts? (the fecundity block's G0 analogue) |
| G1 null | base | survival + nest blocks on | h <= .01; Sa, Sj, fecundity, founders, w_scale inside the seed band |
| G2 recovery (key) | planted J1 h = .03 | survival + nest blocks on, founders pinned | h in [.015, .045] with Sa / Sj / fecundity inside G1's band. If it fails, report where h went (habitat amplitude?) |
| G2b | planted h = .03 | survival block only (no nests) | which compensator takes it when only survival is pinned |
| G3 bias | base, r rising with urban Z (+50% across the Z range) and +30% post-1994 eastern r | survival block | where the bias goes (Sa gradient, disease term) |
| G4 dispersal | known kernel (MDD 200) + known rho (pop density^1) | distance block | recovered band masses within 5 points; gamma recovered |
| G5 real fit | -- | -- | read against G1-G4: Sa level and regional contrasts, Sj x Q-bar, the J1 / J2 / attraction recovery reruns |

## 6. Cost and order (L)
- **Data prep:** one CPU script (minutes). It joins bands, assigns cells on the 54 km grid, and builds m-arrays per
  (cell, banding year, age) and a recovery table. It stores the m-arrays next to the model input with a manifest.
- **Model code:**
  - The phase-1 likelihood is a gather of Sa / Sj / Q-bar at cohort cell-years and a cumulative product over up to ~60
    intervals. That is negligible next to the forward simulation.
  - Q-bar needs the kernel-averaged journey survival per origin, already computed inside the step.
  - The phase-2 distance block needs the kernel row K_j(c -> .) per natal record (~300 rows; cheap).
- **Order:**
  1. G0 (offline, against step29_yb_native and step3_native);
  2. phase-1 block + G1 / G2 / G2b / G3 on TACC, beside the nest-block guards;
  3. G5 real fit;
  4. phase 2 (distance + live stream) and G4.
- **Not recommended:** a CJS model. The release lacks same-station recaptures. MAPS station data would be needed
  (separate request to The Institute for Bird Populations; U).

## 7. What this would and would not settle (L)
- **Would:**
  - remove adult survival as a free compensator (level and broad regional contrasts);
  - bound first-year survival x journey survival;
  - with nests and CBC founders, make a planted front cost either recoverable or attributable to habitat. That is the
    precondition named in DISPERSAL_SYNTHESIS s.2.5.
- **Would test:**
  - the juvenile kernel's tail, against direct movement data with a reporting model;
  - western vs eastern departure.
- **Would not:**
  - identify the within-region survival habitat field;
  - measure the Mycoplasma survival effect with useful precision;
  - say anything about 1940-59 (no bandings in the East before 1960).
