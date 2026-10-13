# Plan: effort-aware use of raw eBird and raw CBC (2026-10-13)

Requested by the user after E062 ("what you've been doing so far in not properly accounting for effort is not ideal ...
look for examples of how people use it from the literature"). Each phase below is a separate pre-registered experiment
with a gate; nothing proceeds past a failed gate without the user.

## 1. Why the current approach is not good enough

- **eBird (E062):** a reporting rate per cell-year, after only the standard filters, ignores how checklists were made.
  eBird grew ~25x from 2008 to 2024 and its checklist mix changed with it, so the index's change is partly effort. The
  signature is in the data: the species-level change correlation with BBS was **-0.19**, against **+0.59** for the
  effort-adjusted S&T trend (E058). Removing eBird's sampling noise did not help (0.044 vs 0.038), so the problem is
  bias, not noise.
- **CBC (E061, E063 pending):** E061 used Audubon's trend product only inside the training era (winters 2001-2024), the
  one setting where CBC adds nothing a BBS-trained model lacks. CBC's distinctive value is coverage of years BBS
  lacks. Audubon's product starts in 1966; anything earlier needs the raw circle data and our own effort model.
- **Leakage:** both products smooth across our held-out areas. Raw data lets us drop every checklist and circle inside
  held-out and buffer blocks before building any target, which removes that caveat.

## 2. What the literature does

### eBird
- **Filters and effort covariates.** Use complete checklists only (zero-filling). Restrict to stationary/traveling
  protocols, <= 5 h, <= 5 km, <= 10 observers. Then model the remaining effort explicitly as covariates: duration,
  distance, number of observers, start time, day of year, protocol. Spatially subsample to reduce the dominance of
  well-sampled places. Johnston et al. 2021, *Diversity and Distributions* 27:1265-1277 (doi 10.1111/ddi.13271), and
  the companion book *Best Practices for Using eBird Data*. Their finding: the value of each refinement varies by case,
  so test it.
- **Observer skill.** Model skill explicitly, e.g. an expertise score from each observer's species accumulation (the
  checklist calibration approach; Kelling et al. 2015), or as a covariate (Johnston et al. 2018, *Methods in Ecology and
  Evolution* 9:88-97). List length (species per checklist) is a common proxy for effort and skill.
- **Standard checklist.** S&T predicts relative abundance at fixed, standardized effort, so places and years are
  compared on equal terms. Our index should do the same: predict at one standard checklist, not average raw rates.
- **Trends.** Fink et al. 2023, *Methods in Ecology and Evolution* 14:2435-2448: double machine learning adjusts for
  year-to-year change in how observers sample (a propensity-type adjustment), giving 27 km trend maps. This is the
  effort control our E062 index lacked.
- **Agreement with BBS.** eBird and BBS trends agree broadly at continental and regional scales; opposite significant
  trends are rare (~1.3%; *The Condor*, "Same view through a different lens"). Agreement is weaker at finer resolution
  (PLOS ONE 2021, Massachusetts, town level). Expect E062-type cell-level agreement to stay modest even after effort
  modelling.

### CBC
- **Effort as a power function of party-hours.** Expected count proportional to (effort / mean effort)^p, species-
  specific, with diminishing returns allowed (Link & Sauer 1999, "Controlling for varying effort in count surveys").
  Simple per-party-hour scaling is wrong: the relation is nonlinear and differs by species (Dunn et al. 2005, *Auk*).
- **Hierarchical models.** Circle and year effects with effort parameters varying among strata as random effects
  (Link, Sauer & Niven 2006, *Condor* 108:13). The North American winter trends (1966-2013, 551 species) used a
  Bayesian hierarchical model with effort adjustment (Soykan et al. 2016, *Ecosphere*). Spatially varying-coefficient
  models (Meehan et al. 2019, *Ecosphere*) and spatially explicit hierarchical models (Smith et al. 2024,
  *Ornithological Applications*) are the current Audubon method.
- **Known problems.** Circles are not randomly placed. Participation varies greatly (Sauer et al. 2009). Feeder counts
  and species detectability matter (Audubon's "survey suitability"). Audubon starts its trends in 1966 for sample size
  and consistent methods (M29).
- **Before 1966 (our profile of the raw data, circles inside the study box):** party-hours were essentially NOT recorded
  before 1940 (0% of counts 1900-1919, 6% in the 1920s-30s, 55% in the 1940s, 92% in the 1950s, >= 97% from the 1960s).
  The number of field counters is recorded for ~100% of counts in every decade. Volume: ~70 circle-counts per year in
  the 1900s, ~150-180 per year in the 1910s-1930s, against 1,000-2,000 per year from the 1970s.

## 3. Plan for raw eBird (gate before any GPU time)

- **R1 Effort-standardized index (CPU).** Per species, a checklist-level model of detection (and, secondarily, count)
  with: effort covariates (log duration, log distance, number of observers, protocol, start-time smooth, day of year);
  an observer-skill term (an expertise score from each observer's species accumulation, Kelling-style, or a per-observer
  random effect); and a place x year term on our 27 km grid. Fitted with spatiotemporal subsampling (cap checklists per
  locality per cell-year so hotspots do not dominate). The index for a cell-year is the predicted encounter rate of one
  standard checklist (e.g. 1 h, 1 km traveling, one observer, peak morning, mid-June) -- the S&T convention.
  Implementation candidates: per-species GAMs (mgcv-style, in Python via pygam or a custom IRLS), or gradient-boosted
  trees with effort covariates plus a partial-dependence standardization; start with the GLM/GAM because its effort
  terms can be inspected.
  Checklists inside held-out and buffer blocks are dropped first (no leakage).
- **Gate R-a (re-run E062 on the R1 index, same yardstick):** species-level change correlation with BBS must turn
  clearly positive (>= +0.3; S&T reached +0.59), and centred cell-level change E_att >= +0.10. Fail -> no temporal use
  of eBird; record and stop.
- **Gate R-b (spatial):** disattenuated level agreement with BBS >= 0.6 overall (E062's simple index: 0.59; common
  species 0.76).
- **R2 (only if R-a or R-b passes) -- the user's design:** effort-standardized eBird communities for the 96 reference
  species, converted to the BBS scale with the existing calibration code (bbs_ebird_calibration.py), projected into the
  BBS ESK basis, supervising z at non-BBS cells at reduced weight (BBS stays primary). Graded by readouts on both
  targets, selected and late states, two seeds, as E057/E060.

## 4. Plan for raw CBC

- **C0 (now, E063):** with Audubon's product, does CBC regional change over the withheld era (1966-86 -> 2005-24)
  predict withheld-era BBS change better than extrapolating BBS's later trend backward? Its result decides whether the
  rest of this section is worth doing.
- **C1 Circle-level effort model (CPU).** Per species, a negative-binomial GLMM of the circle count with: a
  Link-Sauer power function of party-hours, stratum-varying; number of field counters; feeder hours and counters;
  nocturnal hours; weather (temperature, snow, wind); a circle random effect; and year effects per stratum (or a smooth
  in year by stratum). For counts WITHOUT party-hours (nearly all before 1940), effort enters through the number of
  field counters (and min/max parties where present). That relation is calibrated on 1950+ counts that have both, and
  its uncertainty is carried into the early indices. Circles inside held-out and buffer blocks are dropped first.
- **Gate C-a (validation, 1966+):** our stratum trends must reproduce Audubon's product -- species-level and stratum-
  level correlation of 1966-2024 change >= 0.8 for medium/high-suitability species. Fail -> our model is not yet
  trustworthy; fix before any early-era use.
- **Gate C-b (pre-1966 stability):** in 1950-1965, where both effort measures exist, indices from the counters-only
  model must match the party-hours model (correlation >= 0.8); leave-years-out prediction of circle counts must hold
  up before 1940 as well as after. Coverage is reported as a map per decade: early strata without circles get no index.
- **C2 Bridge test (GPU, after C0 passes):** the CBC head supervised THROUGH the withheld winters (1967-1995) in the
  T0 1996 setup, at a lower weight (aux share <= 30%), graded on withheld-era backcasts by readouts on both targets.
  The rival is a CBC-only readout (stratum change mapped to cells, no DESK), so we learn whether DESK integrates CBC or
  merely relays it.
- **C3 Production prospect (only after C-a, C-b and C2):** effort-modelled pre-1966 CBC as a temporal signal for
  1902-1939, where nothing else exists. Restricted to species whose winter and breeding ranges overlap (residents,
  short-distance migrants). Its fit with the age model's absence-only pseudo-zeros is assessed before any production
  change.

## 5. Guards (all phases)
House Finch never; sealed species never; data inside held-out and buffer blocks dropped before any target is built;
one yardstick (readouts on community and dev-species targets, held-out cells, selected and late states, paired with
seeds); a pre-registration and a skeptic round before any result becomes a finding (PROTOCOL).

## 6. Order and rough cost
E063 (minutes) -> C1 + gate C-a (a day of CPU work, mostly building the model) -> R1 + gate R-a (similar) -> C-b ->
C2 (4 GPU runs) -> R2 only if a gate passed.

Sources: Johnston et al. 2021 (St Andrews repository; ebird-best-practices on GitHub); Johnston et al. 2018; Fink et
al. 2023 (St Andrews repository; ebirdst trends vignette); Link & Sauer 1999 (USGS); Link, Sauer & Niven 2006 (Condor);
Dunn et al. 2005 (Auk); Soykan et al. 2016 (USGS); Meehan et al. 2019 (Ecosphere); the Condor eBird-BBS comparison;
the PLOS ONE Massachusetts comparison. Pre-1966 coverage numbers: our own profile of the raw CBC request (2026-10-13).

## 7. Additions (2026-10-13, after discussion with the user)
- **Community-level early CBC.** Little or no published work uses EARLY CBC for community similarity (community analyses
  start in the 1970s: composition vs climate, community temperature indices; the early record is used for species
  trends). Two effort strategies, used side by side and checked against each other where both apply (1950s-60s):
  (a) MODEL-BASED: assemble each circle-year's community from E064's per-species expected counts at standard effort;
  (b) COVERAGE-BASED RAREFACTION (Chao & Jost): compare communities at equal estimated sample completeness, estimated
  from the counts themselves -- no recorded effort needed, which suits pre-1940 counts. Caveats: flock counts violate the
  independent-individuals assumption (compute coverage from detection incidence across a circle's parties or years, not
  raw individuals), and rarefaction fixes sample SIZE, not effort TYPE (feeder, nocturnal, boat effort add particular
  species -- those still need the effort record). Then the community version of E063: does a circle's WINTER community
  change track the BREEDING community change (BBS ESK) at the same place, against extrapolating BBS's later trend?
- **Season test (CONSIDERATIONS G11).** House Finches are year-round residents. Test whether CBC-community features
  predict PROXY residents (sedentary dev species per AVONET, feeder-associated / finch-like preferred; never House Finch
  data) better than BBS-community features, scored in winter (CBC) and summer (BBS) counts.
