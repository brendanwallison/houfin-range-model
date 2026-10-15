# DIGEST

## CURRENT STATE (rewritten each round; read this first -- the dated log below keeps the history)
Last updated 2026-10-11 22:06 (E048 decided: CropAT in DESK NO HELP, all 8 estimates ~-0.02; route-scale test E053b: no sign the 27 km grid limits covariate gains; covariate tally: crop types and observed land cover +0.01-0.03 in open-habitat guilds; CRP, forest age, reservoirs 0; E048 CropAT-DESK training). "Solid" =
independently re-derived, replicated on a second split or seed, and survived a skeptic review. "Tentative" = one run,
or checks still pending. Experiment numbers point to research/experiments/EXXX.md.

WHERE THINGS STAND (2026-10-11 ~11:00, plain summary; details below)
1. DESK as a HABITAT MAP (space): a real but modest improvement over its inputs. Under the age model's own linear
   readout, DESK's 24 dimensions beat the top 24 principal components of its covariates by ~0.03-0.06 of level
   skill for well-observed species, beat the 19 bioclim variables by ~0.10 / 0.20, and beat the strongest SDM-style
   set tried (bioclim + squared terms + land cover) by ~0.03 / 0.08. SKEPTIC (corrected): that matched-rank edge is
   mostly WHICH directions DESK keeps -- bird-relevant ones rather than variance-ordered PCs (compression); with all
   ~290 covariate components and plenty of data the commonest species tie; the genuinely beyond-linear part is
   ~0.02-0.04. ANSWERED (E039g, block-bootstrap CIs, two models): a bird-supervised LINEAR projection of the
   covariates (the 24 linear combinations that best predict the community coordinates DESK is trained on) nearly
   matches DESK spatially -- DESK leads it by only 0.00-0.02 at 24 dims and 0.01-0.03 at 64; robust to leaving out
   any held-out block and to fitting the readout on places DESK never trained on.
   IN TIME, measured as backcasting held-out places' community change from 2005-25 back to 1966-86 (E039h-i): the
   pooled linear map does badly (0.09-0.11 against DESK's 0.22-0.37) because it is pure space-for-time.
   - Fair rivals match or beat DESK in all four model x start-year pairs (E039i): a linear map with separate
     within-place coefficients (0.27-0.35), a linear "kind of place x time" term (0.31-0.41), and a position x time
     trend with NO covariates (0.29-0.40).
   - "DESK's network earns its place in time" (E039h, 12:00) is WITHDRAWN. On this measure, DESK's temporal skill is
     not beyond extrapolating each region's recent trend; E016 found the same on the TACC models.
   - CORRECTION (your question, 14:00): the trend placebo says nothing about WHY regions changed. A regional trend
     can be regional HABITAT change, as M6 already says, and community-level change is change shared across
     co-occurring species, the habitat signature. So "DESK ~ placebo" does not mean DESK misses habitat or that the
     trend belongs to population dynamics. It means DESK, from covariates alone, predicts about as much as
     extrapolating the neighbours' observed trend.
   - ANSWER so far, revised by a skeptic (E044/b/c + skeptic, 15:05):
     - Covariates reach about TWO-THIRDS of the predictable regional trend, mostly through LAND-USE CLASS change
       (climate less; management adds nothing).
     - All of that reach runs through the pattern "places of this environment type changed in a typical way".
       Habitat change structured by habitat type (e.g. abandonment or intensification wherever a land use is) and a
       type-indexed clock (M26) are not separable from that.
     - About a third of the trend is reached SPECIFICALLY: beyond equally smooth misplaced fields, size +0.04-0.10,
       fragile.
     - About a third is NOT reached by the covariates.
     - So: nothing shows the trend is dynamics, and nothing yet proves it is habitat.
     - Scale: all predictors together explain only 9-22% of community change (noise removed).
     - Within a type, places whose covariates changed MORE than typical also changed more in birds (E044d), the
       "acts where it acts" signature M26 asks for.
       - Within type: corr 0.16 for all covariates (mostly climate) and 0.08 for land use.
       - Within type AND region: 0.06-0.08 overall, 0.04-0.06 for land use.
       - Real but small, and correlational.
     - So: the evidence is consistent with substantial habitat involvement in the regional trend and is against
       calling it dynamics. It does not prove habitat.
   - DESK's direction of change does carry information: a readout fitted on within-place change lifts it by
     +0.04-0.07.
   - Beyond the regional trend, DESK adds little.
     - At the selected training state (the production one), the gain is +0.01 to +0.04 against an adequate placebo.
       A skeptic showed a SUPERVISED linear covariate map carries about half of it; against that, 1 of 4 pairs meets
       the bar (E039k-l + skeptic). The selected-state claim is withdrawn.
     - At the LATE training state, it HOLDS against the strongest linear base (trend + kind-of-place clock +
       supervised covariate maps), wherever the readout is fitted (E039n). The gain is +0.03-0.04 at T0 1986 and a
       borderline +0.01-0.02 at T0 1996. VERIFIED: the cell-permuted null is ~0 (E039o), and it replicates in 8/8
       pairs over FOUR seeds (E046: +0.02-0.04). At the production-like selected state it is ~0.01.
     - WHERE it comes from (E039m): entirely DESK's response to REGIONAL LAND-USE MANAGEMENT change. The
       continent-wide intensification shift carries more of it than where it intensified more; climate carries none.
     - At the species level, E033c found DESK's increment ~+0.01.
   Implication for the downstream (not a decision now): DESK's temporal value for the age model is at most a small
   increment on top of a regional trend, and it runs through the management trend. That trend is flat before 1950
   (E035/E041), so nothing tested so far supports DESK's movement in the 1902-1939 backcast window. Before 1966
   there is no observed trend to extrapolate. E044 says the regional trend's predictable part is largely
   covariate-carried, so a covariate-driven habitat block is the right route to it. How well that holds before 1950,
   when the covariates that drive it (management) were flat, is the open question.
   Implication for the downstream (not a decision now): DESK's 64 dimensions beat its 24 by ~0.02-0.04.
   NEW COVARIATES, first test (E047/E047b; county-census-based cropland history CropAT-US, 1 km, 1850-2021):
   - LUH and the county-census record agree on WHERE cropland is (0.94). They do NOT agree on where it CHANGED,
     1966-86 -> 2005-21 (~0.01).
   - Net crop amount carries no community-change signal in either source.
   - Crop TYPE composition (corn, soy, wheats, cotton, ...) adds +0.04 beyond all of LUH's land use within
     environment type, but nothing at the 150 km local level.
   - Skeptic (E047b): against ALL the covariate change DESK sees, not just LUH, the community-level gain is only
     ~+0.012-0.016, mostly in the West. CropAT's local information is county-level census totals; below the county it
     is another allocation.
   - WHERE IT ACTS (E049b/c, species level): census crop types predict GRASSLAND birds' change beyond all of DESK's
     covariates. The gain is +0.015-0.018 locally and beats every null in which CropAT's field is moved in space. It is
     exactly 0 for FOREST birds, at both start years. Against LUH's land use alone the grassland gain is ~+0.12.
     Small, but it is the ecological signature of a real habitat driver.
   - Implication: census crop-type history is genuine habitat information for open-country birds, at a scale LUH
     misses, back to 1850. It is diluted at the community level, so its value is mainly in the species readout.
     E048 (DESK retrained with CropAT, 4 pairs): NO HELP. The backcast is slightly WORSE in all 8 estimates (~-0.02),
     with no change beyond the placebo. Likely CropAT's post-2006 crop-type shifts act as a clock that misdates the
     past. Keep CropAT for the species readout, not DESK's inputs.
   - E050: two more candidates add NOTHING beyond DESK's covariates for the guilds they should act on:
     - CRP county enrollment (1986-2008), for grassland birds;
     - a forest stand-age back-cast (USFS 2011 map), for forest birds. A woodland-bird hint does not meet the bar.
     - Likely reasons: the CRP map is collinear with the Plains crop and climate patterns; the back-cast's change is
       mostly uniform aging.
     - A real succession test needs observed forest-age history: FIA remeasurements, or Annual NLCD 1985+.
   - E052, OBSERVED land-cover change (Annual NLCD, satellite, 1990 -> 2019), beyond everything DESK sees:
     - small, guild-appropriate gains for open-habitat birds: grassland +0.011-0.018, wetland ~+0.025, woodland +0.013;
     - none for forest birds (cover fraction is probably not what they track: age and structure) or for
       human-modified species.
     - Rule fails (forest).
     - Even observed land cover at 27 km is a modest lever (~+0.01-0.03), about the size of census crop types.
   - E051: reservoir creation (92k NID dams by completion year) adds nothing for wetland birds.
   - TALLY of new covariates tested against everything DESK sees, in the guilds they should act on:
     - census crop types +0.015-0.02 (grassland);
     - observed land cover +0.01-0.03 (open-habitat guilds);
     - CRP 0; forest-age back-cast 0; reservoirs 0.
     - At 27 km, better land-use inputs move local community-change prediction by a few hundredths at most.
   - E053/E053b, IS THE 27 km GRID THE LIMIT? Land cover inside each BBS route's 400 m buffer vs the whole 27 km cell,
     both from Annual NLCD 1990 -> 2019, predicting route-level change in 369 species, beyond all DESK covariates:
     - 2,121 routes, all scored out-of-fold. Forest birds (70 species, the stable case): route ~ cell ~ 0.
     - Grassland: E053's route advantage did not replicate (+0.009, CI includes 0).
     - Shrubland: a route advantage (+0.027 in E053b, +0.145 with a 32-PC baseline in E053c). It swings with the
       baseline, its moved-route nulls are nearly as large, and it vanishes on routes with the same observer in both
       epochs. An observer-turnover artifact fits all three (likely).
     - Wetland and human-modified: unreadable under either baseline (corrs past +-1). Likely too little resolvable
       route-level change.
     - No sign that the grid's coarseness is what holds the covariate gains to a few hundredths. Steers only; not
       verified.
2. DESK's TEMPORAL signal: after 1950 it is mostly the land-use management trend (fertilizer, irrigation) -- partly
   real intensification (it scales with cropland), partly a clock; that trend is flat before 1950. DESK still moves
   before 1950 (~0.03/decade vs ~0.05 after) through land-use classes, climate and population, in a different
   direction -- unvalidated (no surveys before 1966).
3. The SPACE-FOR-TIME LEAN (backcasts drifting toward places that look like the past today) is STRUCTURAL: holding
   temperature, management, or both static at training never removes it at the matched training state -- it moves
   into the remaining channels and the backcast loses skill (E038b, E041, E042, all measured on one yardstick).
   (The "climate freeze" -- encoding backcast years with climate held at a reference -- is a DIAGNOSTIC, not a
   remedy; you would not use it in production.) A remedy would have to give DESK temporal supervision or change the
   readout. E043 (your choice) over-sampled same-place, cross-year pairs in DESK's training loss. NO LEVER: the
   late-state backcast moved +0.001 to +0.005 (bar +0.02), at a spatial cost in all four runs. Likely reason:
   DESK's dominant stabilizing term already supervises each cell-year's own target, so extra same-place pairs add
   little temporal constraint.
4. BASELINES are repaired: the covariate and spacetime GPs now fit sensibly; a spacetime GP on neighbours' observed
   abundance beats DESK on CHANGE at held-out places. It extrapolates each species' observed regional trend, whatever
   its cause (habitat or dynamics, not separated -- M6).
YOUR ANSWERS (2026-10-11 ~11:25): the sealed species set stays in use for now (revisit later); 32 dimensions is a
reasonable eventual readout size, but production choices wait for the end of the understanding phase; no climate
freeze in production; go ahead with the same-place pair over-sampling (E043, queued); address the skeptic's points
(E039g, queued).

FOR YOUR ATTENTION (2026-10-11 morning)
- I stopped mid-task last night (~22:00) without a waiter armed; the GPU sat idle 02:25-06:20 and the CPU ~22:20-06:20.
  Resumed when you asked. The waiter rule is in memory and PROTOCOL; the slip was ending a turn on an edit.
- SEALING GAP (escalation): the validation suite reads the raw survey data, not the research cache that keeps the
  sealed half of the species apart, so every suite run so far (E001, E006) graded the sealed species too (only
  aggregate medians were seen). From now on every suite run grades only the 273 dev species
  (research/specs/dev_species.txt). Your call whether the sealed set is still fit for its final-confirmation role.
- The GP repairs passed in their harnesses, but a skeptic found the suite port was NOT running the validated
  baselines (shapes fitted on ~460 mostly rare species instead of the 136 validated; the covariate GP carried each
  place's lasting quirk onto withheld decades). Both fixed, with tests; the suite re-run (E039) restarted 06:34.
  It also found that GPs with place memory get credit on "change at trained places" from sharing the modern epoch's
  noise -- probably part of why the old spacetime GP "beat" DESK there (E006 corrected).
- WITHDRAWN (07:40): the E038 "production candidate" escalation. The skeptic showed I graded the static-temperature
  models against a different yardstick: the "analog direction" is built from each model's cached covariates, which
  for the static models had no temperature change -- that alone halves even the controls' lean. On one common
  yardstick the lean falls only to 48-85% of the controls', and at matched training state (the late epoch) the
  static models lean MORE and match real change LESS in 4 of 4. The inference-time climate freeze already does at
  least as well. RE-MEASURED (E038b, one common yardstick, both training states, paired CIs): static temperature
  trims the lean at the selected epochs (to 0.5-0.75 of the controls') but leans MORE at the late state, and costs a
  little backcast skill (7 of 8 paired differences negative, 3 significant). Fails its pre-registered rule: NOT a
  candidate. WHY (likely): temperature is one route to the analog lean; when it is closed, land-use pattern and
  built-up channels carry it.

SOLID
- The local machine reproduces TACC for DESK, the spacetime GP, no-change and the oracle. The covariate-GP baseline
  does NOT reproduce (fragile optimizer; defect A13) and is not relied on -- it is being repaired (E037, below).
- DESK's temporal change on held-out cells correlates 0.41 with the true community change in trained years and
  0.21 in withheld decades, and it moves more than that accuracy justifies (a calibrated DESK would be 0.62x /
  0.37x as large). In absolute terms it moves less than the communities did; the problem is direction.
- Read through the age model's readout, DESK's change into withheld decades predicts species change worse than
  assuming no change.
- ESK (64 dims) keeps the regionally shared part of a 40-year change about as well as comparable spatial
  structure; it drops route-level idiosyncrasy, temporal or spatial. Truncation is not selectively discarding time.

TERMS (fixed 2026-10-10 after you asked):
- "Change" is always between multi-year AVERAGES (6-21-year windows strip weather and survey noise), never year to
  year. Gaps between window centres: readout/oracle (E013, E030) and ESK retention 1966-86 vs 2005-25, ~40 years;
  corrected suite (E006) 1966-75 vs 2005-25, ~45; span/reach backcasts (E023-E029) a 6-year window vs the first ten
  trained years, ~8-32 (1966-71 vs 1996-2005 = 32); reach curve (E019, E025) vs 2005-25, ~22-46; observer fading
  (E028) 2005-14 vs 2016-25, 11. "Decadal" below means over one to several decades, mostly 20-45 years.
- "Regional" = smooth over a few hundred km: the trend placebo and E030's regional field are Gaussian-smooth maps at a
  300 km scale (places 300 km apart correlate ~0.6, 600 km ~0.14); E028's surface uses 150 km. "Local" = one 27 km
  cell (one or a few BBS routes). "Continental" = the whole grid.

NEW SINCE YESTERDAY EVENING (tentative unless noted; each finding carries WHY: established / likely / unknown)
- Why the readout of change does no better than "no change" -- even for the TRUE community (E030, E011, E013).
  Squared-error skill against no change = (variance of the predicted change / variance of the true change) x (2k - 1),
  where k is the slope of truth on prediction: a prediction must have k > 1/2 to win, whatever its correlation.
  WHY (established, one run): three factors multiply --
  - information: the true community's ~40-year change (1966-86 vs 2005-25) carries only ~5-7% of a species' change
    over the same interval (corr 0.23-0.27). Species change is strongly regional (a 300 km smooth map of the species'
    own change, fitted on training places, explains 23% on held-out places), but species-specific: the rest of the
    community does not share it;
  - scaling: the readout's coefficients come from differences between places, which are large, so applied to change
    over time they predict changes ~2x too big for their accuracy (k ~0.5, the break-even point);
  - noise: the oracle's half-window features carry survey noise the large coefficients amplify (k -> ~0.4).
  Rare species carry almost no information and pull the median down. DESK in withheld decades has almost none
  (corr 0.07, k 0.19); in trained years it reaches corr 0.22, k 0.70 and beats no change.
  Is the low information a defect? Mostly not (E031, your test): about 9 in 10 parts of a species' 40-year change
  are not shared even with the five species it lives with most closely -- its own story, which your population model
  is meant to carry. The ~1 in 10 that is shared is real (random species share ~0.3%; it survives where the same
  observer counted throughout) and the community summary already carries most of it (5-7% vs 8-9%). So ESK's loss to
  no change in the score comes from over-scaling and noise, not from missing habitat. Note: the score asks habitat to
  predict a species' whole change, which your model does not ask of it.
- WHERE DESK'S BACKCAST GOES WRONG (E035b/E035c, replicated on four models): re-encoding the withheld years
  with land use or climate swapped for their later values splits DESK's 1966-86 change in two:
  - the LAND-USE-driven part is most of it (~90% of its size) and carries ALL of its correlation with the real change
    (0.25), with no lean toward analog places;
  - the CLIMATE-driven part tracks nothing real (0.03-0.05) and carries the whole lean toward places that have that
    climate today (+0.25 to +0.31, far beyond its null).
  WHY (established across four models): DESK reads a place's climate change through spatial climate gradients -- "become
  the place that has this climate now" -- and real communities did not change that way over 1966-2025. Its land-use
  reading points the right way (still over-scaled ~3x). Open: whether the land-use signal is each place's own
  land-use change or a regional land-use trend. At the species level too the land-use part carries the signal
  (0.06-0.10 vs 0.03); the land-use part has a small lean of its own in two of four models (+0.04, +0.09).
  Sharper (E035d, per covariate family, four models): the lean is DESK's TEMPERATURE reading (+0.29, corr with the
  real change ~0.01) -- "a place that warmed should look like places that are warm today"; moisture leans less
  (+0.15, a little species signal); population and built-up area carry little. The real signal sits in the LUH
  land-use stream (corr 0.24-0.36), mostly in its regional (300 km) change.
  SKEPTIC (2026-10-10 evening) -- partly WITHDRAWN: I called that stream "land-use classes", but only 12 of its 33
  channels are class fractions; the rest are management channels (fertilizer, irrigation, harvest, protection) and
  the age/biomass of secondary land, and those dominate its change -- fertilizer on C3 annual crops alone ~31%,
  secondary-land age ~24% -- with more than half of it the same shift everywhere. A uniform shift is a clock: DESK may
  be reading TIME from these trends and applying it through place-specific responses (as E027c found for the
  covariates in general), not responding to where land use changed. The 300 km "regional" win is then partly built
  in, since smoothing leaves a uniform shift intact.
  RESULT (E035e, four models): mostly a near-uniform trend (read largely as intensification -- E035g below). Re-encoding with only the continent-wide average land-use shift (the
  same change everywhere) reproduces ~90% of the land-use part's match with real change (0.23-0.32 of 0.24-0.36);
  the management channels carry most of it (0.22-0.33), the class fractions less (0.09-0.16); a placebo that keeps
  only how far each place moves along one common direction gets about half. A real but smaller part follows WHERE
  land use changed more or less than average (0.14-0.27), and that part leans toward analog places. It is not one
  region's (the west contributes little; leaving any zone out changes it by <= 0.05).
  WHY (established across four models): DESK's real signal comes through a near-UNIFORM, continent-wide shift in
  land-use management (chiefly fertilizer, irrigation, the age of secondary land), turned into place-specific change
  by place-specific responses (the "kind of place x year" structure E027c found in the covariates). NOT yet
  established: whether DESK treats that shift as a mere clock (time) or as real intensification -- fertilizer use did
  rise almost everywhere and plausibly matters to farmland birds. They differ in WHERE the response sits: a clock
  responds even in forest places with almost no cropland, an intensification effect scales with cropland (test
  E035g, next). Precipitation alone carries no consistent lean, so the moisture lean came from the
  temperature-derived channels: DESK's space-for-time error is its THERMAL reading.
  PRODUCTION (E035f/g, all 3,902 places, one model): holding channel groups at their 1950 values, DESK's movement
  since 1950 is mostly its reading of land-use MANAGEMENT -- fertilizer + irrigation alone about half of it, all
  land use two-thirds or more; the class fractions and temperature (180 channels) barely matter. Those channels are
  flat from 1902 to ~1950 (fertilizer on C3 annual crops 0-2 kg/ha until 1950, 77 by 2025), so production's
  pre-1950 movement lacks that signal. CORRECTED (E041 movement profile): production is NOT frozen before 1950 -- it
  moves ~0.03 per decade from 1902 to 1950 vs ~0.05 after, driven by the channels that change then (classes,
  climate, population) -- but in a different direction from its post-1950 movement, which is why its distance from
  TODAY stops growing (the "saturation").
  Clock or intensification? (E035g, rule fixed in advance): the response scales with cropland (correlation +0.65;
  2.5x larger in the most-cropped third of places than the least) -> mostly read as agricultural INTENSIFICATION,
  real habitat change; but places with almost no cropland still respond at ~40% of the typical size (a clock-like part,
  or neighbours through DESK's 5x5 view -- not separated).
  WHAT THIS MEANS FOR 1902-1939 (likely): intensification really did not happen before 1950, so a backcast that stops
  there is partly RIGHT. What DESK may miss is the land-use CLASS change that dominated early change (prairie
  conversion, farm abandonment, regrowth): its class reading tracks real change only weakly. Next (E041, GPU): DESK
  trained with the management channels held at each place's average -- can it learn the class-driven change?
  RESULT so far (E041, start year 1996, two seeds): spatial fit unchanged; backcast skill drops -- to about half at
  the late state (0.144 -> 0.070, 0.133 -> 0.061), less at the selected epochs (0.120 -> 0.073, 0.113 -> 0.100) --
  almost all in the east; the Great Plains and west hold up. DESK compensates partly: on a common yardstick (M28)
  its land-use CLASS reading gets significantly more truthful (+0.06 / +0.10) and so does its CLIMATE reading
  (+0.07 / +0.08, with less lean), the full backcast's lean is unchanged, but it loses the clock's real post-1950
  signal (full match -0.02 / -0.04). (An earlier "leans harder on temperature" was a yardstick artifact.) At the
  matched LATE state the whole backcast is clearly worse (match -0.07 / -0.03) and leans MORE (+0.10 / +0.07): with
  the management signal gone, the lean-carrying climate part makes up more of it -- the same route substitution as
  static temperature (E038b). Restricting DESK's inputs moves the space-for-time lean around; it does not remove it.
  So without the management trend DESK does not fully recover its skill,
  and it gains NO movement before 1950 (same as the controls): removing the trend costs post-1950 skill and buys
  nothing for 1902-1939. The management trend is not a defect to remove on this evidence. Replication at the 1986
  start running; E042 (temperature AND management static) queued.
  This is the first concrete target for extending extrapolation: a backcast in which temperature does not drive change.
  Training-time version (E038, four runs): DESK retrained with each place's temperature held at its own 1966-2025
  average (temperature can say where a place is, not when). No loss of spatial fit (at or below every control);
  backcast skill slightly lower (7 of 8 comparisons, mean -0.013, within seed spread); on the first model the
  backcast's lean toward analog places HALVES (+0.153 -> +0.072) while its match with real change barely moves (0.243
  -> 0.233); on all four it seemed to fall to 30-47% of the controls' -- WITHDRAWN: measured against a yardstick
  that changed with the treatment (above).
- Repairing the covariate GP (E037, your request; criteria fixed in advance: healthy fit, sane predictions, and on
  NEW places at least as good as a plain linear covariate model, within 0.01). WHY it failed (established): it was
  fitted on patches of neighbouring places, each patch sitting at its own regional level; centred on the continental
  average, the only way the fit could absorb a patch-wide offset was a nearly flat kernel with a hugely inflated scale
  (26x the data's variance) -- in effect a local polynomial that extrapolates wildly. Fitting on random places from
  across the continent fixed that (scale 0.65x, predictive variance ~1x, skill on new places -0.07 -> +0.28) but left
  it 0.011 behind the linear model (+0.305): it still trusts each training place's lasting quirks, because ~30 years
  of the same place look like 30 independent observations. Round 3: an explicit same-place term (repeated years of
  one place count as one place's quirk), fitted on random places with a few years each, PASSES all four criteria
  and REPLICATES on a second seed (beats the linear covariate model by +0.017 / +0.026 on new places). Fitting
  whole places instead (30 years each) ties the linear model: the fit then learns year-to-year wiggles within places.
  Ported into the suite as an option, with tests, plus a "covariate_linear" arm -- the linear model the age model
  uses, on covariates instead of DESK features: the clean representation comparison.
- Repairing the spacetime GPs (E040, same approach, fixed criteria vs simple references). WHY they failed
  (established): fitted on patches, each place's ~30 very persistent years dominated the fit, and one smooth field
  cannot be both that persistent within a place and only moderately alike between neighbours -- so the spatial
  range collapsed to zero and new places got the continental average (skill 0 vs +0.28 for a nearest-neighbour
  average). Fitting on random places fixes the sum kernel outright; the product kernel also needs the same-place
  term (its spatial field then lengthens from 77 to ~390 km). Both pass all five criteria and replicate on a second
  seed: on new places they beat the nearest-neighbour average (+0.03 to +0.06), on trained places' withheld decades
  they beat each place's own average (+0.06 to +0.07), because a place's lasting quirk fades over ~45-50 years
  rather than persisting forever. Ported (--st-mode repaired).
- DOES DESK ADD ANYTHING OVER ITS OWN INPUTS? (E039, both models, all baselines repaired; skeptic-reviewed).
  CORRECTED: my first reading ("about a tie") was a median diluted by rare species with no signal. For
  well-observed species, read through the same linear readout, DESK beats a linear projection of its own 302
  covariates consistently -- +0.03 to +0.04 of level skill at held-out places, favoured by 84-97% of those species
  (151 of 204 species overall) -- but modestly: ~95% of DESK's level information is linear in the covariates, and a
  nonlinear covariate GP closes most of the rest. So DESK is roughly "a good nonlinear model of a place's own
  covariates". On change both are near zero; a spacetime GP on neighbours' observed abundance beats DESK on change
  (-0.039 / -0.032) -- each species' own regional trend (G3), not habitat.
  IN THE AGE MODEL'S REGIME (E039b: 24 dimensions, the same linear readout, training data cut to 10% and 3%):
  DESK's 24 dimensions beat the top 24 principal components of its covariates by ~0.03-0.06 of level skill for
  well-observed species (75-92% of them), the SAME at every data size -- so it is information, not just compression:
  DESK orders the covariates' information by what matters to bird communities, a better compact habitat summary.
  With all 291 components and plenty of data the covariates nearly catch up for the commonest species; with little
  data they overfit and fall well behind. Side finding: DESK's 64 dimensions beat its 24 by ~0.02-0.04 -- the age
  model's cut at 24 leaves some skill unused (worth your attention for the downstream).
  AGAINST STANDARD SDM INPUTS (E039c/d, your request): DESK's 19-24 dimensions beat the 19 WorldClim BIOCLIM
  variables (built from the same states, same smoothing) by +0.06 / +0.16 of level skill for species at 5-20% /
  >= 20% prevalence (84-100% of species); adding the 12 land-cover classes recovers about half the gap; giving bioclim
  hump-shaped (quadratic) responses recovers more. Against the strongest SDM-style set tried -- bioclim + squared terms
  + land cover, 50 variables -- DESK still leads, by ~0.02 for moderately common species and ~0.06 for common ones
  (66-97% of species), with 24 dimensions, and slightly more with little data. Both models agree.
  CORRECTION (E039f): I had given bioclim DESK's ~10-year output smoothing, which the other arms did not get; with
  the smoothing matched, bioclim scores 0.01-0.03 lower and DESK's lead over that strongest set is ~0.03 / ~0.08
  (the smoothed numbers above are what multi-year climate normals would give an SDM user).
  Spatial context does not explain DESK's edge (E039e): giving the covariate readout each place's 5x5-neighbourhood
  covariates (DESK's convolution footprint) adds only ~0.01; DESK still leads it by +0.03 to +0.05 for
  well-observed species. The edge is mostly how DESK orders the covariates' information, not what it sees nearby.
- Readout scaling (E033, E033b, skeptic-reviewed):
  - shrinking DESK's change by a factor learned on the BBS years (~1/3) stops the readout doing harm -- but only
    because there is almost no information to lose: DESK's change is about as uninformative in its own trained years
    as in the withheld decades, and almost any shrink would avoid harm (WHY established: the information is flat);
  - a readout that learns, inside each place, how a species moves with DESK's change looks much better (correlation
    0.22-0.42) -- but an adequate placebo with NO DESK (smooth position x year, 96 features) beats it by 0.07-0.12, and
    DESK adds ~0.01 on top. WHY (established for two models; confirmed over five placebo draws, E033c): it learns
    each species' own regional trend and extrapolates it backwards -- the population model's job (G3), not habitat.
- What production DESK does before 1940 (E035, descriptive, skeptic-reviewed): its backcast movement is mostly driven by
  LAND USE (HYDE, LUH, BUI), not climate -- swapping 1920s land use for modern values cuts it from 0.33 to 0.08 -- and
  it SATURATES: z's distance from today stops growing around 1960 while the covariates keep drifting, so 1920-39 looks
  like the 1950s. Its lean toward places that look like the past today is marginal for 1920-39 (p ~0.05; +0.05
  measured like-for-like vs about -0.01 for real communities) and clear for 1966-86 (in-sample). The space-for-time
  error sits mostly in the small CLIMATE-driven part. WHY it saturates: unknown. Next (E035b): the same swap on a
  model with withheld decades -- which part of DESK's change tracks the real change, and which carries the error?
- The "kind of place x year" trend vs observers (E034b, skeptic-reviewed): the trend-predictable part of 2005-14 ->
  2016-25 change is the SAME where observers stayed or changed; a change of observer only adds change the trend cannot
  predict. So the trend is not an observer artifact -- but ecology vs a drift common to all observers (e.g. ageing) is
  not separated (E034c: the predictable part is ~27% larger where observers had counted longer -- suggestive of some
  within-observer drift, no CI yet). About a third of the trend's apparent skill was in-time smoothing.
  At a coarser grain (81 km, E034a) a place's own covariate change still adds nothing clear (+0.06 / +0.01 / 0.00, CIs
  spanning 0): grain is not an obvious explanation for why covariate change carries so little.
- Training span vs reach (your question), E023 / E024 / E024b:
  - cheap linear surrogates: at a fixed reach a SHORTER span next to the target backcasts better (strong: 0.22 vs
    0.08 for 20- vs 40-year spans);
  - DESK: two seeds of a 20-year span next to the target beat the 40-year model by +0.038 [0.020, 0.057] at the 1986
    start year (pre-registered rule met after a second seed overturned my first "no"); nothing at the 1996 start
    (one seed).
  - TIME-WEIGHTED DESK (E029): training on all years but weighting the years near the target (weights halving every
    10 or 20 years away from 1986) recovers about three quarters of that gain: +0.029 [0.013, 0.041] and +0.035
    [0.020, 0.047] over the same seeds unweighted, also at epoch 490, with no loss of spatial fit. DECISION FOR YOU
    (escalation): its production analog is weights rising toward 1966 for the 1900-1940 backcast. One start year, two
    seeds per weighting. Downstream checks: slightly better species readout (+0.01), slightly less over-confident,
    and about a third less pull toward climate analogs -- no downside found.
    REPLICATION at the 1996 start (E029b, your choice): NULL by the pre-set rule (-0.003 at the selected epochs,
    muddied by one seed selected at epoch 74); at matched training state (epoch 490, what production ships) +0.018
    [0.002, 0.035]. So: not yet a general lever; a small late-state benefit (+0.02-0.03 at both starts) is suggested
    but not established.
    SETTLED (E032, four seeds per arm, final epoch): +0.032 [0.012, 0.051] at the 1986 start, +0.016 [0.001, 0.029]
    at the 1996 start -- real but small and setup-specific (the pre-set bar of +0.02 at both starts is not met).
    Not a production candidate on this evidence.
  WHY (likely, not shown for DESK): decadal trends drift; a long span averages in later decades whose trends do not
  describe the earlier ones (shown for the surrogates, E023).
  RE-SCORED ON THE FAIR-RIVAL YARDSTICK (E056, 2026-10-11, prompted by your "full span carries more signal" point):
  - 20-year spans vs long spans: no difference beyond the trend placebo (-0.009 to +0.004). The raw proximity lean
    is +0.001 to +0.019, and short spans cost ~0.01 of spatial skill.
  - Time weighting: ~+0.01 raw at the late state, ~0 beyond the placebo.
  - So neither span nor proximity moves DESK beyond the regional trend. Their earlier raw gains were a near-target
    trend the placebo also has (likely).
  - Your case, adding the decades just before the target, is the withheld window itself: untestable inside BBS. It
    needs a deeper series. Per M29 (user, 2026-10-12): not pre-1966 CBC (its own curators start trends in 1966), but
    Grinnell resurveys, Breeding Bird Census plots or historical range edges.
- Where the covariates' temporal signal is (E027, E027b-d, two skeptic rounds): their backcast skill is largely matched
  by a placebo using them only as a STATIC position (a place's average environment) times LINEAR YEAR. A place's own
  covariate change adds at most ~0.05 in correlation (two independent splits); the direction the continent's
  covariates moved does not matter; slow 30-year history adds nothing; DESK already holds most of what the surrogates
  extract.
  WHY (unknown). Candidates: local covariate change at 27 km is small or smoothed relative to its noise (interpolated
  products: decadal HYDE, LUH, coarse climate); responses are lagged and heterogeneous (slow averages did not help);
  the drivers are not in the covariates. Whether the linear trend is habitat or observation is also open.
- DESK keeps changing after the epoch we select (E025): by epoch 490 its backcast direction is better (+0.01 to
  +0.08), its pull toward climate analogs halves (+0.14-0.21 -> +0.03-0.09), but it moves even more than its accuracy
  justifies (squared error vs no change 1.27 -> 1.47 at 27.5 years) and fits held-out places worse. CORRECTION
  (2026-10-10): production ships the state at EPOCH 200 of a 500-epoch schedule (production.json), not epoch 490;
  at epoch 200 the logged direction is near its late plateau and the movement is in between.
  WHY (partly likely): early in training DESK has learned the spatial mapping and moves along it (space-for-time);
  later epochs add movement in other directions. Why those later directions are better, and why movement keeps
  outgrowing accuracy, is unknown.
- The metric term's pair count is not a lever (E026: 4,096 and 262,144 pairs backcast like 65,536).
  WHY (likely): the term's gradient noise was not what limited temporal learning -- the logged withheld-decade
  direction accuracy was the same in every run from epoch ~100 on.
- The corrected validation suite (E006), your three concerns: on HELD-OUT places DESK is the best arm (beats the
  spacetime GPs and the true-community oracle), but still ~no change on species change; the oracle is also ~no change
  (WHY: above). DESK "loses to the spacetime GP" only on trained places, where the GP's degenerate fit predicts that
  each place's modern quirks were smaller in the past.
  WHY (established, E028/E028b, three eras, skeptic-reviewed): where observers changed, 55-60% of that fading is the
  change of observer; the rest (~7-8% per 11 years) is real fading of local deviations of unknown cause -- it does not
  track local covariate change. The covariate GP's fit is broken (A13).
- Replications on fresh seeds of the current code: the decline with reach holds (seed spread 0.02-0.03); the pull
  toward climate analogs, the readout bottleneck and the over-narrow intervals all hold. Current-code retrains track
  the direction of change better than the archived ones at long reach, but are no better than no change in squared
  error. WHY the archived models are worse: unknown -- not the metric-pair count (E026); epoch selection explains at
  most part of it.

TENTATIVE (earlier, still standing)
- As DESK reaches further back, its useful temporal signal fades and its movement increasingly follows
  space-for-time (alignment with the climate-analog direction +0.02 / +0.16 / +0.21 at 1-10 / 11-20 / 21-30 years of
  reach; real communities ~0 or slightly against it).
- Space-for-time fails at the covariate stage: coefficients learned by comparing places at one time predict none of
  the decadal change. Regional trend persistence predicts some of it; linear covariate maps add only +0.03-0.05.
- The readout cannot tell when it is guessing: change intervals far too narrow, and leverage does not flag the
  errors.
- About 30% of what every change metric counts as "true change" is observer turnover.
- Rank 64 would add little over 24.

OVERTURNED (kept so the corrections are visible)
- "A change-fitted readout captures 25-29% of species change from community change": spatial interpolation.
- "The true community's change, read through spatial betas, is worse than no change": errors-in-variables; ~zero.
- "The split kernel works through DESK's temporal information": a placebo does as well.
- "ESK keeps 23% of temporal change because time is small": confounded by the route nugget and survey effort.
- "DESK's backcast errors are not pulled toward space-for-time analogs": the corrected test shows they are.
- "Intervals ~3x too narrow": ~2x on a stable floor.
- "Train DESK on within-cell covariate variation": the within map was a regional-trend extrapolator.
- "Longer reach fails because the tempho models have fewer training years" (my working assumption before E023/E024):
  true for linear surrogates; for DESK at most a small effect.
- "With all history, slow and nonlinear covariate features keep backcast skill where linear maps lose it" (E027,
  first reading): reference, random-draw and warm-up artifacts.
- "A shared continental covariate trajectory x environment carries the signal" (E027c, first reading): survey
  composition plus linear year; random directions do as well.

WHAT THIS DOES AND DOES NOT TELL US
- Everything above characterizes the current DESK, its readout, and linear or simple nonlinear surrogates on 10-50
  training years. It diagnoses where today's system loses temporal signal; it is not a limit on how well
  extrapolation in time can work.
- Fallbacks to keep in view, NOT the plan: recalibrating DESK's temporal deviations on its trained years; holding
  pre-BBS habitat near its BBS-era state with uncertainty that grows with reach.

OPEN QUESTIONS (the plan: what limits extrapolation in time, and what would extend it)
1. Training state: later epochs give better direction and less space-for-time pull but more over-movement (E025);
   gradient noise in the metric term is not a lever (E026). What sets the over-movement, and can it be calibrated
   out downstream without losing the direction gain?
2. Covariate content (T5): the covariates' decadal temporal signal at this grain is small beyond "kind of place x
   linear year" (E027c/d: at most ~0.05 in correlation). What inputs or grain would carry it, and is the linear trend habitat or observation?
3. The readout: even the true community's change, read through the spatial readout, predicts species change no
   better than no change. A readout that lets local deviations fade with reach (E028: half of that fading is real)
   is a candidate.
4. The scoreable target: observer turnover inflates "true change" and rewards models that predict a return to the
   regional mean; grade on the observer-adjusted part.
The age-model sensitivity fit comes last, once there are candidate backcasts worth comparing.

## GATE 1 REPORT (2026-10-10 ~03:24, draft for your review; all UNVERIFIED in the protocol's sense unless noted)
1. Your three concerns, on the corrected validation suite (E006: audit fixes A1-A8, tempho1995 and the base model):
   - "DESK loses to the spacetime GP": only on TRAINED cells in withheld decades, where the GP's (degenerate) fit
     pulls each cell's modern quirks toward its region -- and about half of that advantage is observer turnover
     (E028, replicated in three eras). On HELD-OUT cells, the setting a backcast needs, DESK beats or ties both
     spacetime GPs: +0.025 to +0.07 in trained years, +0.049 / ~0 in withheld decades.
   - "DESK ~ no change on change": still true: +0.010 (held-out cells, trained years), -0.008 / -0.010 (withheld
     decades).
   - "The oracle is weak too": yes. The oracle built from the TRUE community is also at or below no change (-0.002,
     -0.014, -0.029), and DESK beats it on held-out cells (+0.025 to +0.034). Why (E030): a species' decadal change is
     mostly its own regional dynamics -- the community's change carries ~5-7% of it; the readout's coefficients,
     learned from differences between places, over-scale what little it carries (truth-on-prediction slope ~0.5, the
     break-even); and the oracle's survey noise, amplified by those coefficients, tips it below no change.
   - The covariate-GP baseline is broken in the corrected suite too (absurd predictions; A13) and must not be quoted.
2. Which stages lose temporal signal:
   - ESK: no selective loss; it keeps the regionally coherent part of change like comparable spatial structure.
   - DESK: direction of withheld-decade change correlates 0.2-0.4 with the truth, falling with reach; it moves more
     than its accuracy justifies, leans toward climate analogs, and captures ~5-9% of the noise-free temporal
     variance. Training longer (production ships final-epoch weights) improves direction a little and over-movement
     more (E025).
   - Readout: even the true community's change, read through the spatial readout, correlates only ~0.23 with species
     change and is no better than no change in squared error; DESK's reaches 0.07-0.10.
   - Inputs: a place's own covariate change adds little to a backcast -- at most ~0.05 in correlation beyond "kind of
     place x linear year" (E027c + skeptic + a second split, E027d). This is the sharpest constraint found so far: at
     27 km and decadal scales, the covariates as they are carry little temporal signal beyond a trend surface.
3. How much real change there is: most species have resolvable change between epochs (323 / 401 and 249 / 372),
   noise ceilings 0.62-0.67; ~30% of change energy is observer turnover (E021); local deviations from the regional
   surface also fade for real, by ~7-8% per 11 years, unrelated to local covariate change (E028).
4. What I would look at next (your call at this gate): (a) T5 directly -- which drivers or which grain would carry
   temporal signal the current covariates do not (e.g. finer land-use or habitat products, or a coarser grain where
   local noise averages out), and whether the "kind of place x year" trend is habitat or observation; (b) the
   readout, which loses the most; (c) over-movement and epoch selection (production ships the epoch-200 state,
   between the selected epochs and epoch 490); (d) the second span seed (E024b, running) and the metric-pair runs (E026, running) settle the
   time-weighting and optimization questions. (An "environment x continental trajectory" backcast model, which I
   floated earlier tonight, is withdrawn: it reduces to extrapolating a linear trend.)

---- dated log (newest first) ----

## 2026-10-09 night: overnight run (E024 decided; E006, E019b, E027, E028 in; E025-E027b running)
- E024 (DESK on 20-year spans): pre-registered rule NOT met -> no time-weighted DESK. E023b table in E024.md.
- E019b + seed re-runs of E013/E017/E018/E020: everything replicates on the current code (records updated).
- E006 tempho1995 (corrected suite): desk -0.008 / oracle -0.014 (TIME), desk -0.010 / oracle -0.029 (SPACE_TIME) vs
  no change; spacetime_sum +0.119 on TIME is a degenerate damped-persistence fit; covariate GP broken (A13).
- E028: damped persistence by observer continuity: slope -0.182 (observers replaced) vs -0.084 (unchanged).
- E027: slow / nonlinear covariate features hold up at long spans (+0.07 to +0.11 over linear+trend), not at short.
- Training logs: selected epochs 65-162; withheld-decade direction flat-to-rising and movement doubling to epoch 490
  -> E025 (late states); archived models trained with 4,096 metric pairs vs 65,536 now -> E026.
- Archived DESK sweep logs: 16 optimizer steps per epoch (tiles32) and heavier metric weights did not raise held-out
  direction accuracy in trained years (weak evidence against "more steps" for T11).
- Skeptic round (~00:00): E024's "no" rests on one window and one seed (pooled +0.03 at T0 1986) -> second seed of the
  1986-2005 span queued; E027's headline withdrawn (weak 64-PC reference, lucky RFF draw, lag30 warm-up, trend decay)
  -> E027c; E028 survives (observer share ~57% disattenuated) -> earlier-era replication E028b.


## 2026-10-09 ~09:40: GATE 0 (reproduction and timing)
- Reproduction: DESK, the spacetime GP, no-change and the oracle reproduce TACC's reports (base 76/77 medians
  SAME or within CI; tempho1995 173/198, every miss in the covariate-GP baseline). The covariate GP's ARD
  shape fit is fragile across machines (different optimizer terminations; ~0.04 level skill) -- new defect
  A13. The three concerns are reproducible; DESK-vs-covariate-GP is not, until A13 is fixed.
- Timing: full suite 38 / 58 min locally vs ~3 h on TACC; a research cache ~60 s; readout arms 3-9 s;
  route-level variogram 18 s; DESK training reproduction (E010) queued behind the corrected suite.
- R1 measured directly (E014, replicated): ESK keeps the regionally coherent part of a 40-y change as well
  as structured spatial differences; it drops route-level idiosyncrasy. Strong form not supported.
- E016 (development run): covariate->community coefficients learned ACROSS space predict none of the
  temporal community change; within-cell coefficients predict it as well as DESK, better calibrated.

## 2026-10-09 ~09:00: the skeptic's review (verification step d) overturns three of the morning's readings
- R1 is OPEN again. Adjacent-cell pairs are different routes; temporal pairs mostly the same route. The
  spatial curve carries a between-route nugget (fits: 35% of the lag-1 difference, up to 70%) that ESK
  discards. Against the structured spatial part, temporal retention may be LOWER (your hypothesis's strong
  form), and the 40-y change is ~100 km of structured turnover, not "< 27 km". Next: measure the nugget
  directly from route-level counts (N1). The retention numbers also depend on years per half (fixing).
- "Level readout of community change is worse than no change": an errors-in-variables artifact for the
  TRUE community (it is ~0, not negative). Only DESK's change, read through level betas, harms the
  backcast -- that is DESK backcast error. E013 says the same with noise-free correlations: 0.23 (true
  community) vs 0.06 (DESK, inside the planted null band).
- The split kernel's gain is NOT shrinkage, and a placebo deviation block with no DESK content (regional
  position x linear time) matches it. So DESK's temporal deviations may add nothing; regional trends may
  be the whole effect (B8). Testing with a paired placebo arm.
- DESK's atlas correlations were uncentered; centered: 0.41 (trained), 0.21 (withheld decades).
- Robust and now the clearest result: DESK over-moves for its information (a calibrated DESK would be
  shrunk to 0.62x / 0.37x), and the gap to a perfect readout is ~85-90% DESK's backcast, not the readout.

## 2026-10-09 morning: first real-data results (ALL UNVERIFIED -- they steer, they do not conclude)
- Size, not direction (R1, E004): a cell's 40-year community change (1966-86 -> 2005-25) is smaller than the
  difference between adjacent 27-km cells. ESK r24 keeps 23% of it, and the same 23% of adjacent-cell
  differences; r64 keeps 39% of both. Truncation drops small differences, temporal or spatial alike.
- DESK (held-out cells, r24): its temporal change correlates 0.49 with the truth (slope 0.36) in trained
  years and 0.30 (slope 0.20) into withheld decades (tempho1995). Small spatial differences fare no better
  (adjacent cells: slope 0.18, corr 0.30). Given its correlation, DESK moves MORE than a calibrated
  predictor would, so "under-moves 2-4x" looks like the wrong diagnosis; the limit is information.
- The age-model readout (BLR, raw z r24, iid prior; E005) captures ~0 of resolvable dev-species change, and
  into withheld decades is significantly worse than no change (-0.016 [-0.029, -0.008]).
- The readout FORM looks like the larger loss (E011 development run): a beta fitted on LEVELS captures
  4-10% of available species change (negative into withheld decades); a beta fitted on CHANGE captures
  25-29% from the same features, and DESK's backcast change does as well as the observed community's. New
  hypotheses B7 (space-for-time at the readout) and D0 (DESK's information is not the first bottleneck).
  CORRECTED later the same morning (placebo arms): position alone captures 0.30 and the static community
  0.33, so the change-fitted 25-29% is spatial interpolation of where species changed, not species
  following community change. Survives (verified (b)+(c)): the level readout of community change is worse
  than no change into withheld decades, while for planted species whose truth is linear in z it
  recovers 98% -- the failure is the real species' space-for-time mismatch, not the instrument.
- Split kernel (E012): a separate deviation block with a long reference window turns the common-species
  backcast from worse than no change (-0.02/-0.04) to modestly better (+0.02/+0.04, CIs exclude 0), place-
  specific. The data give the deviation block a tenth of the level block's amplitude: it works mostly by
  NOT forcing spatial betas onto temporal change. A short (2011-2025) reference window loses the gain.
- Route turnover inflates the change budget by ~3% (dev species), so it is not a major factor (O3).
- Planted pilot: even a perfect-form readout shows change attenuation 0.2-0.75 depending on prevalence;
  below 10% prevalence the true change is <3% of the observed change variance (E1). E009 adds power.
- Queued: E009 planted v2, E011 change oracle x4, E010 DESK reproduction x2 (bcba64c8, ~100 min each).

## 2026-10-09 (data landed)
- TACC artifacts transferred and verified: 880 processed files (27 GB on TACC; history_vectors excluded)
  and 93 raw files (BBS 2026 release, AVONET, masks), every one matching its TACC sha256.
- The concerning TACC numbers, for the record (old suite, A1-diluted): out-of-community DESK-vs-no-change
  change skill +0.0002 (base), -0.0007 / -0.0037 / -0.0100 (tempho 1975/1985/1995); even the 96 community
  species score +0.009 / +0.011 / +0.002 / -0.004. Those reports were written by ef7b6a2, which is what
  the reproduction runs.
- 18 jobs queued: caches, ESK projections, reproductions, the first experiments (atlas, readout, planted,
  route ceiling), the corrected suite.

## 2026-10-08 (setup, no data yet)
- Local loop infrastructure is up in WSL2: systemd-managed pueue queue (jobs survive sessions; a 15-min
  canary ran to completion across wsl.exe exits), immutable code snapshots, append-only registry, tick
  journal/LOCK/STOP, generated STATE.md. Full test suite: 938 pass; 2 fail -- one needs data (ref grid),
  one was a real optimizer fragility, now fixed.
- GP-species suite audited and fixed (A1-A8): change skill now only over species whose change is
  resolvable above split-half noise; in-sample rows get exact leave-one-out (local GPs were reproducing
  their own modern noise); noise-only back-transform (local GPs predicted spurious declines); baseline
  scales from every training row; a persistent+changing spacetime kernel as the honest temporal rival;
  persistence baseline; block-safe bootstrap; oracle no longer reads withheld decades.
- Found while testing: the per-species amplitude likelihood is multimodal -- a single-start fit can
  "collapse" a species' amplitude where a far better optimum exists. Every scale fit now multi-starts.
  This also touches DESK's own BLR fits; how much it changes past "rare species collapse" readings is an
  open question for the corrected-suite run (E006).
- DESK training can resume from checkpoints; tempho epoch selection no longer reads withheld years (A9).
- Pre-registered before any data: E004 (atlas) and E005 (downstream-matched readout + kernel variants).

2026-10-11 21:22 -- "Is DESK over-fitting?" (user question; read from existing training logs plus E048's held-out
spatial skill. Steers only, not verified.)
- SPACE: yes, classically, after ~100-150 epochs.
  - Training z-MSE falls 0.17 -> 0.12 (epochs 100-500) while held-out-place error bottoms at epochs 82-114 and rises
    15-19% by epoch 500 (4 logs: E022 t1985 s1/s2, t1995 s1, E048 t1985 s1).
  - Early stopping on val_kernel catches most of it. On independent held-out cells, the selected state beats a
    linear covariate map by +0.03 (0.814-0.820). The late state only ties it (0.785-0.792; 3 E048 models,
    fair_rivals space_level).
- SELECTION NOISE:
  - Mid-training, val_kernel moves epoch to epoch by sd ~0.001, about 12% of its value and ~15x its reported
    over-cells SE. The noise dies as the learning rate decays (sd 0.00003 at epoch 400-500).
  - The chosen epoch sits ~15% below the median of its 20 neighbours. So best val_kernel is optimistic, and
    run-to-run comparisons at the 0.0005 level (E048's tolerance) are within this noise.
- TIME: over-fitting is not what limits it.
  - The late, more over-fit state backcasts community change as well or better (+0.01-0.03; E039n/E046).
  - Year-out error rises only 3-7% after its minimum.
  - Low-capacity linear rivals match DESK in time (E039i), and regularizing levers did nothing (E026, E043).
  - Likely limit: the information in the inputs.
- Implication: the selected state is best for space and the late state for time. No single snapshot gets both.
  Checkpoint averaging or selection on a smoothed val curve would make production's state less a matter of luck.

2026-10-11 21:26 -- "Is the metric loss stable? How reliable was E026?" (user question; existing logs only; steers)
- The metric term trains smoothly. Its training value falls steadily (0.0039 -> 0.0021 over epochs 60-500), and it
  is ~9% of the total loss throughout.
- The epoch-to-epoch instability is NOT from the metric term's pair sampling. Mid-training val_kernel jitter
  (epochs 101-200) is the same at every pair count, although 64x more pairs cut that term's gradient noise ~8x:
  - 4,096 pairs: 14.6%
  - 65,536 pairs: 11.9% / 13.1% (two seeds)
  - 262,144 pairs: 14.0%
- The jitter tracks the learning rate (likely the step size):
  - 12-15% at lr 1.4e-3 to 1.1e-3 (epochs 100-200);
  - ~5% at 6e-4 (200-300);
  - <1% at <=2.2e-4 (400-500).
  - DESK takes one optimizer step per epoch, so each epoch is one full Adam step.
  - Random input-stream masking may add to it (untested).
  - The selected epochs (104-165) fall in the noisiest phase.
- E026's reliability:
  - Sound for what it tested. It found no effect bigger than the ~0.02-0.03 seed spread across a 64x range, on two
    backcast measures, at both the selected and late states. The training logs agree (withheld direction cosine the
    same at every epoch), and so do the archived sweeps (16 steps per epoch, or a heavier metric weight, gave no gain).
  - Limits:
    - one seed per arm;
    - the T0 1996 split only;
    - the older estimands (raw backcast corr, not gain beyond the placebo);
    - not verified (no skeptic, no replication).
    - It tested gradient NOISE, not supervision CONTENT. Same-place cross-year pairs stay ~0.035% of pairs at any
      count. Content was tested separately by E043 (oversampling them): no lever.
- Implication: weight averaging (EMA/SWA) over the high-lr phase, or a lower peak lr, would make the selected state
  less a matter of luck. Untested.

2026-10-11 21:36 -- CORRECTION and follow-up (user questions)
- WITHDRAWN: "4,096 pairs over-fits slightly less". It rested on one run's late held-out-place error. The training
  error is the same in all four runs (0.114-0.116 at epochs 450-499), so the 4,096 run's lower held-out error
  (0.2518 vs 0.2578-0.2613) is a slightly smaller gap (2.17x vs 2.25-2.29x).
  - The user's reading is as consistent with the logs: 65k seed 2 is best on held-out years mid-training (0.2115 vs
    0.2146-0.2148).
  - But 65k seed 1 is not (0.2147), so the seed gap is as big as any arm difference.
  - Correct statement: no difference between pair counts can be read from one seed per arm.
- WHY THE STABILIZING TERM IS ~76% OF THE LOSS VALUE: it is the per-element MSE between DESK's EMA output and the
  precomputed ESK coordinates of each training cell-year's observed community.
  - Its weight (64 per element) is the historical 1.0 on a 64-dim sum, kept for numerical continuity (config
    comment), not tuned for time.
  - Its VALUE is large partly because single cell-year targets carry sampling noise that no model should fit
    (likely). Value share is not gradient share; the gradient shares were never measured.
  - Heavier metric weights (sweep mw20-100) did not improve held-out direction in trained years.
- OVER-FITTING REGIME: yes. Held-out-place error is past its minimum by the selected epoch; the gap is 1.5x there
  and 2.2x at epoch 500.
  - Dropout and weight decay do not change it (sweep). Narrower width and a heavier metric term do.
  - Whether less over-fitting helps WITHHELD-decade extrapolation was never tested: E054 (hidden width 64, both
    T0s) is pre-registered.

2026-10-12 00:25 -- over-fitting program, first results (E054, E055f/f2, E056; steering, not verified)
- Less capacity (width 64): over-fitting falls (late gap 1.65-1.75x vs 2.12-2.27x) and spatial skill rises
  (+0.004-0.012). But change prediction FALLS (-0.060 / -0.028 at the selected state). Capacity reduction is not the
  remedy.
- Seed ensemble (mean z of 4 seeds) vs the mean single seed:
  - late state +0.026-0.032 change, +0.012-0.013 beyond the placebo, +0.016 space;
  - selected +0.016-0.017 change, +0.007 space.
  - The first lever to improve both.
- Reading (likely): DESK's temporal increment uses the same spare capacity that over-fits places. Averaging away the
  seed-specific part keeps it; cutting capacity loses it.
- Span and time weighting re-scored (E056): nothing beyond the trend placebo (see E023/E024 entry above).
- Queued overnight: the trajectory with weight averaging (E055a), short schedule, metric weight 40, stronger masking,
  and the grokking probe.
- E055a TRAJECTORY (01:38, one seed, snapshots of the control itself): held-out SPACE peaks at epoch ~70 (0.821) and
  falls to 0.790 by 490. Backcast CHANGE rises monotonically the whole way (0.142 at epoch 40, 0.243 at the selected
  162, 0.264 at 490; beyond the placebo +0.033 over that range). Space and time trade off along one run: the
  over-fitting stretch is also where temporal structure is still being learned (likely learning order).
  - Weight averaging within a run gives nothing (it scores like its middle snapshot).
  - Averaging SEEDS does help (E055f2).

2026-10-12 08:45 -- MORNING SUMMARY: the over-fitting program (E054, E055a-h, E056; one seed per arm unless stated;
steering, not verified)
- The SHAPE (E055a, snapshots of the control):
  - held-out spatial skill peaks at epoch ~70 and falls as DESK over-fits places;
  - backcast change rises monotonically to epoch 490;
  - the spatial selection rule stops at 162, temporally under-trained.
- What does NOT help the backcast:
  - less capacity (width 64): better space, worse time (-0.06 / -0.03);
  - a short schedule (150 epochs): stable space, worse time (-0.02 to -0.04);
  - metric weight 40: earlier selection, worse selected time (-0.06 / -0.14), the same by epoch 490;
  - stronger masking: the same model at matched epoch;
  - weight averaging within a run: scores like its middle snapshot;
  - span / time weighting (E056): nothing beyond the trend placebo.
- The "grokking" probe (2000 epochs + metric 40 + same-place pairs + weight decay): NO SIGN.
  - Raw change at held-out cells is flat, and space collapses (0.762).
  - Its large training-cell "gain" (+0.09) is memorization of the training cells' MODERN communities, which the change
    estimand rewards there (end-swap diagnostic E055h). The backcast end is no better.
- What DOES help: a SEED ENSEMBLE (mean z of 4 seeds) vs the mean single seed.
  - Late state: +0.026-0.032 change, +0.016 space; beyond the strong base +0.005-0.010 (buffer readout).
  - Selected state: +0.016 change, +0.007 space.
  - Each seed over-fits differently, so averaging cancels the seed-specific part (likely).
- Implications (not decisions):
  - (1) A seed ensemble, especially of late states, is the one candidate that improves both space and time. It needs
    verification (joint bootstrap vs the mean seed; skeptic).
  - (2) Selecting the epoch on spatial validation picks a temporally under-trained state.
  - (3) Any training-cell evaluation of change is contaminated by modern-end memorization: use held-out cells or a
    withheld modern epoch.
  - (4) The span question cannot be settled inside BBS. A deeper series (per M29: Grinnell resurveys, BBC plots or
    historical range edges; not pre-1966 CBC) is the
    only test of "train on all of BBS".

2026-10-12 12:20 -- FIRST VERIFIED FINDING (F1, FINDINGS.md) -- ESCALATION (production implication, not a decision)
- Averaging the z of several independently trained DESK seeds improves, over the mean single seed on held-out cells:
  - place-specific backcast change: +0.010-0.015;
  - spatial level: +0.011-0.026;
  - DESK's gain beyond the trend placebo, late state: +0.016-0.018.
- It passed all four checks: a code sanity check, an independent re-derivation, replication on 4 NEW seeds, and a
  skeptic.
- The raw-change gain alone is NOT special: a 54 km smoother on one seed matches it.
- The gain grows like (1 - 1/k) and is mostly reached by 4-6 seeds.
- WHY: variance reduction (established, arithmetic). The removed variance is seed-specific, cell-scale over-fitting
  (likely).
- Implication: the age model could read the mean z of 4-6 DESK seeds (late states best for time) at 4-6x the DESK
  training cost, with no change to the BLR form. Effects are a few hundredths, not a step change.

2026-10-14 06:00 -- OUT-OF-COMMUNITY SPECIES SUMMARY (E059b/g/h, E067; the production-relevant case; steering)
- DESK (single seed, selected) trails the strong base (trend + covariates) by ~0.10 on species change and adds
  ~+0.005 beyond it.
- SEDENTARY species: the gap halves (-0.045 / -0.055 vs migratory -0.10 / -0.14; mean over 8 / 4 seeds).
- Spatial level is slightly BELOW a linear covariate map for every stratum (~-0.02).
- Community-level gains do NOT transfer to these species:
  - late training (flat or worse);
  - grokking probe (-0.07);
  - CropAT (-0.02);
  - metric weight 40 (worse).
- The seed ensemble helps by about its variance gain (+0.016 / +0.024 vs the mean seed), not beyond the base.
- E067 (surrogates): dropping migrants from the training community does not help sedentary targets. MORE resident
  species does (+0.06-0.07). The sedentary edge is likely biological, not an artifact of the selection.
- Production implications:
  - (1) the split-kernel readout (E059e: species change 0.089 -> 0.22, scale fixed);
  - (2) selected or early-stopped state, not late; the short 150-epoch schedule is free for species;
  - (3) DESK's change for out-of-community species needs a trend term or shrinkage. The CBC check (E069a/a2, other
    session) finds z's backcast shift 3-5x too large even with the split readout.

2026-10-15 -- E073 ANCHORED-DELTA SUITE: SYNTHESIS (research/experiments/E073.md; the user asked: "test it comprehensively,
understanding first")
- 1920 vs the BBS era: climate in range; management and human pressure the same direction ~2.4x further; land COVER
  novel in direction and size (Great Plains most).
- A clock-free, linear, community-learned delta of covariate change (no year input, g(0) = 0) carries DESK's change
  direction and more (all species +0.06-0.07 corr), costs nothing to share across species, and needs 25-50 y of
  covariate history. Spatial gradients add nothing (kappa ~0).
- It over-predicts magnitude (estimation variance). Prior shrinkage x~0.1 or in-era CV rescaling fixes most of it
  without withheld data.
- Land-cover extrapolation fails for EVERY representation when training lacks places of large cover change (-0.10 to
  -0.18). DESK's apparent edge came from having seen those places. A random 30% cell loss costs nothing.
- Place-dependent readout slopes: robust inside BBS reach, but worse on early CBC lists even reach-decayed (E074,
  other session). Not a backcast tool.
- Levers that remain: cover-change supervision, and novelty-scaled uncertainty on delta's cover component (E072). A
  two-part DESK is a structural, not a skill, argument.

2026-10-15 -- E072 HONEST CHANGE UNCERTAINTY (research/experiments/E072.md; the user's option 2; UNVERIFIED)
- Change errors at held-out places are ~0.25-0.45 of a species' spatial variance (S) over ~35 y, mostly unpredictable
  at any distance. The readout's posterior is 45-1000x too confident.
- Honest law: a constant term (~0.3 S per ~35 y) plus the covariate delta's own errors-in-variables (block bootstrap
  of its community map, x1.4-5). The latter grows beyond the BBS-era cover range and ~7x for 1920, and it calibrates
  the beyond-range places (R 1.04-1.05, sedentary). Cover novelty adds nothing on top. DESK's seed spread is NOT
  novelty-aware and fails as an option-2 term.
- 1920 projection: change-error variance ~0.5-0.6 S (vs 0.26-0.33 inside BBS), i.e. corr(H1920, H2000) ~0.7 relative
  to the spatial contrast, lowest on the Great Plains (0.68). The predicted 1920->2015 change is about as large as its
  uncertainty.
- Strict rule: no model calibrates both sedentary species halves (one half 1.26-1.33): species heterogeneity.
- E073 SKEPTIC 2 revisions:
  - the covariate delta's "contains DESK" gain is pooled-corr and carried by ~5 widespread species (MSE cost at
    T0 1996);
  - over-movement = cross-place overfitting (confirmed, CV-chosen shrinkage fixes it);
  - the cover-specific extrapolation loss is ~0.10 beyond an equally regional loss (~0.06), not ~0.16;
  - the DESK leg is inconclusive.
