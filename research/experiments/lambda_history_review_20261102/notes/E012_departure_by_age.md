# E012 -- departure readout + age-specific departure (pre-registered 2026-10-16)

Context: E008 (ifd3) fitted departure p(x) = sigmoid(-4.52 - 8 (x - 0.8)), x = N/K: 87% at x=0,
1% at 0.8, ~0 above K -- birds abandon sparse cells and never leave crowded ones. User: sticky
cores may be legitimate, but a crowded patch above K should send dispersal pulses; the current
single-slope form cannot express both.

Step 1 (readout, exp/departure_readout.py, 23a vs ifd3): abundance-weighted N/K at dispersal time
by region x era; departing birds by x band, adults vs juveniles.
- If N/K > 1.1 holds < ~2% of birds everywhere: a crowding arm (beta_hi) would be prior-only ->
  do not build the U-shape; the slope is about the low-density end.
- If N/K > 1 is common (disease K drops, good years): the single slope is actively mis-specified
  there -> U-shape is warranted.

Step 2 (structure, dispersal_departure.form = age; overlay map_run26): adult logit offset,
prior N(-2, 1.5), on the shared curve. Warm start ifd3.
Predictions:
- H_sticky_adults: offset goes well negative (< -2), the shared slope RELAXES toward 0 or positive,
  dU < 0 (better fit), n50 / lambda0 roughly unchanged.
- If the slope stays ~ -8 with the offset near its prior: the negative density dependence is a
  JUVENILE (natal) signal -- low-density natal emigration / conspecific attraction -- not adult
  fidelity; then step 3 (U-shape) only if step 1 found N/K > 1 common.

Step 1 RESULT (2026-10-16, readout_23a / readout_ifd3, pulled to results/age_model/E012):
- N/K > 1.1 is COMMON: 20-48% of birds in established region-eras (23a), 6-34% (ifd3); east
  1994-2007 (post-epizootic K drop) 48% / 51%. -> a crowding arm IS identifiable; the U-shape is
  warranted by the decision rule above.
- ifd3: only ~1% of birds depart per year in aggregate (most live at N/K ~ 1, where p ~ 0.2-1%);
  the 87% departure at N/K ~ 0 applies to the few birds at the front (east 1966-79 median N/K 0.07)
  and in sinks. So movement IS mostly settlement, plus a front exodus (refines the E008 correction).
- 23a: flat ~35% departure; the kernel dumped birds into low-K cells (99th pct N/K 4-17) -- removed
  by K-weighted settlement, a likely large part of E008's gain.
