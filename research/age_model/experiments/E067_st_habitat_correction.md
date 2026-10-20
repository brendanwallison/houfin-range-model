# E067 -- smooth spatiotemporal habitat correction (pre-registered 2026-10-20)

Restores, as a general misspecification term, the original (March) st_basis correction to the habitat fields that was
repurposed into the K/disease term in July and then removed. Motivation: every refine converts the plausible MAP into
higher reproductive turnover (F ~9-15) to match TIMING the model cannot otherwise represent (E057: a real temporal
signal; E063: invasion timing ~20 y per 5% of F); DESK's change is spatially coherent but poorly aimed, and a no-DESK
position x time placebo matched the split readout's gain. A smooth REGIONAL time-varying habitat term is the direct
way to let habitat, not vital rates, carry regional timing.
Form (population_model.st_correction): h(cell, t) = sum_b S_b(cell) w_b g(t); S = 15 land-centred cosines (n_freq 3);
g linear over 1966-2023, held before, observation-centred; w ~ N(0, .1). h is SHARED by all four habitat fields (a
habitat-quality correction, as originally) and enters on H -- not inside K's link (the failure of the July version).
Path consistency: journey survival gets the correction's path image sum_b (A S_b) w_b g(t), A = the land-normalised
path operator, applied once to each basis function (src/data/preprocess/st_correction_basis.py; the builder checks it
reproduces the stored Z_disp). Zero weights reproduce the base model exactly (test).
Arms (cold quick90 MAP -> MAP plots -> float64 refine -> plots + sink audit):
- 67a B1 (62b) + correction, real.   - 67c B3 (62b + route SD .7) + correction, real.
- 67n NULL: B1 + correction on synthetic counts at 62b's MAP (plausible truth, no correction). Pass: |w| within ~2
  prior sds, land-RMS h small vs 67a's.
- 67r RECOVERY: truth = 62b MAP + an east-west correction growing over 1966-2023 (basis cos(pi x), w .5). Pass: the
  injected basis weight recovered within ~30%, others near 0.
Read (only if 67n/67r pass): does the refine keep MAP-like demography (F, Sj) and the plains trough; where and when h
is large (regional timing it took over); K-trend pressure; west-sink audit. Compare with E068 (same truth, no
correction) to separate "the correction absorbs real misspecification" from "the refine contorts regardless".
