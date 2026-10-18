# E052 / E053 -- reproduction: yearling breeding, and the Veit & Lewis parallel model (pre-registered 2026-10-17)

MODEL BUG FOUND: reproduction_age_structured births = S_a*N_a_post*F -- only surviving ADULTS breed, so first breeding
is at ~2 y. House Finch yearlings breed (V&L 1996: juveniles 9-12 months breed). Under the delayed matrix lambda0 =
(Sa + sqrt(Sa^2 + 4F Sa Sj))/2, so reaching a given lambda0 demands a much larger F -- a candidate cause of the F ~10 the
fits keep asking for.

Switches (off by default; population_model.reproduction):
- yearling_breeding: births = N_a_new * F (all winter survivors breed); lambda0 = Sa + Sj*F; F_repl = (1-Sa)/Sj; c
  re-solved as Fmax*Sj/(1-Sa) - 1 so the no-dispersal equilibrium is still K (unit-tested).
- form = "veit" (the PARALLEL model): V&L eqs. 5-8. Breeders B = all survivors; pairs P = B^2/(r + 2B), r = 2*n50
  (our Allee n50 -> half of breeders mated at B = n50); nest sites H = P/(1 + P/delta), delta = K/2 site-pairs; births
  2*Fmax*H. K becomes nest-site capacity, the population equilibrium is EMERGENT (B* solves Sa + Sj*2Fmax*H(B)/B = 1).
  Release exemption sets P = B/2. Same inputs, priors and switches as our model otherwise, so adjustments stay comparable.
  Guarded off: fitness/r0 settlement weights, subcell_strata, crowding capacity (their algebra assumes adult-only breeding).
- equilibrium_age_quantities, age_model_math.local_growth_lambda / allee_viability (F(N) curve, F_repl) follow the form.

Runs (cold start: MAP -> viz -> float64 refine -> viz):
- 52a: run_50a (literature dispersal, dispersers-only capacity settlement) + yearling_breeding.
- 52b: run_51a (V&L dispersal, no settlement) + yearling_breeding.
- 53a: run_51a + form = veit, demography free (biology priors).
- 53b: 53a pinned to V&L: Sa .72, Sj .64, Fmax .75 per breeder (c 1.5 per pair) -> lambda0 1.20.
Predictions: yearling breeding lowers fitted Fmax toward 4-6 (vs E050/E051) at similar U; veit gives a weaker effective
Allee (mate-finding only below n50) and K-limited births rather than K-limited fecundity, so the plains can be sinks
through survival without high-K contortion. Read: MAP vs refined plains sink share / median lambda vs W and E, Sa/Sj/F.
