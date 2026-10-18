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

## Results so far (2026-10-18). lam0 = Sa + Sj*F (yearlings breed). CONUS lat<49 corridor zones.
| run | U | Sa | Sj | F | lam0 | n50 | corr | plains sink / med lam_f / med K | west | east | Allee-dead W/P/E |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 52a MAP | (204,733) | .66 | .26 | 3.93 | 1.67 | .064 | .734 | .81 / .73 / .0125 | .28 / 1.19 | .32 / 1.30 | .02/.01/.10 |
| 52a ref | 189,546 (CONVERGED 3374 it) | .49 | .18 | 8.42 | 2.01 | .206 | .743 | .38 / 1.18 / .053 | .19 / 1.58 | .38 / 1.44 | .23/.27/.01 |
| 52b MAP | (203,535) | .64 | .23 | 3.93 | 1.53 | .090 | .729 | .77 / .79 / .0117 | .20 / 1.31 | .32 / 1.30 | .07/.04/.14 |
| 53a MAP (veit repro) | (205,106) | .52 | .26 | 4.49 | 1.68 | .031 | .724 | .78 / .63 / .067 | .17 / 1.60 | .36 / 1.28 | .06/.07/.08 |
Yearling breeding does NOT fix the refine: 52a refined is 1.8k nats WORSE than 50a (189,546 vs 187,729) and still contorts
(F 8.4, Sa .49, n50 .21, Allee-dead .27 in the plains, plains trough flattened). F demand drops (8.4 vs 11.8) only because
lam0 per unit F is higher. NEW NEGATIVE: at MAP, yearling breeding cuts median K ~7x (plains .0125 vs .0885 in 50a) and
Allee-dead cells appear even before refinement (E .10-.14). MAP plains trough survives in all three (.77-.81 sinks).
