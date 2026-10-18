# E051 -- borrow more from Veit & Lewis 1996 (pre-registered 2026-10-17)

From the paper (~/Downloads/veit1996.pdf): juvenile kernel a .468, b 41.056 km (mean 329 km) and ADULT kernel a .859,
b 76.893 km (mean 98 km), modified Weibull fitted to Massachusetts band recoveries INCLUDING non-dispersers (27% of
juveniles / 22% of adults recaptured where banded) -- the kernel IS the dispersal model. Eq. 18: departure p = N/N2*
below capacity, 1 above, both ages (positive density dependence: sparse fronts stay together, dense cores push out ->
slow-then-accelerating spread). Weak mate-finding Allee P = N^2/(r + 2N), r .15. Survival .67 (adults .72, juveniles .64),
c 1.5 offspring/pair surviving summer -> lambda_max ~1.24. ~80 founders. Reproduces ~40 km/yr spread and core CBC density.
CONFIG BUG FOUND: our adult_shape is 0.468 (the juvenile shape); V&L adult a = 0.859.
Changes: model_inputs guard ignores the adult kernel keys (Z_disp uses only the juvenile stack); dispersal_departure.form
= "veit" (eq. 18, smoothed). Inputs rebuilt into numpyro_input_vl with adult_shape .859 (path features reused).
- 51a: V&L dispersal (kernels incl. adult fix; eq. 18 departure; no settlement), demography free under biology priors.
- 51b: 51a with demography pinned to the V&L equivalent: Sa .72, Sj .64, Fmax 1.40 -> lambda0 ~1.24.
Cold-started: MAP -> MAP viz -> float64 refine -> refined viz. Compare with E050 (literature kernels, departure fixed .98).

## Result (2026-10-18; 51a MAP+refine, 51b MAP). CONUS lat<49 corridor zones.
| run | U | Sa | Sj | F | lam0 | n50 | corr | plains sink / med lam_f | west | east | Allee-dead (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 51a MAP | (196,987) | .57 | .31 | 5.21 | 1.28 | .058 | .759 | .68 / .86 | .49 / 1.01 | .46 / 1.06 | 0 |
| 51a ref | 186,068 (CONVERGED, 3508 it) | .66 | .29 | 14.40 | 2.02 | .063 | .810 | .53 / 1.00 | .53 / 1.00 | .43 / 1.19 | .18 |
| 51b MAP (V&L pinned) | (201,162) | .72 | .64 | 1.40 | 1.24 | .042 | .733 | .70 / .86 | .39 / 1.07 | .44 / 1.07 | 0 |
V&L dispersal (adult shape fixed + eq. 18 departure) gives the best MAP loss of any literature-dispersal run (~7.5k better
than 50a) and its refine CONVERGES to 186,068 -- matching run_40's 186,051 with NO learned dispersal. MAP trough is shallower
than E050's (.68 vs .9 sinks) but still plains-worst. Refinement again flattens plains to = west, Allee-dead .18, F 14.4.
51b: V&L's own demography (pinned) fits only ~4k worse than free at MAP with the same landscape -- the data do not demand
F ~5 at MAP; the F 14 demand arises only in refinement.
