# E050 -- literature dispersal, fixed; sensitivity around it (pre-registered 2026-10-17)

User: E044-E048 were all run on a base whose learned departure layer clamped dispersal (~13-19% departure on top of
literature kernels that already encode staying -> ~85% of juveniles never disperse vs ~5% in the kernel). Those results
are NOT viable; nothing built on that base is a candidate. Literature dispersal is acceptable fixed, not learned; do a
sensitivity analysis around literature values.
All cold-started (MAP quick90 -> viz of the MAP point -> float64 refine -> viz of the refined point), run_40 structure,
departure pinned ~98% both ages (logit 4, slope 0), no density dependence, band weights off:
- 50a: juvenile mdd 330 (literature), adult 100, dispersers-only settlement
- 50b: 50a with settlement OFF (pure kernel)
- 50c / 50d: juvenile mdd 250 / 450 (inputs from the Oct juv_mdd_single_field sweep; derived tercile splits)
Read: MAP vs refined source/sink (plains vs flanks, K-weighted west), demography / lambda0, fit, whether the east and
modern plains are reached without habitat reshaping once birds actually move.

## Result (2026-10-18; 50a/50b MAP+refine, 50c MAP; 50d running). CONUS lat<49 corridor zones.
| run | U | Sa | Sj | F | lam0 | n50 | plains sink / median lam_f | west | east | Allee-dead (P) |
|---|---|---|---|---|---|---|---|---|---|---|
| 50a MAP | (quick90 loss 204,437) | .55 | .27 | 4.75 | 1.16 | .023 | .91 / .66 | .31 / 1.10 | .53 / .97 | 0 |
| 50a ref | 187,729 (8000 it, unconverged) | .69 | .20 | 11.75 | 1.67 | .126 | .42 / 1.06 | .32 / 1.08 | .44 / 1.09 | .22 |
| 50b MAP | (205,941) | .55 | .28 | 4.86 | 1.19 | .023 | .88 / .64 | .26 / 1.15 | .53 / .97 | 0 |
| 50b ref | 186,939 (unconverged) | .86 | .07 | 15.32 | 1.45 | .066 | .49 / 1.00 | .29 / 1.12 | .43 / 1.14 | .25 |
| 50c MAP (mdd 250) | (205,506) | .54 | .28 | 4.70 | 1.16 | .022 | .90 / .74 | .28 / 1.09 | .46 / 1.03 | 0 |
MAP under literature dispersal is the most realistic point so far on every axis: plains a deep trough (~90% sinks,
median lam_f .64-.74 vs west 1.10, east ~1.0), no Allee-dead, demography inside the target (Sa .55, Sj .27-.28, F ~4.8),
weak Allee. Refinement contorts it again: plains trough flattened to ~W/E, Allee-dead 22-25% appears, F 12-15, Sj .07-.20.
Settlement (a vs b) barely matters at MAP. Refine gain is the usual ~17k nats. Not structural (cf. refinement-drives-contortion):
the question is now WHAT the refine buys with the contorted demography -- E052 (yearling breeding) tests whether the F demand
is the delayed-maturity artefact.
| 50c ref (mdd 250) | 188,203 (unconverged) | .62 | .27 | 10.56 | 1.68 | .104 | .33 / 1.14 | .34 / 1.10 | .35 / 1.27 | .19 |
50c refine: same contortion (plains trough fully gone, Allee-dead .19, F 10.6) with shorter juvenile dispersal -- kernel
reach (250 vs 330 km) does not change the refined basin.
| 50d MAP (mdd 450) | (199,748) | .58 | .27 | 5.07 | 1.23 | .043 | .81 / .68 | .29 / 1.19 | .52 / .97 | 0 |
Longer juvenile dispersal (450 km) gives the best E050 MAP loss and log1p corr .75, trough intact.
