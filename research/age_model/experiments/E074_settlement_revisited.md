# E074 -- ideal-free settlement revisited on the clean holdout base (pre-registered 2026-10-21)

Why: settlement gave the largest structural gains of the early rounds (ifd3 -1,729; dispersers-only + coupling +
density-dependent NB = that era's best, Sa .70 Sj .51 F 5.7), then was dropped at E050 on ONE pair (50a with
dispersers-only settlement vs 50b without: MAPs alike, both refines unconverged and contorted) on the legacy-pad base
(regime A: 1e-9 pseudo-zero immigration + float32 phantom floor). E061a (fitness settlement + V&L departure) failed
from a cold start and was not pursued. E070 now locates the demand: very low abundance over large areas the model
rates viable, made by thin kernel seeding of marginal cells that the Allee term then empties. Settlement places
arriving birds by capacity instead of seeding every cell thinly -- a direct, mechanistic candidate.
Arms (base 70z: 69a structure, holdout, no floor, no noise; cold MAP -> refine):
- 74a dispersers-only, capacity-weighted, 3x3 blocks (50a's form);
- 74b ALL birds, capacity-weighted, 3x3 (ifd3: residents re-equalised every year -- the early winner);
- 74c dispersers-only, K x R0^tau, tau ~ HalfNormal(2) (imperfect sorting on breeding quality; ct_ds_r0 form);
- 74d dispersers-only, capacity-weighted, 5x5 blocks (~135 km; scale sensitivity).
Read with E070's gates (G2 demography, G3 plains trough + Allee-dead < 5%, G5) and R1 held-out density vs 70z refined
(-1.818 / -1.178); plus the counterfactual release attribution (does settlement hold the west back for real?) and the
spread of log lam (E070 readout: refines sharpen fecundity's habitat amplitude ~3x).

## RESULTS (2026-10-22)
| arm | MAP plains/west/east | refined Sa/Sj/F | refined plains/west/east | AD w/p | plains K ref | holdout t/s |
| 74a ds, K, 3x3 | .74/1.13/1.11 | .68/.30/12.3 | .93/1.01/.98 | .07/.00 | .076 | -1.846/-1.197 |
| 74b all, K, 3x3 | .74/1.12/1.12 | .58/.56/10.3 | .94/1.15/1.20 | .03/.02 | .066 | -1.831/-1.196 |
| 74c ds, K*R0^tau | .74/1.11/1.09 | .63/.32/13.1 (tau .09) | .94/1.03/1.03 | .16/.00 | .211 | -1.836/-1.195 |
| 74d ds, K, 5x5 | .70/1.24/1.05 | .81/.10/9.0 | .95/1.00/1.18 | .14/.01 | .071 | -1.844/-1.205 |
| 74e fitness (smooth) | MAP NaN at step 13 (float32 negative round-off; rerun with float32-only floor) |
Settlement gives the best-looking MAPs (deep plains trough .70-.74, native West a source: west sink .25-.40) and keeps
plains K from collapsing (74c .211), but every refine still contorts and held-out density is slightly WORSE than 70z.
74b's MAP spatial holdout -4.94 (re-equalising all birds mispredicts withheld blocks badly at MAP).
