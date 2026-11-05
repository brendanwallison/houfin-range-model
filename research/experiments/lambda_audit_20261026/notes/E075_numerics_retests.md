# E075 -- retests of approaches rejected under confounded numerics (pre-registered 2026-10-21)

An audit of E001-E069 against the three numerics regimes (A: legacy 1e-9 pad + float32 phantom floor, until the 62b
lineage; B: zero pad + hard floor, 62b descendants only -- E063, E066, E068a, part of E069; C: zero pad, no floor,
69a onward) found the refine contortion itself regime-independent (55a/62b/69a/69b and 64k/69c agree), so converged
"did not fix it" verdicts mostly stand. The confounds are stalls (B), f32-vs-f64 MAP disagreements (A), and
timing/front readings. Note: the 58a/59a/55a lineages (E057-E059, E061, E064 incl. 64k/64l, E065) are regime A
despite their dates; 64k is legacy pad, not hard floor (corrected in E067/E069).
Retests on the 70z holdout base (no floor; same gates and held-out bar -1.818 / -1.178):
- 75a occupancy (E063; tested only under the floor, whose exact extinction is the regime occupancy addresses -- and
  E069 names stochastic establishment as the structural remedy for the Allee separatrix).
- 75b yearling breeding (E052; a biological correction -- yearlings breed -- rejected on disagreeing f32/f64 MAPs).
- 75c / 75d split readout at rank 32 / 64 (E066b/c; stopped only by floor stalls; rank has no valid refined result).
- (E074e) fitness settlement with smooth allocation (E061a failed cold with the unsmoothed equal-fitness root).
Deferred: E057 P2 covariate drift (never validly run; do on a split readout after 75c/d), E065 guards on a clean
truth point, Veit reproduction (E053/55b).

## RESULTS (2026-10-22)
- 75a occupancy: MAP NaN at step 10 (float32 negative round-off without the floor); rerun with float32-only floor.
- 75b yearling breeding: MAP K low (.010 plains) with Allee-dead at MAP (as 52a); refined .57/.13/10.5, Allee-dead
  .19/.19, holdout -1.825/-1.191. The 52a rejection STANDS on clean numerics.
- 75c SPLIT READOUT RANK 32: refined Sa .55 / Sj .37 / F 6.47, lam0 ~1.46 -- the FIRST real-data refine inside G2
  (plausible demography). Plains 1.04 / west .94 / east 1.02 (trough flattened, west lowest; G3 fails on the trough);
  Allee-dead .00/.02; plains K .088; holdout -1.833/-1.183 (~70z). Deviation/level ratio wild (6.9 survival, .25
  fecundity). G5 marginal (Sj +.09, F +33%).
- 75d split readout rank 64: refine STALLED (|grad| 1.4e5).
