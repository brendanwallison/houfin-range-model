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
