# E071 -- Allee sensitivity on the E070 holdout base (pre-registered 2026-10-21)

Why: every E070 refine, with or without process noise, empties large plains/west areas the model rates viable
(K collapse + Allee-dead 14-23%), and E069 located the Allee establishment switch the optimiser exploits. The Allee
strength is pinned to a literature value (n50 = .0592 route counts; user: pin + sensitivity). If the contortion
depends on the Allee term, weakening it should change the refined demography and the plains/west emptiness.
Arms (base 70z: 69a structure, holdout 2021-2025 + 10% blocks, no floor, no noise; cold MAP -> refine):
- 71a n50 / 4 (.0148);   71b n50 / 100 (.00059; Allee effectively off).
Read with E070's gates and ranking (G2 demography, G3 plains trough + Allee-dead < 5%, G5 MAP->refine; R1 holdout vs
70z refined -1.818 / -1.178). No synthetic guards: pinned-value sensitivity, no new free term.
Interpretation: if 71a/71b refine to plausible demography with a plains trough, the Allee term was the lever; if they
contort as 70z with the emptiness carried by K alone, the demand is K-level absence and the Allee is incidental.
- 71c RELATIVE Allee (form relative, n50/K = .0197 pinned; added 2026-10-21 at the user's request): 55e removed
  Allee-dead by construction on the old base for +923 nats with Sj = Sa .44 / F 14.5; rerun here so its held-out density
  and gates are comparable with 70z and the other arms.
- 71b numerics (2026-10-21): with the Allee term off nothing kills float32 round-off seeds, so 71b's float32 MAP uses
  the 10-ulp floor and the float64 refine does not (numerics.fft_floor_float32_only, new): the MAP is protected from
  phantom immigration, the refine from the exact-extinction switch.
