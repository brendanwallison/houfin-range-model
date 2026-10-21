# E068 -- does the refine contort when the truth is plausible? (pre-registered 2026-10-20)

Every synthetic guard so far drew its data at a CONTORTED refined point. If counts are simulated at a PLAUSIBLE point
-- 62b's MAP (V&L dispersal, n50 pinned, numerics; Sa .57, Sj .31, F 5.1, lam0 1.28; plains .67 sinks med .86 vs
west 1.05; no Allee-dead) -- and the same structure is fitted cold (quick90 MAP -> float64 refine):
- if the refine still moves to F ~10-15, Sj -> Sa or Sj collapse, and a flattened plains trough, the contortion is an
  identifiability/optimiser artifact (the data do not demand it);
- if it stays near the truth, the real data contain something the structure cannot represent (the two-pressure
  reading: spatial absence + demographic/timing), and corrections (E066/E067) are the right path.
Read: refined vs truth demography, plains/west/east lam, Allee-dead, U(truth) vs U(refined) on the synthetic data.

## RESULT (2026-10-20): the refine does NOT contort on plausible-truth data -- the real data demand it
Truth 62b MAP (Sa/Sj/F .57/.31/5.12, lam0 1.28). Refit: MAP .55/.29/4.96 (lam0 1.21); refined .53/.28/5.57 (lam0 1.20),
U 201,706 -> 200,662 (~1k gain vs ~13k on real data); plains / west / east med lam .86 / 1.04 / 1.08; Allee-dead 0.
The optimiser and identifiability do not produce the F ~10-15 pull. The real data contain something this structure
cannot represent and the refine fakes it with demography -- supports the correction-term direction (E066/E067).

## 68b (pre-registered 2026-10-20): 68a without the hard floor
69a showed the hard floor's U jumps are exploitable; they could also have PINNED 68a's refine at the truth. 68b repeats
68a on the no-floor structure (truth = 69a MAP, Sa .57 Sj .31 F 5.2). Same read: if the refine stays near the truth,
68a's conclusion (the real data demand the contortion) stands; if it contorts, the contortion is (partly) an
optimiser/identifiability artifact and E068's reading is withdrawn.
