# E068 -- does the refine contort when the truth is plausible? (pre-registered 2026-10-20)

Every synthetic guard so far drew its data at a CONTORTED refined point. If counts are simulated at a PLAUSIBLE point
-- 62b's MAP (V&L dispersal, n50 pinned, numerics; Sa .57, Sj .31, F 5.1, lam0 1.28; plains .67 sinks med .86 vs
west 1.05; no Allee-dead) -- and the same structure is fitted cold (quick90 MAP -> float64 refine):
- if the refine still moves to F ~10-15, Sj -> Sa or Sj collapse, and a flattened plains trough, the contortion is an
  identifiability/optimiser artifact (the data do not demand it);
- if it stays near the truth, the real data contain something the structure cannot represent (the two-pressure
  reading: spatial absence + demographic/timing), and corrections (E066/E067) are the right path.
Read: refined vs truth demography, plains/west/east lam, Allee-dead, U(truth) vs U(refined) on the synthetic data.
