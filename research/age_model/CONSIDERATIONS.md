# Considerations (user's standing points; read before every pre-registration and interpretation)

- C1 Allee effect for House Finches is ABSOLUTE (mate-finding by density), not relative to K. The
  relative form (run_22) was a fit test only; it also slows the dense eastern core, forcing contortions.
- C2 Fecundity ~10 per adult and juvenile survival >= adult survival are absurd.
- C3 Allee-dead areas and lambda-vs-Allee trade-offs are fundamentally non-identifiable: what is
  inferable is persistently negative effective growth (sinks), by whatever mechanism.
- C4 Avoid hacky special-case penalties; prefer existing terms or structural statements.
- C5 Very high K is genuinely possible; extremely small K should not be cheap.
- C6 Learnable dispersal distance conflicts with precomputed Z_disp; distance is a discrete sweep.
- C7 The century-scale habitat trend is overfit; temporal-habitat weighting (learned eta) is
  deferred until the DESK experiments settle.
- C8 Softplus is liked for loose amplitude priors that still discourage absurd K.
