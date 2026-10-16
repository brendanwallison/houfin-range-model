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
- C9 Simplifying (fewer parameters / more parsimonious structure) needs NO user approval. A simpler model
  that fits as well is an important data point -- but fit alone never adopts it: simple models can
  contort into biological implausibility, so read demography and maps, not just U.
- C10 Limited, biologically reasonable added complexity is fine to TRY; dispersal is a known weak
  point (the user had no efficient smarter dispersal). Z_disp (path-dependent mortality) is the user's
  key innovation but imperfect; the user worries about path-AVERAGED features interacting with weights
  SHARED between Z and Z_disp. (Mechanism: true path survival is a product dominated by the worst
  stretches; sigmoid(mean path habitat) smooths over barriers.)
- C11 Multi-step / multi-scale dispersal with "oracle" settlement (redistribute local population in
  proportion to K or growth after dispersal) is a candidate the user considered; zero extra parameters.
- C12 DESK features will change (research/desk-temporal). Pin conclusions down now where possible, but
  instability may come from imperfect features: check leading findings against the alternative DESK
  cubes precomputed on TACC.
