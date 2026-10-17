# Age-model research loop (Phase 1: diagnostic)

Question: what in the data forces the fit into implausible demography (lambda0 ~ 2 at typical
habitat, fecundity ~10, juvenile survival >= adult), a strong absolute Allee effect (n50 ~ 0.28
route counts), capacity used as an absence switch, and a century-scale habitat trend -- and what is
the least extra structure that explains the data with plausible demography?

Anchor: plausible demography, interpretable niche maps, fit that generalizes -- never U alone.

Rules (adapted from research/desk-temporal's PROTOCOL.md):
1. Every experiment tests a named hypothesis (hypotheses.yaml) with its predicted outcome and kill
   criterion written in experiments/EXXX.md BEFORE it runs.
2. Localize before intervening: likelihood decomposition (exp/loglik_decompose.py) and warm-started
   profiles (L-BFGS from a refined point with one thing pinned) before any structural change.
3. Cheap tests for mechanism; cold MAP + refine + probe + held-out validation only to verify.
4. A falsification pass on the leading hypothesis every few rounds; one open-ended residual look.
5. The user's points in CONSIDERATIONS.md are binding; anything that changes an ecological claim
   goes to the user before adoption.
6. Every evaluation is logged in ledger.csv (counts forking paths). DIGEST.md, newest first, for the user.
7. Budget: DEB23008 ONLY (DEB23007 is off limits); it had 381 SU on 2026-10-16, shared with the
   user's other work. Overall age-model research <= 150 SU (raised from 100 by the user 2026-10-16); Phase 1 = one ~12 h SSH window, <= 75 SU (raised from 25, then 50, by the user 2026-10-16),
   diagnostic only. Prefer one gpu-a100-small reservation (~1.5 SU/h) over dev (full-node rate); hold a
   node only while experiments run. SU logged per experiment in ledger.csv.

Heavy outputs: $WORK/houfin/research/age_model/<exp>/ on TACC, never in the repo.
