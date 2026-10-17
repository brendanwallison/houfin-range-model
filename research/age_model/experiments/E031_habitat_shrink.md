# E031 -- learned shrinkage of covariate change (pre-registered 2026-10-17)

Candidate 01b: intrinsic growth rose up to +0.4 across the east-central US since 1940-55 (lambda ~1.3 -> ~2
where the invasion surged) and fell up to -0.4 in the north; the continental mean is flat, which hid it.
So part of the invasion is explained as habitat improving under the birds. DESK features are known to
over-move in time ~2.2x vs observed abundance change (memory: desk-direction-accuracy-is-the-ceiling).
User guess: the model over-predicts habitat change. habitat_change.shrink = learned:
Z_eff(t) = Z_mean + eta (Z(t) - Z_mean), eta ~ Beta(2,2), local + path features (user: learned, not fixed).
Variant ct_hab on cand_tight (eta starts at 0.5).
Predictions: eta well below 1 (~0.3-0.6); early plains over-fill (obs/fit .42/.60 in 1966-93) and early
west lag (1.36) improve; lambda0 / exodus may shift (the invasion surge must come from dynamics instead).
Read: eta, U, plains/west/east obs/fit by era, Great Plains corridor, 01b change map.
