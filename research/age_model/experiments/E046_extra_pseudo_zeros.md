# E046 -- extend pseudo-zeros through the unsupervised 1940-1965 window (pre-registered 2026-10-17)

User: early invasion (1940-~1965) stayed local to New York; the zero fill stops at 1940. Keep the same subsampled
pre-invasion pseudo-zero sites (outside the native hull + 700 km), drop any within 700 km (the same buffer) of the
release cell, and emit zeros 1940-1965. Built: 1814 sites -> 1570 kept -> 40,820 zero observations (side file
extra_pseudo_zeros.npz; appended at load via observation.extra_pseudo_zeros).
Variant c40_pz on cold run_40 (+ viz). Expect: margins -- less freedom for pre-BBS habitat shifts and early
invasion speed; plains trough possibly firmer. Combine with the E045 winner next.
