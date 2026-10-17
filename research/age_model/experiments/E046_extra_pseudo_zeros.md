# E046 -- extend pseudo-zeros through the unsupervised 1940-1965 window (pre-registered 2026-10-17)

User: early invasion (1940-~1965) stayed local to New York; the zero fill stops at 1940. Keep the same subsampled
pre-invasion pseudo-zero sites (outside the native hull + 700 km), drop any within 700 km (the same buffer) of the
release cell, and emit zeros 1940-1965. Built: 1814 sites -> 1570 kept -> 40,820 zero observations (side file
extra_pseudo_zeros.npz; appended at load via observation.extra_pseudo_zeros).
Variant c40_pz on cold run_40 (+ viz). Expect: margins -- less freedom for pre-BBS habitat shifts and early
invasion speed; plains trough possibly firmer. Combine with the E045 winner next.

History check (2026-10-17): first wild record Jones Beach Apr 1941; range grew ~3.5 km/yr in the 1940s; colonies
on Long Island + Greenwich CT by 1949; ~280 birds on LI by 1951; abrupt acceleration ~1960, ~21 km/yr 1962-79;
NJ/CT then PA in the 1960s. Occupied radius ~200-250 km by 1965.
User: must be conservative AND must not penalise juveniles PROSPECTING from occupied cells. Juvenile kernel mean
~320 km (long tail), so the zero-free zone = occupied radius (x2 safety) + ~330 km: ~480 / 630 / 780 km for the
1940s / 1950s / 1960-65. The flat 700 km buffer is therefore about right (conservative through the 1950s, near the
bound in 1960-65). A tighter history-only schedule (150/300/450 km; extra_pseudo_zeros_hist.npz) was built and
WITHDRAWN -- it would put zeros where prospecting juveniles plausibly were. Residual risk is small: under the NB a
zero against a small predicted density costs ~mu.
Sources: Veit & Lewis 1996 (Am Nat); allaboutbirds.org House Finch; Wikipedia "House finch".
