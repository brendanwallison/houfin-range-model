# E043 -- the literature kernel IS the dispersal model (pre-registered 2026-10-17)

User: the kernels are literature-based and already include non-dispersal; the fitted departure probability
on top is a massive clamp. Kernel self-mass at 27 km: adult (mdd 100) 37% own cell, 59% within 54 km; juvenile
(mdd 330) 5% own cell, 15% within 54 km. Cold run_40 departure ~13-19% => ~86% of juveniles never use the
kernel (literature: ~5% stay home). Same clamp in every fit so far (exodus regimes: ~0.1% at K).
- c40_kern: departure pinned ~98% for both ages (int 4, slope 0): pure literature kernels.
- c40_kern_juv: juveniles ~98%, adult logit offset learned (site fidelity beyond the adult kernel).
If cheap: remove/reparameterise the departure layer (departure 1 by default, density dependence a modest
modulation). Read: cost, lambda0 (prospecting mortality), plains sinks vs flanks, west core.
