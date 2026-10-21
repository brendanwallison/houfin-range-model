# E077 -- a more sedentary native range (pre-registered 2026-10-21)

Why (user + E076): the West carries ~1/3 of the misfit on ~15% of observations and its dense cores are under-predicted
at every point (mu 8.2-10.7 vs 11.2 observed); every refine sharpens habitat amplitudes (~3x fecundity) and collapses
marginal K -- habitat recreating sharp abundance gradients the dispersal kernel blurs. The kernel is Veit & Lewis's,
estimated from the eastern invasion; western House Finches are largely resident while the introduced eastern population
became partly migratory (Able & Belthoff 1998). Caveat: migration is seasonal movement, not natal dispersal; the
link is indirect (migratory populations tend to disperse farther), so this is a hypothesis, not a literature pin.
Form (population_model.native_departure): departure probability (both ages, Veit rule) x (1 - r w(x)), w = the native
core/margin map (the 1902 seed shape) scaled to [0, 1], r ~ Beta(1.5, 1.5). r = 0 is the current model (test).
Kernel distances unchanged; fewer birds leave the native range.
Arms (70z holdout base): 77a real. Guards: 77n NULL (r = 0 at 69a MAP; pass: fitted r < .25 and demography within the
truth's band); 77r RECOVERY (r = .6; pass: r within .15, demography in band).
Prediction if dispersal smoothing is the problem: r well above 0; western core under-prediction shrinks; refined
habitat amplitudes and western K collapse relax; held-out density improves on 70z (-1.818 / -1.178).
Companion diagnostic (E076 probe 2): western residuals by observed density and habitat edge, with 68b as the
selection-artifact control.
