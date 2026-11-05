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

## RESULTS (2026-10-22)
- 77n NULL: PASS (r .27 at MAP -> .06 refined; demography .59/.27/4.92 vs truth .57/.31/5.22).
- 77r RECOVERY: PASS (r .56 vs .6; demography .59/.28/4.98).
- 77a REAL: r = .38 at MAP, .65 refined -- the data want a markedly more sedentary native range (departure in the core
  reduced ~65%). Refined .41/.40/12.7 (contorts), plains .92 / west .93 / east 1.04, Allee-dead .00/.03, plains K
  .153 = its MAP (.144) -- NO plains K collapse, the first refine without it -- holdout -1.826/-1.183 (~70z); the
  release's share of the modern population rises to .15 (the West fills the East less).
