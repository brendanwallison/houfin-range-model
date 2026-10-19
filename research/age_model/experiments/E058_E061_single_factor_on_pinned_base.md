# E058-E061 -- single factors on the pinned-Allee base (pre-registered 2026-10-18)

Base run_55a (V&L dispersal, standard reproduction, n50 pinned 0.059, single release, disease as mortality). One factor each:
- 58a subcell_strata: within-cell urbanisation slices -> can urban "stars" persist inside a plains sink?
- 59a route effects, sd PINNED 0.7 (free sd went to 7.5 = off switch): a bounded home for route-scale heterogeneity.
- 60a capacity manifold amplitude 1.02 -> 0.3: K less able to chase route-scale detail (hypothesis 1a).
- 60b manifold coupling .85 -> .5: K's detail stops dragging the growth fields.
- 61a fitness (ifd3-style) dispersers-only settlement.
Read against 55a (same base): MAP vs refined plains/west/east sink share + median lam_f, Allee-dead, vital rates, U.
Combinations only after single factors read, and only informed pairs (e.g. 58a x 59a: towns vs routes through towns).

## Roadmap -- not yet built (state 2026-10-18)
- E057 covariate-effect drift: P0 intercept drift (all fields) -> P2 built-up drift (generic / Plummer / Plummer-shifted)
  -> P1 agnostic rank-1. Null-calibration guard FIRST (zero-drift synthetic refit, then known-drift recovery).
- Front-speed diagnostic: simulated vs observed invasion front speed at MAP (does a lagging front explain the lam0
  inflation every free refine shows?).
- CBC observation model: 1940-1965 first (pre-BBS front), then all years; winter mapping, party-hour effort offset,
  feeder_hrs as effort covariate (recorded ~1975+), own scale and concentration.
- MAP optimisation adequacy: pinned V&L demography beat free at MAP by 7.5k (53b vs 53a) -> quick90 MAP is not a
  reliable optimum; free-vs-pinned MAP losses are not evidence.

## E059a RESULT (2026-10-18): bounded route effects remove most of the refine's contortion
Route effects, sd pinned 0.7 (log scale; fitted raw effects sd ~1, i.e. effects ~0.7). MAP clean (189,743 f32 / 189,755 f64).
| | U | Sa/Sj/F | lam0 | plains sink / med lam / med K | west | east | Allee-dead | corr | K trend z |
|---|---|---|---|---|---|---|---|---|---|
| 55a ref (base) | 186,101 | .64/.31/14.7 | 2.06 | .49/1.00/.012 | .52/1.00 | .42/1.20 | .16/.18/.01 | .809 | 5.6 |
| 59a MAP | (189,743) | .57/.31/5.16 | 1.28 | .67/.88/.147 | .45/1.04 | .43/1.10 | 0 | .862 | 7.8 |
| 59a ref | 169,804 (conv. 597 it) | .49/.44/8.44 | 1.62 | .50/1.00/.137 | .42/1.06 | .37/1.39 | 0/0/0 | .898 | 2.8 |
Giving route-scale heterogeneity a bounded home: NO Allee-dead cells after refinement, plains K stays at its MAP
level (.137 vs base .012), F demand roughly halves (8.4 vs 14.7), the K-trend pressure drops (z 2.8), and the refine
converges in 597 iterations. Confirms E054: the plains zeros the base refine bought with K-collapse are route-level
heterogeneity. Remaining distortions: Sj ~ Sa (.44 / .49), F still above 4-6, eastern lam 1.39, plains trough
shallower than at MAP (.50 vs .67 sinks). (U not comparable with the base: different likelihood.)

## E060a RESULT (2026-10-18): K-field amplitude prior cut to .3 changes nothing
Refined U 186,160 (base 186,101); Sa/Sj/F .66/.29/14.65, lam0 2.03; plains K .0099, Allee-dead .15/.17/.01 -- the
base refine almost exactly. The capacity manifold's amplitude prior is not the lever: the plains K collapse happens
regardless (hypothesis 1a closed in this form). Contrast E059a: giving the zeros a ROUTE-level home removed it.

## E060b RESULT (2026-10-18): manifold coupling .85 -> .5 changes nothing
Refined U 185,993; Sa/Sj/F .66/.28/14.87, lam0 2.03; plains K .0095, Allee-dead .15/.18/.01 -- the base refine again.
Decoupling K's field from the growth fields is not the lever either. Both E060 arms closed; the route-level
heterogeneity (E059a) is what moved the refine.

## E058a RESULT (2026-10-18; re-refined with the drift override): within-cell urban strata
MAP f32 205,568 / f64 206,134 (+566). MAP: plains .82 sinks, med lam .76 vs west 1.16, east 1.05 -- the deepest
clean-MAP trough so far; Sa/Sj/F .56/.28/4.94; town tilt of K beta_k .19.
Refined 186,220: Allee-dead ~0 (.03 plains) and plains K KEPT (.105) -- like route effects, sub-cell structure gives the
zeros a home -- the town tilt grows to beta_k .76 (towns hold e^.76 = 2.1x capacity per sd of built-up). But the WEST
collapses to a sink (.65, med .83 = plains .83), Sa = Sj = .35 and F 13.4: the contortion moves to demography and
the native west.

## E058b/c pre-registered (2026-10-18): town SURVIVAL tilt (user: feeders act on survival)
subcell_strata.sigma_s_scale: adult + juvenile survival in slice s shift on the logit scale by sigma (u_s - area-mean u)
(cell mean ~preserved); optional sigma_s_drift_scale: sigma(t) = sigma0 + sigma1 g(t), g linear 1966-2023 (held
before, obs-centred). 58b = 58a + tilt (N(0,.5)); 58c = 58b + drift (N(0,.2)). Read vs 58a: does a town survival
advantage (and its growth) let the refine keep the west a source and Sa/Sj realistic, instead of 58a's west sink and
Sa = Sj = .35? Sign and size of sigma0, sigma1.

## E061a RESULT (2026-10-18): fitness settlement on the V&L base -- MAP failed
MAP loss 2,743,226 (f64 U 2,935,882), log1p corr .22, plains .20 sinks / east .02 -- a pathological basin; refine
stopped at 226,874 after 277 iterations. Fitness ideal-free settlement + V&L density-dependent departure did not fit
from a cold start (earlier ifd3 runs used the learned/fixed departure). Not interpretable; not pursued unless needed.

## E059b RESULT (2026-10-18): route effects + K trend loosened x10
MAP clean (189,751 / 189,771). Refined 165,876 (59a 169,804: -3.9k). K trend x.26 -> x3.31 over 1902-2025
(z 2.8 under the loose prior) -- route effects cut the demand from 56a's x.12 -> x7.9 but a ~13x continental rise
remains. Refined Sa/Sj/F .67/.50/6.7, lam0 1.86 (F closest to 4-6 of any refine); plains .49 sinks med 1.01 vs west
1.14 (plains still below west), east 1.76 with K 1.4; Allee-dead .11 in the west only. The temporal K demand is real
and continental-scale: E057's window-held drifts are the principled version.

## E058b RESULT (2026-10-18): town survival tilt -- refine STALLED in the most realistic state yet
MAP 202,177 (f64 202,578). Refine stopped on the tolerance rule after 254 iterations at 200,332 (others reach ~186k):
plains .71 sinks med .78 vs west 1.10, east 1.04; Allee-dead 0; Sa/Sj/F .57/.30/5.1, lam0 1.26 -- all in range.
The survival tilt is unused (sigma .00); the K tilt went NEGATIVE (beta_k -.50: towns hold LESS capacity).
Continuation queued (0037h: warm from this point, tol_du .005, 8000 it): local optimum or plateau before the usual
contortion? If it holds near 200k the town structure has made a realistic basin locally stable.
## E058c RESULT: tilt + drift -- also stalled early (180 it, U 204,523) in a realistic state: plains .82 sinks med .71
vs west 1.13, east 1.00; Allee-dead 0; Sa/Sj/F .56/.28/4.98. sigma and its drift both 0 (unused). 58b and 58c (both
with the survival-tilt parameters, unused) stall in the same realistic region while 58a went on to the usual
contortion; continuations from the stalled points (copied, 10x tighter tolerance) queued as 0037h.

## E058b/c continuation RESULT (2026-10-19): the stalls are OPTIMISER FAILURES, not optima
Continued from copies of the stalled points (tol_du .005): 58b moved .00026 nats in 21 it, 58c 10 nats in 60 it --
with gradient norms ~2.0e6 / 1.3e6 (genuine refined optima: 55a 271, 58a 557). L-BFGS cannot find a descent step
along a huge gradient: the gradient is inconsistent with the function. Suspect: the town survival tilt (slice-varying
Sa/Sj) inside subcell_reproduction's allocation, whose gradient is an approximate implicit one (stop_gradient bisection
+ one Newton step); 58a without the tilt refined normally. The "realistic stalled state" is WITHDRAWN as evidence.
Fix before reuse: check the allocation's gradient against finite differences with sigma != 0 (unit test), and if it
fails, make the implicit derivative exact (envelope/IFT through the equal-lambda level) or tilt survival outside it.
Follow-up: the subcell allocation's gradients match finite differences exactly (sigma 0-1, beta_k 0/-.5, dense and
sparse; local test), so the implicit-gradient suspect is cleared. Leading hypothesis now: a KINK -- the allocation
switches slices on/off as the equal-lambda level crosses their thresholds, making U continuous but non-smooth; L-BFGS
stalls on a kink with a large gradient. 0037i (gradient anatomy) reports per-site |grad| and U along steepest descent
at 1e-7..1e-1 for the 58b stall and 58a's optimum.
## Gradient anatomy RESULT (0037i): at the 58b stall |grad| 2.03e6 is almost all subcell_sigma_s (1.91e6) and
subcell_delta_f (6.9e5); every other site < 3e3. A 1e-7 step along -grad RAISES U (+.18 vs -.20 predicted): the
gradient for the two tilt parameters is wrong there. Mechanism (likely): the allocation's single Newton correction
w - (S - M)/dS -- where the allocation curve is nearly flat dS -> 0 and the derivative explodes ~1/dS^2 (the local FD
test used uniform cells and never hit it). Strata results with sigma_s or delta_f free are suspect until the
correction is guarded/smoothed (e.g. stop the correction when |dS| is small relative to M, or soft masks in alloc).
