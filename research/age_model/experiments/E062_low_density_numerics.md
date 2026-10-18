# E062 -- low-density numerics: legacy pseudo-zero pad + float32 phantom floor (pre-registered 2026-10-18)

Found in code: rightpad pads every dispersal FFT with 1e-9 (not 0) -- a pseudo-zero the kernel spreads back into the
grid each step (continent-wide trickle of immigration, both precisions); FFT round-off negatives are clamped only after
reproduction and positive round-off is never removed (the float32 floor, E054c).
Switch population_model.numerics = {fft_pad: 0, fft_floor_ulps: 10}: zero padding; convolution output below
10 * eps(dtype) * max(output) set to 0 (stop-gradient threshold). Absent = legacy. 50 MAP steps under JAX_DEBUG_NANS
in both precisions on synthetic inputs: finite (tests/test_numerics_floor.py).
- 0034d forward check: 50c MAP params under 62a numerics, f32 and f64. Pass: eastern trajectories agree across
  precisions (vs legacy: f32 93 vs f64 1.1 in 1982).
- 62a = 50c + numerics, 62b = 55a + numerics: cold MAP -> refine. Read: f32 MAP loss vs f64 U agreement; spread timing;
  does the refine still need lam0 ~2 (spread-rate deficit) once phantom immigration is gone?
Not yet addressed (methodology, with the user): bird-scale extinction / demographic stochasticity (atto-foxes, and
stochastic establishment of small founder groups that a continuous density cannot represent).
