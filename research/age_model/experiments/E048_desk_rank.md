# E048 -- DESK rank 48 / 64 (pre-registered 2026-10-17)

User: higher-rank / more robust representations gave marginal abundance gains in DESK experiments; more
flexibility might let the model capture e.g. an urbanisation signal that rank 24 drops. The model uses the top 24
of DESK's 64 latent features. Inputs rebuilt at latent_dim 48 and 64 into SEPARATE dirs (numpyro_input_r48/_r64;
shared rank-24 inputs untouched); cold MAP (quick90) -> refine -> viz of BOTH the MAP and refined points, run_40
structure. Rank 64 runs with XLA mem fraction .95 (Z_disp ~13 GB float64).
Read vs rank-24 cold run_40 (MAP: plains 70% sinks, lam .85; refined U 186,050.7, corr .808): fit; MAP realism;
whether refinement still erases the plains trough (more flexibility could capture urban signal OR give refinement
more room to contort); demography; east under-prediction at the MAP.
Budget: Phase 1 cap raised to 150 SU (DEB23008 at 309 SU on 2026-10-17 afternoon).

2026-10-17: rank 48 MAP realistic (P .66 sinks / median .88; W .55/.25/.96). Rank 48 float64 refine OOM twice (a
transpose duplicates f64 Z_disp [206508,124,48] = 9.8 GB); rank 64 float64 OOM (12.2 GB alloc) even at mem .95;
rank 64 float32 refine stopped at 166 its on float32 noise (U rose 25.7 over the 20-it window), U 188830 (unconverged).
Re-queued all three ranks (24/48/64) as float32 refines from their MAPs with a noise-tolerant stop (tol_du 1.0 over
200 its) for an apples-to-apples comparison + viz.

Float32 refines (tol 1.0 / 200 its) make no progress (r24 197844 -> 197844; r48 195244 -> 198029, worse; r64 195126 -> 195123):
float32 L-BFGS sits on its noise floor, as found in the HMC work. BUT the MAP stage (same Adam schedule) is informative:
rank 48 / 64 MAPs are ~2,600-2,700 nats better than rank 24 (float32 U 195,244 / 195,126 vs 197,844), and the rank-48
MAP stays realistic. Next: float64 refines with HOUFIN_Z_FLOAT32=1 (Z kept float32 -- lossless, halves memory).

r64 float64 (Z float32) refine: U 195,814.9 -> 187,516.4 in 651 its (stopped early; worse than r24's 186,051). Source/sink
REFINED: W .51/.41/.99 | P .61/.57/.90 | E .48/.18/1.07 -- the PLAINS TROUGH SURVIVES REFINEMENT at rank 64 (plains worst:
median .90 < W .99 < E 1.07), unlike rank 24 (P .42/.45/1.05). West still drifts toward sinks (K-weighted .41 vs .20 at MAP).
r64 MAP: W .39/.20/1.09 | P .64/.46/.86 | E .49/.14/1.02. r48 refine near convergence at 184,509 (best of the project).

r48 float64 (Z float32) refine: U 195,395 -> 184,477.3 in 2456 its (BEST FIT OF THE PROJECT; log1p corr .816), lambda0
1.68 (Sa .48, Sj .46, F 9.1), n50 .017. Source/sink REFINED: W .41/.57/1.09 | P .24/.25/1.29 | E .34/.24/1.31 -- the plains
trough is INVERTED (plains median lam 1.29 > west 1.09), west core still K-weighted sinks .57. More rank = more room for
the habitat field to contort (user's concern), not a clean urban channel. Rank 64 kept the trough only in an early-stopped,
worse basin (187,516). Higher rank does lower lambda0 (2.02 -> 1.68 -> 1.41).
