# E073 -- E070 stage 2: persistent lambda(t) corrections with detrended process noise (pre-registered 2026-10-21)

Design: E070 (terms, gates G1-G5, ranking R1-R5, holdout mask). Stage 1 found transient noise alone does not release
the contortion; the user asked to run stage 2 alongside E071 (Allee) and E072 (route zero inflation).
Terms (base 70z: 69a structure, holdout 2021-2025 + 10% blocks, no floor):
- N: process noise sigma .05 from 1902, DETRENDED (process_noise.detrend_degree 1: each coefficient's yearly series is
  projected orthogonal to {constant, linear} over the active years -> eps carries no sustained regional trend).
- S: split readout (rank 24, ref window 1996-2025), deviation/level amplitude ratio ~ LogNormal(log .14, .7) per
  field (DESK's measured split; route 3 / 66m).
- H: free smooth habitat correction from 1902 (15 land cosines x linear time, w ~ N(0, .1); shared by the four
  habitat fields + its path image).
Arms: 73s0 (S), 73h0 (H) -- ablations, no noise; 73s (N+S), 73h (N+H), 73sh (N+S+H).
Guards (simulated at 69a MAP through 73h's structure, fitted with it):
- 73xh inject H only (w[0] = .5, cos(pi x)): pass if w[0] recovered >= 70% and noise RMS u at the 70n null level
  (<= .25).
- 73xe inject a noise pulse only (70r's, detrended as fitted): pass if every |H w| <= 2 prior sds (.2) and w[0] does
  not absorb the pulse (|w[0]| <= .1).
Read with E070's gates and ranking against 70z refined (-1.818 / -1.178) and stage 1.
DEVIATION from E070's design (recorded, not hidden): H's prior scale is NOT yet calibrated from DESK's out-of-community
temporal error; it is 67's .1 per weight. Reason: the calibration needs DESK's century habitat change in link units
on the full Z, a separate computation. R3 compares H's realised size (st_correction metrics, 1902-2023) with the DESK-driven change (z_attribution_since_early, figure 01a), so the
size can be judged after the fact; a calibrated rerun follows if H is used heavily.

## RESULTS so far (2026-10-21)
- 73xh CROSS-RECOVERY (inject H w0 = .5): H recovered w0 .29 (58%; bar 70% -> FAILS amplitude, like 70r: shrinkage
  under the N(0, .1) prior); other H weights <= .13; noise RMS u .16 (null level; PASS -- noise does not absorb H).
  Demography .52/.27/5.81 (truth .57/.31/5.22).
- 73xe CROSS-RECOVERY (inject noise pulse): H stays ~0 (w0 .02, max |w| .07): PASS -- H does not absorb a transient.
  Noise RMS u .18. Demography .55/.23/6.07.
  -> The two terms do not trade in either direction; H is shrunk (58%), the noise likewise (70r 29%).
- 73s0 SPLIT READOUT ONLY (no noise): MAP .55/.28/4.87 with the strongest plains trough of any MAP (plains .75 vs
  west 1.07, east .99; deviation/level ratio .19-.27 ~ DESK's .14) but poor MAP holdout (-1.997 / -1.332).
  Refined (2,533 it) .46/.44/13.9; plains .99 > west .93, east .97 (trough INVERTED); ratio overruled to .9-2.5
  (as 66m); holdout -1.820 / -1.170 -- the best SPATIAL held-out density of any arm (70z -1.178), temporal = 70z.
- 73h0 H only: refine STALLED at iteration 210 (|grad| 1.7e4-4.8e4) in the clean regime; continuation queued.

## Stage 2 arms (read 2026-10-22; 70z refined: .50/.43/14.7, plains .90/.99/1.10, holdout -1.817/-1.178)
| arm | refined Sa/Sj/F | plains/west/east | AD w/p/e | holdout t/s | notes |
| 73s N+S | .65/.16/13.9 | 1.00/.99/.88 | .05/.11/.03 | -1.827/-1.180 | ratio .2 -> 1.4-3.9 |
| 73h N+H | .37/.36/13.0 | .86/.96/.96 | .09/.16/0 | -1.830/-1.790 | spatial holdout collapses (check: pre-f64-viz-fix?) |
| 73sh N+S+H | .61/.39/13.1 | .99/.98/.92 | .03/.03/.01 | -1.815/-1.228 | best temporal by .002; spatial worse |
| 73h0 H | stalled at it 210; continuation stalled again (|grad| 1.8e5) | | | | |
None viable (G2 fails everywhere). Stage 2 as designed does not release the contortion either.
