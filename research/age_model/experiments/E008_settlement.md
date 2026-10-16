# E008 -- ideal-free settlement (pre-registered in plans/experiment_map.md, 2026-10-16)

settlement.form = ideal_free: after dispersal each year, each b x b block (alternating half-block
offset) redistributes its population in proportion to K. Zero new parameters. Warm start run_23a
refined; NOT a profile -- the model changes, so the refine walks far (U at warm start 204k / 583k).

Results (worker 3514078):
| variant | U | dU vs 23a | dll | Sa | ratio | F | lam0 | n50 | disp int/slope | conc |
| 23a | 199370.9 | 0 | 0 | .54 | .95 | 10.9 | 2.02 | .283 | -0.58/+0.26 | .412 |
| ifd3 | 197642.1 | -1729 | +1452 | .52 | .87 | 10.6 | 1.85 | .180 | -4.52/-8.00 | .434 |
| ifd5 (4000 its, not converged) | 197650.9 | -1720 | +1376 | .62 | .57 | 10.0 | 1.83 | .217 | -2.10/-3.93 | .434 |
Gains: west zeros (all eras, +227..+291), west 1966-79 positives (+244 -> H9 partly relieved), east
1966-79 zeros (+239: the front no longer leaks ahead), east 1980-93 positives (+145).
Losses: east 1980-93 zeros (-277), east 1966-79 positives (-200 ifd3 / -375 ifd5), plains 1966-93 zeros.

Interpretation:
- By far the largest single improvement in the project; predicted direction (H1 relief: n50 falls
  .28 -> .18) holds, H2 relief (lambda0 falls) only weakly (2.02 -> 1.85), fecundity unchanged.
- The FFT kernel is effectively switched off (ifd3: sigmoid(-4.5) ~1% dispersing, slope -8): block
  settlement has REPLACED dispersal, at a fixed scale set by b (81 km at b=3) and the alternating
  offset. So the gain is "habitat-directed short-range movement" vs "habitat-blind kernel + path
  mortality". Front speed is now partly a structural constant of b -- a caveat for reading n50/lambda0.
- ifd5 reaches juvenile/adult ratio 0.57 at almost the same U: the juvenile split becomes free.
Follow-up: E010 (profiles warm-started from ifd3: n50 0.03, dispersal at prior means, ratio 0.5);
ifd3_f3 (Fmax 3 under settlement) still running.
