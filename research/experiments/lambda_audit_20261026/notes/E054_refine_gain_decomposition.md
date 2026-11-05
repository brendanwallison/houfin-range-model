# E054 -- what does the refine's ~11-18k nats fit? (pre-registered 2026-10-18)

Every E050-E053 refine buys 11-18k nats and pays with a wrong structure whose location moves with what is free. Before any
further structural change: per-observation NB log-likelihood at MAP vs refined (exp/loglik_decompose.py --compare), for
50a (lit dispersal, Fmax contortion), 51a (V&L dispersal), 53a (V&L repro, Sa .95), 53b (V&L repro pinned, flattened).
Regroup per_obs.npz offline by GP corridor zone (west/plains/east, lat<49), era, route, count==0 vs >0.
Questions: is the gain concentrated (few routes/years, e.g. early eastern invasion front or modern plains positives, as
E047 found) or diffuse? Is it the same observations across all four refines (then it is one data feature the structure
lacks) or different ones?

## Result (2026-10-18). Per-obs NB loglik, refined minus MAP; zones = GP corridor (lat<49); nats as pos/zero counts.
TWO REGIMES, split by a float32/float64 disagreement at the MAP point (quick90 float32 loss vs float64 U at the same
params, from the refine logs): 51a 196,987 vs 197,002 and 53b 197,654 vs 197,660 AGREE; 50a 204k vs 234k, 50b +15k,
50c 206k vs 462k, 52a +11k, 53a 205k vs 303k DISAGREE (50d +7k, 51b/52b <1k).
- Disagreeing runs (50a, 53a): loglik gain 47k / 117k, 57% / 80% of it EASTERN POSITIVES 1980-1993 (26.7k / 93.3k);
  top 1% of obs carry 60-75%. The float64 trajectory at the float32 MAP params misses the eastern expansion; the
  refine is mostly repairing that, not contorting a good fit. Per-obs gains correlate .82 between 50a and 53a.
- Agreeing runs (51a, 53b): gain ~11k, broad (top 1% carry 31%), the SAME observations in both (per-obs corr .86):
  east 5.4k (positives 1980+), west 3.1-3.7k (zeros 1994+, positives 1966-79), plains 1.5-1.9k almost all ZEROS.
  MAP UNDER-predicts the early native west (mean y 21.9 vs mu 13.7, 1966-79; refined 18.0) -- the west halves over
  the record (21.9 -> 9.9) faster than MAP's dynamics allow -- and OVER-predicts the plains (y .30/.49 vs mu .59/.72
  in 1966-93); the refine buys plains zeros by lowering plains K (median .156 -> .010 in 51a) with Allee-dead
  patchwork -- a cell-level way to make sparse occupancy, i.e. the speckle/sub-cell heterogeneity hypothesis.
Implications: (1) "MAP realistic, refine contorts" is confounded with float32 numerics in 6 of 10 E050-E053 runs;
only 51a/53b are clean. (2) In the clean runs the refine's targets are a regional TEMPORAL decline in the west (drift /
K trend, E056-E057) and plains ZEROS (sub-cell/route heterogeneity, E058/E059). E054b (float32 vs float64 per-obs at
the 53a and 50c MAPs) queued to localise the numerics.

## E054c -- float32 phantom floor CONFIRMED (2026-10-18; 50c MAP params, forward sim in both precisions)
Eastern-third population (relative units): 1966 f32 .82 vs f64 .34; 1974 7.3 vs .46; 1982 93 vs 1.1; 1990 380 vs 4.9;
1998 585 vs 58; converging only after 2006. Minimum positive land density: f32 jumps from ~1e-13 to 4e-7 (1974) and
3e-5 (2000s) -- EVERY land cell carries a floor -- while f64 keeps ~1e-12 tails and 4% of cells at (0, 1e-6] through
the 1980s. The float32 FFT round-off floor acts as phantom immigration ahead of the front; under a weak Allee effect
it grows, and the float32 MAP's eastern expansion is ~20 years faster than the same parameters in exact arithmetic.
Consequences: (1) the 50-series / 52a / 53a MAP fits owe their eastern fit to the floor; in float64 their spread is far
too slow, and the refine compensates with lam0 1.4-2.0 (the "F 10-15" demand) -- the spread-rate deficit hypothesis
(2d) in concrete form. (2) Clean runs (51a/53b/55a: V&L density-dependent departure) do not need the floor -- but even
float64 tails (1e-13) are atto-foxes: fractions of a bird that regrow under a weak Allee effect.
Fix to decide (methodology): an extinction threshold in BIRD units (density below ~1 bird per 27-km cell -> 0), the
standard cure for the atto-fox problem, makes both precisions agree and the front speed honest. Then re-check
every MAP's f32/f64 agreement.
