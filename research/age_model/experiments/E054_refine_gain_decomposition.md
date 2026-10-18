# E054 -- what does the refine's ~11-18k nats fit? (pre-registered 2026-10-18)

Every E050-E053 refine buys 11-18k nats and pays with a wrong structure whose location moves with what is free. Before any
further structural change: per-observation NB log-likelihood at MAP vs refined (exp/loglik_decompose.py --compare), for
50a (lit dispersal, Fmax contortion), 51a (V&L dispersal), 53a (V&L repro, Sa .95), 53b (V&L repro pinned, flattened).
Regroup per_obs.npz offline by GP corridor zone (west/plains/east, lat<49), era, route, count==0 vs >0.
Questions: is the gain concentrated (few routes/years, e.g. early eastern invasion front or modern plains positives, as
E047 found) or diffuse? Is it the same observations across all four refines (then it is one data feature the structure
lacks) or different ones?
