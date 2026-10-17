# E042 -- does the structure express the plains as a demographic trough? (pre-registered 2026-10-17)

User: the plains should have far MORE sinks than the flanks; the correlative SDM is not the target.
Cold run_40: sink share west .37 / plains .42 / east .34; K-WEIGHTED west .55 (!) / plains .45 / east .36;
lam_f medians 1.08 / 1.05 / 1.31. The plains are demographically ~identical to the native west; the west's
high-K core is largely sinks -- free coupling (rho .63) lets K and lambda decouple perversely.
Within the corridor: High Plains (105-100W) 53% sinks, median .99; eastern plains (97-94W) 15%.
- c40_plains3: plains likelihood x3 (diagnostic) -- can the structure produce a distinct trough?
- c40_rho95: coupling prior target .95 (logit sd .15) -- does reining in K/lambda decoupling restore western
  sources and push the plains below both flanks?
Viz both; compare unweighted per-region decomposition.

Results: plains x3 and coupling .95 leave the plains ~= the west (sink .40-.42 vs .37-.40; K-weighted west .55-.58).
CORRECTION (user, 2026-10-17): I first read this as "the structure cannot express a plains trough" -- WRONG.
Across the many MAP (Adam, prior-continuation) runs the plains were overwhelmingly worse than the west, with
broadly realistic demography; Allee-dead was uncommon. It is the L-BFGS REFINEMENT that moves into unrealistic
arrangements to eke out likelihood. E042 only shows the refined basin is sticky. The goal: a model whose
REFINED optimum stays biologically realistic. Next: E044 -- run_40 MAP point viz + decomposition of the
MAP -> refined gain (which observations the contortion buys).
