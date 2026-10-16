# Experiment map (Phase 1 -> Phase 2), written 2026-10-16 before E001/E002 results

Local, no-GPU checks run as soon as E001's decomposition lands (run_23a = point A):
- exp/baseline_checks.py: H8 (within-cell between-route spread vs NB variance; calibration by
  region) and H9 (observed vs fitted annual mean by region, 1966-1985).
- exp/plot_decomp.py --meta: dll by years-since-first-positive (H2), years-since-disease-arrival
  (H6), km-from-release (H5), plus maps and region x year curves.

Branches (each next step is a warm-start profile unless marked):

1. E001 cost at Fmax=3 SMALL (< 30 nats)
   -> lambda0 ~2 is prior/structure-driven (H7). Next: E003 confirm with Fmax=3 + ratio pinned 0.5
      (both components plausible) -- if still cheap, Phase 2 = adopt plausible demography by
      structure/strong prior, cost-free.
2. E001 cost LARGE and concentrated in [0,10) years since first positive, positives
   -> H2 fill-in. Next: E004 dispersal/immigration -- warm-start with more low-density dispersal
      (pin dispersal intercept high) at Fmax=3: does immigration buy back the fill-in?
3. E001 cost large, concentrated after disease arrival (east 1994-2007)
   -> H6. Next: E005 pin disease recovery parameters fast/slow at Fmax=3.
4. E001 cost large, early west / 1966-1979 (and H9 check shows the west lag)
   -> H9/H5: the established western population. Next: E006 the 1902 native seed level
      (initpop core fraction) and the western K level -- warm-start with initpop pinned higher.
5. E001 cost diffuse over positives everywhere + H8 check shows var/mean far above NB
   -> H8/H4: count model. Phase 2 candidate (structural, needs the user): a persistent route-level
      random effect (or cell-level overdispersion field) in the observation model -- expected to
      relieve the extreme fields, the K switch AND possibly lambda0.
6. E002 cost concentrated on zeros beyond fronts (plains, 1966-1993)
   -> H1 front pinning. Next: E007 is it the Allee or the habitat? pin n50 weak AND let the plains
      habitat sensitivity vary (or a static-habitat check, deferred per C7).
7. E002 cost small
   -> strong Allee is a soft optimum; Phase 2 can adopt a weak Allee prior safely.
Falsification (every ~5 rounds): rerun the leading result from a COLD start (MAP + refine via
38_model_experiment) and on a second warm start (run_21) to check it is not a warm-start artifact.
