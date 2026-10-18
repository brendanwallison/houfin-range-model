# E050 -- literature dispersal, fixed; sensitivity around it (pre-registered 2026-10-17)

User: E044-E048 were all run on a base whose learned departure layer clamped dispersal (~13-19% departure on top of
literature kernels that already encode staying -> ~85% of juveniles never disperse vs ~5% in the kernel). Those results
are NOT viable; nothing built on that base is a candidate. Literature dispersal is acceptable fixed, not learned; do a
sensitivity analysis around literature values.
All cold-started (MAP quick90 -> viz of the MAP point -> float64 refine -> viz of the refined point), run_40 structure,
departure pinned ~98% both ages (logit 4, slope 0), no density dependence, band weights off:
- 50a: juvenile mdd 330 (literature), adult 100, dispersers-only settlement
- 50b: 50a with settlement OFF (pure kernel)
- 50c / 50d: juvenile mdd 250 / 450 (inputs from the Oct juv_mdd_single_field sweep; derived tercile splits)
Read: MAP vs refined source/sink (plains vs flanks, K-weighted west), demography / lambda0, fit, whether the east and
modern plains are reached without habitat reshaping once birds actually move.
