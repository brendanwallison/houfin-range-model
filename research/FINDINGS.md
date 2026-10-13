# FINDINGS (verified only -- see PROTOCOL.md gate)

Each entry: title line, registry job ids, number with CI, caveats, eval_version.

## F1 (2026-10-12): Averaging independently trained DESK seeds improves place-specific backcast change and spatial level beyond any single seed; the raw-change gain alone is generic smoothing
- Jobs: E055/seed-ensemble-618a20970f67 (E055f), E055/single-seed-reference-53d12786e977 (E055f2),
  E055/smoothed-single-7fefc5e5e898 (E055k), E055/ensemble-size-0a785424d9fa (E055j). Interactive: E055i
  (research/exp/ensemble_gain.py) and the skeptic's scripts.
- Estimands: held-out cells, early withheld epoch vs 2005-25, ESK r24, noise-free ABBA corr, joint 6x6-block bootstrap.
- Ensemble (mean z of k seeds) minus the MEAN single seed:
  - local change (minus the 150 km leave-one-out average):
    - late: +0.012 [+0.005, +0.020] (new seeds 5-8), +0.015 [+0.007, +0.025] (8 seeds);
    - selected: +0.010 to +0.011, CI > 0.
  - modern-epoch level:
    - late: +0.022 [+0.020, +0.026] (new seeds), +0.026 (8 seeds);
    - selected: +0.011-0.012.
  - beyond the trend placebo (4 seeds vs seed 1, late): +0.016 / +0.018 (T0 1996 / 1986), CI > 0. A smoothed
    single seed gives +0.005 (CI includes 0).
  - held-out spatial skill (space_level): best of every arm tested (0.825-0.827 at the selected state).
- NOT unique to ensembling: the raw backcast-change gain (+0.016 selected, +0.026 to +0.031 late). A 54 km smoother
  on one seed matches it, because both shrink the prediction's variance.
- Size: gain ~ (1 - 1/k), ens8 / ens4 = 1.19-1.24. Most of it arrives by 4-6 seeds.
- Mechanism:
  - Established (arithmetic): the gain is variance reduction. The covariance with the truth is linear in the
    prediction, so averaging adds none; all of the gain comes from removing between-seed variance (13-23% of a
    seed's predicted variance).
  - Likely: that variance is seed-specific CELL-SCALE over-fitting (memorized places). It keeps 0.46-0.59 of its
    variance under 54 km smoothing, and E055a/E054 place DESK's temporal learning in the same over-fitting stretch.
- Caveats:
  - T0 1996 for the size curve and the replication; T0 1986 only for the 4-seed result.
  - Seeds 1-2 and 3-4 were trained on different code versions; seeds 5-8 on one.
  - Against the BEST single seed chosen after the fact, the gain is smaller (+0.004 to +0.022).
  - Effects are a few hundredths. eval_version pre-v1.
- Downstream (an implication, not a decision): the age model's readout could take the MEAN z of several DESK seeds,
  preferably late-state ones, at k times the training cost. It keeps the BLR form (z is still a linear feature map).
- CAVEAT (2026-10-13, E059b): F1 holds for the 96-species COMMUNITY composite. For the 273 OUT-OF-COMMUNITY species
  (the production-relevant case):
  - the ensemble lifts late-state species change by +0.034 (T0 1986, CI > 0) / +0.017 (T0 1996, CI includes 0)
    over seed 1, and spatial level by +0.006-0.009;
  - it adds NOTHING beyond the strong linear base (the same +0.002-0.004 as a single seed);
  - for these species every DESK configuration trails the trend-plus-covariates base by 0.09-0.13.
