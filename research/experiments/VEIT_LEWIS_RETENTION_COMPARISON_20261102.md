# Why Veit–Lewis spreads while strict overflow stalls

The main difference is the departure rule, not an absence of slowing growth in Veit–Lewis. The new excess-only rule is an additional project hypothesis. The paper's equation18 allows movement throughout the approach to equilibrium.

## Checked against the original paper

Visually checked printed pages 260–267, equations 5–18 and figures 3–6. The source copy is retained at `.research_runtime/veit_lewis_1996/paper.pdf`; author-hosted copy: https://www.math.ualberta.ca/~mlewis/publications/17Veit1996AN.pdf.

The paper models mate finding, then competition for nesting sites. Its stable no-movement population is derived from those processes and survival. Departure probability rises linearly with density relative to that population, reaching 1 there. Reproduction slows well before that point; the reported example has a maximum annual growth multiplier about 1.24. Its reproduction and movement phases are therefore compatible without needing a large surplus above equilibrium.

The nesting-site parameter and total equilibrium population have different meanings and units. With their parameters the stable population is about 3.758 individuals/km², not the nesting-site parameter 1.5. Our current K is likewise a population-equilibrium reference, not a measured number of breeding territories. Treating K as a strict territorial retention budget needs an additional biological mapping.

## The departure contrast

| Population as fraction of reference capacity | Paper's fraction entering movement | Shared smooth overflow fraction entering movement |
|---|---:|---:|
| 25% | 25% | 0.0000061% |
| 50% | 50% | 0.000454% |
| 75% | 75% | 0.0448% |
| 90% | 90% | 0.705% |
| 100% | 100% | 3.466% |

These are entry into the movement kernels, not guaranteed off-cell exports. The adult-priority variant allocates the retention budget by age; its actual rates are measured below rather than inferred from this shared-age table. Both models' distance kernels retain some birds locally.

## Measured in the actual fitted model

Replayed the accepted control and adult-priority seed 41 fits from their own frozen source/checkpoints. Counts below use actual pre-departure stage pools. N/K quantiles are weighted by modeled population; attempt fractions are total attempted movement divided by total starting population. Modern means 2010–2025, pre-disease means 1970–1989. These are model results, not independently measured movement rates.

| Region | Typical population/capacity before disease | Fraction attempting movement before disease | Typical population/capacity modern | Fraction attempting movement modern |
|---|---:|---:|---:|---:|
| East | 79.8% | 0.297% | 20.6% | 0.0000113% |
| Great Plains | 99.3% | 11.34% | 40.5% | 0.00142% |
| West | 94.5% | 2.33% | 56.5% | 0.0480% |

For comparison, the gradual-departure control attempts movement in about 31.7%,55.3% and67.6% of modern eastern, Plains and western starting individuals respectively. The stage/cell weights differ across fits. The table is not a paired biological estimate of true bird movement.

There is an important structural interaction with disease. The crowding coefficient is computed from survival before the additional disease mortality is applied. Departure still uses the original K. Disease can thus lower the sustainable population far below the point where overflow permits substantial movement.

Under frozen 2020 fields with movement removed, the adult-priority fit gives these population-weighted median positive equilibrium levels, relative to K:

| Region | Disease disabled | Fitted disease enabled | Actual modern population/K median |
|---|---:|---:|---:|
| East | approximately 100% | 20.6% | 20.6% |
| Great Plains | approximately 100% | 39.8% | 40.5% |
| West | approximately 100% | 51.0% | 56.5% |

Roots were located on a dense imposed-density grid, so tiny deviations above100% in the disease-disabled result reflect numerical interpolation/calibration tolerance. These are fixed-environment local equilibria, not forecasts of the changing spatial population. The close eastern match shows that the low population is approximately where the fitted disease-modified dynamics can sustain it, not simply a population that has not yet had enough time to fill the cell.

Crowding near the reference density already limits growth, as the user suggested. Weak mate limitation can lower the equilibrium further, and moving habitat/capacity can keep populations below a moving threshold. In this completed adult-priority fit, disease is a quantitatively large additional mismatch. It does not explain the entire low-departure behavior: eastern departure was already low before disease.

## Controlled numerical experiment

Implemented the paper's scalar reproduction, adult survival and separate adult/young kernels in a one-dimensional mechanism experiment. Hold reproduction, kernels and initial population fixed and change only gradual versus smooth-overflow departure. Repeat with a 5% additional annual population loss while leaving the original departure reference fixed.

With gradual departure, the population expands; with smooth overflow it remains almost confined to its initial patch over 50 years. The contrast persists at 2.5,5 and10 km grid spacings and across 6000/8000 km half-domains. The initial patch is integrated over cells so its total mass is identical at each resolution. An extra 5% loss lowers the no-movement equilibrium to 74.5% of the original reference. Gradual departure still permits 74.5% entry into movement there; overflow permits only 0.0408%.

This isolates a mechanism. It is **not a quantitative reproduction of the paper's historical simulation**: original reported speeds are not reproduced, spread distances remain resolution-sensitive, the founding patch is represented as a one-dimensional projection, and our finite-domain/kernel integration differs from their unpublished numerical implementation. Do not interpret the figure's absolute speeds as a model comparison against real invasion observations. The departure contrast and the local-equilibrium calculations are the relevant evidence.

![Controlled growth/departure comparison](VEIT_RETENTION_BENCHMARK_20261102.png)

Reproduction source: `research/exp/veit_retention_benchmark.py`; outputs `VEIT_RETENTION_BENCHMARK_20261102{,_fine,_coarse}.json/.npz`. Actual fit replay: `research/exp/run_retention_mechanism.py`, outputs `RETENTION_MECHANISM_step39_{adult_priority,veit}_fixed_s41_20261102.json`. No accepted fit or frozen source was edited.

## Other differences and implications

The original demonstration has fixed demographic parameters and a simplified homogeneous setting. Our model fits a changing continental landscape, western and introduced populations, disease and multiple observation sources. Its opportunity to compensate is much greater. The paper's reproduction/phase ordering is written between spring censuses; ours is written with movement/survival before the next births in postbreeding states. That distinction needs a census-aligned comparison; it is not by itself proof that our implementation introduces an extra biological year of delay.

The immediate lesson is not to increase growth until cells overflow. Departure needs to be compatible with populations that are viable and reproducing below nominal K. The original gradual rule already supplies that. If adult-first territory allocation remains the target, retention must instead be tied to a clearly defined breeding-space quantity and the timing of recruitment/births, rather than silently equating breeding slots with total-population equilibrium. Merely weakening crowding or lowering a threshold may improve exchange, but neither automatically resolves lambda inflation, mate limitation, disease or the full pathology checklist.

The earlier E051 experiment already implemented the paper's gradual density-dependent departure. Its apparent MAP benefits did not prevent later refinement compensation. Therefore the finding explains the failure mode of the stricter new rule; it does not establish that restoring the old rule is a complete solution to the project.
