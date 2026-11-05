# Evening review, November 2: new completions and departure tests

The new departure proposal is implemented as six additional matched fits, with two existing Veit controls. It is separate from the completed overflow experiment. The primary movement comparison inherits the fixed nominal 1951 state, NestWatch and original pseudozeros, without CBC, a K floor or native-specific departure. This isolates movement rather than simultaneously changing historical supervision.

## Newly completed results

The following are population-weighted modern medians over each fit’s own population. They describe the fitted population; they are not fixed-cell map agreement or proof that the growth process is identified. Potential annual growth is Sa+Sj*Fmax before crowding, mate limitation and movement. Values near one for realized contribution do not clear potential-growth inflation.

| Fit | Adult survival | Young/adult survival | Maximum offspring output | Potential annual growth | Fitted arrival error, years |
|---|---:|---:|---:|---:|---:|
| Earlier no-CBC reference | 0.562 | 0.477 | 8.31 | 2.79 | -3.0 |
| Overflow adult priority, seed41 | 0.731 | 0.999 | 14.85 | 11.57 | -3.0 |
| Overflow adult priority, seed42 | 0.702 | 0.996 | 15.56 | 11.60 | -3.0 |
| Historical range only, seed41 | 0.524 | 0.538 | 7.74 | 2.71 | -3.0 |
| Historical + invasion range, seed41 | 0.481 | 0.650 | 7.19 | 2.73 | -3.0 |
| Historical range only, seed42 | 0.531 | 0.533 | 7.80 | 2.74 | -3.0 |

Negative arrival error means earlier than observed, using the existing 10%-of-occupied-density definition. The lower detection threshold is also retained in the underlying front audit; an unchanged median does not imply unchanged spread everywhere.

## Regional demographic processes

Contribution is the expected next-year breeders attributable to a starting breeder, including successful recruits elsewhere. Local replacement below is total-population replacement with movement removed at the fitted state. Export is successful off-cell next-year breeders per starting breeder. These are different quantities and retain their existing census definitions. K/Allee nonviability denotes cells with positive potential growth suppressed through the capacity/mate-limitation interaction; it is separate from genuinely poor demographic habitat. Regional contributions are ratios of totals, not percentages of source cells.

| Fit | Region | Contribution | Local replacement | Successful export | K/Allee-nonviable land | Potential-growth sink land |
|---|---|---:|---:|---:|---:|---:|
| Earlier no-CBC reference | East | 0.990 | 0.896 | 0.261 | 0.9% | 58.9% |
| Earlier no-CBC reference | Great Plains | 0.877 | 0.822 | 0.388 | 1.5% | 66.1% |
| Earlier no-CBC reference | West | 0.988 | 0.834 | 0.499 | 1.5% | 55.6% |
| Overflow adult priority, seed41 | East | 1.002 | 1.005 | 8.27e-08 | 21.7% | 11.6% |
| Overflow adult priority, seed41 | Great Plains | 0.989 | 0.999 | 2.66e-05 | 9.5% | 8.7% |
| Overflow adult priority, seed41 | West | 0.972 | 0.982 | 0.000398 | 19.5% | 6.2% |
| Overflow adult priority, seed42 | East | 1.001 | 1.004 | 9.91e-08 | 24.0% | 8.7% |
| Overflow adult priority, seed42 | Great Plains | 0.985 | 0.996 | 5.94e-05 | 10.2% | 7.8% |
| Overflow adult priority, seed42 | West | 0.971 | 0.981 | 0.000294 | 19.7% | 5.0% |
| Historical range only, seed41 | East | 0.991 | 0.896 | 0.257 | 0.5% | 60.5% |
| Historical range only, seed41 | Great Plains | 0.904 | 0.840 | 0.381 | 0.6% | 67.2% |
| Historical range only, seed41 | West | 0.990 | 0.855 | 0.487 | 1.6% | 57.9% |
| Historical + invasion range, seed41 | East | 0.991 | 0.908 | 0.244 | 0.0% | 61.2% |
| Historical + invasion range, seed41 | Great Plains | 0.922 | 0.872 | 0.331 | 1.3% | 68.6% |
| Historical + invasion range, seed41 | West | 0.991 | 0.886 | 0.465 | 0.7% | 60.4% |
| Historical range only, seed42 | East | 0.990 | 0.892 | 0.262 | 0.7% | 60.5% |
| Historical range only, seed42 | Great Plains | 0.906 | 0.838 | 0.39 | 1.5% | 66.3% |
| Historical range only, seed42 | West | 0.990 | 0.851 | 0.49 | 1.9% | 57.4% |

## Interpretation and outstanding checks

Historical CBC range supervision has not produced the near-total movement shutdown or order-of-magnitude potential growth seen in the first overflow fit. The seed41 historical and invasion-supervised fits retain successful exchange and population-weighted potential growth around 2.6–2.7. That is not a clean bill of health: this growth remains high, the fitted invasion remains about three years early under the main timing definition, and separate survival components shift. The invasion-supervised fit also loses some spatial prediction in the Plains.

| Fit versus earlier no-CBC reference | Temporal predictive difference [90% block interval] | Spatial real-record difference [90% block interval] |
|---|---:|---:|
| Historical range only, seed41 | +0.0009 [-0.0006, +0.0024] | -0.0032 [-0.0074, +0.0006] |
| Historical + invasion range, seed41 | -0.0015 [-0.0049, +0.0021] | -0.0074 [-0.0189, +0.0008] |
| Historical range only, seed42 | +0.0005 [-0.0010, +0.0021] | -0.0032 [-0.0070, +0.0003] |

These are exploratory prediction intervals conditional on fitted optima. Raw optimized objectives across different observation terms are not ranked as if they were the same likelihood.

The full checklist remains applicable: historical growth opening, frozen-environment invasion, potential-growth inflation, age-survival extremes and compensation, western immigration dependence, Plains K/Allee artifacts, establishment at observed sites, source bridges, effective mover/stayer survival and distances, reproductive agreement, founder/seed sensitivity, predictive performance, parsimony and HMC geometry. The tables above provide regional levels; fixed-cell cross-model reversals, bridge topology, finite-propagule establishment, source-isolated timing, and curvature for the new historical-range fits still require their follow-up outputs. They are not implicitly passed.

No candidate is selected from these preliminary completions. The second historical seed and remaining invasion seed provide the next replication checks. The six new movement fits will be collected and scored by the same persistent watchers.
