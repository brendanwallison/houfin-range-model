# Original versus derived historical pseudozeros

The original model input contains 68,932 rows, but observation masking leaves effective weight 60,610 over 1,595 cells. The replacement has exactly the same pre-1940 total weight, spread across 2,107 cells. These are explicit range constraints, not independent surveyed absences.

The full cached old footprint, before model-input subsampling/masking, contains 1,961 cells. The derived footprint shares 1,961, adds 146, and removes 0. Thus the difference between 1,595 active original sites and 2,181 derived cells is partly subsampling/masking, partly true geometry. Full coarse-cell areas are nominal 2,916 km², not coastal land area.

| Region | Original full footprint | Original active sites | Derived sites | Original weight | Derived weight |
|---|---:|---:|---:|---:|---:|
| East | 677 | 515 | 677 | 19570.0 | 18746.8 |
| Great Plains | 129 | 95 | 154 | 3610.0 | 4169.5 |
| West | 177 | 133 | 260 | 5054.0 | 3912.3 |
| Other | 978 | 852 | 1016 | 32376.0 | 33781.3 |

## Strength at matched population

The derived NB2 calibration is fixed at the accepted no-CBC seed41 concentration 0.437242. The original pseudozeros used learned density-dependent dispersion c(mu)=k*max(mu,.001)^p, with reference p=0.859118. Original zero loss is c(mu)*log(1+mu/c(mu)); the derived loss uses constant k. The corrected curves include both this dispersion difference and the census/readout difference. No new CBC calibration scale or effort coefficient is fitted. This comparison is conditional on that calibration, not proof that all original fits had the same strength.

The old observation census is previous total plus fitted phi times current young; the derived constraint uses annual total. For a time-constant population with young fraction .5, the old expected count is 1.13295 times total density. The original density-dependent dispersion weakens sparse-population zero penalties relative to a constant-dispersion comparison. Consequently the derived penalty can be stronger at low matched densities even with identical weight; the earlier fixed-dispersion curves omitted this and are superseded. The table includes young fractions 0, .5 and .8 in the machine-readable output.

| Total route-scale population | Original loss (young .5) | Derived loss | Derived/original |
|---|---:|---:|---:|
| 0.0001 | 6.551 | 6.060 | 0.925 |
| 0.001 | 49.266 | 60.541 | 1.229 |
| 0.01 | 449.268 | 599.273 | 1.334 |
| 0.1 | 4026.969 | 5458.243 | 1.355 |
| 1 | 35468.208 | 31536.327 | 0.889 |
| 10 | 307071.746 | 84079.125 | 0.274 |

## Actual reference trajectory

At the identical accepted demographic point, original pseudozero loss is 7.462; derived pre-1940 absence loss is 81.838; added presence loss is 139.795. This includes geometry/readout differences at a common trajectory, not effects of refitting. Old pseudozero weight remaining in the replacement fit is exactly 0.

Derived absence loss decomposes into 5.389 at originally active sites, 1.502 inside the old full footprint but not actively weighted, and 74.947 in genuinely added geographic cells. Thus the larger reference loss must not be described as an elevenfold uniform increase in the zero penalty.

![Extent, regional allocation, loss and gradient](HISTORICAL_PSEUDOZERO_COMPARISON_20261102.png)

Coverage weighting is 1+log(1+distinct CBC circle/decade units within 216 km), normalized to the fixed weight budget. Regions without CBC coverage retain baseline historical-prior weight; they are not called surveyed absences. These choices and the fixed dispersion need sensitivity checks if they determine the scientific conclusion. A wider extent with lower per-cell weights may constrain new locations while reducing pressure at previously weighted sites.
