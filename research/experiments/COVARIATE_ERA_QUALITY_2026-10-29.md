# Covariate era quality: does starting in 1940 buy better covariates? (2026-10-29)

**Question.** Is there a year when the environmental covariates get clearly better, so that starting the downstream
House Finch simulation in 1940 (not 1902) would buy better inputs?

**Short answer.** No: 1940 is not a break in any stream over the US. Over the US, climate is not
measurably degraded in any decade. Human population (HYDE) becomes genuinely year-to-year in **1950**. Land-use type
fractions (LUH-3 crop/pasture/urban) are straight lines between decade values over the **whole** 1903-2024 span.
Built-up area (BUI) is straight lines between 5-year values throughout. The one era effect we can measure is precipitation
**outside the US**: in 1903-1919, 46% of eastern non-US cells (Hudson Bay lowlands) and 15-18% of western non-US cells
(Sonora / Baja) have muted year-to-year rainfall. That is mostly gone by the 1940s in Mexico and by 1970 around Hudson Bay.

Script: `research/exp/covariate_era_quality.py`. Output JSON and log are in the session scratchpad (`cov_era/`).
Input: stored yearly states `.../encoder/states/yearly_states` (1902-2025, float64).

## Method

- **Undoing the input smoothing.** The stored states are EMA-smoothed along years. The schema does not record
  `ema_tau` (built before it became provenance), so it was estimated. I inverted
  `s_t = a x_t + (1-a) s_{t-1}` for a range of tau values and kept the tau that makes BUI/HYDE exactly piecewise-linear.
  **tau = 2 exactly** gives a median relative second difference of 6e-14 for BUI, against 3e-3 at tau 1.9 or 2.1.
  The EMA started before 1902, so 1902's raw value cannot be recovered and the raw series runs 1903-2025.
  Every number below is computed on the recovered raw series. (established)
- **Climate** (Tave/Tmin/Tmax/PPT, median-elevation point, 12 monthly channels). Measured per decade, West (lon < -100) vs East:
  - year-to-year variance within the decade, relative to the mean within-decade variance of the 1990s-2010s;
  - share of cell-months within 0.05 sd and within 0.001 sd of the cell's 1961-1990 mean. The 0.001 sd test is the
    fingerprint of CRU's "no station in range, so insert the climatology";
  - anomaly correlation with the cell 108 km and 405 km to the east;
  - share of cells whose variance is below 25% of modern, split into US and outside-US cells.
- **LUH-3 / HYDE / BUI.** For each year, the share of changing cell-channels whose second difference is non-zero
  (> 1e-3 of their typical yearly step). Years inside a straight-line segment score about 0. Knot years score about 1.

## Numbers

**Climate, US and whole regions.** Median variance ratio by decade:
- Tave, West: 1.00 (1903-09), 1.42, 1.21, 1.49, **1.16 (1940s)**, 1.34, … 1.01 (2010s).
- PPT, West: 1.28, 1.18, 1.08, 1.03, **1.17**, 1.00, … 0.88.

The share of cell-months sitting exactly on the 1961-90 normal is 0.04-0.09% in every decade, against about 0.08% expected by chance.
The only exception is PPT East in 1903-1919, at 0.20% and 0.16%. Coherence is flat across the century: about 0.98-0.99 at 108 km for temperature,
about 0.92 for precipitation, and about 0.55 for precipitation at 405 km. Across all decades, **no US cell** has temperature or precipitation variance
below 25% of modern.

**Precipitation outside the US.** Share of cells with variance below 25% of modern (median location of those cells):

| decade | West non-US | East non-US |
|---|---|---|
| 1903-09 | 14.5% (29N, 110W) | 46.0% (52N, 84W) |
| 1910-19 | 17.5% | 25.6% |
| 1920-29 | 3.6% | 3.4% |
| 1930-39 | 1.5% | 4.5% |
| 1940-49 | 0.2% | 1.7% |
| 1950-59 | 0 | 1.8% |
| 1960-69 | 0 | 0.5% |
| 1970- | 0 | 0 |

For temperature, the only non-zero value is 0.2% of eastern non-US cells in 1903-09.

**HYDE (population density, urban and rural).** Kinks fall only at 1910, 1920, 1930 and 1940, with interior years at exactly 0.00. From **1950 to 2013** about 90% of cells
change shape every year, which is genuine annual data. 2014-2019 and 2021-2024 are again straight lines, with a knot at 2020.

**LUH-3.**
- c3ann, pastr and urban have knots only at decade years (1910 … 2020) and at 2024, with interior years at 0.00-0.04 in every era, including after 2000.
- primf and secdf (primary and secondary forest) change every year in every era (70-88% of cells), driven by LUH's own transition and harvest
  accounting rather than by its inputs. The 1944-48 and 1951-52 dips are small.
- No change at 1940.

**BUI.** Knots fall every 5 years (1905 … 2020) and nothing lies between them in any era. Share of US cells with any
built-up area: West 0.23 (1903), 0.49 (1940), 0.52 (1950), 0.68 (2020). East 0.76, 0.87, 0.88, 0.91. The growth is smooth, with no step.

## Documentation (web)

Verified means I fetched the page this session. Recalled means from memory, not re-checked.

- **CRU TS v4** (Harris et al. 2020, Sci Data, doi:10.1038/s41597-020-0453-3; read via PMC7125108), verified:
  - The correlation decay distance (CDD) is 1200 km for temperature and 450 km for precipitation.
  - "Land grid cells where no observation reaches are set to 0 (representing the climatology in anomaly space)."
  - Filling with climatology "has the unavoidable side effect of decreasing variance".
  - Station numbers change over time and can produce trend artefacts.
  - The `stn` variable (0-8 stations per value) flags the relaxed cells.
  - The paper maps coverage for 1910-19, 1940-49 and so on, but its text gives no North America counts.
- **climr** (bcgov.github.io/climr, `guidance_timeSeries` vignette), verified:
  - The `cru.gpcc` option is CRU temperature plus GPCC precipitation, 1901-2022. The anomalies are interpolated by angular-distance weighting, are about 0.5 degrees, are relative to
    climr's 1961-1990 baseline, and are added to the 2.5 km climr mosaic (change-factor downscaling).
  - The documented quality problem is a **late** one: CRU station numbers drop sharply at the end of the 1990s. The vignette gives no
    pre-1960 guidance.
  - We use `cru.gpcc` (`src/data/acquire/climate_climr.R`).
- **GPCC Full Data**: about 75,000-85,000 stations with records of 10 years or more, 1901 onward (Becker et al. 2013, ESSD 5:71, verified abstract only).
  The per-year station counts were not verified. Recalled: global counts are low before about 1920, rise strongly through the 1950s and peak around 1970-1990.
  The US and southern Canada are dense from the early 1900s. Northern Canada and Mexico are sparse early.
- **HYDE time steps**: in HYDE 3.2, 10 years over 1700-2000 and annual over 2000-2015 (Klein Goldewijk et al. 2017, ESSD 9:927), verified.
  HYDE 3.5 runs to 2025 (Utrecht/PBL). Its exact time steps were **not verified** on the web, but our measurement shows decade steps up to 1950,
  annual values 1950-2013, and then 2020/2025 anchors.
- **LUH2 method** (Hurtt et al. 2020, GMD 13:5425), verified: HYDE crop, pasture, rangeland and urban "were then linearly
  interpolated in time to produce annual maps". Wood harvest uses FAO from 1961 and is hindcast to 1920 from FAO per-capita rates, with a 1920
  anchor. LUH3 (v1.2, CMIP7) method text was not found. Recalled: it uses the same approach on HYDE 3.3/3.5.
- **HISDAC-US** (Uhl et al. 2021, ESSD 13:119), verified:
  - It is built from the ZTRAX year-built field of surviving structures, in 5-year steps.
  - Survivorship bias "increases towards early points in time" because demolitions are not recorded.
  - Year-built data are missing for the northern Midwest, Vermont, Louisiana and New Mexico.
  - Recall "drops almost logarithmically" going back in time. Correlation with census counts is above 0.8 since 1900 for dense counties.
  - None of this describes a step at 1940.

## Reading

- **CRU/GPCC relaxation does not affect the US in any decade** (established for our stack, measured). Over the
  US, the early climate has full-size year-to-year variability and no excess of exact-normal values. *Why*: the US has enough stations from the
  1900s to cover every cell within the CDD (likely, per recalled US station history). The CDD is 1200 km for temperature, so even a
  sparse West is covered. Note that full-size variance does not prove accuracy. With few stations, the anomaly can be the right size and
  still sit in the wrong place, which this test cannot see (unknown size of effect).
- **Precipitation outside the US is relaxed in 1903-1919** (established, measured). The affected areas are the Hudson Bay lowlands and Sonora/Baja.
  *Why*: precipitation's 450 km CDD and few stations there before the 1920s (likely). Most of it is gone by the 1920s and essentially all of it by 1940.
  A 1940 start does avoid this, but it is a non-US, mostly off-range issue.
- **HYDE improves in 1950, not 1940** (established, measured). *Why*: HYDE stores population as decade snapshots before 1950 and
  annual values after (likely: the input census series become annual in 1950). Before 1950 the covariate is exactly a straight line
  between decades.
- **LUH-3 land-use types never become annual** (established, measured). They are decadal-linear throughout, matching LUH's
  documented linear interpolation of HYDE (established). Forest primary/secondary fractions are annual throughout. No break in any era.
- **BUI has no break** (established, measured). It is 5-yearly throughout, and its documented early bias (survivorship, recall) fades gradually
  rather than at a step (established from the documentation; its size by era is unknown).

**Start-year implication (reported, not decided):** a 1940 start buys essentially nothing over the US. The only
year where a stream changes qualitatively is **1950** (HYDE becomes annual), plus precipitation outside the US, which becomes reliable from about 1920 (Mexico)
or about 1970 (Hudson Bay). LUH and BUI never change their temporal resolution. The CBC/BBS-era considerations
(CONSIDERATIONS M29) do not change this.
