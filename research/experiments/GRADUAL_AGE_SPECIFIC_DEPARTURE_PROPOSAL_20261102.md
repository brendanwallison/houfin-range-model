# Revision relative to the long-used Veit departure baseline

Implemented and queued on November 2 after full-data validation; fitted results are pending. The comparison is with gradual Veit departure, not the recently tested strict overflow rule. This is a minimal movement revision; a breeding-slot allocation model would be a separate, substantially larger model change.

## Concrete first candidate

Let x=N/K use exactly the existing pre-departure population and equilibrium-density reference. Replace the approximate clipped-linear shared-age rule with:

    p_young = x/(1+x)
    p_adult = rho*p_young

Start with rho fixed at 0.2 as an explicit sensitivity assumption, not a House Finch estimate. Retain the existing stage-specific convolution kernels, survival accounting, reproduction and census timing. K retains its current meaning; it is not renamed as a territory count. No new habitat field, threshold, curve-slope parameter or native-specific process is introduced.

The juvenile curve follows if the relative propensity to leave versus remain is x: departure odds are proportional to crowding. Equivalently, departure and local-retention events have relative rates x and 1. This is a simple behavioral assumption yielding a smooth saturating response, not a uniquely established House Finch law. The adult multiplier represents stronger breeding-site fidelity; it is also an assumption requiring sensitivity analysis.

| N/K | Veit approximation, both ages | Proposed young | Proposed adults, rho=0.2 |
|---|---:|---:|---:|
| 0.10 | 10% | 9.1% | 1.8% |
| 0.25 | 25% | 20.0% | 4.0% |
| 0.50 | 50% | 33.3% | 6.7% |
| 1.00 | 100% | 50.0% | 10.0% |

The actual smoothed project baseline reaches about 93%, rather than exactly 100%, at x=1. These are entries into kernels, not successful movements to another cell; kernel self-cell mass and subsequent survival remain relevant.

At low density, x/(1+x)=x+O(x^2): juvenile departure matches the baseline to first order. At high density it avoids routinely sending nearly every individual into movement. Adults receive greater retention at every density. This is statistical priority for adults remaining, not an exact allocation of breeding vacancies, and it does not create a surplus or guarantee population replacement.

## What would make it an improvement

The hypothesis is that retaining more established breeders while preserving low-density juvenile movement can reduce repeated redistribution from occupied sources and still allow expansion. It also reduces export from mature populations, which could help excessively early invasion or could instead starve the invasion front. Adult movement does not have an automatic additional mortality penalty in this project, so reduced departure cannot simply be counted as saved deaths. Measure source retention, successful off-cell recruits, destination survival and fitted spread.

No algebra guarantees lower potential growth. The current fecundity/crowding normalization can still conceal high unregulated fecundity behind near-replacement realized output. A lower fitted movement fraction is not evidence that this pathology has improved.

## Matched comparison

Use a two-by-two comparison: existing gradual curve versus the rational curve; equal-age departure versus the fixed 0.2 adult multiplier. Keep the 1951 initialization, shared habitat, fixed weak absolute Allee, NestWatch, K-floor choice and historical supervision identical within each comparison. Repeat seeds. This separates a known age effect from the curve change rather than attributing everything to a combined treatment. Adult-multiplier sensitivity at 0.1 and 0.5 is secondary to that comparison.

E012 and E037 already explored lower adult movement on earlier configurations. Adult fidelity is not a newly discovered mechanism. The specific question is whether the revised density response and age contrast improve the current yearling configuration under matched observations and initialization. This note does not claim an exhaustive code-history proof that the rational curve was never tried.

Compare the full pathology checklist, including potential growth levels and historical shifts; realized contribution and replacement maps in East/Great Plains/West; K/Allee artifacts and source bridges; arrival/establishment timing and frozen-habitat spread; actual reproductive output and effective survival; predictive performance; initialization/seed sensitivity; and curvature/computational suitability. Smooth probabilities help differentiation but do not establish HMC mixing or identifiability.

## Execution receipt, November 2 evening

Six new immutable runs are named `step41_{veit_adult20,rational_shared,rational_adult20}_s{41,42}`. Existing `step39_veit_fixed_s41/s42` supply the two shared-age Veit controls. All use the same fixed nominal 1951 initial abundance, yearling breeding, shared habitat, fixed absolute clumped Allee and NestWatch; original pseudozeros remain, CBC is disabled, the capacity floor is zero and native-specific departure is disabled. This primary comparison isolates movement; it does not simultaneously adopt the historical-range branch or a K floor.

The probability checks passed boundedness, stage allocation conservation, monotonicity, low-density behavior, exact inherited Veit-curve equivalence and finite-difference derivatives. Each new configuration type passed a full-data float64 likelihood/gradient evaluation and forward/contribution replay: maximum balance errors were 9.99e-16 (Veit with adult retention), 5.55e-16 (rational shared), and 8.88e-16 (rational with adult retention). Frozen source/module hashes and population configurations were checked before publishing; each run has a source manifest. The existing contribution override freezes the new age-specific probabilities for attribution, so it does not silently substitute the original Veit rule.

The six runs use queue priority 0614, after the remaining historical-range fit and before floor sensitivities. The single-A100 worker can claim them; all three lanes of development/regular workers were extended to include step41. Persistent collectors now download frozen sources, rescore accepted fits and schedule movement follow-ups. Local watcher tasks were restarted as 1145 and 1146.

At dispatch, small allocation 3544518 was running the final historical-range replicate; four-hour three-GPU allocation 3547292 was pending on priority. Additional two-hour three-GPU development allocation 3548096 was submitted and pending on resources. Workers claim shared queue files atomically, so simultaneous allocations do not duplicate fits. No fitted result from step41 is yet claimed.

Validation: `.research_runtime/gradual_smoke/VALIDATED_LOCAL.json`; remote monitor receipts `gradual_departure_VALIDATED.json` and `gradual_departure_specs.json`. Implementation: `research/exp/gradual_departure.py`, `stage_gradual_departure.py`, `verify_gradual_departure.py`.

## References

Veit and Lewis (1996), especially equations 13-18, provide the original stage-specific convolution framework and the shared clipped-linear departure example: https://www.math.ualberta.ca/~mlewis/publications/17Veit1996AN.pdf .

Greenwood and Harvey (1982) review the biological distinction between natal and breeding dispersal: https://doi.org/10.1146/annurev.es.13.110182.000245 . Bowler and Benton (2005) review condition-dependent emigration and distinguish departure, transfer and settlement: https://doi.org/10.1017/S1464793104006645 . These support the modeling structure, not rho=0.2 or this exact density curve as an empirically calibrated House Finch relationship.
