# Reducing inflated potential growth: retrospective and proposed synthesis

**Recommendation ordering superseded November 3:** [Mechanism-based synthesis](MECHANISM_SYNTHESIS_20261103.md) re-examines the joint compensations, adds fixed-support temporal accounting and specifies discriminating tests. The original evidence and recommendation history remain below; the prior growth-ceiling argument alone does not establish a path to reducing temporal lambda inflation.

> **Objective correction, November 3:** In this project, "lambda inflation" means excessive increase in lambda through time. High absolute potential growth is a separate pathology. This survey correctly retained historical-change evidence (including approximately -.20 for long-tail redistribution and -.08 for reproductive deviations), but its prioritization overemphasized reducing the low-density reproductive ceiling. The crowding algebra demonstrates a possible reduction in growth level, not a reduction in historical growth change. Consequently, the claim that curvature should be the first test specifically for temporal lambda inflation is a working hypothesis, not an established conclusion from this review. Evaluate all queued mechanisms using modern-minus-early lambda on identical cells and equal weights, separately for East, Great Plains and West; report levels separately. Historical habitat gating, temporal response restrictions, redistribution and reproductive variation must be weighed by their direct evidence for temporal change, alongside the full pathology checklist. The original review is retained below to make the scope and prioritization error auditable.

November 2, 2026. Scope: the original October 16â€“23 age-model experiment records, October 25â€“30 habitat and 54-km ladders, and October 31â€“November 2 auxiliary-data, initialization and movement fits. Sixty original age-model notes have been copied without editing and SHA256 recorded in [the source manifest](lambda_history_review_20261102/SOURCE_MANIFEST.json). This review also uses the newer experiment reports and checked contribution outputs. It is a review of the recorded experiments, not a fresh numerical reanalysis of every historical checkpoint. Older withdrawn and unconverged results remain explicitly qualified.

**Recommendation:** make the next structural test the *shape of density regulation*, crossed with the already queued gradual age-specific departure comparison. Retain later established-state initialization, uncertain reproductive conversion, weak fixed absolute mate limitation and historical range supervision as independently assessed components. A small K floor addresses a separate absence artifact; it is not by itself a growth constraint. Do not respond to inflated growth primarily by adding travel mortality, shortening every movement, strengthening the Allee brake or restoring unrestricted demographic fields.

The central working hypothesis is that the current model can demand appreciable realized reproduction at intermediate densities while allowing a much larger maximum fecundity to disappear into its automatically strengthened crowding brake. Slow colonization and rapid growth after establishment then have to be fitted with a poorly matched combination of crowding, departure and habitat change. The hypothesis is supported by the code and a new frozen-fit diagnostic, but the proposed replacement has not been fitted.

## Keep two meanings of inflation separate

1. **Excessive growth level:** in the current yearling life cycle, potential annual growth is `Sa + Sj*Fmax`, with mate limitation, crowding, disease loss and movement removed. This can be excessive even if the fitted population barely replaces itself.
2. **Exaggerated historical change:** a landscape that moves from widespread early sinks to strong modern sources may be supplying invasion timing that movement and reproduction should explain.

An experiment reducing early-to-modern land-mean lambda change by .2 has not necessarily lowered occupied-cell maximum growth by .2. Many historical notes called the former â€œlambda inflation.â€ Tables below preserve that distinction. Delayed-breeding and yearling eigenvalues, Veit site-capacity reproduction and standard reproduction also cannot be pooled as one scale.

## What the less successful experiments actually contribute

All numerical contrasts in this table belong to their original matched family and definitions. They are not a league table of current candidates.

| Evidence | Useful gain or clue | Cost, qualification and implication |
|---|---|---|
| **E001/E002, fecundity and Allee profiles** | Lower fecundity partly lowered growth; weakening the Allee brake moved pressure into departure. The observations preferred a slower beginning followed by faster accumulation. | Fixing F at 3 moved adult survival to .73 and juvenile/adult survival near .97; lowering F alone did not identify growth. Low F is not the endpoint. |
| **E008/E013, settlement and crowding departure** | Habitat-directed settlement lowered delayed-life-cycle lambda about 2.02â†’1.85; a crowding departure arm recovered much of its count-fit gain without settlement. Local crowding and redistribution matter. | Settlement plus sparse-cell exodus became a compensating pair. Near-total retention in cores and perfect regathering are not acceptable evidence that real movement is low. |
| **E014/E030, perfect sorting and movers-only settlement** | Keeping residents in place distinguished dispersal from annual redistribution of every bird. | Perfect fitness sorting flattened vital-rate fields because observed birds occupied lower-quality cells. The lesson is bounded information and limited search, not that habitat-sensitive settlement is absent. |
| **E016/E019, release simplification** | Freeing the old overstructured founder input removed roughly 65% of the remaining cost of a low-fecundity fit; one historical release fit better than 90 founder coefficients. | Maximum growth remained high and shifted into survival. Later-state initialization solves a real resolution problem but cannot be expected to fix all subsequent growth dynamics. |
| **E020, disease as mortality rather than capacity loss** | Improved the recorded fit substantially and lowered delayed-life-cycle lambda about 2.24â†’1.80; plausible fecundity became much cheaper. This is an example of a useful structural change despite remaining pathology. | It did not solve survival/fecundity compensation. Disease-modified equilibrium also lies below the original K, which makes K a poor hard movement threshold. Retain a biologically meaningful disease model rather than deleting it to fix departure. |
| **E026, breeding-season observation** | Corrected a mismatch between survey timing and a post-breeding state; creates a distinct route for relieving reproductive demand. | It did not complete the solution. Census alignment remains a prerequisite, not a license to reinterpret all fitted fecundity as observed fledglings. |
| **E027/E028, plausible demographic corners and cold starts** | A plausible split could be close in likelihood, and a cold positive-density-departure basin had lambda about 1.36 versus 1.78 in the warm exodus basin. | The cold basin fit worse; a purported nearly free plausible corner partly reflected the old prior. Both objective decomposition and distinct starting basins matter. A better prior alone was overwhelmed. |
| **E029/E042/E060, shared fields and K/growth coupling** | Some partial decoupling lowered growth and improved counts; higher habitat rank sometimes lowered lambda. | It could manufacture high-K western sinks or invert the Plains trough. Conversely a single positively shared field reduced western sink problems but did not remove growth inflation. Restore only a specific missing response if residuals require it, not the whole private-field system. |
| **E043/E050/E051, departure and literature kernels** | Correcting the adult kernel and using gradual Veit departure removed an arbitrary movement clamp and produced clean numerical controls. | Refinement still reached delayed lambda around 2.0 and tiny-K/Allee artifacts. Lower adult departure had already been tested elsewhere; it is not a new discovery. |
| **E052/E053, yearling breeding and Veit reproduction** | Yearling breeding corrects the life cycle. Veit reproduction kept F modest in one refined fit. | Yearling breeding alone left high growth; Veit reproduction moved the distortion to adult survival .95, while pinned demography spread near-replacement sinks across the continent. This was not a test of changing only the curvature of the current crowding response. |
| **E055/E056/E071/E079, Allee scale and form** | Weaker or absent mate limitation sometimes lowered F and retained a Plains trough; smoother forms improved some optimization behavior. | Weakening did not consistently remove aggregate-growth demand. Relative Allee removed Allee-dead cells by construction but often moved pressure into survival or historical habitat change. Keep a weak fixed absolute effect and inspect actual establishment. |
| **E017/E058/E064/E065, subcell urban structure** | Some capacity-heterogeneity versions preserved Plains K and reduced Allee-dead area. This supports distinguishing local abundance support from cell-wide biological quality. | Survival tilts were entangled with perfect-information allocation, invalid gradients/stalls or failed/misspecified guards. Corrected versions still had unstable moderator signs or demographic compensation. These results reject those implementations as solutions; they do not rule out heterogeneous establishment or a small suitable area inside an inhospitable cell. |
| **E059 corrected by E064k/E069c/E076a, route effects** | Genuinely bounded effects reduced capacity collapse and Allee artifacts; temporal forecasts improved. | Early apparent rescues were withdrawn because â€œpinnedâ€ SDs were actually learned. Corrected route effects still left F and survival distortion and lost spatial transfer. Observational patchiness is one pressure, not the entire cause of high growth. Current routeâ€“observer effects already address part of it. |
| **E062/E063/E069/E075, numerics and occupancy** | Cleaning FFT artifacts established that some early timing conclusions were numerical. Occupancy gave a principled way to distinguish sparse expected abundance from a realized group. | Independent Poisson arrivals made establishment harder, and fine sweeps did not establish improved conditioning at the transition. Converged clean fits still contorted. A clustered-arrival test is not equivalent to repeating the failed independent-arrival closure, but it is a larger secondary branch. |
| **E068 and later static outside-truth checks** | Plausible simulated data could be fitted without automatically creating huge growth. The numerical optimizer alone is not a sufficient explanation. | These are conditional model checks, not proof that real biological growth is high. Later auxiliary-data simulations had centering-contract problems, so not every old recovery verdict can be treated as clean information about identifiability. |
| **E066/E075/E078, split readout** | E075c reached delayed lambda about 1.46, F 6.47 and little Allee-dead area with broadly comparable prediction: a substantive lead, despite missing other gates. | It flattened the regional trough and required extreme temporal response directions; route-effect combinations undid its demographic benefit. It shows that how spatial information is translated into rates matters, but does not justify unrestricted temporal coefficients. |
| **E067/E073, persistent habitat corrections** | A regional correction reduced the western sink and could retain the Plains as the least favorable region. | F could rise from about 8.7 to 10.7 while spatial structure improved. Genuine spatial improvement and worse reproductive distortion can coexist. The components should be evaluated separately before combining them. |
| **E070 and 54-km abundance noise** | Some versions lowered potential growth or maximum fecundity while preserving prediction. | They often moved the demand into adult survival, juvenile collapse, low K or correction-driven growth. Adding noise is not uniformly ineffective, but a low fitted F does not establish a better biological model. |
| **E080/E081 and October 26 shared readouts** | Tied/static temporal responses greatly reduced century growth change with small forecast differences; shared summer coefficients improved the habitat-change/prediction compromise. | Static habitat did not fix F demand and is not the scientific target. The old .14 calibration is not portable to improved DESK. These are evidence against equating every fitted historical change with real habitat change. |
| **54-km long-tail weights, two seeds** | Reduced the historical lambda increase by about .20 and F, with essentially unchanged spatial prediction. | Required mean juvenile movement around 625 km and massive tail weight, conflicting with the season-filtered banding diagnostic. The data benefit from redistribution/export, but the geographical mechanism used to provide it is suspect. Do not adopt the tail; preserve the clue. |
| **Shorter kernels and added travel costs** | Biologically relevant alternatives that test permanent movement versus mixed seasonal recoveries. | Shorter kernels raised F/founders and worsened prediction; fitted hazards were absorbed by other rates. More death or less reach is not a direct treatment for high reproductive demand. |
| **54-km fecundity deviations, count corrections and resets** | Two good-seed fecundity-deviation fits reduced the historical lambda increase by about .08 with neutral prediction. A modest persistent biological component remains worth considering. | The larger count-only improvement lost spatial prediction; repeated resets and low-density noise could take over the invasion. Mean growth cannot be hidden in a correction and then excluded from the diagnostic. |
| **NestWatch and its annual-conversion sensitivities** | Constrains realized reproduction and its pattern; with/without nests changes output at the same nest sites even when Fmax moves the other way. | Annual conversion remains consequential, and high Fmax can be hidden by crowding. Keep the data, estimate or bracket the conversion honestly, and examine which densities the records inform. |
| **1951 initialization and historical CBC range fits** | Removes the release-cell exemption, avoids the old founder backcast and supplies direct historical geography. Recent range-only replicates have similar broad demographic solutions. | Fixed broad abundance scenarios can push juvenile survival to its ceiling; range supervision alone still leaves potential growth around 2.7 and early arrival. It informs initial history and timing, not an annual survival/reproduction census. |
| **Strict overflow, two adult-priority seeds** | Isolated a real mismatch between nominal K, disease-modified equilibrium and departure. | Potential growth near 11.6, negligible successful exchange and more K/Allee nonviability repeat across seeds. This rejects this rule with our current K; it does not explain the older gradual-baseline inflation by itself. |

## A concrete new lever: crowding curvature

For the current yearling branch, temporarily ignore mate limitation and disease to isolate the algebra. Define:

```
F_rep = (1-Sa)/Sj                 # reproductive output needed for replacement
R     = Fmax/F_rep
x     = reproductive-stage density / K
F(x)  = Fmax / [1 + (R-1)x]       # current curve when R > 1
```

The important density here is the one actually passed to reproduction, after movement and the model's disease treatment of arrivals. It is not necessarily the pre-departure density.

Increasing Fmax also increases `R-1`. At x=1, F is exactly F_rep in this simplified case, however large Fmax becomes. Holding survival, K and density fixed, the partial logarithmic sensitivity of realized F to maximum F is:

```
d log F(x) / d log Fmax = (1-x) / [1 + (R-1)x]
```

Thus data near replacement can have very little direct leverage on Fmax. Above K the partial derivative can even be negative. This is a property of the equilibrium-normalized curve, not evidence that a high Fmax is biologically required. It also explains why annual reproductive observations need not directly shrink maximum fecundity. It is a conditional derivative, not the full response of a refitted spatial population.

**Proposed first alternative:** replace x by xÂ²:

```
F_2(x) = Fmax / [1 + (R-1)xÂ²]
```

This preserves the low-density maximum and the same mate-free, disease-free equilibrium reference in cells with R>1, while changing how strongly reproduction is suppressed at intermediate density. For R<=1 retain c=0, as now; do not force a source into existence. Keep absolute mate limitation and actual mortality outside this expression. K is still not a measured territory count.

This is the standard Maynard Smithâ€“Slatkin type of smooth density response, adapted to our age-specific fecundity. It is not the Hassell form with an exponent outside the whole denominator, and not the previously tested Veit pair/site-capacity reproduction. [Fronhofer et al., Oikos, doi:10.1111/oik.09824](https://doi.org/10.1111/oik.09824) derive nonlinear density regulation from resource dynamics and discuss this family. The literature supports testing curvature; it does **not** establish exponent 2 for House Finches. Fix 2 for the first sensitivity rather than add another weakly identified fitted scalar.

### Why this could actually lower lambda

At fixed Fmax, the alternative increases reproduction below K. That alone is not a reason to adopt it. The useful comparison is after allowing Fmax to refit.

Illustration: adult survival .55 and juvenile survival .25. Suppose the needed realized output at half K is 2.7 young per breeder. Both curves can supply that output and the same equilibrium:

| Curve | Maximum output per breeder | Realized output at half K | Potential annual growth |
|---|---:|---:|---:|
| Current crowding | 5.40 | 2.70 | 1.90 |
| Squared-density crowding | 3.24 | 2.70 | 1.36 |

These are illustrative algebraic values, not fitted House Finch estimates. In this example reproduction at the observed intermediate density is preserved while the previously extrapolated low-density growth is much lower. Both exact two-age equilibria were checked numerically and are locally stable. Source code and outputs: `research/exp/lambda_density_shape_derivation.py`, [JSON](lambda_history_review_20261102/DENSITY_SHAPE_DERIVATION.json).

![Two curves matching reproduction at half K](lambda_history_review_20261102/DENSITY_SHAPE_DERIVATION.png)

The revised curve has stronger overcompensation above K. It could create oscillations or undesirable density responses in a full fit even though the illustration is stable. Check the full age-structured feedback Jacobian and trajectories, not just the fixed-density growth formula. A fitted exponent or a much sharper transition is not the first step.

### What the new frozen-fit diagnostic adds

`research/exp/lambda_crowding_probe.py` reconstructs three frozen current fits, verifies their potential-growth fields against saved contribution arrays, and verifies realized reproduction where breeders exist. It measures the derivative above at actual reproductive-stage densities. It also computes approximate maximum local fixed-density growth with both crowding and mate finding present, with and without disease. These maxima are stage-projection multipliers at imposed density, not derivatives of the density-feedback map, and exclude dispersal.

The output JSONs are in [the review directory](lambda_history_review_20261102/). On the Veit control, modern breeder-weighted median sensitivity is about .37 in the East, .54 in the Plains and .18 in the West. Under the failed adult-priority overflow fit it falls to about .087, .039 and .018. For example, a small 10% increase in Fmax changes realized reproduction by only about .18% at the western median of that overflow fit, holding density and other quantities fixed. The curve really does conceal much of the inflated reproductive ceiling.

This does not make the overflow result harmless. On the same modern positive-survey cells, its 2020 mean potential growth is about 10.4/9.3/9.8 in East/Plains/West; even after allowing both mate limitation and crowding, the best local fixed-density growth remains much higher than the control. Disease does not erase that distinction. We should report both potential and attainable productive growth rather than dismiss either one.

The new common-cell support uses cells with positive BBS counts during 2010â€“2025, with equal weight per cell in each region and latitude 25â€“49 N: 206 East, 179 Plains, 662 West. It differs from older all-era positive-cell supports. Do not splice its levels into those tables. The density-sensitivity summaries instead use next-breeder weights within each fit; they describe where reproduction occurs, not fixed-support map agreement.

The approximate density-grid maxima below include both mate limitation and crowding and the fitted disease response, but exclude movement. Each cell is allowed its own best imposed density, so these are not a simultaneously attainable continental state or observed annual population growth.

| Frozen fit, 2020 | East | Great Plains | West |
|---|---:|---:|---:|
| Gradual Veit control | 1.858 | 1.082 | 1.483 |
| Historical-CBC range only | 1.858 | 1.082 | 1.475 |
| Failed adult-priority overflow | 4.913 | 2.897 | 4.227 |

This is a useful higher-level distinction. Historical-range supervision changes individual vital rates while these regional productive-growth summaries remain close to the control. The failed overflow rule changes productive capacity substantially as well as its component rates. Agreement of these regional means alone is not map-level identification; retain the full fixed-density surfaces and cellwise comparisons. Nevertheless it is precisely why survival/fecundity movements should not automatically be interpreted as changes in the growth process.

## The proposed synthesis, in order

**1. Complete the current departure comparison.** It tests whether stronger adult retention plus gradual natal movement reduces reproductive replacement demand without starving expansion. It does not guarantee lower lambda, and the earlier age-offset experiments make that uncertainty substantive. Keep both the curve-only and age-only contrasts.

**2. Cross density curvature with movement on the same observation design.** First run current Veit versus the proposed rational age-specific departure, each with crowding power 1 versus 2, using matched seeds. Two cells of this comparison already exist or are queued; power-2 fits are new proposals. If the combined movement result is poor, still read the power-2 Veit arm: a useful crowding result should not depend on declaring the new movement law a success. Preserve the 1951 initialization and other shared assumptions; do not simultaneously add extra habitat freedom, change K floors and change historical supervision.

**3. Transfer the useful components to the corrected historical-CBC design.** Keep a matched control with that design. Use the K-floor sensitivity to remove the tiny-K/absolute-Allee absence route where appropriate, while allowing truly poor demographic habitat and immigration-dependent populations. Check whether lower lambda returns pressure to low K or habitat gating. Historical range constraints and the new crowding form have complementary jobs; no reason to assume either solves the other's problem.

**4. Use reproductive and survival information jointly.** Retain NestWatch with annual-conversion uncertainty. Separate records in weakly and strongly crowded predicted conditions, and compare observed reproductive patterns rather than just a continental mean. Add the existing BBL recovery likelihood through the marked-bird transition so it constrains effective survival. Do not equate the regional pilot survival with raw Sa at origins, and do not multiply Sj by Q: the current Q branch replaces resident juvenile survival. Adult survival information alone can push demand into young or births; it needs the reproductive and movement accounting beside it.

**5. Revisit one small source of remaining variation only if needed.** The old fecundity-deviation term deserves a fresh current-base comparison because its useful two-seed historical-change reduction was modest but prediction-neutral. It should not be discarded merely because it did not solve every pathology. Use one low-rank, centered, bounded term and include its effect in total reproductive-growth diagnostics; do not hide its contribution by reporting only a stripped skeleton. Do not begin by restoring all split rate fields or continent-wide biological corrections. If fixed-cell residuals instead clearly implicate the shared habitat response, the E075 result motivates a single targeted response contrast rather than wholesale return to private fields.

**6. Keep clustered establishment as a second, larger branch.** The failed independent-Poisson occupancy experiment and the absence of support for a particular urban-strata/fitness-sorting model are not tests of clustered successful dispersal into small suitable patches. A compact stochastic or moment-based representation of group arrivals could separate rare successful colonization from average density without a full seasonal movement model. It must retain real sinks and use finite-propagule checks. Do not sell it as a proven smoothing cure; previous threshold sweeps explicitly contradicted that claim.

## Measure whether lower lambda is a real improvement

Every candidate should have one common scorecard, with East, Great Plains and West at the same locations and eras. Report potential growth levels and upper tails, historical changes, and maximum growth attainable with mate limitation and crowding. Also report actual reproduction, effective adult/juvenile survival, Runge contribution and local replacement, gross successful exchange and immigration dependence. These distinguish a genuinely more economical demographic explanation from a parameter simply moving elsewhere.

Preserve fixed observed-cell and all-land comparisons, source/sink reversal counts, source-bridge topology, K/Allee nonviability, finite-propagule establishment at observed sites, fitted and frozen-habitat invasion timing, and source-isolated invasion replays. A Plains trough is expected biology; the failure is using tiny K plus deterministic mate limitation to create persistent empty pseudo-sinks in otherwise productive habitat. Neither all sources nor no immigration is the target.

Use profiles that constrain **aggregate growth**, then reoptimize the compensators, to see the likelihood cost and which observations resist lower growth. A first profile at a fixed reference habitat can be useful but cannot substitute for mapping growth across fixed cells; field slopes must not evade it. Separate likelihood loss from changing priors. A transparent growth prior may be appropriate eventually, but increasing penalties on F alone has already been tested repeatedly and moved the pathology.

For HMC, consider a parameterization in reproductive surplus or aggregate growth rather than a huge fecundity counterbalanced by crowding. A mere coordinate change or renaming K as a competition coefficient does not change the likelihood family or resolve identifiability. Preserve the intended prior measure/Jacobian if it is only a reparameterization; explicitly label a changed prior as such. Local curvature, directional finite differences and the two-age density-feedback spectrum are cheap screens. Actual sampling remains necessary to assess mixing, but should be spent on biologically useful configurations.

## Decision

My first new biological test would be **the current standard reproduction with a modest, fixed change in crowding curvature, crossed with gradual age-specific departure**. The earlier failures make this more compelling: they repeatedly reduced one fitted rate while leaving productive capacity, habitat gating or movement dependence wrong. This proposal directly targets the gap between useful reproduction at intermediate density and an unnecessarily large low-density ceiling, while retaining one habitat field and adding no fitted spatial component.

The next question is empirical: does that gap shrink in a full refit without returning to extreme survival, tiny-K Allee artifacts, implausible Plains bridges, fast invasion or poor prediction? The algebra demonstrates an available route, not that the data will choose it. No new full-fit configuration was launched by this retrospective; existing queued movement and historical-range work was left intact.
