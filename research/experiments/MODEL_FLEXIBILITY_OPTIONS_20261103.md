# Small structural departures, local process error and movement contrasts

November 3, 2026. Design response to the user's four proposed directions. This is a proposal, not an implementation or queue receipt. Read with [the mechanism synthesis](MECHANISM_SYNTHESIS_20261103.md). Historical lambda inflation means change through time; absolute growth level remains a separate readout.

## Recommendation

Return to a limited mixture of strongly correlated demographic fields and small model discrepancies, while retaining the original gradual departure control. The shared-field restriction and regional low-rank discrepancy terms are themselves substantive assumptions; failures of their flexible predecessors do not establish that those particular restrictions are correct. Begin with interpretable contrasts, then combine components whose measured benefits are complementary. Do not enable all proposed flexibility simultaneously.

Three different problems require distinct probes:

1. Demographic processes may respond similarly but not identically to ecological similarity.
2. The growth/transport equations may omit small local deviations, including deviations poorly represented by a few smooth regional modes.
3. A mean-field transport operator may spread tiny fractions everywhere instead of generating rare groups of successful colonists. A small continuous error is not automatically a model of this third problem.

## 1. Strong correlation rather than exact identity of demographic fields

The most parsimonious first departure is a small private capacity component around the shared field. It allows places with similar demographic performance to differ somewhat in resource capacity, without initially adding three new private vital-rate fields. The next alternative is a small reproductive component, or a survival-versus-reproduction grouping, if the first contrast points there. The correct ordering should depend on residual growth/density patterns, not an assertion that K must be biologically independent of survival.

One example on the capacity link scale is eta_K = alpha_K + a_K H + tau_K G_K, where H and G_K are unit-scaled fields and G_K uses the same DESK basis with independent, strongly shrunk coefficients. If G_K is independent of H, the latent correlation with H is a_K/sqrt(a_K^2+tau_K^2). A private SD equal to .1 or .25 of the shared SD corresponds to correlations about .995 or .970. These are transparent sensitivity values, not literature-estimated House Finch correlations. Correlation after the nonlinear link need not equal the latent correlation.

Keep shared and private field scales separately identifiable under the chosen parameterization; do not freely fit an amplitude times another unconstrained amplitude. Fixed private budgets are a first screen; a learned hierarchical scale needs its own regularization and geometry checks. A private temporal field unrestricted from the spatial field would reintroduce the historical extrapolation problem. Initially use one private response on the same local and path features, rather than a new free time trend. If survival is later given a private field, its path image must enter juvenile movement consistently; Q still replaces Sj for movers.

Historical evidence is mixed, not a reason to forbid this branch. E029 partial decoupling improved fit and plausibility in one lineage; later variants produced high-capacity western sinks. E075's split spatial/temporal responses reduced growth level while flattening the Plains trough and inflating temporal survival directions. The latter is not the same experiment as a tiny private K field on the current shared habitat base. The K floor now permits asking whether small decoupling helps without using arbitrarily tiny K to express absence. It does not prevent high-K genuine sinks, so contribution and immigration audits remain essential.

## 2. Tiny independent cell-year abundance corrections

In the reviewed records/code, I have not found a completed full-fit test of independent abundance innovations for every land cell-year at a very small fixed scale. E070 uses regional spatial basis functions with annual coefficients; the 54-km screen used related regional abundance, reproductive and low-density terms plus resets. Their consequences cannot stand in for an exhaustive fine-scale iid test.

A direct state-space alternative is

    N_next(i,t) = N_deterministic(i,t) * exp(sigma*z(i,t) - sigma^2/2),
    z(i,t) iid Normal(0,1).

The -sigma^2/2 gives a unit prior-mean multiplier conditional on the deterministic state. It does not prevent the fitted innovations from supplying systematic growth, guarantee a zero long-run effect under density feedback, or make the posterior mean trajectory equal the noise-free trajectory. A common multiplier for both ages in a cell-year is the simplest initial abundance discrepancy and preserves the immediate age ratio; separate stage errors are an additional hypothesis.

Start with fixed annual log-scale sensitivities such as .005, .01 and .02, not a freely optimized variance at joint MAP. These are diagnostic sizes (about .5%, 1% and 2% local annual SD), not calibrated ecology. Use no additive offset that introduces birds into empty cells. Multiplication preserves exact zeros and does not solve rare arrival/establishment by itself. It can still amplify infinitesimal deterministic tails, which must be checked in front counterfactuals.

Small yearly error is not necessarily small lifetime error: without demographic feedback, independent log-errors with sigma .01 accumulate to log SD .10 over 100 years. With selection by the likelihood and nonlinear feedback, that random-walk approximation is not a bound. The prior cost of a systematic .01 annual correction at sigma .01 is substantial per cell but can still be outweighed by enough correlated count information. Inspect cumulative, regional and realized population effects rather than declaring a small sigma safe.

This adds about 237,490 latent innovations over 1966-2020 on 4,318 cells, or 535,432 over 1902-2025. Parameter count alone does not prove HMC infeasible, but full nonlinear differentiation and weak process/observation separation make this a substantial computational comparison. Noncentered standardized innovations are a natural starting parameterization, not an HMC guarantee. A restricted-window full-cell pilot is more defensible than assuming a regional-basis test answers the question. Collapsing or approximating latent innovations is a separate inferential implementation and must not be presented as exact marginalization of the nonlinear model.

Compare against the existing regional abundance formulation at a common marginal cell-year variance, the modest reproductive formulation and the no-discrepancy control. Do not equate raw basis-coefficient SDs across forms. Read held-out real BBS records, historical growth, full/no-innovation trajectories, innovation maps/cumulative effects, age survival, immigrant deficits and fronts. An observation-only counterpart can diagnose whether count noise offers the same benefit, but cannot by itself identify the cause of the residuals.

## 3. Small dispersal corrections

**Conservative local redistribution.** Perturb how already-arrived birds are divided among adjacent land cells, with nonnegative probabilities summing to one and a small explicit reallocation budget. This changes locations rather than creating abundance. One version is unguided local mixing; another uses a weak capacity or habitat preference. Comparing unguided and guided mixing at the same movement budget tests whether directional information is useful. The queued 25% capacity-weighted local placement is one guided version, not a completed test of tiny random transport discrepancies.

**Small probability changes around the existing transport kernel.** A mathematically transparent form is T_ji proportional to P0_ji*exp(sigma*xi_ji), normalized over destinations j for each origin i. P0 is the conditional transport distribution. This guarantees positivity and preserves the number transported before separately represented mortality and boundary loss. Never silently normalize away existing sea/off-domain loss or pay juvenile annual survival twice. Full independent edge-year noise is too large a first design; a small set of local directional or band perturbations is a more practical screen. Some origin-direction/kernel-band variants were already tested, so a new version must state what differs.

**Destination-weight perturbation.** Using xi_j rather than xi_ji gives T_ji = P0_ji*w_j / sum_k P0_ki*w_k. For a translation-invariant kernel the denominator and arrival sum can be computed with convolution/adjoint convolution, retaining FFT structure. It may still be costly per band and requires explicit boundary/survival accounting. It changes where arrivals go; a noisy preference need not be biologically better than unbiased movement. A bias toward high K alone may direct birds toward demographic sinks. A bias toward net demographic quality can manufacture self-confirming source selection. Neither is established by fitting better.

**Departure fluctuations.** Small origin-year perturbations of departure logits change the amount exposed to movement and Q, so they also alter effective annual survival. This is less clean as a test of geographic redistribution. Use only after explicitly separating survival exposure from destination choice. A free Q offset is not a small movement correction: earlier fits used it as an alternative juvenile-survival process.

Prefer the conservative spatial correction as the first transport discrepancy. Keep the diagnostic contrast unguided versus guided so the test does not assume that bird knowledge of better territories is the missing process.

## 4. Rare establishments ahead of the front

Better destination choice and stochastic establishment are different mechanisms. The current mean-field convolution already deposits small expected masses ahead of the main front, but that is not a realization in which one place receives a group and most receive none. E063 tested an independent-arrival occupancy approximation; it made establishment harder and did not remove threshold sensitivity. It did not rule out clustered propagules.

For a conceptual contrast, arrivals of mean m can be represented as groups of size b with group count Poisson(m/b). Expected arrivals remain m and variance becomes b*m. At fixed m, increasing group size makes arrivals less frequent but larger when they occur. This can help cross a mating/establishment threshold conditional on arrival; it can also reduce establishment probability or accelerate the front too much. Group arrival does not guarantee suitable sex composition, surviving mates or viable habitat.

A cheap first test is repeated forward simulations of independent versus clustered arrivals at the same expected movement budget and fixed demographic parameters, preserving stage/sex assumptions explicitly. Measure successful satellites, extinction, arrival timing and continental spread, including truly poor Plains habitat. Such a contrast can establish whether the missing stochastic mechanism is capable of the desired pattern. It is not a fitted likelihood or a drop-in differentiable HMC model. A discrete-state full fit would require marginalization or an explicit approximation whose bias is checked; smoothing alone does not establish biological equivalence.

## 5. Reaction-diffusion is a feasible contrast, with a different interpretation

Replacing the entire inference stack is unnecessary for a first comparison. Constant-coefficient diffusion over a fixed interval is a Gaussian redistribution operator, so a Gaussian kernel can replace the current long-tailed kernel while retaining the age structure, birth timing, observation model and convolution machinery. This tests the movement distribution/scale assumptions, not the claim that continuous-time demography is preferable.

A simple local population equation would be dN/dt = local_growth(N,x,t) + div(D*grad(N)). Habitat-directed movement would need another specified flux term. Spatially varying Fickian diffusion div(D*grad(N)) and ecological diffusion Laplacian(D*N) have different stationary behavior and should not be treated as interchangeable habitat models. Land boundaries and source/sink mortality also matter.

For a controlled kernel contrast, report second moments, self-cell retention and tails; matching mean distance alone does not match spread behavior. A Gaussian local component plus a small fixed long-distance component offers an intermediate option. The latter fraction should not be freely increased until it recreates the old extreme-tail fit. Allow the demographics to refit, since the earlier short-kernel fits compensated through reproduction and founders.

Deterministic diffusion still spreads fractional density and cannot itself represent rare successful groups. Replacing a fat-tailed integral operator with diffusion can slow satellite formation, but is not an automatic cure for lambda inflation, K/Allee artifacts, or partial migration. A discrete sparse diffusion operator may be computationally feasible at this grid; a new positivity-preserving solver, boundary treatment and gradients would need validation before production inference.

## An economical sequence

1. Read the existing floor x reproductive-variation and floor x local-placement contrasts, including the departure-curve/age controls.
2. Screen two currently missing restrictions separately: a tiny private K field, and genuinely local cell-year abundance innovations at fixed small scale. Reuse one common floor/base and compare actual variance budgets.
3. Compare weak unguided versus habitat-guided conservative local transport. This decides whether improving direction is needed, rather than assuming it.
4. Use matched-budget stochastic forward simulations to assess clustered establishment, and a Gaussian/diffusion kernel as a separate deterministic movement contrast. These are cheap capability checks before a new likelihood/inference implementation.
5. Combine only components that address distinct residual patterns. Re-evaluate historical lambda change, all P1-P15 concerns, predictive transfer and inference geometry together. Preserve historical-range versus original-supervision controls rather than adding count likelihoods simultaneously.

These are proposed tests; none is newly queued by this note. A new flexibility term earns its place by the compensation pattern it changes, not simply by lower lambda change or a better training objective.

## Established methodological context

Small process innovations and their separation from observation error are standard state-space modeling, with important estimability problems: [Auger-Methe et al. 2021](https://doi.org/10.1002/ecm.1470). [Morales et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2894961/) discusses diffusion, redistribution kernels, and the limits of mean-field approximations when discrete individuals and stochastic movement matter. These sources support the model classes, not the particular correction scales, strong-correlation budgets or propagule group size proposed here. Project evidence is archived in E029, E042, E043, E063, E066, E070, E074, E075, the 54-km misspecification suite and step27 directional-choice registration.

Implementation update: the initial field/abundance designs and matched weak-arrival tests are now implemented, validated and published. See [the execution register](OUTSTANDING_WORK_20261103.md) and [live receipts](OUTSTANDING_WORK_STATUS_20261103.md). Conditional branches above remain conditional; this original proposal is retained for provenance.
