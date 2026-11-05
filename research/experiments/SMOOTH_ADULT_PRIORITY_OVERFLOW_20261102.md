# Prototype: smooth excess-only dispersal with adult priority

The change exports excess above a retention capacity rather than making nearly all birds enter dispersal at capacity. It is implemented in `research/exp/priority_overflow.py` and integrated into new immutable step39 forks. Older frozen fits are unchanged. See [execution and validation](OVERFLOW_EARLY_EXECUTION_20261102.md) and [live queue status](OVERFLOW_EARLY_LIVE_20261102.md).

For nonnegative stage abundance x, capacity c, and positive width tau, define the bounded smooth minimum:

    S(x,c) = -tau * logaddexp(-x/tau, -c/tau)
             + tau * softplus(-(x+c)/tau)

This is equivalent to x - tau*softplus((x-c)/tau) + tau*softplus(-(x+c)/tau), but the log-sum-exp form avoids subtracting large nearly equal terms when abundance greatly exceeds capacity. It approaches min(x,c), is zero when either argument is zero, and lies between zero and min(x,c). It has no min/max kink at x=c.

For a single pool, retained birds are S(N,C); exported birds are N-S(N,C). With tau=.05*C, half-capacity abundance exports about 0.00045% of its birds; abundance equal to capacity exports about 3.47%; abundance at 1.5*C exports about one third. The present Veit-form rule instead selects about 50% at half capacity and 93% at capacity. Both refer to input to the spatial kernel, not successful off-cell movement.

Adult priority is a sequential allocation:

    retained_adults = S(A,C)
    retained_young = S(J,C-retained_adults)
    exporting_adults = A-retained_adults
    exporting_young = J-retained_young

The same positive tau is used in both allocations, including when the remaining young capacity is zero. In the sharp limit, 70 adults and 50 young with capacity 100 retain all 70 adults and 30 young, exporting 20 young. Adults themselves disperse when adult abundance exceeds capacity. Priority softens only near the capacity boundary.

The allocator conserves each stage and caps total retained abundance. Exported stage pools can feed the existing convolutions directly, without dividing by a near-zero pool to construct a departure probability. Actual mover survival and destination effects remain separate. An integration would also need an attribution adapter that freezes the ambient allocator before following origin tags; it must not use its nonlinear Jacobian as Runge contribution.

## Biological and inference choices still required

- Capacity and stage abundance must refer to the same census. The current K is a no-dispersal total-population equilibrium, not automatically a number of breeding territories. A territory interpretation requires an appropriately derived breeder/adult-equivalent capacity and explicit treatment of when resident survival opens vacancies. The prototype deliberately leaves this capacity conversion outside the allocator.
- Yearlings continue to breed when they obtain places; adult priority must not accidentally restore delayed maturation.
- Stronger retention can protect incipient populations and favor natal over breeding dispersal without an added native-range scalar. Its appropriateness for House Finch must be tested; it is not guaranteed to resolve CBC/census disagreement or historical invasion timing.
- Fix width initially (e.g. 5% of the original capacity). Making tau arbitrarily small sharpens curvature and can hurt HMC even though derivatives exist. First and second derivatives are finite in tested empty, saturated and zero-vacancy cases; HMC performance remains untested.
- Separate the stronger-overflow effect from adult priority in matched comparisons rather than attributing both effects to one mechanism.

The allocation follows the established resident-preemption idea: [Modelling migration in birds: competition's role in maintaining individual variation](https://pmc.ncbi.nlm.nih.gov/articles/PMC8580437/) uses first-arrival priority to occupied breeding territories. That supplies a conceptual precedent, not a validation of absolute age priority in House Finches or this particular smoothing formula.

Three tests pass: conservation/retention bounds and sharp-limit allocation, finite Jacobians/Hessians with a directional finite-difference check, and below-capacity leakage/boundary behavior. Prototype and tests remain separate from current model configurations.
