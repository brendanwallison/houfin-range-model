# CBC compensation and the filling/departure mechanism

The first joint-CBC fits are a worse overall biological configuration than their matched no-CBC free-abundance controls. This is not evidence that CBC measures adult survival near .28. It exposes tension between observation/process blocks and leaves open whether the winter link, circle heterogeneity, reproductive conversion or demographic structure is responsible.

## What the likelihood actually pushes

At the completed seed-41 free-abundance joint fit, the CBC negative-log-likelihood and remaining negative log density have opposing gradients:

| Raw coordinate | CBC loss derivative | Remaining likelihood/prior derivative | Local preference of CBC alone |
|---|---:|---:|---|
| Adult-survival intercept alpha_a | -9,278.6 | +9,296.8 | Increase |
| Juvenile-ratio intercept alpha_j | -2,184.1 | +2,183.5 | Increase |
| Fecundity intercept alpha_f | -1,218.1 | +1,220.9 | Increase |
| Log initial abundance | -252.8 | +256.1 | Increase |

These are partial derivatives at one fitted point, holding other coordinates fixed, not a causal intervention or a profile likelihood. The survival intercept also affects linked juvenile survival. The near cancellation is direct evidence of conflicting pressure. The optimizer's global response includes shifting capacity, reproduction, age composition and survival. A high-production/low-carryover allocation is its current compromise; lower adult survival is not a direct inference from CBC recoveries or individual survival observations.

CBC currently observes a static seasonal redistribution of A_t + q_w J_t, with source-specific scale/effort/circle terms. The inherited BBS breeding readout observes total_{t-1} + phi J_t before its observer effects. They are different census conventions, not an explicit winter-to-spring transition. Numerical parity with the existing BBS implementation does not establish biological compatibility of these conventions. The joint fits put winter young availability near .99; the observation-only calibration at the old population had about .096. This flexibility reinforces the need to validate age/census mapping rather than interpreting the vital-rate changes literally.

Current discriminating tests already in the queue are uncertain NestWatch annual conversion, a different winter-displacement scenario and broader persistent circle heterogeneity. They test specific candidate explanations; no one cause has yet been established.

Gradient receipt: [CBC_LIKELIHOOD_PRESSURE_20261102.json](CBC_LIKELIHOOD_PRESSURE_20261102.json).

## Is filling before export still implemented?

Yes, the shared density-dependent departure rule is active in these fits. Native departure is absent, but this does not disable density dependence. The actual rule is a smoothed approximation to p(N)=min(N/K,1), for both ages:

    x = N / max(K, 1e-6)
    p = clip(x - softplus(10*(x-1))/10, 1e-6, 1)

At 10% capacity about 10% are selected for dispersal; at 50% capacity about 50%; at capacity the smoothed value is about 93%. The number entering dispersal is approximately N^2/K at low density, so sparse populations export much less in absolute numbers. However, it is not a hard waiting threshold and does not model individual territory occupancy. The nominal 0.8 target fraction is not an active threshold under this Veit-form switch.

Selected birds enter the stored spatial kernel. Some remain in or return to their origin grid cell, and some die; p is not the successful off-cell emigration fraction. The rule operates on a 54-km cell's average density, not actual nesting territories. [Veit & Lewis (1996)](https://doi.org/10.1086/285924) motivates the density-dependent/Allee model family; our smoothing and grid-scale implementation must be distinguished from a literal territory-allocation model.

## Does it provide an early quiet phase in the actual fits?

The seed-41 release-cell replay gives:

| Fit | Year | N/K | Fraction selected for dispersal |
|---|---:|---:|---:|
| CBC free abundance | 1951 | .492 | .492 |
| CBC free abundance | 1955 | .399 | .398 |
| CBC free abundance | 1965 | .536 | .535 |
| No-CBC free abundance | 1951 | .466 | .465 |
| No-CBC free abundance | 1955 | .213 | .213 |
| No-CBC free abundance | 1965 | .282 | .282 |

The founder cell first reaches 80% of fitted capacity in 1977 with CBC and 1984 without CBC, yet substantial departure is already allowed from the first initialized year. Thus the mechanism is present, but it does not produce "remain inside the release cell until it fills" in these fits. This is a concrete concern for the slow-then-fast explanation, separate from whether an extra native-range multiplier is useful.

Both reconstructed trajectories were checked against the saved fitted adult states to tolerance 1e-10. Receipt: [CBC_FOUNDER_DEPARTURE_20261102.json](CBC_FOUNDER_DEPARTURE_20261102.json). Only the two seed-41 matched fits were replayed for this diagnostic; this is not an all-seed conclusion or an isolated causal test of a different departure law.
