# Missing field and cell-year discrepancy comparisons: execution specification

Registered November 3 before reading results from these experiments. The earlier proposal documents did not implement or queue them. This record will carry separate staged, validated and published receipts; none of those states means a fit is running or accepted.

Primary question: can small departures from exact habitat-field identity or from deterministic abundance dynamics reduce exaggerated historical growth change while improving the joint demographic, establishment, movement and predictive explanation?

## Common design and controls

Use the original gradual Veit departure, yearling breeding, shared DESK basis, weak fixed absolute clumped mating, NestWatch with inherited annual conversion, fixed nominal 1951 established state and original absence supervision. Fix the common larger K floor at four times the pinned mating scale for this first comparison. Do not simultaneously change crowding power, CBC supervision, native departure or kernels. The two completed step39 larger-floor seeds are the existing matched controls, despite their differing fitted explanations. Add a third cold control, seed 43, rather than treating either original solution as the preferred basin.

Every new configuration gets cold seeds 41, 42 and 43. These measure optimizer/initialization sensitivity, not posterior uncertainty. New experiments have priority over remaining older secondary fits. Existing sources/checkpoints are not edited.

## Field comparisons: nine new fits

- K-private SD .10 relative to the shared coefficient SD: fixed strong prior correlation with the other fields, about .995 under the prior.
- K-private SD .25: corresponding correlation about .970.
- All four fields exchangeably correlated at fixed rho .99, using the already implemented shared-plus-private Gaussian coefficient construction. No fitted correlation scalar or independent time field.

Private K coefficients are iid over DESK features, preserving the same feature kernel; the amplitude budget is fixed. The private K construction preserves its shared coefficient and adds the private component, so total prior K coefficient variance rises by 1+tau^2. Report that explicitly rather than claiming unchanged marginal amplitude. Other fields are untouched. Actual fitted field/rate correlations can differ from prior correlations and must be reported.

## Abundance comparisons: eighteen new fits

Two forms at sigma .005, .01 and .02, with three seeds each:

- Independent innovations for every active land-cell/year, 1966-2020 inclusive: 55 x 4318 = 237490 standard-normal latent variables. Store only active years; do not construct a dense identity spatial basis.
- Regional eight-mode abundance innovations over the identical years, normalized to unit variance in every land cell. This is the matched-scale comparator, not the older unnormalized-basis experiment.

Apply the same multiplier to both age classes after the existing annual demographic transition: exp(eps), eps=sigma*z-sigma^2/2. Regional z has per-cell prior variance one by row-normalizing its basis. No detrending is applied to the iid formulation: temporal detrending would destroy the stated independence. Both forms are disabled in held-out forecast years and preserve exact zeros. Mean-one prior multipliers do not guarantee zero fitted growth contribution or zero long-run effect under density feedback.

Use fixed sigma; do not learn a variance that can collapse at joint MAP. Compare actual variance and cumulative innovation effects, not just a named sigma or total objective across models with different numbers of latent variables.

## Accounting and mandatory readouts

Process multipliers must be included in replay and BIDE closure as a separate process adjustment, not mislabeled as births, migration or mortality. Report both biological and full fitted origin contributions, and both skeleton potential growth and growth including the process multiplier. Regional isolation replays retain the fitted innovations; zero-innovation replays remove them explicitly. Finite-propagule intrinsic checks are labeled deterministic biological checks, not stochastic establishment probabilities.

The same East/Great Plains/West masks, equal-cell weights and era definitions are required for historical lambda comparisons. Also report fixed observed-cell support, fitted and frozen-environment invasion timing, effective mover/stayer survival, juvenile ceiling, K/Allee artifacts, source bridges, western regional persistence and internal exchange, innovation dependence, paired real-record prediction, source/precision provenance and seed behavior. HMC feasibility remains separate and is not established by a converged MAP.

## Validation and dispatch

Before publication require independent small-array checks of prior covariance, active-year indexing, mean-one correction, positivity/zero preservation and post-transition accounting; full real-data finite likelihood/directional-gradient and replay checks with nonzero new coefficients; and immutable per-run manifests. Local float32 and TACC float64 lanes must explicitly register the new families, controls and diagnostic adapters. A successful test or upload is not a completed scientific fit.

Expected new fit count: 9 field + 18 abundance + 1 third-seed common control = **28**. Two completed controls are reused. Conditional combinations, century-window noise, historical-CBC transfer and survival integration are listed separately in the outstanding-work register; they are not silently included here.
