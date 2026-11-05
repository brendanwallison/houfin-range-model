# First structural screens: exploratory results

Four joint fits, one mapped-reference start each, 500 L-BFGS iterations per arm.
Every fit is finite; none demonstrates stationarity. The fitted source is
3d33f31. Compact evidence and checkpoint/readout hashes are in
`TAKEOVER_CAMPAIGN_01_EVIDENCE_20261105.json`; coefficient geometry is recorded
separately. Bulk checkpoints and traces remain outside Git.

All arms retain the same prepared features, observations, nominal priors on
existing active sites and initialized population. F2 adds an independent capacity
direction while Sa/Fmax/Sj remain shared. D1 fixes a three-year compact local
onset and removes the unused fitted sigmoid-width site. No direct prior on
historical growth or vital-rate change is used.

| Arm | Total data NLL | Spatial development NLL | Future development NLL | East potential medians 1966 → 2010 | East disease fraction, 2011–25 |
|---|---:|---:|---:|---:|---:|
| F0 / original onset | 114023.79 | 32109.90 | 21555.85 | .481 → 1.526 | .264 |
| F2 / original onset | 113616.53 | 30774.80 | 21661.48 | .541 → 1.824 | .268 |
| F0 / compact onset | 114191.48 | 32166.93 | 21501.18 | .619 → 1.232 | .151 |
| F2 / compact onset | 113752.22 | 30828.77 | 21625.06 | .755 → 1.616 | .168 |

Lower NLL is better. Total data includes all configured count rows and NestWatch;
the regional/development metrics use real BBS at 25–49N. Development blocks and
2021–25 have been repeatedly inspected and are not fresh confirmation. Potential
growth is mate-unlimited, disease-free and crowding-free; the table is not actual
population growth or the N→0 limit with the Allee effect. The two potential numbers
are spatial medians on the same ever-positive cells, not the median of cellwise
growth changes. Disease is the equal-cell/year mean fractional reduction applied
to survival at the actual forward density, not a fraction of all birds dying.

## What these screens change

Independent capacity improves in-sample and spatial development prediction, but
increases the potential-growth swing and worsens the future development score.
It does not yet support the hypothesis that the shared capacity direction alone
causes the inflation. Its fitted coefficient-direction cosine is .761 with
original onset and .802 with compact onset; these are realized coefficient
cosines, not the prior coupling parameter. Useful likelihood flexibility may
still improve lower-growth profiles, which these point fits do not measure.

Compact onset shifts the shared-field fit toward a substantially smaller growth
rise and disease toll. It costs about 168 total-data likelihood units at this
iteration cap; future development NLL improves about 55 and spatial development
NLL worsens about 57. This is a promising discriminating result, not a selection
verdict. Current gradients, one start and unequal progress through each basin
leave convergence and prior-versus-likelihood attribution unresolved.

## Diagnostic correction

The readout scan adds outputs at the actual post-dispersal, pre-reproduction
disease step. Its population states reproduce the unmodified forward scan
exactly for these fits. An independent NumPy calculation reconstructs its
mortality values, and a planted disease-off control returns exactly zero
mortality without changing the scan identity.

For the reference screen, occupied-East 2011–25 disease fraction is .2641 at
that stage; the older end-of-year-density approximation gives .3102. New births
and stage ordering matter to the density-dependent Hill term. Earlier disease
toll summaries require that qualification; the correction is not an improved
fit. Arriving juveniles already include journey survival Q and must not receive
an extra local Sj factor in movement accounting.

## Registered next actions

Campaign 02 distinguishes compact-start from compact-midpoint and a sigmoid
matched to compact 10–90% width. It then refines both original and compact-start
shared-field fits from their saved endpoints. The exact case settings are on the
structural branch in `TAKEOVER_CAMPAIGN_02_DESIGN_20261105.json`, committed before
launch at b92694b. Priors and nuisance freedom remain matched.

An additional CPU start is registered here before launch: compact-start F0,
seed 20261106, jitter .10 in the runner's support-respecting/whitened coordinates,
initialized at campaign_01 F0/D1, at most 1500 iterations. It tests start/basin
sensitivity alongside the GPU refinements; it is not yet the three-start standard
required for a stable scientific claim. GPU cores 0–7, CPU fit 16–31 and collector
8–15 use separate affinities. Device and all source/input/checkpoint hashes are
recorded by the fitter. No source changes are made to the running GPU worktree.

Do not select a model for attaining a preferred lambda. Before a structural
verdict, compare refined starts, date/width controls, output-level likelihood
profiles, actual disease timing/recovery, capacity/Allee escape routes, movement,
and training-only centering/nuisance sensitivity. Keep direct output priors as
labeled diagnostic benchmarks rather than the proposed solution.
