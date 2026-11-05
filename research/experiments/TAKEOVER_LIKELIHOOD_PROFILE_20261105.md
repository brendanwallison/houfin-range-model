# Conditional data profiles: what is actually learned about late growth change?

Question: does rapid local disease onset change the data-supported cost of
reducing historical growth change, or primarily select a different fitted
decomposition? No direct output prior is proposed for the production model.

The profiled scalar is RMS over the fixed ever-positive eastern cells of
log(lambda_potential_2010)-log(lambda_potential_1993). Potential is the
mate-unlimited, disease/crowding-free yearling index Sa+Sj*Fmax. RMS prevents
positive and negative cell changes cancelling as they did with regional-mean
constraints in R103. Report paired delta quantiles and level/rate maps as well;
an RMS alone cannot establish a biologically sensible decomposition.

Initial grid: original learned sigmoid onset versus compact-three-year onset;
each gets an unconstrained conditional-data fit and target RMS .30, .15 and .05.
These are diagnostic output constraints, not fitted priors or selection targets.
An augmented Lagrangian uses up to four 400-iteration stages, with target
tolerance .005. Unconstrained fits get at most 1500 iterations. A target not
reached is explicitly invalid, never a likelihood cost estimate.

All ordinary structural, disease, NestWatch and count-dispersion parameters can
adapt. Observer-route effects and the first-year effect are fixed at the SAME
original-model training endpoint for every arm/point. Thus these are conditional
likelihood profiles, not full likelihood profiles with integrated random effects.
They deliberately exclude all parameter priors and the age regularizer from the
objective. They do not establish identifiability under the full Bayesian model.
A second common observer reference at RMS .15 in both families is reserved as a
nuisance-conditioning sensitivity before interpreting a difference (two more
development contrasts, ten total for this question).

Remove the exact shared-factor/amplitude scale redundancy by fixing the Sa
amplitude to one and absorbing it into the factor, adjusting the other relative
amplitudes. For F2 an analogous K-factor gauge is available. This preserves every
effective coefficient and the actual count/nest likelihood; no prior measure is
asserted equivalent. Constrained-site posterior runs keep their old coordinates
and objective. The new conditional coordinate set is small enough for dense
BFGS, allowing full curvature history without a dense 19,000-coordinate matrix.

Compare likelihood costs within a family against its best unconstrained
conditional-data fit; a constrained fit beating that fit diagnoses optimizer
incompleteness. Report residual, raw data/prior decomposition, gradients,
numerical failures, field norms, disease timing/recovery, vital rates, movement,
source/sink behavior and existing development scores. No finite cap or relative
function stop is treated as stationarity. Conditional-data fits can escape into
extreme rate levels: such behavior is evidence about the ridge, not adoption.

After these checks, repeat informative contrasts with joint nuisance refits,
proper nuisance regularization/integration and recovery/transfer verification.
The model answer remains structural evidence with honest uncertainty; attaining
a chosen lambda value is never the promotion rule.

Activation checks: 23 profile/coordinate/controller/graph tests pass. Actual-data
evaluation reproduces the reference posterior 135590.798028349 and its likelihood
blocks. The conditional coefficient gauge passes an independent before/after
count/nest-likelihood identity check; a three-iteration real-data constrained
BFGS smoke run is finite. It is a mechanics check, not a profile result.
The profile supervisor is chained after the numerical control's verified
termination and also checks its workers/children, so two GPU owners cannot
overlap. The compact unconstrained conditional fit runs concurrently on CPU.
