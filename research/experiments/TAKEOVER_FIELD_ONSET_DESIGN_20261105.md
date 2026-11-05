# Capacity versus demography and bounded disease onset

Registered before any fit of these constructions. Mechanics tests and identity
replays are validation, not scientific results. Parent: step43 factor/power-two/
Veit model, reconstructed at 1127a8c. No production adoption is proposed here.

## Questions and rival explanations

1. Does requiring abundance capacity and vital rates to read the same habitat
   direction force excessive demographic change? F2 gives K its own direction
   while Sa/Fmax/Sj still share one. If this explanation matters, the raw
   likelihood profile over historical demographic change should become less
   one-sided after all nuisance parameters adapt. A smaller MAP swing alone is
   insufficient evidence.
2. Does an indefinitely widening local disease ramp let historical habitat
   growth and disease cancel? D1 replaces that ramp with a fixed finite onset.
   If this matters, the fitted growth/disease decomposition should become more
   stable across starts without moving the compensation into recovery, severity,
   density response, observation effects or capacity.

## Exact constructions and historical novelty

F0: beta_j=a_j*f for [Sa,Fmax,K,Sj]. F2: beta_Sa=a_Sa*f,
beta_F=a_F*f, beta_Sj=a_Sj*f, beta_K=a_K*g; f and g iid N(0,I).
Feature kernel, input Z/path arrays, centering, intercepts and amplitude priors
remain the reference's. F2 adds 64 coordinates and an extra scale ridge; it is
not advertised as an identified parameterization or an amplitude-free model.
It contains F0 predictions at g=f, but the prior density there differs.

The implemented correlated extension uses beta_K=a_K*(r*f+sqrt(1-r^2)*g),
with coefficient correlation r in [0,1]. Its r=1 identity arm omits g entirely.
An estimated-r follow-up is earned only if the independent contrast is useful.
R101 loosens every field with private coefficients; historical rank2 groups
Sa/Sj versus F/K. Neither isolates this shared-demography/K contrast. E029
already showed that general decoupling can help and let K become an absence
switch; that failure mode is a required readout here.

D0: historical sigmoid gate with fitted tau. D1: x=clip((t-onset)/3,0,1),
gate=x^3*(10-15*x+6*x^2). This is exactly zero before local onset and one after
three years, with two continuous derivatives. Geographic lag remains fitted;
recovery, its timescale, severity modifiers and current-density dependence
remain active. There is no disease_tau_raw site in D1. The old disease_tau
deterministic is a compatibility placeholder in compact arms, not an inferred
onset timescale. Recovery retains its existing years-since-onset clock.

The three-year duration is an externally motivated structural sensitivity,
not a measured mortality-gate duration: the cited disease studies describe
abundance stabilization, which need not equal onset saturation. R88 tightened
sigmoid tau, retaining tails; R97 combined sharpness and altered habitat change.
Required controls before interpreting a D1 gain: compact gate centered on the
old onset midpoint; and fixed sigmoid matched to compact 10–90% width. Report
both first-affected-season versus midpoint semantics. Do not equate the compact
gate's saturation with constant mortality or elimination of endemic disease.

## Staged comparison

First validate unchanged F0/D0 and the F2 r=1 identity on actual inputs. Then
the smallest fitted matrix is F0/D0, F2/D0, F0/D1, F2/D1. All primary arms
exclude direct priors on historical lambda/vital-rate change, establishment
augmentation and sandbox forward monkey patches. Native field checkpoint
unavailability does not justify silently rebuilding or renormalizing Z.

Use the same constrained-site log posterior as the original replay, with
support-respecting optimization coordinates; exclude Jacobians from that MAP
objective and label it. Report data likelihood and priors separately. Do not
compare total posterior densities across different latent dimensions as a
model ranking. Full joint fits must allow all co-adapting disease, observation,
NestWatch and habitat sites to adapt; omit exactly inactive Veit logistic sites.

Short screens can choose which fits merit refinement, never establish a
negative biological verdict. Promising contrasts require at least three
dispersed starts, including lower/higher demographic-change states, and
matched softplus/exp-capacity and training-only nuisance controls. Keep inherited
full-support centering explicit; a training-only centering sensitivity must
relevel intercepts and distinguish the resulting prior-anchor change.

## Readouts and decisions

Fixed occupied-cell East/Plains/West support; equal-weight era means and paired
cell contrasts over 1966–1990, 1993–2010 and modern years. Also report all-land
and population-weighted views separately. Required: raw BBS/NestWatch likelihood,
existing spatial/future development scores, potential and finite-density growth,
Sa/Sj/Fmax/K, disease onset/peak/chronic toll and recovery, fronts/never reached,
Q/movement budgets, source/sink and immigration dependence, K-as-absence switches,
factor/amplitude/effective-coefficient norms, finite gradients and optimizer
termination diagnostics. Relative objective stop does not establish stationarity.

For informative arms, profile output-level historical and post-1993 growth
contrasts with nuisance refits; show likelihood-only slopes/curvature separately
from penalties. Upstream amplitude/coordinate controls follow with their induced
prior made explicit. The exact rescaling audit already establishes that merely
fixing a redundant factor amplitude cannot improve the likelihood family.

Retain a structural candidate only if its decomposition and prediction survive
starts, transfer and recovery checks. If F2 only uses extreme K, revise or reject
that explanation. If D1 only changes arrival-date semantics, report that control
rather than attributing the effect to bounded onset. If neither improves the
geometry, derive equilibrium/return-speed regulation as the next structural
question; do not search for a penalty that produces a preferred lambda swing.

## Outstanding scientific launch gate

The 391-result index is complete as an artifact inventory, not as a verified
reproduction of every historical intervention. Anchor replays and exact
source/specification novelty checks remain required before scientific fits.
The new mechanics can be tested and committed while that audit continues.
No fitted result is reported in this registration.
