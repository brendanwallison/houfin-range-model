# Executed-model audit — November 5, 2026

This records source and numeric evidence, not an adoption decision. Source origins
and hashes are in `TAKEOVER_BASELINE_SOURCE_MANIFEST.json`; aggregate numeric
evidence is in `TAKEOVER_NUMERICAL_AUDIT_20261105.json`. The recovered code is isolated
from the original checkout and frozen run. Full pipeline/history audit is still
open; no new scientific fit has crossed that launch gate.

## Reproduction and source ownership

Frozen and reconstructed GPU replays both give objective 135590.79802835, regional
real-BBS NLL 110044.700261, spatial held-out NLL 32105.457889 and 2021–25 NLL
21557.483856. Each launch's repeat canary is byte-exact and finite. Hashes differ
across GPU launches while aggregate likelihoods agree to floating-point precision;
cross-process hash equality is not a numerical equivalence test. Imported module
hashes verify the reconstructed launch uses the isolated worktree.

R104 was alive during initial inventory and completed normally at about 04:48 CST.
Its saved objective is 135881.253412, BBS NLL 110139.082821, and eastern occupied-cell
potential lambda is .54455 in 1966 and 1.34933 in 2010; modern eastern disease toll
is .2590. Free n50 changed from raw -2.79690 to -2.80793. Its convergence flag is a
relative-function-reduction stop with gradient norm about 384, so stationarity is
not demonstrated. It retains the per-cell rate-change penalty and is an expanded
partial refit, not a structural solution or a full all-site joint fit. The extra
near-zero establishment slopes remain in its optimization vector under an extremely
tight penalty, a conditioning concern for the replacement runner.

## Active path, as executed

1. `src/config_utils.py` deep-merges defaults, overlay inheritance and explicit
   overrides; expands root variables. Overlay comments alone are not resolved
   behavior. `age_priors` binds many switches at import: different configurations
   require isolated interpreter processes.
2. `data_loading.load_data` loads prepared arrays and validates the feature/path
   contract. It builds observer/mask metadata and optional inputs, controls dtype
   and explicit device residency, and uses feature-rank truncation. Input hashes,
   feature lineage and upstream preparation need their own manifest beyond model
   module hashes. The large Z/path arrays are not truly streamed merely because a
   scan indexes one year at a time.
3. Occupied-habitat centering subtracts a constant feature vector before links.
   Audit warning: `occupied_habitat_center` chooses positive observation cell-years;
   its support can include subsequently masked observations. Distinguish centering
   and prior anchoring from a held-out likelihood leak, and inspect the prepared
   center's provenance before claiming a fully training-only test.
   Direct input audit: the used center exactly matches all 33,519 positive
   cell-years; only 25,600 remain under the fitting mask. Training-only and full
   centers differ by .01719 in feature-vector norm (largest coordinate .00748).
4. `sample_priors`: fixed coupling one gives beta_j = a_j f for j in ordered
   [adult survival, reproduction, capacity, juvenile survival]. a_j is
   softplus(w_scale_raw_j); per-rate gamma slopes are deterministic one. There are
   no private fields in this reference. The shared direction can make K and growth
   move together, but nonlinear links and juvenile/adult multiplication mean their
   effective temporal changes are not proportional.
5. `project_and_scatter_age_structured` projects local and path features. Adult
   survival is sigmoid(alpha_a + H_a). Juvenile survival is adult survival times
   sigmoid(alpha_j + H_j) on this ratio-form base. Fmax = softplus(alpha_f + H_f).
   Disease-free capacity = softplus(alpha_k + H_k), plus the configured positive
   floor in final K. The optional continental K trend is disabled.
6. Q is the juvenile-survival readout evaluated on path habitat, multiplied by
   the corresponding adult-survival path readout under ratio form. Thus changing
   survival coefficients changes local rates and movement survival together. A
   test of local rates alone must explicitly state whether Q changes.
7. Disease amplitude combines a bounded severity modifier, spatial basis,
   arrival-date modifier, native-lineage modifier, an arrival/lag gate and recovery.
   `b_late` multiplies each cell's *arrival-date* covariate, not current calendar
   year. Temporal change in amplitude comes from onset/recovery. The active target
   is survival; old capacity-target comments are inapplicable.
8. The yearling crowding coefficient is max(Fmax*Sj/(1-Sa+1e-6)-1, 0). Actual
   fecundity divides Fmax by 1+c*(N/K)^2 and multiplies absolute aggregated
   mate-finding. K is the no-movement, no-disease, mate-unlimited equilibrium scale
   when replacement is feasible, not universally the realized equilibrium. If
   c=0 or mate limitation/disease persists, the usual K-equilibrium claim fails.
9. Initial native abundance is tied to unfloored baseline capacity and the native
   seed map; the established eastern initialization is in 1951. The full scan still
   contains earlier years: those are not an empirically simulated release history
   for this arm. The initial amount is pinned; release Allee exemption is zero.
10. The forward step adds the invasion pulse, disperses, optionally settles,
    applies survival/disease and reproduction, clips/masks, and applies any enabled
    discrepancy/state correction. Those corrections are disabled in the reference.
    Veit departure replaces logistic departure with its smoothed N/K rule.
11. Adult and juvenile resident/arrival streams are distinct. Juvenile arrivals
    have already undergone Q survival; reproduction does not multiply their count
    by local Sj a second time. Disease applies its density-dependent added loss to
    resident survival and arriving juveniles. Audit origin accounting against this
    order, not a convenient one-rate scalar approximation.
12. The breeding observation mean uses the previous state (or seed at the start)
    plus a fitted fraction of current fledglings, then the route-count gauge.
    Zero-sum observer-by-route effects and a first-year effect multiply means.
    Concentration is phi*mean^kappa with quality adjustment. The likelihood mask
    excludes 2021–25 and pinned spatial blocks; raw distributions must score them.
13. NestWatch uses realized fecundity at each nest location/year, broods fixed at
    2.5 on this base, attempt covariates, daily survival/exposure and fledgling
    counts. It does not directly observe Fmax. CBC is disabled. The representative
    local age-structure Beta term is a regularizer, not another count dataset.

## Direct likelihood/parameterization audit

Trace accounting closes to 3e-11. Count likelihood (including all configured rows)
is -112057.684904; NestWatch is -1965.800447; their combined NLL is 114023.485351.
This differs from regional real-BBS NLL by pseudozeros and geographic support.
Pinned n50 still contributes its conditioned prior density; an observed trace
flag does not by itself mean independent observed data.

Rescaling f by .5 or 2 and a by its reciprocal changes effective coefficients by
at most 5.6e-17 and leaves both data likelihood terms unchanged to about 4e-10.
The total objective instead worsens by about 62.13 / 59.68 through factor/amplitude
priors. This is an exact likelihood-scale redundancy in the active model. Removing
it can improve coordinates; it cannot produce a new likelihood optimum. Direct
coefficient priors, fixed sensitivities and separate field directions are different
interventions and must be labeled separately.

A real-data gradient audit covers all 29 sampled sites / 19,275 scalar entries.
Every derivative is finite. The only exactly zero data-gradient sites are the
logistic dispersal intercept and slope, confirming the Veit dead-site finding
without inferring activity from parameter presence. Disease, habitat, observation
and NestWatch parameters all have nonzero data derivatives. Nonzero derivatives
are evidence of influence, not of identification.

## Disease-specific audit and candidate question

The configured mortality ceiling also sets n_max = .85*(4/ceiling-2), the upper
bound of the disease Hill exponent. Lowering the ceiling changes the exponent's
support and transformed prior as well as mortality. R90 therefore is not a pure
output-cap comparison with every other response family held identical.

The fitted onset width (tau about 3.9 years, 10–90% gate width about 17 years) can
express a long ramp. Older R88 sharpness tests constrained tau while leaving the
sigmoid's tails and recovery tradeoffs. A distinct structural candidate is a
finite-duration local onset with zero effect before arrival, a short transition
and an endemic phase; geographic arrival uncertainty is modeled separately.
This is a proposed mechanism, not a result or an asserted exact biological form.
Cross it with capacity-versus-demography field structure, without any direct
lambda-change penalty, after the source/history gate closes.

External primary evidence: [Hochachka & Dhondt 2000](https://pubmed.ncbi.nlm.nih.gov/10792031/)
reports stabilization within three years of local epizootic arrival and
density-dependent decline. This motivates a rapid-transient sensitivity; it does
not identify an annual survival-mortality gate numerically. [Hochachka et al.
2021](https://pubmed.ncbi.nlm.nih.gov/33821505/) describes the initial large decline
followed by stability or slower decline despite pathogen evolution. A rapid onset
does not imply zero chronic disease or forced recovery.

## Remaining gate work

- Complete upstream native encoder/export, kernel/path, mask/centering and raw
  input provenance audits; confirm source support of external biological anchors.
- Finish the historical code/artifact crosswalk and explicit novelty checks for
  field topology, amplitudes, regulation and disease. Preserve invalid and
  confounded comparisons rather than averaging their estimates.
- Replace the long monkey-patched partial-refit harness for new scientific runs
  with an explicit model configuration and correctly transformed active-site
  optimization. Remove inactive/unused experimental coordinates structurally.
- Preregister the smallest informative structural comparisons and output-level
  likelihood profiles, then launch the committed designs with measured resources.
