# Local autonomous experiment takeover plan

Prepared November 5, 2026. Planning only: no fits launched, running jobs stopped, branches switched, or commits pushed by this review.

## Objective and interpretation

Understand and improve the model's likelihood geometry and biological decomposition while retaining real historical habitat change, invasion timing, abundance, and source–sink behavior. A small lambda swing is not the selection objective. Direct priors on historical rate change are diagnostic comparators, not the presumed final solution.

Prior changes are legitimate candidates where they express defensible beliefs about the upstream field construction, number and coupling of fields, amplitudes, and spatial versus temporal responses. Every such change must be labeled as a prior change. A coordinate change cannot, by itself, improve an otherwise identical likelihood profile.

The takeover has two gates before new scientific experiments: reproduce the executed model from identified sources, and complete a code-to-experiment audit of every active component and relevant historical alternative. Unknown pieces remain explicitly unresolved; summaries or comments cannot substitute for executable evidence.

## Findings already checked in source

- The local sandbox runner imports the individual run's frozen code from `E:/Datasets/houfin/grid54/runs/<step>/code`, with a separate JAX runtime and the sandbox scripts on its import path. The Windows checkout is not the executed model. For example, frozen step43 `age_priors.py` is about 122 KB versus 52 KB in this checkout; `age_forward.py` is about 59 KB versus 12 KB. Reconciliation is essential before making changes.
- The current reference is `step43_factor_p2_veit_s41`: standard yearling reproduction, crowding power two, absolute aggregated mate limitation, Veit departure, initialization in 1951, a positive capacity floor, NestWatch, and observer-by-route effects. Stale overlay comments still describe older disease-on-capacity and release behavior.
- In the executed prior, four field coefficients have order `[adult survival, reproduction, capacity, juvenile survival]`. With coupling fixed to one, `beta_j = softplus(w_scale_raw_j) * f`: one shared direction and four positive amplitudes. Juvenile survival and journey survival share coefficients; capacity also controls departure through N/K. Field changes therefore propagate through more than the named local rate.
- Duplicate per-rate slope parameters are already fixed to one because they formerly multiplied the same amplitudes. The likelihood still admits a common factor/amplitude rescaling in the single-field construction; the priors price that direction.
- Existing rank-two groups pair adult/juvenile survival and reproduction/capacity. This is not a dedicated capacity-versus-demography split. Exchangeable R101 loosens all four fields together, and did not remove the growth swing.
- The old fixed-amplitude ladder did not prevent inflation: the shared factor norm increased. Repeating that intervention without distinguishing effective coefficient size from its factorization would add little information.
- Local fields use occupied-habitat centering; adult survival has a logistic link, juvenile survival can be a ratio of adult survival, fecundity has a softplus link, and capacity has softplus/exp alternatives plus its floor. The density brake is derived from disease-free rates. Disease acts later on survival at current density; arriving juveniles have distinct survival accounting.
- An existing `crowding_capacity` helper documents and implements delayed-breeding, linear-density algebra. It must not be assumed valid for current yearling breeding and power-two crowding merely because it is an available switch.
- The upstream community field is another model layer: `Z = M + A D + R`, including a covariate-change recalibration and a regional space-time factorization. Its C/L factorization has its own scale penalty. This layer must be distinguished from downstream demographic fields and their amplitudes.
- Git currently has 25 modified tracked paths and 2,069 untracked files, counted without collapsing directories. The active branch is `research/desk-temporal`; `origin` points to `brendanwallison/houfin-range-model`. Substantial model evolution exists outside this branch's tracked source.

## Phase 0 — Transfer ownership without losing experiments

1. Inventory Windows, WSL, E: run stores, frozen source trees, queue scripts, process IDs, completion monitors, logs, checkpoints, and available CPU/GPU/disk. Read the actual environment and resolved imports; record Python/package versions and source hashes.
2. Collect R104 and all R100/R101/R103 continuations if finished. A trace or queue entry is not a completed experiment. Preserve partial checkpoints and failure records. Do not rerun or terminate an owned task merely because a report is stale.
3. Establish a handoff boundary with a single writer/dispatcher. Existing processes may finish against their immutable inputs; competing launchers must be reconciled before taking the GPU. Keep local and TACC ownership distinct. This plan concerns local execution and creates no new TACC submissions.
4. Honor the STOP/lock mechanism and existing resource limits after checking their scope and current applicability. `research/budget.json` currently lists 22 GPU-hours/day, 30 working ticks/day, 10 development evaluations/question and 150 GB disk. Do not silently interpret an old cap as permission to exceed another active limit.
5. Create an append-only takeover manifest: task ID, parent, code/config/input/checkpoint hashes, exact command, execution state, output location, and collector ownership. Start/resume/finish/fail are separate states.

Exit: every active job has one owner and a collector; no unknown source is eligible for a new fit.

## Phase 1 — Reconstruct and audit the complete executed model

Produce `MODEL_EXECUTION_MAP` and an experiment crosswalk, both versioned. For each component include equations, units, shapes, order in the forward pass, resolved config, source location/hash, latent sites, priors/pins, data support, active gradients, diagnostics and historical interventions.

Trace all of the following end to end:

1. Data ingestion and provenance: BBS counts/zeros, pseudozeros, NestWatch exposure and annual conversion, banding cohort and reporting likelihood, CBC/seasonal optional blocks, holdout masks, centering, gauge, initialization and seed maps.
2. Encoder and export: raw covariates, transforms/coverage/fills, DESK reference features, M/A/D/R components, seasonal coupling, EMA/interpolation, raw versus averaged/native grids, feature truncation, feature kernel and normalization, and path-feature construction. Confirm which required artifacts are local; missing encoder checkpoints remain a concrete dependency.
3. Habitat coefficients: single, exchangeable and rank-two constructions; field order; factor norms and amplitudes; former gamma products; level/deviation readouts; centering and intercepts. Identify exact and approximate scale, rotation, intercept and coupling symmetries.
4. Local demography and regulation: actual yearling life cycle, adult/juvenile ratio, census state, fecundity, mate-finding, crowding coefficient and power, capacity link/floor, and equilibrium where rates do not permit self-replacement. Derive rather than assume which quantity K represents.
5. Movement: cell-integrated kernels, band/direction weights, path operator, edge/land normalization, departure, Q, resident versus arriving juveniles, and settlement. Print per-target gradients so inactive logistic departure sites cannot recur as experiments.
6. Disease: documented arrivals, fitted lags, onset gate, recovery fraction and timescale, severity fields/modifiers, current-density Hill term, age-class order and compensation with crowding. Distinguish transmission scale from a resource scale.
7. Observation model: breeding-season count readout, fledgling availability, observer/route/year effects, quality multipliers, density-dependent concentration, all masks, nest and optional survey likelihoods. Score raw distributions before masking.
8. Inference and numerics: continuation schedule, MAP coordinates, priors and deterministic factors, transformations/Jacobians, optimizers, precision, clipping/floors, FFT handling, checkpointing and diagnostics. Explain discrepancies between frozen and live implementations before adopting either.

Read historical E001–E081 notes and original artifacts by component, including withdrawn/stalled cases; the grid54 ladder and structural controls; R0–R104 code, specifications, raw summaries and relevant traces. Build rows recording what was actually changed/fitted/held fixed, not just an experiment's title. Classify evidence as reproducible, conditional, confounded, invalid or missing. In particular audit E029/E060, E066/E073/E075/E080, grid54 steps 2/4, R101, and all amplitude/field comparisons.

Mandatory replay anchors: current reference, an early fixed-amplitude arm, exponential-link control, disease-cap arm, R97 late-change restriction, R101, and R103b. Compare forward arguments, initial/final states, likelihood blocks, priors and diagnostics. Use short checks first; no full refits merely to recover an existing readout.

Exit: an independent reader can reconstruct every active model contribution, and each proposed experiment has a defensible novelty statement. This is the prerequisite to Phase 3, not work deferred until after launching it.

## Phase 2 — Preserve and organize the repository

Do the inventory and preservation early; create the runnable baseline before experimental implementation.

- Record HEAD, local refs, status, tracked diffs and untracked inventory; preserve local-only source and run manifests before any staging. Identify concurrent writers. No reset, clean, forced checkout, wholesale snapshot overwrite, or deletion of historical outputs.
- Compare the Windows checkout, WSL tree, frozen step43 source and later experimental snapshots. Reconstruct a canonical runnable source through reviewed dependency groups, preserving provenance and distinguishing patches from historical copies. Verify a baseline replay before calling the reconstruction complete.
- Ignore runtime/generated material with narrow, documented rules: `.research_runtime`, Python/test caches, disposable root dispatch/probe scripts, local environments and downloaded R libraries, telemetry/logs, raw/processed data, checkpoints, fitted arrays, generated bulk maps and duplicate run trees. Use `.git/info/exclude` for purely machine-specific scratch. Relocate useful automation into maintained script paths rather than hiding it.
- Keep source, tests, experiment specs, preregistrations, small numerical evidence, hashes and portable manifests tracked. Do not blanket-ignore `research/`, JSON, CSV, shell scripts, or all experiment outputs; those include essential evidence. Inspect existing global binary/PDF exclusions and explicit selected artifacts. Avoid automatically removing already tracked files.
- Check proposed uploads for credentials, restricted data, local account paths, large binaries and accidental vendored dependencies. Use explicit staged path lists and staged diffs; never `git add .` over this tree.

Proposed branch/commit organization, adjusted after the dependency audit:

1. `research/local-baseline-reconstruction`: scoped ignore rules; portable input/source manifests; reconstructed active model and required data/config changes; meaningful regression checks; historical record index. Separate dependency-coherent commits, with source attribution in commit bodies. Split encoder/data preprocessing into an independent prerequisite branch where feasible.
2. `research/local-loop-control`: isolated-worktree runner, single-writer ownership, immutable run manifests, collectors and verification/reporting corrections.
3. `research/habitat-field-geometry`: field topology and amplitude implementations, algebraic identity checks, preregistered experiments and compact results.
4. `research/regulation-observation-structure`: only earned follow-ups on capacity/disease/observation structure, based on Phase 3 results.

Use worktrees for new branches so active experimental processes do not import a changing checkout. Commit provenance and preregistrations before launching their jobs. Push reviewed branches to origin and open draft PRs against the appropriate prerequisite branch; never force-push or merge to main as part of autonomous experimentation. Each PR states scientific scope, executable changes, validation and unresolved evidence. Large results remain outside GitHub with hashes and reconstruction commands in the repository.

Exit: a reproducible committed baseline and reviewable changes, with no accidental data upload or loss of local work.

## Phase 3 — First scientific campaign: field topology and amplitude geometry

Direct temporal-change priors are OFF in the main structural comparisons. Keep reference and R103b as labeled benchmarks. Freeze unrelated model choices initially; use matched softplus/exp capacity controls so an observation-link improvement cannot be attributed to field topology.

### 3A. Remove redundant coordinates before changing biology

Derive the likelihood in effective coefficients and identify invariances analytically and with directional gradients. For the single field, test a nonredundant representation of its shared direction and per-rate sensitivities. Compare common learned scale, fixed common scale, and independently parameterized relative sensitivities without letting a free factor norm silently restore the removed scale.

Separate three cases explicitly:

- Equivalent coordinates: same forward predictions and likelihood family, with induced prior measure handled correctly. Expect better numerical conditioning, not a different likelihood optimum.
- Different coefficient prior: for example direct Gaussian coefficients versus a scale mixture. Label the changed prior and compare its contribution separately.
- Different model family: for example fixing relative sensitivities or separating K from demographic responses. This can change what the likelihood can represent and must earn its added/restricted flexibility empirically.

Do not normalize exported Z ad hoc or silently replace the isotropic feature/kernel contract. If an identified radial/directional representation requires an induced radial prior or Jacobian, derive and test it.

### 3B. Test field topology with the amplitude issue controlled

Use a small staged matrix, not every possible cross-product:

| Arm | Structure | Question |
|---|---|---|
| F0 | Current one direction, positive rate-specific amplitudes | Reproduced reference |
| F1 | Same family in identified coordinates | How much poor geometry is parameterization alone? |
| F2 | Two fields: capacity separate from one shared demographic direction | Does abundance stop forcing joint rate changes without granting four private fields? |
| F3 | F2 with a shared component plus a capacity-specific residual and estimated coupling | Does limited independent capacity structure help without becoming an absence switch? |
| F4 | Existing all-four exchangeable construction, reproduced from R101 | Matched historical comparator; not presented as new |

For F0/F2, screen learned amplitudes versus a nonhierarchical/fixed-scale coefficient specification after mapping its implied prior. Extend only the informative contrast to F3/F4. Retain an original-coordinate control so optimizer improvement is not mistaken for new biology. Existing rank-two survival-versus-reproduction/capacity grouping is an additional comparator only if the audit shows it answers a distinct remaining question.

Coupling/amplitude prior sensitivities are secondary, upstream tests: report field correlations, effective coefficient norms, scales and likelihood/penalty decomposition. A narrower prior that selects a smaller swing does not establish a better structural likelihood.

For promising arms, profile the *output* contrasts jointly with nuisance refits: invasion-era change, post-1993 change, modern potential growth and disease toll. Include both lower- and higher-swing starts. Assess boundedness, slope and curvature of likelihood-only profiles separately from priors; compare predictive trajectories and whether the same decomposition persists across starts.

Primary success: comparable or better prediction with a more stable, interpretable likelihood decomposition; no long compensating escape into K, Q, factor norm, Allee, disease, founders or observation effects. A larger model need not have a worse raw likelihood; it must justify its flexibility through transfer, recovery and geometry. No requirement that lambda attain a chosen numerical target.

## Phase 4 — Structural follow-ups earned by the field results

1. If topology changes do not resolve the ridge, derive a capacity/resource or regulation formulation in which equilibrium and return speed are distinguishable rather than imposed together. Start with one-cell equations and age-specific stability, then a heterogeneous spatial test. Account for current yearling breeding, power-two crowding, absolute mate limitation, disease and immigration. The old growth-derived-capacity helper is not a valid ready-made implementation for this base.
2. Revisit adult-side versus recruitment-side density feedback only if it tests an explanation left open by R86. Do not repeat juvenile-DD variants already known to relocate the swing into Sj.
3. Improve the disease transient comparison: route-paired and observer-continuous pre/post trajectories, several windows, regional uncertainty, disease-off counterfactuals and the missed Plains decline. Disease caps remain diagnostic profiles, not automatic production assumptions.
4. Isolate observation-link and variance effects with matched controls. Verify the exponential-link gain by count bins, level versus spatial pattern, integrated treatment of unseen route effects, and dispersed starts. Constant-concentration NB already failed to solve the ridge; do not repeat it as a new explanation.
5. If encoder attribution is necessary, separate M/A/D/R and covariate-family contributions on the native field, with training-only references and source masks. Model regional field scale/coupling only after checking its own factorization and community-level evidence. Missing TACC artifacts are documented, not replaced by inferred attribution from downstream correlations.

## Phase 5 — Autonomous execution and evidence standards

Each loop iteration: collect completed tasks; validate artifacts; update the model/experiment crosswalk; state competing explanations; preregister the smallest discriminating contrast; commit its code/spec; enqueue within resource limits; arm a completion monitor; report findings and unresolved failures. No claim of execution without a task receipt and process/output evidence.

- One GPU fit initially, bounded CPU work alongside it, measured memory headroom and capped threads. Restore more concurrency only from measured capacity. Short screens triage; serious candidates receive adequate float64 refinement, continuation and at least three dispersed starts.
- Mandatory mechanics checks for structural changes: zero-change identity where applicable, targeted finite differences, active-site gradients, positivity/mass accounting, equilibrium and life-cycle consistency, source hash/import identity, mask counts and objective/readout agreement.
- Fit all co-adapting parameters before a structural verdict. Partial refits are diagnostics; their failures or improvements are not final biological verdicts. Record exact fixed/free sites and each likelihood/penalty block.
- Common geographic support and equal-weight era comparisons, with all-land, ever-observed and population-weighted views separately. Use 1966–1990 and 1993–2010 changes alongside longer backcasts; 1940 output is not observed establishment dynamics for a 1951-initialized fit.
- Report potential growth, local finite-density growth curves, disease mortality, actual realized population change, source–sink/immigration dependence, K/Allee nonviability, all vital rates and NestWatch fit, fronts plus never-reached share, and movement/export budgets. Read banding predictions at cohort locations/years with actual forward disease density.
- Use training-only nuisance fits or full joint refits for temporal tests. Report fixed-latent diagnostics separately. Existing repeatedly inspected 2021–25/spatial blocks are development evidence, not a fresh final test. Preserve an untouched evaluation split if one exists; otherwise declare the selection history and use preregistered blocked/time replication and paired uncertainty without claiming untouched confirmation.
- Estimate optimizer/start variability and predictive uncertainty for the actual comparison. Do not use a universal 500/1,000-unit threshold as an equivalence test. Compare likelihood and prior separately; total MAP objectives across different dimensions or likelihoods are not a model league table.
- Failed gradients, line-search stalls, OOMs, unfinished jobs and invalid diagnostics retain their status. Negative scientific results require a working fit and an intervention that actually touched the intended process.
- Before promotion, apply the repository verification gate: planted recovery on the real design, independent re-derivation, replication and a skeptical review of source plus artifacts. HMC follows a plausible, reproducible full-fit mode; it does not repair an unexamined structural ridge.

## Delivery and stopping points

First deliverables: preserved run/ownership inventory, reconciled baseline commit, complete execution map, code-to-experiment crosswalk, corrected scoring/replay anchors, and committed Phase 3 preregistrations. Then deliver small batches of structural evidence with GitHub branch/commit links and explicit next decisions.

Prefer a structural candidate when its improvement survives matched likelihood scoring, nuisance refits, starts, transfer and recovery, and when its geometry is improved for a stated reason. Retain output-change priors only as transparently labeled regularization if the structural evidence still leaves genuine uncertainty. Stop or revise a branch when its discriminating prediction is falsified; do not keep searching its hyperparameters for a smaller lambda swing.

Open dependencies at plan approval: current task ownership/resource availability; final R104/continuation artifacts; complete source differences across Windows/WSL/frozen trees; availability of native encoder components; GitHub authentication/branch protections; and the existence of any genuinely untouched evaluation data. Resolve by inspection before requesting information the local environment may already contain.
