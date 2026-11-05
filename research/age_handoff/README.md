# Final DESK -> House Finch handoff (diagnostic, 2026-10-25)

The age-model base is frozen at d9bd47a. These are isolated overlay patches and tasks, not changes to the other agent's working branch or production configurations.

## Ensemble and kernel choice

Use the latest DESK candidate plan's arithmetic mean **features**, the artifact already evaluated by FINAL_HELDOUT. This supersedes E081's older proposed mean-kernel stacking/projection for this diagnostic. Those constructions are different; no top-64 projection is applied to the ensemble and no implicit first-64 cut is allowed on v2.

All four seeds share hashed coordinate targets in each block. The builder verifies these identities and preserves the observed seed-correlation diagnostics. The old correlation-only .8 gate is insufficient to distinguish prediction disagreement from coordinate rotations when fixed targets anchor the axes; shared-target mode is explicit and tested. This is a diagnostic use of the current candidates, whose CBC completeness still includes indirect sealed/focal information.

V1 is 64 summer dimensions. V2 is 64 summer + 64 winter. The existing isotropic feature-coefficient prior gives equal prior variance to corresponding coordinates, hence a SUM of the two seasonal kernels. The existing correlated spatial/temporal readout is retained; no new season-specific amplitude parameter is introduced. Matched-width v2 selects the leading 32 coordinates within each half explicitly, with full source/selected block provenance.

## Four matched diagnostic MAP/refinement cases

- fd_control64: earlier CBC1902 encoder, readout rank64 (the E081g readout inherited rank32, so it is not the width-matched comparator).
- fd_v1_r64: final v1 four-seed mean, rank64.
- fd_v2_r128: final v2 four-seed mean, all128 features.
- fd_v2_r64: same full v2 input, explicit summer32+winter32.

All use the unchanged 80h biology and correlated temporal readout, cold quick90 MAP followed by float64 L-BFGS with a 16000 iteration ceiling and existing convergence/non-finite guards. Inputs/results are isolated under fd_* paths. Refined diagnostics require valid=true. No candidate is promoted to production by these runs.

The patched source extends kernel validation/provenance and explicit seasonal feature selection only. It does not change the population likelihood, demographic process, dispersal, observation model, or optimizer objective. Path integration retains the whole feature layout. Tests verify the sum-kernel identity, source/selected widths, and rejection of global cuts that would remove winter.

The later `code_memoryfix1` snapshot additionally casts large Z inputs to their requested float32 dtype on host before GPU transfer. The original loader temporarily allocated the full float64 array and OOMed at128features. The changed transfer preserves final values/dtypes and density; its recovery wrapper records and explicitly permits that source-hash difference. Original snapshots, stalled fits and drift-guard rejection logs remain retained. No convergence threshold is relaxed.
