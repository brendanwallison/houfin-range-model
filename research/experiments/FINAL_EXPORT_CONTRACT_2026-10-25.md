# Final DESK export contract correction — 2026-10-25

Applied locally and to the idle TACC desk-research worktree. Corrected all eight existing TACC cube_meta.json files, preserving cube_meta.json.pre-contract-v2 beside each. Checked checkpoint and desk_meta hashes and all 124 array headers per cube (992 total). Arrays and fitted checkpoints were not modified.

- v1: kernel=ruzicka, latent_dim=64, summer block [0,64).
- v2: kernel=ruzicka_sum, latent_dim=128, summer [0,64), winter_half [64,128). Dot products approximate a SUM, not the average, of the season kernels.
- Each block names its basis and hashes meta.json, esk_projmat.npy, and esk_landmarks.npy. Winter also names and hashes its coordinate targets.
- Future private-block exports require explicit coordinate-target basis provenance and identity/private-only heads. Unidentified private channels cannot silently be called an ESK basis.
- The mean-cube builder now checks layouts, basis identities, timelines and array widths, tests alignment separately in each block, and writes cube_meta.json with all member checkpoint identities.
- Path integration accepts and preserves the explicit sum contract. The existing age-model loader explicitly rejects a sum cube until block-aware feature selection/priors are implemented; it cannot silently discard the winter block. Production rank and prior configuration were not changed.

Repair utility: scripts/repair_cube_contract.py --config <overlay> [--write]. Default is dry-run; writes are atomic and retain the original metadata. The isolated TACC repair code and previous source archive are in /work/07980/bwa386/ls6/houfin/research/contract_repair_20261025/.

Validation: 6 focused tests passed in WSL and on TACC. Combined with existing kernel-contract tests: 21 passed, 1 deselected after identifying an unchanged diagnostic's numerical-tolerance failure (landmark_at_rank_rmse_norm=1.23018e-5 versus a 1e-5 bound). The diagnostic implementation and test were not edited. Post-write verification confirmed all eight dimensions/layouts and backup files.

## Holdout inventory (species labels only; no sealed outcomes read)

Original research cache: 96 community + 273 development + 296 sealed species. Current final BBS basis uses community+development=369; CBC auxiliary and presence tables use 333. Verified zero sealed overlap and no House Finch in all these inputs. No research/sealed_reads.csv exists in this checkout. This audit did not inspect sealed response values or claim a global audit of all agent activity.

The prepared CBC table also excludes sealed species. A winter confirmation needs a separate extraction of sealed taxa from the available raw CBC request, with taxonomy, effort, split-affected species, and coverage checks. Not all 296 are necessarily usable in winter. The current cbc_prepare script deliberately restricts itself to reference/development species; do not silently relax that guard. Build a confirmation-only path with the same logged-read discipline as research/adopt.py.

Recommended evaluation: use species folds for iterative recipe choices. Once recipes/readouts/metrics are fixed, one logged sealed confirmation can compare v1 r64, v2 32+32, and v2 64+64, plus summer-only/winter-only ablations. Fit readouts on training cells/years and score independent spatial blocks and temporal windows; report summer BBS and winter CBC level/change separately, full/sparse regimes, uncertainty, and pooled species results with a coverage rule fixed before opening outcomes. Rank-match the r64 comparisons. A full-history candidate can test held-out-species generalization but does not by itself validate early-century extrapolation: use temporal-holdout replicas and pseudo-early winter tests for that claim.

## Queue access

Existing SSH control connection works. Slurm --test-only passed account DEB23008, allocation, and access checks for gpu-a100-small and gpu-a100-dev; both partitions are UP. These checks submitted no jobs and reserve no resources. DESK worker queue has no running/pending tasks and no active DESK worker; jobs can be submitted by starting a new worker. Three separate age workers remain allocated; do not assume their GPU slots are available for DESK. Scheduler estimates are provisional, not guaranteed capacity.

Follow-up source audit: direct sealed-species exclusion does not establish total independence. CBC completeness uses all raw species and may indirectly include sealed observations; see FINAL_EVAL_READINESS_2026-10-25.md before sealed scoring. Also, the existing age workers already support a DESK queue drain; their allocated capacity can be reused without a new Slurm submission while still running.
