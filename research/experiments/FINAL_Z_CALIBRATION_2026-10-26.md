# Z calibration: first local round

Declared before calibration outcomes. Aim: test a small positive-semidefinite Z-metric family, keeping one shared downstream coefficient field. This is exploratory development calibration, not a confirmed correction of misalignment.

Use clean v2 seeds1/2 features and the existing X_dev.npy development-species counts. Exclude House Finch, all sealed species, encoder spatial holdouts and buffers. Use1996–2005 versus2016–2025 only, requiring at least4 distinct surveyed years per epoch. Reuse deterministic ABBA splitting and log1p of mean raw counts from the existing evaluation library. No1902–1995 bird outcomes or sealed count files are read.

Target proxy: community log-abundance level and20-year change Gram matrices, cross-multiplied between disjoint ABBA observations. This directly assesses a similarity prior rather than fitting separate temporal coefficients. It is not the original species-weighted Ruzicka target, and agreement with it does not by itself establish a correct habitat prior. Observer dependence and encoder supervision of development species remain limitations.

First family: identity plus seasonal spatial-prior-variance shares0,.25,.5,.75,1. Each metric is diagonal, nonnegative, constant over space/time, with training spatial trace normalized to1. Raw identity is also normalized by one global scalar; no direction is altered in that baseline. Consequently one shared coefficient vector sees kernel z^T M z'. Source block identities are retained. Direction-dependent metrics are a subsequent stage, not implicitly included here.

Four fixed6x6 spatial-block folds (seed20261026); fit all normalizations only on each fold's training cells. Deterministically split development species by SHA256 into calibration and validation halves. Choose a candidate on calibration-species spatial CV change loss subject to spatial excess loss<=.05 of the observed spatial Gram mean-square. That5% constraint is an exploratory design choice, not a validated ecological tolerance. Report all candidates and held-species scores; do not retune from validation-species outcomes.

For each sampled cell pair, loss=P^2-2PT, omitting the shared noisy target-square term; P is the candidate kernel, T the symmetrized cross-half target. This avoids treating noisy target squares as biological signal. Report level/change losses and cross-half scale/support. Use identical20000 sampled pairs per fold across candidates. Bootstrap uncertainty and additional blocked time contrasts are follow-ups before any adoption; this first run does not claim significance from its selected minimum.

Outputs: fixed folds/species manifests, all candidate CV scores, selected exploratory full-development metric, source hashes and exact support. Local CPU only; independent of House Finch refinement memory. No production Z or prior is modified.

Execution: the first pass completed locally. A conditional paired6x6 spatial-block bootstrap of the selected-versus-identity validation losses was added after the initial point results, without changing selection. Initial and extended results are retained separately; see FINAL_Z_CALIBRATION_RESULTS.md. This uncertainty extension was not part of the original declaration above.
