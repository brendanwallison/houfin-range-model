# Additional overnight tests: reuse the archive before adding fits

Registered after the five new step33 configuration fits were staged. No additional model fitting or new allocation requests are added here.

## Archive availability

The TACC grid54 inventory verified 96 run directories, 70 refined checkpoints, 69 accepted refinement reports and 69 cached source–sink field sets at the check. These include the important native-basis, learned-kernel, native-departure, shorter-kernel and yearling comparisons. The local grid54 tree contains 111 run directories, including some runs not mirrored to TACC. A run directory alone is not proof of a completed fit. The entire pre-grid54 archive has not been exhaustively verified, so “everything is stored” is not yet a justified blanket claim.

Preserve all archives. Inventory files record missing/incomplete artifacts rather than silently excluding them. No fits are overwritten or deleted.

## 1. Cached retrospective checklist — CPU, immediate

Read every accepted fit's saved metrics, modern source–sink/capacity fields, western-sink audit, configuration and available front diagnostic. Publish a table of tradeoffs and a machine-readable inventory. Label source data, life cycle and input provenance; exclude identified synthetic-observation runs from the real-data summary. Do not rank raw objective values across bases or interpret a smaller lambda change as an automatic improvement.

This directly checks whether an old intervention that looked favorable on one headline worsened low-K behavior, Allee-dead regions, source/sink patterns or another recorded concern. It also identifies where richer rate readouts are missing.

## 2. Rich readouts at eight old fixed points — queued

Replay accepted checkpoints with their **own frozen code and resolved configurations**, without optimization:

- Native habitat baseline v1 and v2.
- Learned-band-kernel fits on native v1 and v2.
- Native departure on its original base (explicitly different habitat input).
- Juvenile MDD 110 and 200 km.
- Second seed of the yearling/native base.

Use the current biological readout: actual age-specific survival, juvenile/adult ratio saturation, fecundity, capacity and growth by era/region, population weighting, plus existing map diagnostics. These comparisons revisit the high-value older leads and test whether some apparent improvements were substitutions between pathologies. Inputs and life cycles remain separate comparison groups.

Ten-minute cap per replay (eight maximum); no likelihood modification, training or refitting. Readouts are stored alongside existing diagnostics and mirrored into the overnight report artifacts.

## 3. Fixed-parameter float32/float64 stress test — completion triggered

When accepted diagnostics finish, replay nest-only and the three seed-41 step33 configurations (CBC, CBC+native departure, CBC+absolute Allee) at both precisions with **the identical fitted point**. Fit neither version. Compare expected counts and population trajectories, and the timing of reaching 1% and 10% of one fixed float64 reference population scale. Do not use a different threshold for each precision.

Report population/log-count discrepancies, arrival shifts, and cells reaching the threshold in only one precision. The test distinguishes biological differences between configurations from sensitivity to numerical representation at establishment thresholds. Float64 is the reference; a discrepancy does not make float32 a competing biological model.

Twenty-minute cap per fixed-point pair; four pairs maximum. Existing workers run these checks after the main fit tasks. Completion triggers are deterministic and do not launch another fit. No automatic acceptance threshold or model promotion is introduced.

These tests supplement the already queued generating-mean/centering contract check. If that check reveals a synthetic-test artifact, preserve and qualify the old recovery results before commissioning corrected refits.
