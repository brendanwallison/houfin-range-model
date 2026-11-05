# Direct-change neural pilot and matched continuation control

Prepared/dispatched on2026-10-27 around07:30 CDT. The existing time-conditioned
pilot was still running, at epoch216/500 when checked at07:19. The local GPU
remained occupied. These new GPU runs are queued, not already training.

| Task | Arm | Work |
|---|---|---|
|808|change100|100-epoch warm-start continuation with explicit community change-vector loss|
|809|continue100|Matched100-epoch continuation without that additional loss|

Both depend on the ongoing time-conditioned task790 and use the same completed
500-epoch control checkpoint, random seed, learning rate1e-4, optimizer reset,
observation masks, original seasonal objectives and learned EMA initialization.
Each exports an instantaneous128-coordinate population cube after training.
The comparison with809 is required to separate the new objective from extra
training, optimizer restart or nuisance-parameter reinitialization.

## New target and loss

Only permitted summer training records enter the new targets. The three
training epochs are1996–2005,2006–2015 and2016–2025. A cell must have four
permitted observations per epoch in a contrast; ABBA halves and the sampling
seed are fixed before scoring. The target is the difference of mean annual
coordinates in the same fresh summer teacher used by the control. Repeated
contrasts are weighted so heavily observed cells do not acquire more total
weight just because they supply all three contrasts.

The new loss compares those vectors to differences of predicted community Z
**after the training output EMA**. It does not compare instantaneous population
features directly with lagged observations. The existing winter objectives,
including historical presence supervision, remain; no early-summer labels
are invented and no House Finch/sealed responses enter the target.
The prepared target contains2,303 contrasts from1,245 training cells; its
earliest summer observation is1996.

At the first training step, the added loss's coefficient matches its output-head
gradient norm to the weighted primary coordinate loss's gradient norm, with
explicit numerical bounds. The resulting coefficient and both gradient norms
are saved and reused on resume. This is a declared training-only pilot heuristic,
not validated temporal-capacity calibration or a replacement for blocked
strength selection. It does not impose the rejected0.14 amplitude ratio.

## Verification and scoring

The isolated patched trainer completed two full CPU epochs, including a checked
warm start, checkpointed forward/backward calls, vector loss, gradient balancing
and optimizer updates. A planted-vector test verifies the loss is invariant to
spatial offsets but detects incorrect temporal changes. Existing mean/deviation
prototype and observation-EMA tests also pass. Full-size GPU memory use for the
new loss is not yet tested; the frozen control/source files remain separate.

Additional CPU audits are queued to score both summer and winter observations
using each model's own learned training EMA, and to compare808 with809 on paired
observations. The same audit also covers control_s1 and time_r2_s1. These
observation-space reports remain separate from the original instantaneous-feature
diagnostics and never replace the raw population export. Each audit spans the
full land support, so winter evaluation is not restricted to BBS sample cells.

Outputs will be under
`.research_runtime/community_z_repair_20261026/observation_comparison/`:
`ALL_OBSERVATION_COMPARISON.{md,json}` and
`MATCHED_CHANGE_LOSS_COMPARISON.{md,json}`.
Runtime `OBSERVATION_TASKS.json` records actual audit/comparison queue IDs.
The IDs are810(control),811(time-conditioned),812(change100),813(continue100),
and814(paired comparisons).810 has completed both summer and winter scoring;
its historical winter backcasts still do not demonstrate a gain over the
fixed-modern-feature placebo. The other audits await their model fits.

The competing learned mean-map/deviation architecture remains a tested research
prototype. Its dedicated trainer/exporter integration and staged fit have not
been dispatched. It is retained alongside these arms, not substituted for them.

Sources: `research/exp/z_change_neural.py`, `z_repair_ema_audit.py`,
`z_repair_compare.py`; wrappers `research/bin/z_change_neural_gpu.sh` and
`z_observation_{audit,compare}_cpu.sh`. Training source snapshots are
`change100_code/` and `continue100_code/`; audit source is
`observation_audit_code/`, each with a source manifest.
