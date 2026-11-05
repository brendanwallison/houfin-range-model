# Evening update, 2026-10-26

Current-work pointer: as of2026-10-27 around07:30 CDT, the time-conditioned DESK
pilot is at epoch223/500. The CPU change-learning comparison is complete, and
the direct-vector-loss neural pilot plus matched continuation control are
queued. See `../../COMMUNITY_Z_NEURAL_CHANGE_DISPATCH_2026-10-27.md` and
`../../COMMUNITY_Z_CHANGE_CPU_INITIAL_2026-10-27.md`. The text below is the
retained2026-10-26 downstream-fit status.

Checked at20:33 CDT. These focal-model results do not select community-Z candidates.

The shared-coefficient summer64 fit completed refinement and diagnostics at16:13.
It passed the existing optimizer gate after4,038 iterations, with final gradient
norm776.49 and objective143,925.04 to135,615.52. Winter64 had passed earlier.

| Refined shared readout | Temporal mean LPD | Spatial mean LPD | Land mean early lambda | Land mean modern lambda |
|---|---:|---:|---:|---:|
|Winter64|-1.83019|-1.22799|0.64355|1.10285|
|Summer64|-1.80400|-1.20957|0.65597|0.95581|

Both use the same temporal12,204 and spatial22,548 heldout records. These point
scores do not establish significance, and the lambda increase remains an
ecological concern. Both have `split_readout: null`; neither uses the0.14 ratio.
Summer figures and metrics are available on this PC at
`research/validation_figures/shared_clean_s64_refined/viz_ref/`.
All29 copied diagnostic files were SHA256 checked against WSL originals.

TACC3535018 failed the optimizer gate after4,584 iterations, rather than failing
memory allocation. Objective142,613.71 to133,947.94; recent median gradient2.27e6
at the plateau exceeded the10,000 gate. Final saved-point gradient4,413 is a
different statistic. `r128_refine_report.json` retains the invalid result;
the gate was not relaxed and no accepted validation figures were generated.

TACC3535014 completed the separate gradient audit on the older clean split-fit
state, not the newly stalled shared128 state. Repeated objective evaluations
were identical. At step1e-8, the gradient-direction finite difference differs
from autodiff by about0.10%; smaller steps show more numerical sensitivity.
This does not diagnose or certify the new shared128 failure. The complete
`stalled_gradient_audit.json` is retained beside this note.

The local GPU moved directly to the matched community-DESK control at16:13.
At20:33 it was at epoch207/500; the rank-two time-conditioned model is queued
next. The frozen-control residual and development scoring remain automatic
follow-on tasks. There are no new heldout Z-repair performance results yet.
