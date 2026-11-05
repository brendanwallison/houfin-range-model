# Final held-out evaluation status

Updated 2026-10-25T08:56:35-05:00. Summer tasks complete 26/26.

Specification: research/experiments/FINAL_HELDOUT.md.
Live report: research/experiments/FINAL_HELDOUT_RESULTS.md.
Run root (WSL): /home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025
Run root (TACC): /work/07980/bwa386/ls6/houfin/research/final_holdout_20261025

Completed: geometry/extraction for 5,969 cells and 124 years; all five primary representations; winter confirmation (211 taxa, 6,243 test lists); exact completeness reconstruction; sealed/focal benchmark membership audit; fixed-readout pseudo-early damage check with independent numerical reproduction. Twelve focused tests pass.

Existing allocations 3530553/4/5 execute tasks through queue2. Local pueue 740/746/747 execute the winter-half summer scores. Tasks 9923/9928 were transferred atomically from remote pending to local, so no duplicate worker execution. No new Slurm allocations were submitted.

All 26 scoring tasks are complete. Collector pueue749 completed successfully, copied all products/read ledgers, generated the final report and wrote EVALUATION_DONE at the run root. Its reporting-only snapshot is code_reporting_r2.

Scientific source snapshots: code_data, code_score, code_winter_r2, code_membership_audit, code_damage_r2; each retains source manifests. Failed pre-score attempts were retained: missing package markers in initial extraction; relocated-circle overlap guard; a quoted-argument launcher issue; damage-dependent taxonomy exclusion guard. Fixed before producing the affected results. No real-data outcome was retuned.

Important: direct target exclusion does not imply strict independence. Recognized sealed taxa constitute about 37% of the CBC completeness benchmark; focal House Finch appears in about 80% of circle-years' benchmark sets. Future production adoption needs a clean-completeness training sensitivity. The current evaluation remains provisional. The damage test detects observation-driven inflation of temporal gain, but does not reproduce the large real early v1/v2 gap.

Read ledgers: research/sealed_reads.csv locally; remote_sealed_reads.csv at the run root for TACC scoring. No training checkpoint or age-model production setting was changed.

Final verification 2026-10-25T09:12:21-05:00: all 26 summer DONE markers present; winter, membership and damage checks complete; 12 tests passed. Interpretation is in FINAL_HELDOUT_INTERPRETATION.md.
