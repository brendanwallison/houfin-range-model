# House Finch final-DESK interim assessment — 2026-10-25

The completed diagnostics do not establish a final downstream winner. Final-v1's numerical restart passed the configured plateau gate and produced diagnostics, but its final gradient norm is~65379; the gate uses a recent median, not that final value. Control64, full-v2 and v2(32+32) remain stalled, including their finite-history restarts. Direct gradient audits are running/queued. No convergence threshold is relaxed.

## Existing observation-aware holdout scores

These come from the established MAP diagnostics: NB2 log density, including its route-occupancy zero gate where enabled. Higher mean log density is better. The temporal set has12204 observations; the spatial set22548. Scores condition on a fitted point; they do not integrate parameter uncertainty. All rows below have matching holdout counts, but optimizer stage and readout width differ as labeled. No paired uncertainty interval has been computed here.

| Encoder/readout | Stage | Temporal mean log density | Spatial mean log density | Temporal log1p RMSE | Spatial log1p RMSE |
|---|---|---:|---:|---:|---:|
| Earlier CBC1902,64 | Cold MAP |−1.9743|−1.3470|.9531|.7936|
| Final v1,64 | Cold MAP |−1.9413|−1.3158|.9213|.7491|
| Final v2,32+32 | Cold MAP |−1.9063|−1.2770|.8527|.7060|
| Final v1,64 | Restart-refined, gate passed |−1.7997|−1.2114|.8081|.7035|
| Earlier CBC1902,32 (81g) | Refined |−1.8023|−1.1970|.8192|.6651|
| Earlier matched control,32 (81h) | Refined |−1.8193|−1.2241|.8237|.7191|

At the same cold-MAP budget,32+32 v2 improves both heldout summaries over v1. That is an optimization-stage result, not a comparison of converged fits. Refined final-v1 has nearly the earlier CBC32 temporal score (+.0026 mean log density) and a worse spatial score (−.0145), with a different readout width. It is not evidence for a decisive final-v1 advance.

## Ecological diagnostics remain material

For final-v1, refinement moves the fractions of modern land at the reported low/high K extremes from~.03%/0% to23.5%/17.7%. These are reporting thresholds around the reference K level, **not enforced parameter bounds or clipping**. The diagnostic's western native-seed cells are39.4% fundamental sinks and17.9% Allee-dead after refinement, compared with42.5%/.12% at its cold MAP. These categories describe the model's modern suitability at historically seeded cells, not a direct census classification.

The earlier CBC32 refined model has more fundamental-sink native cells (62.3%) but fewer Allee-dead native cells (1.17%). Thus this is a mixed ecological comparison; the new encoder does not plainly remove the longstanding contortion. Final-v1's no-release counterfactual attributes only~.15% of modern total population to the release, versus~2.72% in earlier CBC32. That diagnostic also needs examination before accepting an invasion interpretation.

## Active numerical checks

Task9964 restarted v2(32+32) from its retained finite point, with fresh history and unchanged model/guards; it stalled again with recent median gradient~22500 above the10000 guard. Tasks9965–9968 audit control64, full128 and final-v1 using the existing density: gradient norms by site, repeated potential values, finite differences along the gradient and largest coordinates. Exact-state variants use the optimizer's saved unconstrained vector and record the difference from reconstructing it via constrained checkpoint parameters. They diagnose scaling, non-smoothness or gradient mismatch without changing the density.

The original encoders in these House Finch fits still used the contaminated CBC completeness benchmark. Clean-completeness training/evaluation is tracked separately in FINAL_CLEAN_FOLLOWUP_2026-10-25.md. Do not promote from this table or compare raw posterior modes across widths as a substitute for heldout predictive scores.

Sources: `/work/07980/bwa386/ls6/houfin/research/age_model/FINAL_DESK_MAP/<arm>/{viz_map,finite_restart1/viz_ref}/{metrics,west_sink_audit}.json`; earlier baselines: `E081/{81g,81h}/viz_ref/`. All original failures and restart outputs remain retained.
