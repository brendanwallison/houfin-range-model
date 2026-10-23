# STATE -- generated 2026-10-09T09:22:53-05:00 by research/tick.py; do not edit (edit intent.md)
phase 0 | eval pre-v1 | research/desk-temporal @ 58cc3d9

## Jobs
- finished  gpu                       E002/cache-desk_tempho_1975-e740027d985a
- finished  gpu                       E002/cache-desk_tempho_1985-1be1f738e96c
- running   gpu     2026-10-09T11:22-05:00 E006/suite-corrected-gp_species_base-67c77ddcc9e7
- queued    gpu                       E006/suite-corrected-desk_tempho_1995-528767bc5599
- queued    gpu                       E003/esk-desk_tempho_1985-e0cd23006456
- queued    gpu                       E003/esk-desk_tempho_1975-adadad9930fa
- queued    cpu                       E011/change-oracle-tempho1985_withheld-388a5de4020d
- queued    cpu                       E011/change-oracle-tempho1975_withheld-97d9ab0ce665
- queued    gpu                       E010/desk-repro-base-r1-e795ff73b46a
- queued    gpu                       E010/desk-repro-base-r2-f76f45f0596f

## Recent events
- 10-09T08:51 done     E011/change-oracle-placebo-base_trained-fde5bfc1d36f 
- 10-09T08:51 done     E011/change-oracle-placebo-tempho1995_withheld-936a57d426c0 
- 10-09T08:51 done     E012/placebo-paired-desk_tempho_1995-10d601e9a7f3 
- 10-09T08:51 done     E012/placebo-paired-gp_species_base-0dc00627c5c0 
- 10-09T08:51 done     E013/readout-corr-tempho1995_withheld-9bf6614c8807 
- 10-09T09:20 done     E001/repro-gpsp-tempho1995-65eb9eb982f0 
- 10-09T09:20 done     E014/route-nugget-k2s1-2e967461f7a3 
- 10-09T09:20 done     E014/route-nugget-k3s0-7c784bd6a1c0 
- 10-09T09:20 done     E016/cov-surrogate-base_trained-892a6ba73b1a 
- 10-09T09:20 done     E016/cov-surrogate-tempho1995_withheld-6a09a946df58 

## Hypotheses
open 26 | testing 11
- testing E1: Most cell-level change is noise, so no_change is near-unbeatable on RMSE even for a perfect habitat model.
- testing T0: DESK's temporal under-movement is the MSE-optimal shrinkage toward unreliable within-cell targets, not a defec
- testing O3: Route turnover between epochs survives the ABBA year split and is counted as real change, inflating every ceil
- testing E4: The independent oracle is weak partly through errors-in-variables (noisy half-window community features) or of
- testing P1: Held-out species' decadal change is mostly continental trend plus colonization fronts, which belong to the mec
- testing R1: (user) Spatial turnover dwarfs temporal turnover, so eigen-ordering and truncation keep spatial structure and 
- testing T4: Space-for-time mismatch - spatial covariate-to-community slopes differ from temporal slopes.
- testing B1: The isotropic prior under-weights low-variance (temporal) directions.
- open, priority 1: C1, N1, B8

## Findings
0 verified findings

## Intent (research/intent.md)
Phase 0 (gate 0 pending) with Phase 1 experiments running on cached arrays. Data landed and verified 2026-10-09.
GATE 0 DUE when E001 base + tempho1995 finish (~08:59 / ~10:15): compare_reports verdicts + timings to the user,
with the morning's (unverified) results as a preview.
Live questions (UNVERIFIED leads, see DIGEST 2026-10-09 morning):
- B7: is the readout's level-fitted beta the main loss of change? E011 x4 queued (1975/1985 pre-registered).
  If yes -> E012: does a temporal beta learned INSIDE the trained era transfer to withheld decades?
- D0: DESK backcast change ~ observed community change for the species readout (half-window comparator is
  noisy -- needs an errors-in-variables-corrected comparator before it can conclude).
- E1/readout ceiling by prevalence: E009 (planted v2) running.
Verification debt (none of the above is a FINDING yet): E004 atlas (b) re-derivation, (c) second split,
(d) skeptic; E011 the same once the jobs land.
Then: E010 DESK reproduction (GPU, after E006) = Phase 0.4 gate.
Cold copy to E: (the nohup'd rsyncs died with their session): pueue group "io", tasks 36/37 (not in the registry).
When done: cd /mnt/e/Datasets/houfin/work_houfin && sha256sum -c --quiet processed_xfer.sha256 (same for data).
TACC baseline (old suite, A1-diluted): out-of-community primary change skill base +0.0002 (401 spp),
tempho1975 -0.0007, 1985 -0.0037, 1995 -0.0100; community mode +0.0086/+0.0108/+0.0017/-0.0036 (~91 spp).
