# STATE -- generated 2026-10-09T08:17:03-05:00 by research/tick.py; do not edit (edit intent.md)
phase 0 | eval pre-v1 | research/desk-temporal @ 1b3a8d0

## Jobs
- running   gpu     2026-10-09T08:59-05:00 E001/repro-gpsp-base-aac7dac563c4
- queued    gpu                       E001/repro-gpsp-tempho1995-65eb9eb982f0
- queued    gpu                       E002/cache-desk_tempho_1975-e740027d985a
- queued    gpu                       E002/cache-desk_tempho_1985-1be1f738e96c
- queued    gpu                       E006/suite-corrected-gp_species_base-67c77ddcc9e7
- queued    gpu                       E006/suite-corrected-desk_tempho_1995-528767bc5599
- queued    gpu                       E003/esk-desk_tempho_1985-e0cd23006456
- queued    gpu                       E003/esk-desk_tempho_1975-adadad9930fa
- queued    cpu                       E011/change-oracle-tempho1985_withheld-388a5de4020d
- queued    cpu                       E011/change-oracle-tempho1975_withheld-97d9ab0ce665

## Recent events
- 10-09T07:47 done     E004/atlas-tempho1995-trained-00f762e4bf66 
- 10-09T07:47 done     E004/atlas-tempho1995-withheld-65b6d651fbc0 
- 10-09T07:47 done     E005/fastarm-desk_tempho_1995-ec296a1aa03a 
- 10-09T07:47 done     E005/fastarm-gp_species_base-6969e4d243e8 
- 10-09T07:47 done     E007/planted-tempho1995-b3091e48eefb 
- 10-09T07:47 failed   E008/route-ceiling-desk_tempho_1995-56635e247f9a 
- 10-09T08:08 done     E008/route-ceiling-gp_species_base-f1f84319eb87 
- 10-09T08:09 done     E009/planted-v2-tempho1995-135104d22dc2 
- 10-09T08:11 done     E011/change-oracle-base_trained-fff5408bcf35 
- 10-09T08:11 done     E011/change-oracle-tempho1995_withheld-e8bd81e6849e 

## Hypotheses
open 24 | testing 10
- testing E1: Most cell-level change is noise, so no_change is near-unbeatable on RMSE even for a perfect habitat model.
- testing T0: DESK's temporal under-movement is the MSE-optimal shrinkage toward unreliable within-cell targets, not a defec
- testing O3: Route turnover between epochs survives the ABBA year split and is counted as real change, inflating every ceil
- testing E4: The independent oracle is weak partly through errors-in-variables (noisy half-window community features) or of
- testing P1: Held-out species' decadal change is mostly continental trend plus colonization fronts, which belong to the mec
- testing R1: (user) Spatial turnover dwarfs temporal turnover, so eigen-ordering and truncation keep spatial structure and 
- testing T4: Space-for-time mismatch - spatial covariate-to-community slopes differ from temporal slopes.
- testing B1: The isotropic prior under-weights low-variance (temporal) directions.
- open, priority 1: C1

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
TACC baseline (old suite, A1-diluted): out-of-community primary change skill base +0.0002 (401 spp),
tempho1975 -0.0007, 1985 -0.0037, 1995 -0.0100; community mode +0.0086/+0.0108/+0.0017/-0.0036 (~91 spp).
