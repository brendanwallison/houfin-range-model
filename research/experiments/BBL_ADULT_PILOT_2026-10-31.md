# BBL adult recovery pilot: preregistration, 2026-10-31

Authorized follow-up to PROJECT_SURVEY_2026-10-31.md. This is an isolated CPU diagnostic; no population fits or production changes. Written before running the new preparation or fitting code on real data.

Question: do adult dead-recovery timings support a broad survival constraint once never-recovered birds, partial initial exposure, follow-up truncation, and reporting changes are represented?

Code: `research/exp/bbl_survival.py`. Public input: the existing local NABBP 2025 group-37 CSV and its lookup tables. Fit adult ages AHY/SY/ASY/TY/ATY (official numeric codes 1/5/6/7/8), normal wild releases, excluding ambiguous repeated release keys, handling mortality and invalid chronology. Use April–August bandings outside the West; retain all months in the sedentary West. End follow-up at 2024-12-31, before the July 2025 retrieval. Retain the never-recovered denominator. Death is assigned to encounter month; release to month midpoint. These are explicit approximations, with old-carcase reporting delay unresolved.

Design: one constant annual survival per origin region, conditional dead reporting by recovery era (1960–79, 1980–94, 1995–2005, 2006–24). First interval half-month, later intervals full months. Multinomial likelihood over recovery month or no recovery before cutoff; no informative survival prior. Three optimization starts; profile survival .20–.85 with reporting reoptimized. No inference that regional origin equals lifelong location. This pilot deliberately cannot constrain juvenile Q or habitat coefficients.

Gates before biological use: probability normalization and closed-form censoring check; independent analytic monthly probability check; synthetic joint S/reporting recovery (S error < .02, reporting <8% relative under high-information design); real m-array denominator reconciliation; optimization-start agreement. A separate skeptic review is required before using real estimates as constraints. Survival-profile breadth, sparse regions, reporting boundaries, origin selection and delayed recovery remain part of the readout, not hidden failures.

Predictions: broad adult survival near the existing crude .5–.6 estimates is plausible, but regional/era reporting changes can shift it. Flat profiles or boundary reporting estimates would argue against a precise anchor. The survival constraint must not be declared useful solely from a point estimate.

Next integration gate: generate the marked-bird transitions from the simulator's stayer/mover branches. Q replaces Sj for movers, and Sa applies after adult movement at destinations. Confirm full-exposure impulse agreement before introducing partial-year approximations. The current NumPy reference transition is only a mathematical guard, not this adapter. Never multiply Sj by Q as the full-year juvenile survival.

Considerations: G3 (population dynamics rather than habitat compensation), M29 (historical observation reliability), P1 (label assumptions), P3 (rules before results). Existing TACC fits and their immutable code snapshots are untouched.

## Sensitivity registration (after first pilot, before sensitivity fits)

The corrected adult cohort contains 296,896 bands / 782 dead recoveries. Initial S estimates are West .488, Plains .669 (15 recoveries only), central-East .514, East .589. These do not select an informative prior. The schema reconciliation corrected an initial coding mistake (4 is local/nestling, 1 is AHY); the initial fit was superseded and must not be used.

Run initial exposure fractions .25 and .75 month against .5, and constant reporting against the four-era reporting fit. A survival displacement > .03 is material for this pilot and prevents treating the point estimate as a stable anchor. This checks only two assumptions, not reporting delay, movement, constant survival, or selection. Report the profile support and all optimization failures, including reporting boundaries in sparse strata.

Additional sampling guard, before running it: simulate constant annual survival .55 at the real release-cohort sizes, using each region's fitted era reporting probabilities only to set realistic recovery frequencies. Seeds 41 and 42; refit S and all reporting parameters. Report whether .55 lies inside the approximate likelihood-profile support (delta NLL <= 1.920729). This is a power/coverage diagnostic with only two seeds, not a coverage study. The very sparse Plains stratum is expected to have broad support; do not require a precise point estimate there.

## Completed results

Machine-readable results: `BBL_ADULT_PILOT_RESULTS_20261031.json`. Raw source SHA256: `3a9b18307987cdad6871039d3b75a87f56b1168eafdd6c2d2c89a990badaaad1`. Compressed m-arrays, full profiles, and synthetic artifacts are in `.research_runtime/bbl_adult_pilot_20261031/` (not committed).

| Origin region | Eligible adult bands | Dead recoveries | Annual S | Grid points within delta NLL 1.920729 |
|---|---:|---:|---:|---|
| West | 171,151 | 430 | .488 | .475–.500 |
| Plains | 6,588 | 15 | .669 | .525–.800 |
| Central-East | 40,299 | 90 | .514 | .450–.575 |
| East | 78,858 | 247 | .589 | .550–.625 |

The last column is an approximate profile-support **grid**, spaced .025; endpoints fall between grid points. It is not a calibrated biological uncertainty interval. It excludes neither structural bias nor movement/selection effects. Three optimizer starts agree to <8e-6 NLL in the reported fits. Plains pre-1980 reporting hits its lower numerical bound and must not be interpreted.

All registered release-fraction/reporting sensitivities shift S by <.02. Conditional reporting ranges roughly .0016–.0050 in the better-sampled region-era combinations; do not confuse that with an unconditional recovery probability. The original .3% center is not automatically numerically unreasonable for old, nearly fully followed cohorts: most birds eventually die. The issue is deriving it on the correct conditional scale and follow-up, not banning that numerical value.

Under the **real release sampling**, planted S=.55 is inside profile support for West, central-East and East in both seeds. Plains seed 41 estimates .292 and excludes .55 (delta NLL 2.497); seed 42 estimates .525 and includes .55. With only 15 real recoveries, this is a warning against a precise Plains anchor, not a two-seed coverage estimate. The high-information analytic/synthetic test independently recovers both S and reporting.

Ten focused tests pass, including:

- independent closed-form monthly recovery/censoring probabilities versus sparse likelihood;
- synthetic joint S/reporting recovery;
- eligible-band denominator, chronology, handling mortality, and official AHY-versus-local age coding;
- propagation of a marked location distribution over multiple intervals, without births or repeated survival factors;
- an explicit finite-grid enumeration versus FFT juvenile dispersal impulses in the **live** ladder `age_forward.py`, including destination Q and finite-epsilon edge loss. The FFT's tiny zero-input padding response is subtracted for this linear-operator comparison.

Interpretation: the adult records have useful information under this coarse model outside the Plains, and favor a broad regional survival constraint as the next integration experiment. **No prior or production likelihood has been changed.** The current estimate is origin-region, constant-S, encounter-month based. It cannot establish a habitat gradient or identify juvenile journey mortality. Source-era selection, death-report delay, age mistakes, and movement still require stress tests before a biological claim or adoption.

## Integration contract and remaining work

`marked_transition` and `marked_recovery_distribution` now provide a tested NumPy reference for the survival block. A juvenile transition has `(1-p) Sj` on its stayer diagonal and `p M Q` on mover destinations. For adults the same construction uses Sa at both stayer and mover destinations. Propagate the full surviving distribution, with no reproduction. Reported-death probabilities use loss of living mass in each interval and a conditional reporting probability; the never-reported outcome also includes unreported deaths.

This is **not yet a differentiable production adapter**. A row of M must be built from the exact fitted departure, edge correction, cohort weights and any enabled steering/attraction. Q must retain its destination/cohort indexing. Time-specific partial exposures must be explicit. Reporting that depends on death location needs a corresponding mortality-location operator; the scalar interval reporting in the reference does not supply one. The full-exposure impulse test currently covers the base juvenile movement primitive, not every optional mechanism or a full-year age-class switch.

Computational scoping correction: gathering Sa at a bird's expected cell is cheap but generally wrong. Propagating tens of thousands of marked spatial cohorts over thousands of cells and decades is not automatically negligible. Batch/coarsen cohorts with a verified approximation or use a low-rank transition representation before promising a cheap joint fit. A broad regional adult-survival block is an explicit approximation and must be tested against full propagation on a subset.

Next decision: collect the pending nest/founder guards, then preregister the joint adapter and planted-J1 test on the chosen life cycle. Keep the pilot estimates as external diagnostics until that mapping and its bias tests pass. Do not use the same BBL encounters independently for survival and movement likelihoods.
