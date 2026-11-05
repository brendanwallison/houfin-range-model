# E025 -- continental year effects on expected counts (pre-registered 2026-10-16)

Base dm_band (U 195728.2; single release, disease as mortality, learned kernel tail; lam0 1.68, F 8.7,
Sj/Sa .5 free). year_effects: per-year log multipliers on expected counts, sd ~ HalfNormal(0.3),
non-centred, centred over observed years.
Variants: db_yr (free), db_yr_plaus (F 5, Sj/Sa .5, Sa .6).
Prediction: if the deterministic model chases survey-wide swings with fast dynamics, lambda0 / F fall
with year effects on and the plausible pin gets cheaper than db_plaus (running). Fitted sd ~0.1-0.2
expected for BBS-wide annual noise; a large sd that soaks up trends would be a red flag.
