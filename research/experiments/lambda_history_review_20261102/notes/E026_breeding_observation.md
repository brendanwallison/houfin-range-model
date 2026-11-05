# E026 -- breeding-season observation (pre-registered 2026-10-16)

Found by reading the likelihood: expected counts were the POST-breeding total (adults + all of the
year's new juveniles). BBS (June) counts mostly breeders. With F ~10 juveniles outnumber adults ~5:1 at
low density vs ~1:1 at K, so modelled counts in a colonising cell grow slower than the breeding
population -- a direct structural push on lambda0.
observation.form = breeding: count(t) ~ breeding population of t (end-of-year total of t-1, all of
breeding age) + phi * fledglings of t, phi ~ Beta(1,4) learned.
E025 year effects (for the record): db_yr -254 vs dm_band but yr sd 0.69 and lam0 1.64 (no change);
with the plausible pin sd 2.25 -- they absorb a large shared TEMPORAL misfit (H3 trend), not lambda0.
Variants on dm_band: db_obs (free), db_obs_f5 (F 5, Sj/Sa .5), db_obs_plaus (F 5, Sj/Sa .5, Sa .6).
Prediction: lambda0 and F fall; the plausible pin much cheaper than without (db_plaus, running).
