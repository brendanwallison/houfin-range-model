# Observed invasion targets for the sandbox (built 2026-11-03)

Builder: research/exp/invasion_sandbox/targets.py (run via sbx_env.sh step3_obsroute; config paths only, no model).
Output: /mnt/e/Datasets/houfin/grid54/sandbox/targets/INVASION_TARGETS.{json,png}. Distances from the release cell
(row 27, col 89; nearest Jones Beach). East = land east of 95W, 25-49N (exhaustive G13 mask).

## East
T1 (eastern CBC circles in cbc_early.npz; the file also spans the native West, excluded): first positive circle beyond
54 km and 100 km in 1955, beyond 200 km in 1956 (one circle; p90 of positive circles is 153 km in 1959). Matches the
ladder record (step 5: first > 54 km 1955, > 100 km 1956).

T2 radius (p90 distance of positive sites):
| source | year: km |
|---|---|
| CBC, eastern circles | 1956: 354 (1 outlier circle); 1959: 153; 1962: 195; 1965: 267 |
| BBS fixed panel (950 eastern routes run >= 25 yr of 1970-2010) | 1970: 54; 1975: 233; 1980: 411; 1985: 550; 1990: 1128; 1995: 1422; 2000: 1436; 2010: 1483 |
Speeds: ~17 km/yr 1955-65 (CBC); ~32 km/yr 1975-85 and ~87 km/yr 1985-95 (BBS) -- the observed front ACCELERATES through
the BBS era until it reaches the 95W boundary of the mask (~1450 km) in the mid-1990s. The BBS radius lags the CBC
radius by ~10 yr around 1965-75 (detection of sparse birds on summer routes; winter migrants in CBC -- Weiss 1997).
Read the two sources separately; never splice them into one curve without that offset.

T3 share of panel routes positive: 1970 .00, 1975 .01, 1980 .10, 1985 .22, 1990 .42, 1995 .64, then ~.55-.66 (2010 dip
.56: Mycoplasma era).

T4 fill-in (mean log1p count by years since a route's first positive, per arrival decade) and T5 (Hovmoller: share of
routes positive by year x 1-degree longitude, 33-43N, 106-88W; both fronts) are in the JSON/PNG; not yet summarised.

Caveats: BBS first positives are detection-biased late (SDM_BASELINE_B6: ~8 yr for late cohorts); the CBC file starts
in the model's cbc_t years (count labelled by the summer year used by the fit); early CBC effort is unreliable (M29).
