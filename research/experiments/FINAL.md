# FINAL DESK candidate (provisional) -- definition, evidence, open items

Started 2026-10-24. Goal (user): a provisionally final DESK candidate to lock in today / tomorrow. The user's stated
defaults are the starting point and must be overturned by convincing data, not by what the pipeline started with:
ensemble builds, rank 64 (higher only if justified), several data sources (BBS + CBC, eBird if feasible), CBC presence
supervising the early years, reference community = the entire community.

## Candidate v1 (queued on TACC 2026-10-24: tasks 01_final_s1..s4)

| Component | Choice | Evidence / status |
|---|---|---|
| Reference community | all 369 non-sealed species | user rule; E086y2-b (observed communities: all > residents > sedentary-at-size); E086ac (all vs all-sedentary through DESK) PENDING |
| Basis | spacetime ESK, BBS raw points, 16k landmarks, rank 64 (`esk_all369_breed`) | E086ab / E086ab-b: through DESK the all-species basis needs ~48-64 dims for sedentary-common species |
| Readout rank | 64 | E086ab-b: at r64 all-species >= old 96 (late); E086d / E086k: beyond 64 barely covariate-predictable, so not higher |
| BBS supervision (primary) | raw route counts of all 369 species on the 27 km grid, 114,172 surveyed cell-years 1966-2025: the stabilizing target (z pinned to each cell-year's ESK coordinates) and the Ruzicka kernel loss (dot(z_i, z_j) ~ BBS Ruzicka over pairs across space and time) -- the production terms, unchanged | the basis and the main supervision are BBS; CBC terms are auxiliary |
| Window | 1902-2025 | covariates 1902+; smoke + TACC 1902 build ran (E086w) |
| CBC Ruzicka, 1966-2024 | all 333 CBC species, calibration refit to the new basis (dot ~ 0.017 + 0.838 Ruz, corr 0.76, weight 2.81) | E086s / E086u: replicated sparse-data change gain for sedentary species (selected state); needs winters inside the era (E086u2) |
| CBC presence, 1902-1965 | 15,727 lists, 333 species, completeness slopes, WINTER observation operator (held-out nll 4.22 vs uniform 5.81), weight 0.01 | user preference; E086v-b: with the winter operator ~inert (neither helps nor harms); E086x: no handoff artifact for the Ruzicka term |
| Ensemble | 4 seeds, delivered as ONE mean cube (`research/exp/final_mean_cube.py`, refuses if seeds are not aligned) | age model reads one cube (downstream contract) |
| eBird | not included in v1 | no all-species eBird product prepared; the E084 eBird metric head (96 species) was safe with small gains (E086n) -- a v2 candidate |
| Training state | selected epoch | E086o: the late-state harm to sedentary breeding change in cover-change cells is within seed noise at the selected state |

## Which result governs E081's harm finding

E081 found that winter (CBC) supervision of z harmed sedentary breeding-season change in high-cover-change cells. The
governing later result is E086o: that harm is late-state-specific (real for residents in cover-change cells at epoch
~490, within seed noise at the selected state). E086s / E086u then show the CBC Ruzicka gain for sedentary species at
the selected state. The candidate ships the selected state.

## Open items (from the age-model research agent, houfin-range-model-9a, 2026-10-24, and own records)

1. The age model truncates z BY POSITION to its first 24 dims (`model_inputs.py:470`, `age_model_config.json`
   latent_dim). Rank 64 is an age-model config change -- the age-model owner's call. Check the eigen-ordering of the new
   basis before relying on a positional cut.
2. LUH3 irrigation channels are broken (M31 / E080); production DESK reads all 33 land-use channels, including the
   management / primary-secondary "clock" channels. Not changed in v1 (an input change is a retrain with its own test).
3. The single-amplitude readout over-moves; the split readout (between / within blocks) is contract-compatible (E072).
4. Honest uncertainty (E072c): change error at a place since ~1920 is comparable to the spatial contrast; span-scaled
   corr(H1920, H2000) ~ 0.1-0.3. The iid z.z' kernel implicitly assumes ~1.
5. Every DESK model, CBC-supervised or not, predicts about half the mean change of sedentary common species (E086x); even
   the observed breeding kernel carries little of it. The winter half of a sum kernel helps the level readout at r64
   (+0.03, E086ab-b) but not the within readout.
6. The early-era presence term is ~inert (E086v-b); before 1966 z is in effect covariate extrapolation.

## Pending tests feeding v1 / v2
- E086ac: all species vs all sedentary basis through DESK (decides the community, if convincing).
- 1902 CBC build vs its matched control (E086w): early-era and modern change, to be scored once the age model needs it.
- Secondary, "anything that works" (user, 2026-10-24): a fixed smooth regional correction on top of DESK; an extremely
  smooth half-century human-bird trend term inside DESK (a stand-in for feeding). Not started.

## Candidate v2 (queued on TACC 2026-10-24: tasks 02_final_v2_s1..s4) -- EQUAL summer / winter halves

User (2026-10-24): several data sources, high rank, and no assumption that summer > winter -- equal unless proven
otherwise. z = [z_summer 64, z_winter 64]; the cube exports all 128 dims (`latent_cube.include_private`), so the age model
can read 32 + 32 or 64 + 64 by position within each half.

| | Summer half (z, 64) | Winter half (private block, 64) |
|---|---|---|
| Basis | all-species BBS spacetime ESK (`esk_all369_breed`) | all-species CBC spacetime ESK (`esk_all369_winter`, 333 species, 1966-2024) |
| Coordinate target | stabilizing term, weight 64 | coordinate target on 81,794 CBC cell-years (70,801 kept), weight 64 |
| Ruzicka pairs | BBS, weight 5 | CBC (identity head on the private block), weight 5 |
| Early years 1902-1965 | covariate extrapolation (+ cross-season term from 1966) | CBC presence lists (13,379 kept), winter operator fitted FROM WINTER COORDINATES (74,115 lists, held-out nll 4.09 vs uniform 5.81) |
| Cross-season | CBC Ruzicka on z_summer, 1966-2024 (the E086s / E086u term) | -- |

Smoke (local CPU, 1 epoch + cube, jobs final-v2-smoke / final-v2-smoke-cube): every term loads and trains (epoch 1:
winter coordinate 38% of the loss, winter pairs 7%, presence 2%); cube (133, 224, 128), winter half varies in 1902 and 2025.

Evidence for v2's design so far: E086ab-b (r64) -- a winter half improves the level readout +0.03 (CI > 0) and costs the
within readout -0.024; untested with equal 64 + 64 halves and a winter coordinate target. Compare v1 vs v2 once built.
