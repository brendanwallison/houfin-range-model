# The model timeline — one definition of "a year"

This is the canonical temporal contract. Every stage — ingestion, EMA states,
the community encoder, path features, the age-structured model, and
visualization — uses it. The machine-readable source is the `timeline` block of
`config/data_config.json`, exposed through `src/temporal.py`. Nothing else
should hardcode start/end years or the invasion offset.

## Model year T

Model years are the contiguous calendar years `first_year … end_year`, indexed
`0 … N-1`. The mapping calendar-year ↔ index is a **lookup**
(`temporal.year_to_index`), never `year - start` subtraction — so a gap in the
timeline fails loudly instead of silently shifting every index.

| knob | value | meaning |
|---|---|---|
| `first_year` | **1902** | first year with a complete climate bio-year |
| `end_year` | **2025** | newest BBS field season (2026 release, 1966–2025) |
| `invasion_year` | **1940** (base); **1951** in the current candidate | NYC House-Finch release (see Invasion) |
| `bio_year_start_month` | **8** (Aug) | start of the climate bio-year window |

## Climate = bio-year Aug(T−1) → Jul(T)

Climate for model year T is the **antecedent 12-month window ending at the
~June breeding-season BBS count**: August of T−1 through July of T. Ending in
July means weather *after* the count never leaks into the predictor for T;
starting in August captures the winter/spring survival + breeding-condition
window that drives the June count. Assemble it with
`temporal.bio_year_months(T)`.

Because observed climate (climr / CRU TS + GPCC) begins **January 1901**, the
first year with a *complete* bio-year is **1902** (Aug 1901 → Jul 1902). That
is why `first_year = 1902` even though data acquisition starts in 1901.

### The 12 months stay 12 channels

The bio-year window is **not** averaged into an annual value. Each base variable
enters the model as **12 separate channels**, one per month position in the
window — `{base}_b{kk}m{MM}`, where `b01` is `bio_year_start_month` (August) and
`m{MM}` names the calendar month it resolves to. So `Tmax_b01m08` is August of
T−1 and `Tmax_b12m07` is July of T.

Ordering by *window position* rather than calendar month is deliberate: channel
index then means "months since the window start", and `b01..b12` also sorts
lexicographically into window order (which matters because sorted variable order
*is* the channel order). Calendar ordering would put Jan–Jul, the window's late
months, before Aug–Dec, its early ones.

Elevation quantiles are asymmetric to keep the channel count in hand: every base
gets `q50`, and only `Tmin`/`Tmax`/`Tave` also get `q10`/`q90`, since within-cell
relief moves a temperature via the lapse rate far more than it moves a flux.
Authoritative channel order lives in `climate_grid_monthly/manifest.json`.

## Land use / soil = calendar-year-T state

LUH-3 and HYDE are the **annual land state as of calendar year T** (soil is
static). Only climate uses the bio-year window; land use is contemporaneous with
the count year. All streams are then EMA-smoothed across model years (τ = 2 yr,
per `states.streams[].ema_tau`). The EMA runs **per channel along the year
axis**, so a monthly climate channel is smoothed against the same month of
previous years and never against its neighbouring months.

## Invasion

`inv_timestep` is **derived** at ingest as `invasion_year − first_year`
(`temporal.invasion_timestep`), so it is never hardcoded. Under the base
`config/data_config.json` (`invasion_year` 1940) the release fires at index 38;
the old 1900 start gave 40.

**Current candidate (revised 2026-11-08; `docs/model_spec/`).** The release fires
in calendar **1951** (index **49**), into one cell, (27, 89).
- **How the index was set.** It was not produced by `temporal.invasion_timestep`
  on the base timeline. The candidate's inputs (`established1951`) were derived
  from the J200 ingest by `research/exp/invasion_sandbox/kernel_sens.py derive`,
  which changed only the metadata `inv_timestep`, 38 → 49
  (`docs/model_spec/DESK_PROVENANCE.md` row 6).
- **Data overlay.** The 54 km data overlay `config/overlays/grid54_data.json` sets
  `timeline.invasion_year` = 1951, so the resolved data config agrees.
- **Base config.** The base `config/data_config.json` still reads 1940.
- **Mismatch risk.** A fresh ingest under the base timeline would therefore
  release in 1940.

## Epizootic window

`disease_start_year = 1993` is the winter mycoplasmal conjunctivitis first
appeared in wild House Finches in the mid-Atlantic. `disease_timestep` is derived
from it exactly as `inv_timestep` is from `invasion_year`
(`temporal.disease_timestep`; index 91). In the current candidate
(`disease_prior.target = survival`) it bounds the **added-mortality amplitude**
Δ(x, t): Δ is identically zero before this index (`src/model/age_priors.py`). No
K-depression term runs in the candidate. The "K depression" reading applies only
to the base-config default `target = capacity`; the same wording in the
data-config `timeline._comment` is stale for the candidate. The per-pixel
*arrival* year is exogenous, not fitted (see
`scripts/build_disease_arrival_map.py`). The fitted onset adds a continental slack
δ_0 and a regional slack to that map, then ramps in over 3 years (smoothstep gate);
see docs/PRIORS.md#disease-onset-and-recovery.

## How the age model uses the timeline (current candidate)

Revised 2026-11-08 from `docs/model_spec/MODEL_SPEC.md` §0, §3.6, §4 and §5.

- **Annual step.** The model runs t = 0 … 123 ↔ 1902 … 2025, one step per year.
- **Spin-up.** It is preceded by a 40-year **spin-up**, with rates frozen at their 1902–06
  means, no disease and no release. The spun-up total becomes the 1902 state.
- **Habitat input.** The habitat input Z_t is the FIELD54 DESK cube with **EMA smoothing of
  the DESK output** (half-life 10.9 yr) plus a field correction. This EMA is distinct from
  the τ = 2 yr state-stream EMA described under "Land use / soil = calendar-year-T state" above
  (`states.streams[].ema_tau`).
- **Readout window.** The split readout's long-run level Z̄(x) is the 1996–2025 mean of Z_t.
- **BBS expected count.** The count for year t uses N_{t−1}, the previous year's
  end-of-year (post-breeding) total. "Breeders of year t" is a modelling convention [L].
- **Pre-1966 evidence.**
  - Range absences: t ≤ 1965.
  - Early CBC counts: 1902–65.
  - Museum-specimen counts: 1880–1940. The 1880–1901 decade is read against the 1902
    state.
  - The 1902–39 pseudo-zero BBS rows. Whether they are kept is a *pending* user decision;
    see docs/PRIORS.md#observation-model.

## End of the timeline

`end_year` is the newest BBS field-season year. Covariates that end earlier
(e.g. LUH-3 ends 2024) are carried forward to `end_year` by the existing EMA /
persistence — human population and land use change slowly, so this is
defensible; it is documented, not silent.

## Where it is enforced

- `src/temporal.py` — `load_timeline`, `model_years`, `bio_year_months`,
  `year_to_index` (gap-safe), `assert_contiguous`, `invasion_timestep`,
  `disease_timestep`.
- `src/data/combine/build_states.py` (+ `combine/streams.py`) — builds the
  bio-year climate stack and the yearly states over `first_year … end_year`.
- `src/data/combine/model_inputs.py` — maps BBS `obs_year → t` via
  `year_to_index`; derives `inv_timestep` and `disease_timestep`; asserts the
  year axis is contiguous.
- `src/data/preprocess/bbs.py` — clips observations to `first_year … end_year`.
