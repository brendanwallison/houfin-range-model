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
| `invasion_year` | **1940** | NYC House-Finch release |
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

## Land use / soil = calendar-year-T state

LUH-3 and HYDE are the **annual land state as of calendar year T** (soil is
static). Only climate uses the bio-year window; land use is contemporaneous with
the count year. All streams are then EMA-smoothed (τ = 10 yr) across model years.

## Invasion

`inv_timestep` is **derived** as `invasion_year − first_year`
(`temporal.invasion_timestep`), so the release always fires in calendar **1940**
regardless of where the timeline starts (1902 → index 38; the old 1900 start →
40). It is never hardcoded.

## End of the timeline

`end_year` is the newest BBS field-season year. Covariates that end earlier
(e.g. HYDE 3.5 ends 2023) are carried forward to `end_year` by the existing EMA /
persistence — human population and land use change slowly, so this is
defensible; it is documented, not silent.

## Where it is enforced

- `src/temporal.py` — `load_timeline`, `model_years`, `bio_year_months`,
  `year_to_index` (gap-safe), `assert_contiguous`, `invasion_timestep`.
- `src/data/combine/states.py` — builds the bio-year climate stack and the
  yearly states over `first_year … end_year`.
- `src/data/combine/model_inputs.py` — maps BBS `obs_year → t` via
  `year_to_index`; derives `inv_timestep`; asserts the year axis is contiguous.
- `src/data/preprocess/bbs.py` — clips observations to `first_year … end_year`.
