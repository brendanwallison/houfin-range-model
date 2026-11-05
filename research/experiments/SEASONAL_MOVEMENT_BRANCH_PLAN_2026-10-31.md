# Proposed branch: separate seasonal movement from breeding dispersal

**November 1 scope update:** the user does not currently justify a fully dynamic seasonal movement process. The preferred first step is now an annual demographic model with a constrained seasonal observation mapping, described with published precedents in [SEASONAL_OBSERVATION_INTEGRATION_20261101.md](SEASONAL_OBSERVATION_INTEGRATION_20261101.md). The explicit breeding/winter dynamics below remain a later alternative, not the immediate implementation recommendation. No model or active queue was changed by this planning update.

Planning assessment prompted by the user, 2026-10-31. No new population fits, production changes, or git branch switches were made. Existing autonomous workers/watchers remain in place. Suggested isolated implementation branch: `research/seasonal-movement` in a separate worktree; preserve the current uncommitted research checkout.

## Recommendation

Develop a minimal breeding/winter state model as a controlled alternative to the annual model. Separate reversible seasonal displacement from natal/breeding dispersal. Do not begin with twelve independent monthly movement/survival fields. The purpose is to test a missing life-cycle structure, not to give the optimizer more ways to fit the same annual observations.

House Finch evidence supports this distinction: [Able & Belthoff 1998](https://pmc.ncbi.nlm.nih.gov/articles/PMC1689500/) reports eastern partial migration and reciprocal southwest/northeast seasonal movement. The existing BBL reconnaissance also reports much larger eastern summer–winter displacement than western displacement, and far shorter breeding-season natal distances than the current juvenile kernel. Those encounter summaries remain reporting-biased; they motivate a test rather than fix the kernel numerically.

## Available observations

Verified local roots:

- `/mnt/e/Datasets/ebird_raw`: at least the Canadian raw EBD and accompanying sampling-event file (`CA/ebd_CA_relAug-2026*.txt.gz`) plus protocol/metadata. Country coverage, House Finch row counts, repeated complete-checklist coverage and historical depth still require a manifest; do not assume all continental countries or all years were verified by this inspection.
- `/mnt/e/Datasets/cbc_raw`: all-country/all-year CBC circle-species, effort-summary, effort-by-type, and weather CSVs, plus existing prepared arrays.
- BBS breeding counts, BBL dated encounters and NestWatch observations already used/scoped by the current work.

Use raw focal House Finch observations for the new observation likelihood. Keep House Finch excluded from community-habitat training. The current community-prepared CBC panels intentionally exclude it; return to raw CBC or a dedicated focal extract. eBird Status & Trends products may be useful for descriptive checks, but they are model estimates, not independent raw observations ([Cornell documentation](https://science.ebird.org/en/status-and-trends/faq)). Do not multiply a raw-checklist likelihood by an independent-looking likelihood for maps derived from those same checklists.

## Minimum life-cycle structure

Keep a breeding census B_t and a winter census W_t, with adult and young cohorts where the timing requires them. Define dates explicitly around the actual BBS and CBC observations, not just a January year label. A CBC count in December t or January t+1 belongs to winter t/t+1 according to its date.

Conceptual sequence (the exact ordering relative to birth and natal settlement is a biological hypothesis to register):

1. Breeding B_t -> summer survival and reproduction -> post-breeding population.
2. Natal/breeding dispersal D changes prospective breeding location; seasonal movement A redistributes a subset into winter habitat.
3. Winter W_t -> winter survival, with disease/weather exposures evaluated there.
4. Spring return tendency R, plus explicitly modeled permanent relocation -> B_t+1.

Movement before mortality conserves birds on the modeled domain. Seasonal movement does not create offspring or automatically establish a breeding population. Define how off-domain movements are treated; renormalizing them into the domain is a substantive assumption.

First movement family:

- a resident component;
- a low-dimensional partial-migration component with directional displacement and modest spread;
- optional short-range local redistribution/diffusion within either component;
- a separate permanent-dispersal kernel constrained by breeding-to-breeding encounters.

Do not model migration as repeated independent isotropic diffusion. It cannot represent the directional seasonal signal well and can accumulate artificial displacement. At 54 km, much local redistribution is sub-grid and may be identifiable only as observation availability, not a diffusion coefficient.

## Return connectivity is the main structural hazard

Two aggregate seasonal abundance maps do not identify which breeding populations use which winter sites. An arbitrary autumn kernel followed by an arbitrary spring kernel can make seasonal travel look like permanent spread. The spring kernel must not be the unconstrained inverse of an autumn matrix either.

Begin with a few breeding-origin regions/cohorts or a constrained connectivity basis that retains a return destination tendency. Keep return fidelity distinct from permanent relocation. Compare plausible connectivity alternatives and preserve uncertainty. Breeding–winter–breeding individual records are particularly useful if available; two-endpoint seasonal banding pairs do not fully identify a round trip.

A full cell-origin × winter-cell × time state would be expensive in the differentiable model. Estimate the state/adjoint memory before implementing it; use a coarse connectivity pilot and checkpointed seasonal scans first. The local GPU can support float32 pilots; TACC remains the default for final float64 refinement and larger state spaces.

## Observation models and identifiability

| Source | Main contribution | Required observation treatment |
|---|---|---|
| BBS | Breeding abundance and establishment | Existing route/observer model |
| CBC | Winter distribution and historical winter presence | Circle support, effort response, weather, feeder effort, calendar date, source-era differences |
| Raw eBird EBD + sampling | Within-year occurrence/abundance and transition timing | Complete checklists, reconstructed nondetections, protocol/duration/distance/time/site/observer effects; repeated-site support |
| BBL | Individual movement, return and survival information | Season/age/date, spatial precision, encounter opportunity and reporting delay |
| NestWatch | Reproductive output during breeding | Existing exposure/outcome and brood-number assumptions |

Do not subtract CBC counts from BBS counts to estimate mortality or recruitment: they have different observation scales and spatial footprints. Within-season longitudinal information and marked birds help calibrate the comparison. eBird seasonal changes in counts also mix movement, births, deaths, site choice and detectability.

Initially retain an annual adult-survival constraint and allow only a tightly structured seasonal allocation of its mortality hazard; do not add free summer and winter survival maps simultaneously. Otherwise only their product may be identified. The constant-region adult BBL pilot is a diagnostic constraint, not yet a transferable seasonal likelihood. Time-stamped dead recoveries can help with seasonal allocation only after recovery-date versus death-date uncertainty is handled.

Use existing seasonal habitat information as a starting point, with explicit feature provenance. Do not presume that slicing the current jointly calibrated summer/winter representation yields validated season-specific vital-rate fields. First hold the habitat representation and most demography fixed when testing movement structure, then release compensators in staged synthetic tests.

## Benefits that an annual view misses

- Distinguish winter occupancy from breeding establishment at the invasion front.
- Test whether apparent annual habitat improvement instead reflects changing winter access, redistribution, or survival exposure.
- Evaluate winter habitat at winter destinations rather than breeding origins.
- Separate production during breeding from losses between breeding seasons, subject to observation calibration.
- Align feeder aggregation and seasonal disease dynamics with the birds actually exposed; [Altizer et al. 2004](https://doi.org/10.1111/j.0021-8790.2004.00807.x) documents seasonal House Finch disease dynamics.
- Make BBL age and partial-year exposure map more naturally to the actual life cycle.
- Test carry-over effects (winter conditions affecting breeding output) later, after simpler seasonal structure recovers reliably.

None guarantees better identification. In particular, a seasonal branch may leave the too-fast breeding expansion unchanged if the problem lies in reproduction, establishment or permanent dispersal.

## Staged experiment plan

1. **Data audit and descriptive CPU pilot.** Build a manifest and focal extracts from the existing EBD/sampling and CBC files. Preserve effort, dates and coordinates. Summarize seasonal distribution changes on fixed site sets, with detection/effort adjustment. Contrast East/West, adults/young where marked data allow, and cold/warm winters. Never interpret a seasonal map difference alone as individual migration.
2. **Movement-only modern pilot.** Hold breeding abundance and demographic parameters fixed/anchored. Compare no seasonal movement, local diffusion, and a resident-plus-directional-return model. Use BBL pairs to constrain movement and eBird/CBC to test seasonal distributions. Fit a few regional parameters rather than a cell-year migration field.
3. **Synthetic stress tests under the real observation design.** Plant seasonal trips with zero permanent dispersal; permanent dispersal without migration; winter mortality without movement; changes in detection/effort; and combined migration/dispersal. Allow annual-model compensators to refit. The key gate is distinguishing return travel from permanent establishment, not just recovering total movement distance. Set numeric gates after measuring the data's achievable precision, before reading treatment fits.
4. **Joint two-season population comparison.** Match the current life cycle, observation holdouts, habitat inputs and demographic data blocks. Compare both seasonal predictions and breeding-front timing; report biological compensation and uncertainty alongside scores. Start on the modern overlap where eBird/CBC/BBS support the distinction, then test historical transfer.

Freeze shared geographic/time holdouts across sources for evaluation. Separate auxiliary-data-aided prediction at BBS-withheld sites from prediction at places withheld from all sources. CBC records used to anchor founders cannot also be counted as independent early-history validation. Shared site/observer effects and community-teacher inputs require leakage checks, even though the focal species is excluded from DESK.

Promotion requires identifiable seasonal structure under realistic synthetic biases and a useful prediction/biological result on withheld data. Improved BBS likelihood alone is insufficient. This branch complements the current nest/founder/survival constraints; it should not replace or silently alter their experiments.
