# Seasonal observation integration: literature and a minimal House Finch proposal

November 1, 2026. Literature review and proposed design only; no population model, fits or active queues changed. This narrows the earlier SEASONAL_MOVEMENT_BRANCH_PLAN_2026-10-31.md in response to the user's preference against a fully dynamic migration process. Do not treat a stronger founder prior as the main route to identification.

There is direct precedent for using BBS, Christmas Bird Count and eBird together in seasonal distribution models, and separate precedent for adding reproduction and marked-bird information to an annual demographic model. I did not locate one directly matching all four named data sources in a partially migratory House Finch population. Combining the published components with our annual mechanistic model is a proposed adaptation, not a published validated House Finch specification.

## Closest precedents and their boundaries

- **Ogawa et al. (2024), Bayesian integrated species distribution models for hierarchical resource selection by a soaring bird.** American white pelicans: summer BBS detections and summer eBird sighting-location counts share summer occupancy states; winter CBC detections and winter eBird location counts share winter occupancy states. Seasonal models have their own environmental associations and survey observation terms; CBC detection includes party-hours. This is direct precedent for the exact three observation sources and seasonal distributions without migration trajectories. It does not connect summer/winter abundance through a shared demographic population or establish partial-migration identifiability. [Paper](https://doi.org/10.1016/j.ecoinf.2024.102787), [open full text](https://eosfcweb01.eosfc-intl.net/cp4810_U95041_Documents/Text%20Collections/WS%20Publications/2022-2025/REP%202024-078.pdf).

- **Robinson et al. (2018), Using citizen science data in integrated population models to inform conservation.** Tricolored blackbirds: an eBird relative-abundance time series was integrated with banding and nesting data. The eBird input was restricted to an early breeding window and integration was sequential. This supports combining opportunistic observations and demography; it does not solve winter connectivity or calibrate absolute abundance automatically. [Paper](https://doi.org/10.1016/j.biocon.2018.10.002), [author-hosted full text](https://tricolor.ice.ucdavis.edu/sites/g/files/dgvnsk3096/files/inline-files/Robinson%20et%20al.%202018.pdf).

- **Link & Sauer (2007), Seasonal components of avian population change: joint analysis of two large-scale monitoring programs.** A hierarchical seasonal model jointly used BBS and Christmas Bird Count with survey-specific sampling covariates for Carolina Wrens. It establishes a simple summer/winter integration precedent, but its resident-species application cannot be assumed valid for local counts of partially migratory finches. [Paper record and abstract](https://pubmed.ncbi.nlm.nih.gov/17489453/).

- **Strebel et al. (2022), Spatiotemporal modelling of abundance from multiple data sources in an integrated spatial distribution model.** A shared latent abundance process and distinct observation models integrated count, detection/nondetection, presence-only and absence-only data. This supplies the general multi-source observation architecture, not a seasonal demographic mechanism. [Author-institution summary](https://www.vogelwarte.ch/en/publications/spatiotemporal-modelling-of-abundance-from-multiple-data-sources-in-an-integrated-spatial-distribution-model/), [DOI](https://doi.org/10.1111/jbi.14335).

- **Robinson, Morrison & Baillie (2014), Integrating demographic data: towards a framework for monitoring wildlife populations at large spatial scales.** Seventeen bird species were analyzed using count, nest-record and ringing information. In this British study, CBC denotes **Common Birds Census**, not Christmas Bird Count. It is relevant to integrating nest measurements with annual demographics, not an example of the exact American surveys. [Paper](https://doi.org/10.1111/2041-210X.12204).

A useful limit is **Nuse, Pavlacky & Latimer (2025), Linked dynamic abundance models as the basis for a full-annual-cycle, integrated population model**. Their separate seasonal models share rates without requiring tracked cross-season trajectories; the application assumes complete migration and uses ancillary demographic data. Simulations found seasonal counts alone insufficient for accurate vital-rate estimation. This is a more elaborate option, not the recommended first implementation. [Paper](https://doi.org/10.1002/ecs2.70485).

The species-specific historical concern is documented: **Able & Belthoff (1998)** reported partial migration, southwest breeding-to-winter movements, northeast winter-to-breeding movements, age differences and changing migration patterns in the introduced population. Their historical banding-based percentages should not be imposed as unbiased current migration probabilities or as a precise description of the 1940s. [Paper/author abstract](https://experts.boisestate.edu/en/publications/rapid-evolution-of-migratory-behaviour-in-the-introduced-house-fi/), [DOI](https://doi.org/10.1098/rspb.1998.0541).

## Recommended starting architecture: annual demography with seasonal observation mapping

Retain the existing annual population process and permanent natal/breeding dispersal. Add source-specific observations of the appropriate seasonal population quantity. Seasonal redistribution changes the winter view of the annual population; winter observations do not redefine breeding establishment.

| Source | Quantity linked to model | Observation terms |
|---|---|---|
| BBS | Existing breeding-season population/readout | Route/observer, effort/protocol and count dispersion |
| Breeding-window eBird | Same breeding population | Complete checklist nondetections, duration/distance, party size, time, protocol, site/observer and sampled-location coverage |
| CBC | Seasonal winter distribution | Circle footprint and coverage, party-hours/method, weather, selection and overdispersion |
| Winter-window eBird | Same winter distribution as CBC | Checklist effort/detection and coverage terms, with its own observation scale |
| NestWatch | Realized reproduction | Existing visit exposure, nest outcomes, conditional counts and annual-conversion uncertainty |

Use raw focal-species records. Published eBird maps derived from those same records are not additional independent likelihood data. Integration can reduce the dependence on particular surveys, but does not automatically remove preferential location selection or make absolute abundance identifiable. All local geographic/year coverage still needs verification before promising continental integration.

A minimal seasonal redistribution view could be written:

    W_t = [I - D_m + K_theta D_m] P_t

Here P_t is the population quantity at the chosen post-breeding census, D_m is diagonal with source-region migratory fractions, and K_theta maps migrants to winter destinations. Its columns sum to one over a defined domain; off-domain movement must be represented explicitly. West versus Northeast might initially have different fixed or very few estimated migration parameters. A single southwest-biased displacement kernel with a few plausible distance scenarios is enough for a first test. This is an analyst-proposed observation mapping, not a formula taken from the papers above.

W_t describes conditional spatial redistribution, not an independently propagated seasonal demographic state. It does not feed a separate migration/return simulation into next year's breeding population. Annual reproduction and survival remain in the original process; we would not infer migration-specific mortality, return fidelity or carryover effects from this construction. This omission is the scope of the approximation, not a finding that such effects do not exist.

The census timing must respect the annual simulator's adult/young pools and reproduction. CBC in December t/January t+1 cannot be attached indiscriminately to the same year's breeding adults; young production and seasonal exposure matter. Any unresolved common seasonal abundance factor is confounded with survey scale and must not be reported as estimated survival. Start with a stated conversion/sensitivity, rather than free winter survival maps.

A simple spatial winter/breeding multiplier is even cheaper but cannot create winter occupancy in cells with zero breeding abundance. It is therefore inappropriate wherever winter visitors extend beyond breeding establishment. The redistribution mapping avoids that particular error while keeping seasonal movement distinct from permanent dispersal. An unconstrained cell-year seasonal correction would create a new escape route and is not recommended.

Historical migration cannot be calibrated precisely by modern eBird alone. Begin with fixed or region-specific movement scenarios, use dated seasonal banding to inform their plausible shapes with encounter-selection sensitivity, and test modern transfer using independent held-out observations. If historical evidence warrants time variation, compare a small number of era hypotheses, not a freely varying migration field. If the early establishment interpretation changes substantially across plausible scenarios, retain that uncertainty rather than choosing the scenario with the best founder story.

## Staged evaluation and expectations

1. Build a focal raw-data manifest and seasonal extracts; verify complete-checklist zeros, effort variables, calendar/census alignment, geographic coverage and duplicated sampling events across sources.
2. Fit breeding-window eBird plus BBS and NestWatch first, keeping the current demographic mechanisms and common geographic/time holdouts. This can improve recent abundance/pattern constraints without any winter-movement assumptions.
3. Pilot CBC/winter-eBird observation terms on modern overlap with a fixed or very small seasonal mapping. Compare no seasonal redistribution and a few directional resident-plus-migrant alternatives. Evaluate held-out winter and summer predictions separately.
4. Test whether these seasonal alternatives are distinguishable from seasonal detectability, recruitment and permanent breeding spread under realistic synthetic observation designs. Establish source/census contracts before interpreting recovery.
5. Apply the accepted observation models to historical CBC with explicit migration and sampling uncertainties. Assess changes in the later initialized release-cell state and expansion history. Do not independently multiply the old backcast founder anchor by a likelihood for the same historical observations.

More abundance observations can constrain late-year maps and reduce individual-survey artifacts, but may still leave fecundity/survival/establishment tradeoffs. NestWatch and BBL measure different components and remain useful. A direct observation likelihood can be more informative than a founder prior, but its calibration and seasonal mapping determine whether it targets the desired state.

The recommended first branch is therefore an **annual demographic model with a constrained seasonal observation mapping**, followed by a more explicit seasonal process only if a demonstrated residual problem requires it. It can use all three count sources and the reproductive observations while making narrower claims than a full annual-cycle migration model.
