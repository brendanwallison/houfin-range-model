# Additional experiment set: which life-cycle census regulates reproduction?

Plan written after the crowding-curvature fits were implemented, validated, published and verified in TACC's queue at20:01 CST November2. This additional set is not implemented or queued here.

## Question and motivation

Should density-dependent reproduction respond to all birds present before annual survival, as now, or to the birds surviving to form the current breeding population?

The present reproductive function computes its crowding term from adult survivors of movement plus juvenile stayers plus journey-surviving arrivals, before applying local adult/stayer survival. It then computes the new breeding population and its births. Thus a young cohort that is largely lost before breeding can still contribute strongly to the density brake on reproduction. That may be appropriate for food or other resources consumed during the earlier interval. It is less clearly appropriate if the modeled fecundity reduction represents competition among breeding birds. This is an alternative biological hypothesis, not a demonstrated implementation error.

E026's breeding-season observation correction and E052's yearling-breeding correction showed that life-cycle timing can change reproductive demand. E053's Veit reproduction already used surviving breeders, but also changed pair formation, the site-limitation function and the meaning of K. Its low F/high adult-survival tradeoff therefore does not isolate this question. The new set would change only the density used by the standard reproductive crowding term.

The general need to align the census used for density feedback with birth/death ordering is discussed explicitly in Chevy et al., *Population Genetics Meets Ecology*, Ecology and Evolution2025, doi:10.1002/ece3.71098, especially Section7/AppendicesB–C: https://doi.org/10.1002/ece3.71098 . Author-hosted text: https://www.benhaller.com/pubs/ChevyEtAl2024.pdf . This supplies a methodological precedent, not evidence that breeding-only competition is correct for House Finches. The published correction concerns a GBIF figure citation, not these density-feedback arguments: https://doi.org/10.1002/ece3.71540 .

## Concrete model contrast

Keep the actual life cycle and all survival/movement terms. Define B as the very same surviving adult/young pool that already produces births in the yearling model:

```
B = Sa_effective * adult_after_movement
    + Sj_effective * juvenile_stayers
    + juvenile_arrivals_after_the_existing_disease_factor
```

The last term already includes journey survival; do not apply Sj again. B equals the simulator's new adult/breeder state before new offspring are added.

Do not divide B directly by the current K, since K is a total-population equilibrium reference. Derive its matching breeding-population reference from the original, pre-additional-disease rates used to construct c:

```
F_replacement_reference = (1-Sa_reference)/Sj_reference
B_reference = K/(1+F_replacement_reference)
            = K*Sj_reference/(Sj_reference+1-Sa_reference)
```

This is the adult/breeder share of the same nominal, mate-free, disease-free equilibrium. Keep the numerical survival guard consistent with the actual c calculation in an implementation. It introduces no new population scale or fitted field.

Compare:

```
current crowding density:  x = N_before_local_survival/K
breeder crowding density:  x = B/B_reference

F_effective = Fmax/[1+c*x^power]
```

At the nominal equilibrium without disease or mate limitation, both normalized densities equal one, so the equilibrium calibration is retained. At low density they both approach zero, so potential growth at fixed vital rates remains Sa+Sj*Fmax. They differ during changing age composition, juvenile losses, immigration and disease—conditions under which the existing model must explain expansion and decline.

Keep the existing absolute mate-finding factor and its census unchanged in this first comparison, as well as departure density and kernels. Moving mate limitation onto breeders would require an explicit conversion of its fixed absolute scale and would confound the crowding test. Keep disease active; use reference rates for the scale conversion and actual disease-modified surviving breeders in B. Do not redefine K as a hard number of territories, and do not impose an overflow rule.

Where potential reproduction is insufficient for replacement, retain c=0 as now. The contrast should not turn genuine demographic sinks into viable sources by construction.

## Small matched set

Begin with the established gradual Veit departure control, so the test remains interpretable even if the new rational departure law performs poorly:

| Density used for crowding | Power1 | Power2 |
|---|---|---|
| Existing pre-survival total | step39 Veit controls, seeds41/42 | queued step42 Veit fits, seeds41/42 |
| Surviving breeding population, properly normalized | two proposed fits | two proposed fits |

This adds four fits and reuses four controls. All retain identical 1951 initialization, shared habitat, weak absolute Allee, reproductive observations/conversion assumptions and historical supervision. Do not simultaneously add a K floor or switch CBC designs. A promising result can subsequently be crossed with the new departure law.

## Predictions and failure modes

The directional hypothesis is that protecting productive breeding populations from excessive crowding attributed to a transient young cohort can preserve useful realized reproduction with a lower maximum. The optimizer may then need less inflated growth and less historical habitat opening. This is not guaranteed: the fit may simply alter survival, increase crowding elsewhere, or spread faster.

Check current versus breeder-normalized pressure by age composition and region before fitting. If they are nearly identical throughout informative cell-years, this is not a useful new lever. Verify the closed two-age equilibrium independently and compare forward derivatives against finite differences. Then inspect invasion trajectories, productive-growth curves, actual reproductive patterns at NestWatch sites and full stage accounting.

Judge the same complete pathology scorecard as the curvature test: aggregate growth and its historical change, survival compensation, actual Runge contribution/replacement and immigration requirements, Plains viability/artifacts/bridges, establishment, predictive transfer, seed sensitivity and HMC geometry. A change in individual survival rates can be acceptable if higher-level growth and movement remain credible and stable. A lower Fmax achieved by increasing adult or juvenile survival without lowering the inflated productive-growth ceiling is not the target.

This set is worth testing because it isolates a life-cycle mechanism already implicated by earlier experiments, uses no additional learned field or scalar, and is distinct from changing the mathematical curvature of crowding.
