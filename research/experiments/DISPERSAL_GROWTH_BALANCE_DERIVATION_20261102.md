# A movement formulation that could reduce required growth

This is a population-balance derivation and a candidate mechanism, not a demonstrated reduction in fitted lambda. It follows the diagnosed crowding/departure mismatch in VEIT_LEWIS_RETENTION_COMPARISON_20261102.md.

## Two competing pressures

Near-zero departure below equilibrium can make observed spread difficult unless the model increases production, founders or favorable habitat change. At the other extreme, frequent movement can deplete breeding populations or lose many birds on journeys, also increasing replacement requirements. Departure probability alone therefore has no monotonic relationship with the growth the fit requires.

The relevant quantity is successful recruitment elsewhere per unit of demographic production, while preserving local replacement. Changing a movement curve can make modest growth sufficient, but cannot guarantee that an optimizer chooses modest potential growth when other compensators remain.

## A source population's annual balance

Consider B breeding birds at a site, no incoming birds, and returning adults that remain locally. Let Sa be their annual survival, F the young produced per breeder, Sj the annual survival of young that stay, and Q that of young that move. These are expected counts, not a decision made after learning which individual survives. Assume mate limitation is negligible in this established example, slots are fully occupied, and all locally available recruits can breed the next year. F is realized production at that site, not necessarily the unregulated fecundity maximum.

Local replacement needs (1-Sa)B new breeders. It therefore requires (1-Sa)B/Sj locally retained young. To produce E successful emigrant recruits, another E/Q young must enter movement. Hence:

    F B = (1-Sa) B / Sj + E / Q

Write e=E/B. Then:

    F = (1-Sa)/Sj + e/Q
    Sa + Sj F = 1 + (Sj/Q)e

If movers and stayers survive equally well, potential growth at this example's reproductive output needs only equal one plus the successful export per breeder. A source exporting20% of its breeding population annually can operate at growth1.2, rather than needing an enormous population overshoot. If moving halves survival, the same export requires growth1.4. Journey survival and successful settlement thus matter as much as departure. If F is still density-suppressed well below its maximum, this calculation does not bound the fitted maximum growth; reducing that gap is part of the proposed change.

Example (illustrative, not fitted House Finch estimates):100 breeding birds,60 adults surviving,200 young produced,30% young surviving. Forty surviving young replace the lost adults and20 recruit elsewhere. The source retains100 breeders while supporting expansion. With only replacement production, there are no young left to export. With lower journey survival, more production is needed for the same successful export, exactly as the equation shows.

## Candidate biological structure

Use the existing habitat/capacity representation to describe breeding space, with its units made explicit. Returning breeding adults have priority; young fill remaining breeding opportunities; surplus young can move before local crowding simply removes their contribution. Some background adult and juvenile movement can remain where biologically justified. Neither perfect habitat knowledge nor guaranteed successful settlement follows from this rule.

Density regulation would chiefly limit local recruitment into breeding space, with measured reproductive variation still allowed. Do not simultaneously normalize fecundity to exact local replacement at the same population reference and demand excess above that reference before departure. Those two controls consume the same demographic surplus twice. A full implementation must align the prebreeding/postbreeding census, survival and movement phases and account explicitly for birds that fail to settle. It cannot be implemented honestly by renaming the present total-population K as a territory count.

Disease changes adult losses and the young available to fill vacancies. It need not leave birds waiting to attain an unchanged, disease-free population threshold. Where disease leaves insufficient recruitment, decline or dependence on immigration is legitimate; the formulation must not force every cell to export.

## Expected benefit and failure conditions

The candidate could support observed spread at modest reproduction by preserving breeding populations and releasing a genuine surplus. It could reduce the need to inflate lambda to overcome the strict threshold. That is a directional hypothesis, not a prior requiring a desired growth value.

It fails as a solution if surplus birds are sent into implausibly costly journeys or sinks, if poor destination settlement causes most to die, if high reproduction is hidden in an unconstrained discarded/nonbreeding pool, or if spatial structure collapses into capacity. Assess actual offspring, retained recruits, successful emigrant recruits, explicit losses and growth capacity together. C near one in a capacity-limited population does not establish that potential growth is reasonable.

## Prior work and methodological basis

This is related to site-dependent regulation and distinguishing natal from breeding dispersal, rather than a new general ecological principle: Rodenhouse, Sherry & Holmes1997, https://doi.org/10.1890/0012-9658(1997)078[2025:SDROPS]2.0.CO;2; Greenwood & Harvey1982, https://www.annualreviews.org/content/journals/10.1146/annurev.es.13.110182.000245. Those sources do not establish the particular allocation law or parameters for House Finches.

The project's E012 tested an adult departure offset; E008/E014 tested capacity/fitness-based destination redistribution. They are relevant predecessors, not an exact test of this recruitment-first balance. E014's perfect sorting flattened vital-rate fields and transferred structure into K, so lossless or perfect-information settlement cannot be assumed beneficial. The existing growth-derived-capacity helper also addresses part of the reproduction/capacity coupling, but its stored formula includes the older delayed-breeding survival factor; it cannot be used unchanged for current yearling breeding.

No full TACC fit of this candidate is claimed here. Existing overflow, floor and historical-range runs remain useful controlled evidence; they are not silently replaced by this derivation.
