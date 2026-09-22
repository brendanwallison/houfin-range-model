"""Correlative-SDM benchmark against the dynamic range model.

The benchmark compares DESIGNATIONS, not map appearance: the dynamic model's
``lam_fundamental >= 1`` (a source/sink call -- can a population sustain itself
here WITHOUT immigration) against a correlative SDM's thresholded probability
surface (a suitability call inferred FROM occurrence).

A correlative model is structurally incapable of populating cells 1 and 3 of the
niche x occupancy 2x2 (see scripts/viz/hypothesis_scenarios.py) -- unoccupied
source and occupied sink -- because it infers suitability from occupancy. That is
Pulliam (2000), and it is what this benchmark demonstrates.

See the plan for the full design and for the two framings explicitly rejected
(a temporal backcast into the invasion transient; stop-bins as replicates).
"""
