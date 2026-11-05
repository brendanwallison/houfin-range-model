# Numerical control: curvature history

Directional finite differences at the best compact-onset endpoint agree with
the JAX gradient at steps 1e-7 and 1e-6. Along the normalized negative gradient,
local secant curvature is about 1.64e6; a step 1e-4 increases the objective even
though a step 1e-6 decreases it. This is evidence of severe stiffness, not an
obvious wrong-gradient failure or demonstrated stationarity.

Registered control: same compact-three-year single-field model, same constrained
posterior, same priors and all active sites, warm start at the selected/refined
day-loop endpoint. Run at most 1500 iterations with L-BFGS maxcor 200 rather than
20. Save raw likelihood/prior decomposition and actual gradient. The question is
whether more curvature history improves descent/conditioning economically.
Do not interpret this as a new model, a growth-change constraint or a biological
negative result. Conditional likelihood profiles follow independently.
