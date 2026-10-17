# E045 -- constrain temporal habitat change so refinement cannot contort (pre-registered 2026-10-17)

E044: run_40 MAP is realistic (plains 70% sinks, median lam .85 vs west 1.11); refinement gains 12,813 nats,
~5,600 of them from eastern positives 1980-2025 (MAP under-predicts the east 1.2-1.7x) -- raising growth broadly
lifts plains lambda .85 -> ~1.05 and erases the trough.
User: refinement also exaggerates century-scale change; DESK/ESK validation says temporal change is low, and
1902-1965 is unsupervised, so the model can posit huge pre-BBS shifts. If such shifts are disallowed, the only
pre-BBS supervision (no eastward invasion from the native west) must be carried by the plains trough.
Measured: mean |dlambda| since 1902 MAP .17-.19 vs refined .25-.48; continental k_trend up to x1.7, fitted
9-19 prior sds out (prior sd .029 per weight).
Variants on cold run_40: c40_nokt (k_trend budget 1e-6 = off), c40_eta (habitat_change learned, eta ~ Beta(4,5),
mean .44 ~ DESK over-move 2.2x), c40_nokt_eta (both). Viz each.
Read: does the REFINED optimum keep the plains trough (sink share / K-weighted / median lam vs flanks) and
realistic demography, and at what likelihood cost; per-region decomposition vs cold run_40.
VRAM (for a later higher-rank test): M=24 float64 refine loads ~9.6 GB (Z_disp 4.9 GB) of a 29.6 GB limit; M=48
fits; M=64 needs mem fraction ~.95 or float32 Z_disp or host residency.
