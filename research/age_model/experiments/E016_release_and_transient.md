# E016 -- release phase (R1, R2) and colonization-transient mask (F1) (pre-registered 2026-10-16)

Synthesis so far: both implausible demands sit in the EASTERN COLONIZATION TRANSIENT -- under
settlement, 53% of the Fmax-3 cost lies within +-10 y of a cell's first positive (peak 5-10 y after),
67% before disease arrival; the Allee/exodus cost sits within +-5 y of arrival. The release is
prior-fixed: founder pulses sit on their prior (0.06-0.10 of q50 in every one of 90 site-years).
All warm ifd3.

R1a budget10: invasion_pulse_prior.log_budget 3 -> 10 (founders free).
R1b budget10_f3: same + Fmax 3. Read: cost(R1b) - cost(R1a) vs ifd3_f3's +314. If it collapses,
    high lambda0 was partly growth from a prior-fixed seed through the unobserved 1940-66 phase.
R2a exempt26: Allee exemption over the whole pre-BBS era. R2b exempt0: no exemption.
    Read n50 / exodus slope: if the strong Allee is serving pre-BBS confinement, exempt26 is costly
    and exempt0 cheap; if it serves the post-1966 front, both leave n50 ~unchanged.
F1a mask10 / F1b mask5: drop observations within +-W y of each cell's first positive (left-censored
    cells kept): 30k / 17k of 192k obs. Read lambda0, Fmax, n50, exodus slope. If they become
    plausible, the transient is the whole forcing. U is NOT comparable (fewer obs); the decomposition
    on ALL obs is (the masked obs become held-out).

R1 results (2026-10-16):
| variant | U | Fmax | lam0 | n50 | founder sum (frac of q50, 90 site-years) |
| ifd3 | 197642.1 | 10.6 | 1.85 | .18 | 6.84 |
| budget10 | 197701.7 (+60: wider prior's normalisation; dll +3) | 10.5 | 1.86 | .185 | 6.67 |
| budget10_f3 | 197812.9 | 3.0 | 1.69 | .14 | 2.22 |
| ifd3_f3 (tight founders) | 197956.3 | 3.0 | -- | -- | 6.18 |
-> Fmax-3 cost with FREE founders: +111 (vs +314 with prior-fixed founders): freeing the seed removes
   ~65% of the remaining fast-growth demand. The fit wants FEWER founders at low fecundity (2.2 vs 6.7):
   the prior-fixed seed (~9 q50 of sustained immigration over 10 y into 9 cells) was too large, and
   the pre-BBS population had to be braked/grown around it. Supports the user's release hypothesis.
