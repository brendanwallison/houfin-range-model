# NestWatch interaction and the actual CBC founder anchor — 2026-11-01

Verified against frozen step33 source (d4435fb), the current run configurations, the full pathology audit and the step31/32 registration in GRID54_LADDER_2026-10-29.md. This note distinguishes fitted quantities from imposed values and historical evidence from model-state observations.

NestWatch enters as a joint likelihood on realized reproduction at observed nest cell-years, not as a direct prior on Fmax. The simulator supplies F_real after density dependence and mate finding. Under paired breeders and a 1:1 sex ratio, the mean output per attempt is `2 * F_real / B_region * exp(X delta_mean)`, with separate logistic-exposure information on daily nest survival and a model for fledged counts conditional on successful nests. The fixed cycle is 32 days. The current central configurations pin attempts per female per season B to 2.5 in all three regions; nuisance effects cover building, nestbox, other artificial substrate, unknown substrate and historical cards. The registered training-period selection is 1,089 rows through 2020, with overlapping exposure/count information, heavily skewed toward modern observations. These observations do not directly measure 1940s reproduction or dispersal.

Consequently the data constrain realized output jointly with attempts, density, establishment and the fitted nest-observation corrections. They do not independently identify Fmax, juvenile survival or an annual breeding-conversion factor. Raising attempts would require greater annual reproduction for fixed per-attempt means and fixed nuisance effects, but a joint refit can also change spatial demography, nest success and observation terms; the fitted mean F_real need not move proportionally.

Matched relative-Allee readouts illustrate the interaction (modern fitted-population-weighted medians):

| Configuration | Adult survival | Resident juvenile/adult ratio | Founder bird-equivalents |
|---|---:|---:|---:|
| Yearling base, no nests or founder pin | .517 | .757 | 44.0 fitted |
| Add NestWatch, no founder pin | .396 | .995 | 29.6 fitted |
| Add central founder pin, no NestWatch | .387 | .991 | 77.46 fixed |
| Both NestWatch and central founder pin | .400 | .995 | 77.46 fixed |

Both kinds of constraint can redirect compensation toward survival on this relative-law base. NestWatch is informative, but it does not by itself prevent a ceiling solution. On the combined absolute-law fit, ratios instead become .465–.479 and adult survival .560–.570. Native departure preserves this broad improvement. However, there is no completed matched absolute-Allee + native-departure with/without-nests ablation in the current audit. The healthy ratios cannot therefore be attributed specifically to NestWatch. The completed attempts sensitivity under absolute Allee shows the interaction is still consequential: B=2.0 gives Sa .534 and ratio .641, while B=3.125 gives Sa .379 and ratio .980. Near-equivalent predictive scores conceal different demographic allocations.

The decisive missing ablation is to hold absolute Allee, founder setting, habitat inputs and other switches fixed and compare nests off/on with native departure off/on, then repeat or propagate annual-conversion uncertainty. This is a proposed discriminating comparison, not a claim that those fits have run. Simply freeing attempts using the same per-attempt data would add a fecundity confound; independent seasonal nesting histories/effort or defensible conversion sensitivities are needed.

The current CBC-informed anchor is still a **single 1940 injection**, not a later density-state constraint. It goes into one 54 km cell nearest Jones Beach, half adults and half juveniles, before annual movement. The inherited release-cell Allee exemption lasts ten years. Metadata inv_timestep=38 with a 1902 initial year gives 1940. The model's `invasion_pulse_prior` has median .754 and log SD .77 recorded, but those prior values are inert in the current pinned arms: `pinned_sites.log_inv_total` fixes the injection.

| Setting | log_inv_total | Injection in model founder units exp(log_inv_total) | 1940 bird-equivalents |
|---|---:|---:|---:|
| Central | -.283 | .754 | 77.46 |
| Low bracket | -1.79 | .167 | 17.16 |
| High bracket | +1.23 | 3.421 | 351.70 |

The conversion is 102.8 birds per founder unit, conditional on the release-cell land fraction (~.647), BBS sampled area (~25.13 km2), and assumed detection .5. Bird numbers are therefore model-conversion equivalents, not observed release counts. The low/high values are an assumption-based sensitivity band, not a posterior interval. They remain fixed even when reproductive assumptions or movement change within a given arm.

Evidence used to construct the backcast (CBC winter years labeled by January):

- 1946/47: six/four CBC circles ran inside the release cell with no recorded House Finches.
- 1948: first CBC detection, 35 birds. Release-cell CBC sums for 1948–55 are 35,104,36,37,66,80,44,49. These are raw sums under changing effort, not cell-population estimates.
- Through 1952 the recorded CBC House Finches lie in the release cell; a neighboring-cell detection appears in 1953.
- The registration cites Elliott & Arbib's approximately 280 known birds in the NYC area by 1951, interpreted as 280–560 after an assumed allowance for undercount. That is not the CBC total: the 1951 release-cell CBC sum is 37.
- The 1951–66 release-cell CBC increase from 37 to 1,400 was used to motivate growth up to ~1.27 annually; it includes changing observer awareness/effort. Together with the apparent late-1940s/early-1950s lag, the backcast assumed average 1940–51 growth roughly 1–1.27. The early zeros and a judgment against an initial decline helped set an approximate 20–300 founder band, enlarged for conversion uncertainty to the 17–352 equivalents used in fits.

Thus the label 'CBC founder anchor' is shorthand for an assumption-heavy historical backcast, using CBC geography/counts plus a historical population tally. It is not a fitted CBC observation likelihood. It does not hold the simulated 1951 state near 280–560, and it does not directly use later densities in the current likelihood.

A preferable next formulation is to retain an uncertain 1940 release and put observation constraints on the simulated state in actual later survey years. Raw CBC circle counts and surveyed nondetections should have effort/detection and winter-to-breeding correspondence represented. A rough 1951 historical tally could provide a broad additional observation constraint if its provenance and spatial coverage are verified. This uses later evidence without fixing an assumed early growth rate in a back-calculated founder pin. It also lets NestWatch constrain reproductive mechanisms and historical observations constrain the resulting accumulation/spread. Moving the injection to 1951 or hard-resetting the state would bypass the early dynamics rather than test them. Reusing the same records as both a founder anchor and an independent likelihood would double-count evidence.
