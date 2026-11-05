# Invasion-shape sandbox: research branches and plan (living document, started 2026-11-03)

Purpose: one place for every research branch, idea, user directive and correction from the 2026-11-03 sandbox
sessions, so nothing is lost.
- Results and pre-registrations: `SANDBOX_INVASION_SHAPE_2026-11.md` ("SBX" below, cited by round ID).
- Original approved plan: `C:\Users\brend\.claude\plans\codex-currently-has-a-serene-peach.md`.
- Considerations: `research/CONSIDERATIONS.md` (G3, G4, G13, P6, M6, M12, C1-C12).

REVISION 2026-11-03 (evening). Rebuilt after an independent audit of the full transcript against this plan. The audit
found: missing branches; promises not kept; stale or contradictory rows; a mis-worded row (A3) that led to a wrong
build; and lessons from user corrections that were not recorded. Every item it raised is folded in below, marked
[audit] where it was missing.

Status labels: DONE (with reading), RUNNING, NEXT (agreed, not started), OPEN (idea, not agreed), BLOCKED (needs
something), PROMISED-NOT-DONE (said it would be done; not done). Evidence labels as in the record: established /
likely / unknown. Base labels: [s3] step3_obsroute (averaged Z, relative Allee); [s3n] step3_native (native field54
Z, relative Allee); [cur] step43_factor_p2_veit_s41 (native field54 Z, absolute pinned Allee, standard yearling
reproduction, crowding power 2, Veit departure, initialised 1951, single shared habitat field). The production-facing
base is [cur]-like; every claim should say which base it rests on.

## Working rules (user directives; read before every round)

Science framing:
- Understanding phase: report production implications, do not decide them.
- Static-habitat tricks are diagnostics only; habitat change must stay learnable.
- A regional trend is predictability, not dynamics.
- Sedentary species first.
- Middle grounds are built ON Z / Z_disp (the community prior), not as free functions of time or density (user, raised
  repeatedly; the assistant sidestepped it more than once).
- Production-facing base = native field54 Z on the current lineage, not the averaged 27 km Z (user: "we should have
  a native 54 km DESK").
- Compare abundance only in the fit's own observation units (the expected-count readout). CBC enters only through a
  CBC observation model, never by converting model units to "birds" against raw counts.
- Reports in plain language (user).

Process:
- Run autonomously. Never idle with planned work; never ask the user to choose between sandbox diagnostics. Run
  them, pre-registered, and report (user: "run autonomously", "go back to working autonomously", "I thought you were
  working autonomously", "say you're running things and you aren't holding them up").
- Arm a monitor for every job and act on completion. Silence is not success.
- Read the records BEFORE claiming novelty or proposing a test (already-tested items: release year, step 5).
- Before building a branch, RE-READ ITS ROW HERE, and the user's words that created it. (A3 was in the plan, but a
  different structure was built.)
- Verify the base's overlay (Allee form, reproduction form, initialisation year, field form) against the current
  lineage before running or interpreting it.
- Do not present known project history as a finding (single field = ladder step 2).
- Efficiency: screen with short runs (about 40 iterations) on the GPU (free to use; Codex has no GPU workers). Run
  CPU arms in parallel. Use float64 for anything that starts at an optimum. Treat about 1,000 log-lik units between
  capped runs as optimizer noise.

---

## 0. What we now believe (one screen; base in brackets)

1. **"Early too fast" is established. "Late too slow" is conditional.** [s3, s3n]
   - Frozen habitat: the release fills the East 20-31 yr early (R0). In BBS observation units the East is about 60x
     too dense in 1966-79 and 2x too sparse by 2010-25 (established).
   - "Late too slow" shows mainly in ABUNDANCE / fill-in. In first arrival the frozen error is a roughly constant
     offset. Every time-homogeneous brake trades one phase for the other (R1, R2, R11-R13).
   - Removing the native West: with frozen habitat the native West crosses the Plains before 1955 (R0); habitat
     gating holds both regions.
2. **The inflation is driven by fitting the post-1966 invasion, not by the empty pre-BBS period.** Ladder step 5
   (inj1960): lambda change unchanged when the release moves 1940 -> 1960. The observed invasion constrains the fit
   strongly (user). Established.
3. **Literature reproductions** (DONE, missing from the earlier plan) [1-D / 2-D uniform landscapes]:
   - L1, Veit & Lewis 1996 reproduced. r .15 gives lag then acceleration in 1-D. In 2-D the same threshold kills
     the release; about a 10x WEAKER threshold (r .01-.02) reproduces lag and late speed with unchanging habitat.
   - D1: the matching mate-finding threshold MOVES up as potential growth rises (lambda0 1.42: n50 about .002 birds /
     km^2; 1.8: .03; 2.2: .1; 3.0: .2) and narrows to a knife edge. Weak Allee suffices if potential growth is low;
     strong Allee is only the patch for high growth, and it fails at the knife edge in the real landscape (R1).
     Production implication (likely): bring low-density potential growth down, not a stronger Allee.
   - D2: cell size (10-54 km) is not what removes the lag.
   - Stochastic twin (1-D, integer birds): spreads FASTER than the deterministic model at V&L rates. The "fractional
     front" explanation fails in its simple form.
   - Density floors stop all spread at .003 birds/km^2 or more (in MODEL units; the birds conversion is a convention,
     see J).
4. **Front timing runs through LOCAL rates from Z, not journey survival from Z_disp** (R15, replicated on [cur]).
   Established.
5. **Which rate carries lambda inflation depends on the base.**
   - [cur]: about 70% survival (East Sa .13 -> .48), fecundity flat (4-5).
   - [s3n]: about half fecundity.
   - Exact Leslie lambda in both.
6. **Regulation differs by base.**
   - [s3n] (Beverton-Holt power 1, relative Allee): fecundity acts almost only at low density; crowding cancels it in
     filling cells (effective F 3.6-4.5 while Fmax swings 6-12).
   - [cur] (C1): power-2 crowding is weak below K. The ABSOLUTE Allee is the front brake (mated fraction .23 in sparse
     cells 1951-55, .60 in 1966-75). The median filling cell in 1976-90 is a SINK filling by immigration. The [s3n]
     front-speed-knob reading does not transfer.
7. **Realised growth vs literature anchors** (r .12-.18 in the 1940s, about .22 in the 1960s).
   - [s3n] matches both (about a doubling, even though potential rates inflate by an order of magnitude; partly
     arithmetic, r = log lambda).
   - [cur] (initialised 1951) stalls 1951-62 near the release (r .01 vs CBC about .24). Likely cause: dispersal drain
     into Allee-dead neighbours (F1). R20 (sedentary founders) raised it only to .15-.16.
8. **Time-shape inverses** [s3, s3n].
   - R5c: with fixed habitat the best one-rate history is a smooth, about 16x rise in eastern fecundity 1940-2010;
     no step near 1960 (no one-off behavioural change); capacity alone cannot substitute.
   - R6b: calendar time beats local settling-in (colonisation age) by 2,155; the age effect flips sign once calendar
     time is present.
   - R7: the West needs a rise too (about 2-3x from mid-century); the East's is steeper (about 40x). Shared in
     direction, not size: supports the PROPOSED G15 (see G4) in direction only. Regional calendar trends reach within
     1,000-1,600 of the full fit: predictability, not cause.
9. **Measured settlement growth (BUI q90)** [s3, s3n].
   - Right timing, and the same coefficient East and not-East. Identified in the native West alone, it transfers
     64-71% of the East's gap (R10).
   - Only fecundity reaches the needed size (about 16-60x); capacity, survival and Allee-relief routes fail at credible
     sizes (R11-R13).
   - BUI and the fitted habitat change are SUBSTITUTES in the likelihood. Spatially, the fitted change is
     ANTI-correlated with BUI growth on [s3] (-.20 to -.35) but correlated on [s3n] (+.46 East, +.48 Plains) (R9d).
     The native field carries more of the settlement pattern.
10. **Flexible forms alone do not remove the demand** (R14 [s3n]): a free low-density time shape needs a 14-21x
    change. User's rule: the rigid form is not the problem.
11. **Establishment readout from Z** (R16-R20 [cur]).
    - With the standard readouts frozen, Z's own time change routed through a low-density boost closes 95-98% of the
      frozen-to-fitted gap. Confined to low density (rho <= .3 K) it needs a continent-wide x6-10 rise. Its East /
      West split is NOT identified (moves with every dispersal rule; the "eastern-specific" clue is withdrawn).
    - PROXY CHECK: its time course correlates .85-.92 with the fit's own habitat-change trajectory. It is the same Z
      time signal rerouted, not new information.
    - JOINT refit (C2c, LOCAL from the fit): the axis adds +88 log posterior but stays near 1, and the field keeps its
      inflation. HOMOTOPY from the establishment basin: as the field sees real Z the fit returns monotonically to the
      inflated-field basin (135,491; establishment solution 137,250). The data choose the shared field, not
      establishment, for Z's time signal. A3 closed for this structure.
12. **Dispersal rules tested so far do not remove the requirement** (R17-R20 [cur]): kernel choice, philopatry
    (worse), an eastern departure ramp, sedentary-then-evolving departure. NOT closed: destination-habitat settlement
    (B3) and a departure floor from near-empty cells (B9) are untested.
12b. **SYNTHESIS (evening):** every mechanism that helped with the standard readouts frozen was abandoned once the
    shared field could use Z's time change: the establishment axis (A3 homotopy), the survival anchor (C2), the
    departure floor (B9c, floor -> .07). The inflation is the shared field passing Z's eastern time signal into
    Sa / Sj / Fmax; mechanisms cannot compete unless the field is denied that signal. Established for these
    mechanisms on [cur].
12c. **Two jobs of the fitted habitat change** (mix_replay, [cur]): K's time change carries most of the likelihood
    (67% alone: abundance / fill-in) but leaves the front 27-44 yr early. The growth rates' JOINT change (Sa, Sj,
    Fmax: the lambda inflation) times the front (without K: -5 to -15 yr). The single shared field couples them.
12d. **Decoupled structure (R21, replay at fitted parameters):** Z's change to CAPACITY only (abundance) plus an
    establishment readout of Z for front timing, with NO time change in Sa / Sj / Fmax, matches the all-rates replay
    (111,682 vs 111,854) with fit-like timing. The establishment change needed is x4.7 East / x2.2 West, replacing the
    fit's joint three-rate inflation. Best decomposition so far. Caveats: still above the about 3x bar; same Z
    signal (proxy question). A9 JOINT REFIT: holds (about 900 worse, not converged), survival constant, but
    low-density potential lambda still rises about +.78 (fit +.87): the inflation is relocated and legible, not
    removed.
13. **Central open questions:**
    - Where does Z's strong eastern time signal come from: the observation-fitted regional field R(s, t) vs
      covariates (E1-E4; TACC files)?
    - Is there a competitive solution in which an establishment axis, not the shared field, carries it (A3 homotopy)?

---

## A. Z-built middle ground

| # | Branch | Status | Notes / next |
|---|---|---|---|
| A1 | Low-density / establishment readout of Z with the standard readouts FROZEN (R16 / R16c-R20) | DONE | See 0.11. Remaining: (i) implied potential-lambda change vs the fit's [PROMISED-NOT-DONE]; (iv) static control from several starts [PROMISED-NOT-DONE]; a time-shuffled-Z placebo [not run]. The old "dims 15 / 6 / 18 / 21" check does NOT apply ([s3]-era indices; [cur] uses native Z); replaced by the overlap check (done) |
| A2 | Crowding-curvature readout theta(x, t) from Z | DONE (R16) | Works only via extreme early compensatory crowding (theta .06-.99 in 1940); static theta .23-21; not credible |
| A3 | Establishment as a real model axis: a SEPARATE establishment / low-density readout of Z added to the ONE shared habitat field, jointly refitted. Neither the single field nor the old separate correlated fields ever had an establishment-specific axis (user) | DONE (C2c local + homotopy): from the establishment basin, letting the field see real Z returns monotonically to the inflated-field basin (135,491 vs 137,250 at alpha 0); field inflation returns in full (lambda .23 -> 1.08), establishment axis flattens (x1.2). The data do not select the establishment route | [Corrected: the earlier wording "beside Sa / Sj / Fmax / K" implied four fields and led to the wrong build.] CLOSED for this structure: an axis built from the same Z cannot pull the time signal from the shared field. Not a promotion candidate. The open lever is what Z's time signal is (E1-E4) |
| A4 | Level / change split for the establishment readout | OPEN | Ties to E059e / E069 / E073 |
| A5 | Nonlinear (GP) link of Z for the readout | OPEN, low | Z information about 95% linear (E039) |
| A6 | Winter-capacity / winter-survival readout from the WINTER Z block | BLOCKED (TACC item 6) | Most direct feeder link; survival carries about 70% of [cur]'s inflation |
| A7 | [audit] State-only learned residual / universal differential equation on growth (Phase D; the user's SciML ask). Inputs: H, N/K, colonisation age, settlement; never calendar year; shape-constrained; read back its shape | OPEN | User: "is there a middle ground ... would we still get source-sink?". Answer given: yes if local demography and movement stay separate; flexibility needs STRONGER guards (no calendar inputs, transfer, step-28 truth, bounds) |
| A9 | DECOUPLED structure, joint refit: capacity reads real Z; Sa / Sj / Fmax read frozen (mean) Z (space only); a separate establishment readout reads real Z; habitat sites + readout refitted on the full posterior | DONE (A9): within about 900 of the fit's posterior (not converged), fit-like timing; survival constant and plausible (Sa .39); establishment G x5.1 East / x3.3 West; low-density potential lambda still rises about +.78 (fit +.87) | Viable alternative structure that makes the inflation legible and confined to establishment at the front; it does not remove it. Next: converge it (more iterations), a second seed / start, then the branch-E question of what the signal is |
| A10 | Abundance innovations with a looser prior (user question after Codex's TACC cell-year results) | DONE (R22): sigma .3 absorbs 97% of the BBS gap (prior cost about 14,800) but fronts stay 33-55 yr early; innovations suppress occupied eastern cells 1951-80 then boost after 1996. They fix abundance, not timing | Agreed with the user: the coherence cost is the anti-takeover guard; the problem is acting on the stock. Mean-one centring compounds into a decline at loose sigma (note for any TACC rerun) |
| A11 | ONE innovation per cell, active only on the wavefront (colonisation clock x low N/K; user: "elegantly target the wavefront") | K-real DONE (R23, N/K gate, 200-it cap): gap .83, prior 704, offsets +3/0/-4/-9; u negative East/Plains, tracks habitat quality and the fit's habitat change (rho -.5 to -.6, R2 .57 East), not distance from release. K-frozen RUNNING; R23b (absolute Allee gate, converged) RUNNING | Alternatives noted: innovations on ARRIVALS' settlement success (can act ahead of the front); front-indexed low-rank (sector x year) speed innovations. Codex's existing process_noise.target = low_density is the closest built option (still multiplies the stock) |
| A14 | Allee STRENGTH vs habitat change as substitutes for timing (R26 T0/T0w, R27 alpha curve, R28 production-structure refits with n50 unpinned; [cur] pins n50 = .059 route counts via age_priors.PINNED). Records: E002 (free n50 .28 demanded by the early eastern front), E055 (pinned: unidentified with habitat change; Allee-dead loop) | T0: frozen rates + K real -> n50 x13.3, F x1.92, BBS +15,000, fronts -3..-6.5, never .156 (Allee-dead modest: Plains 3%, West 2.4%). R27 curve DONE (free n50 worth 15,070 / 10,480 / 6,650 / 2,640 / 0 at a = 0 / .25 / .5 / .75 / 1; fronts fit-like at every a when free). R28a: n50 x.81, lambda rise unchanged. R28b (decoupled): n50 .274 (= E002), G East x4.6 -> x1.9, additive low-density lambda rise +.75 -> +.70. R29: shape irrelevant once n50 free. T1: + small per-cell establishment -> 113,674 (within ~3,600 of fit), n50 x2.9, u loses its habitat tie (R2 .84 -> .20). REVIEW SUBAGENT running | Production implication to report (not decide): the pin + habitat change vs a strong Allee are alternative explanations of timing; the user's Veit 1-D/2-D prior favours weaker |
| A15 | Round 2 (review-ranked, 2026-11-04): R30 const-growth refit, R31 full posterior, R32 region transfer, R33 Z_disp keys, R34 T1 controls, R35 realised growth, R36 multistart, R37 absolute-gated A9 | R31: strong-Allee survives full posterior (NestWatch charges ~1,180). R32: East demands x14.6, native West+Plains x2.9. R33: Z_disp real -> n50 x8.7. R35: anchors reject frozen growth WITHOUT strong Allee, not with. R36: one basin. R34: static per-cell map + n50 x6.1 within ~700 of the fit under the FULL posterior (NestWatch better than fit); map alone (pinned) fails timing; both needed. R30 (plateau): constant growth + n50 x8 is 1,285-2,089 worse than A9/R28b/fit (full posterior). R37 (absolute-gated G, pinned): G suppresses early, relaxes later; 138,309. R38: static Z map (64 coef) + n50 x5.2 within ~2,800 of fit under the FULL posterior; pinned fails timing. R39: K frozen -> nothing fits. R40 (absolute-gated G + free n50): 136,151, best decoupled (fit +560), vital rates constant, establishment of mate-limited birds rises x6.4 East through time -- the establishment / low-density decomposition. R41: shared modern zero over-prediction | Production implication to report: the n50 pin selects between lambda inflation and a stronger-Allee + static-heterogeneity explanation |
| A16 | Round 3 (from the round-2 review): R42 Z_disp real in the decoupled family; R43 static maps added to the FIT; R44 readouts (K*, lambda(N,t), G decomposition); R45 West-only; R46/R48 time-block HELD-OUT; R47 what G may read | DONE. Z_disp confound real but modest (-170 to -430). Fit + map gains only 609 (Z map: 26); matched flexibility keeps the fit ~1,200-2,500 ahead in-sample. Pin is V&L-consistent in [cur]'s East (n50/K* .018); alternatives need 3-4.5x above V&L's top. Decoupled median eastern cells have c = 0 (abundance set by a low-density hump, not K). G's rise ~90% a continental Z clock (dims 14/19/13); detrended G = static G; time-varying G worth only ~440. West alone keeps n50 at the pin. HELD-OUT: fit beats every constant-vital-rate / stronger-Allee alternative by 2,800-3,500 (1994-2025) and 1,130 (2010-25), in every region incl. the native West | Implication to report: the fit's time-varying vital rates carry forward-predictive information; open question = is it habitat or Z's continental clock (E1-E4, blocked on TACC files) |
| A17 | Round 4 (from the round-3 review): R49 score never-fitted rows (overlay observation_mask: 2021-25 + 31 blocks; harness now scores them); R50 fit with free n50 held-out; R51 NestWatch leak; R52 fit with detrended field; R53 vital rates detrended / K real | DONE. R49: on never-seen rows differences are a few hundred (blocks favour time-varying structures ~400-500; 2021-25 indifferent); R41 zero-share "defect" was an artefact (fit matches). R50: fit + free n50 still beats alternatives by 2,300-3,000 held-out; on early data it prefers n50 x.44. R51: removing post-1993 NestWatch TRIPLES the alternative's held-out deficit (2,878 -> 8,543). R52: without Z's continental clock the fit forecasts worst of all (+4,670, West +3,556). R53: ~3/4 of the forecasting edge is K's clock; the vital rates' clock matters mostly in the West; in the East vital rates without the clock forecast as well as the fit | Open: is Z's continental clock habitat or encoder artefact (E1-E4, TACC files) |
| A18 | Round 5 (from the round-4 review): R54 clean-row long forecasts; R55 detrending references (West / per-region); R56 linear-clock placebo; R57 scrambled-years placebo | DONE. 2021-25 (never-seen): fit beats the static alternative by ~1,400 at both splits (East); blocks do not discriminate. West-referenced detrending: East = fit (+118), West +1,292 (2/3 of R52's West loss was the band-mean artefact). Per-region detrending: +2,887 (East +1,131). Linear clock = fit within noise (+565). Scrambled years: +5,942 (worse than no clock). The forecasting information is each region's own Z trajectory; the continental part acts as a linear trend | Next: independent driver test (BUI / HYDE vs observer effort) for the regional trajectories; E1-E4 when TACC files arrive |
| A19 | Round 7 + necessity (user reframing 2026-11-04: lambda RISE expected; find the necessary vs convenient part): R59 observer refit, R60-R64 regional detrending / placebos, R65/R67/R68 vital-rate necessity path (a = share of Z's time change the vital rates read; K real), R66 common-observer comparison, R69 per-rate necessity | R59: refitting observers on training reverses the ranking (both over-fit); R66: with COMMON training-only observers the fit structure still beats the static alternative in both pairings (1,066-3,723 held-out). R60b/R61/R64: the East's forecast information is its own trajectory relative to the West, carried as a regional trend (a straight line suffices). Necessity: a = .75 beat the single-field fit under both observer sets, BUT R71 (structure-matched decoupled a = 1) is the best held-out arm -> the "a = 1 worse" result was the structure difference; a = .75-1 are supported about equally: essentially the FULL eastern swing in occupied-cell potential lambda (~.3 in 1940 -> ~1.5 in 2010) is forecast-consistent. R69: adult survival's time change carries ~86% of the forecast skill; juvenile survival's ~17% and nothing on top; fecundity's change alone is worse than none. R72 (replication) RUNNING | REVIEW CORRECTION: vr_alpha is compensated by refitted weights (w_scale_raw), so the path did NOT vary the realised swing; the necessary swing SIZE is unresolved (1966->2010 occupied-East median rise somewhere ~.66-1.0). Robust: some rise needed; modern end pinned (2010 median ~1.3-1.6); adult survival carries the forecast value. Next: swing path with weights held fixed; swap-aware G-inclusive lambda readout; per-rate partial shares; latent-leak control | Production implication to report: a constraint on vital-rate time change relative to K would be absorbed by larger weights (more extreme spatial spread); juvenile survival / fecundity are the leading convenient-lever candidates |
| A20 | Genuine necessity (after the round-7 review): R73 swing path with the fit's weights FIXED (a truly scales the fit's time change); R74 swap-aware G-inclusive lambda readout of per-rate arms | DONE (one split, one observer set, caps). Clean rows (2021-25, blocks): a swing of ~.6 (a ~ .5; occupied-East potential lambda ~.65 in 1966 -> ~1.26 in 2010) is necessary; a = .5-1 equivalent on 2021-25; blocks prefer smaller swings; only the latent-leaky 1994-2020 rows prefer the full swing. R74: adult survival's change (~.5 swing) carries ~3/4 of forecast skill; juvenile survival's larger swing alone carries little. -> BUT R73/R75 "half suffices" relied on a fitted ~80% low-density fecundity cut (G .16-.19); R77 (G pinned at 1): half the swing FAILS on clean rows (+3,443 / +1,286, fronts 4-7 yr early), the full swing works. RESOLVED: data accept EITHER the fit's full rise (occupied-East potential lambda ~.5 -> ~1.55, 1966-2010) with no low-density penalty, OR ~half the rise + a strong low-density brake; the pin / user prior favour the former. R77/R78 (no brake, two splits): a = .75 equals a = 1 on all clean rows, a = .5 fails -> REVIEW (R71-R78): what is forced is a DEEP 1966 start (occupied-East median ~.45, invariant for a >= .7; pre-1994 clean blocks + timing); minimum swing ~.7; the MODERN level (~1.1-1.55) is weakly identified (non-identifiability, not demonstrated convenience; the paradise TAIL is spatial, in the weights). Deep start rests on adult survival ~.14-.20 (literature check needed). "Adult survival backbone" rests mostly on leaky rows. Conditional on fixed dispersal / disease / pin R79 (fine grid): clean rows conflict on the modern level (2021-25 wants ~1.3-1.4; blocks ~1.1-1.2; never-reached ~1.55). R80 (dispersal + disease refitted per arm): modern clean rows accept a = .5; the INVASION ERA (fronts, training, pre-1994 blocks) rejects it; a = .75 best/tied on every clean score -> necessary = deep start (~.45 in 1966) rising to ~1.2 by 2010; the extra ~.3 to ~1.5 is supported only in-sample (best convenient-lever candidate). BBL anchor: the deep start rests on Sa ~.15 vs banding .56 East 1960-93 (implausible rate). R81b (HARD BBL anchor, Sa ~.51 in occupied East 1966-93): costs ~160 training, NOTHING in the invasion era (pre-1994 blocks better, fronts on time, never-reached better), ~400-600 on modern clean rows; early hostility is NOT moved to other rates -> 1966 occupied-East potential lambda ~.76 (mild sink) instead of ~.47: the "deep start" was mostly implausibly low adult survival. R82: the post-1993 vital-rate rise carries forecast info at known places (with latents fixed), not at new places | Next: per-cell swing readout; replicate R81b at 2005 and with latents refit; replicate at the 2005 split and with the other observer set; regional necessity; a leak-free scoring of the 1994-2020 rows |
| A13 | Allee SHAPE (smoothness): aggregation k in {.5, 1, 3, Poisson} at fixed gamma, with and without per-cell arrival corrections (R25). Records: k fixed at 3 since E079 (27 km; 79a/79c never written up), never swept at 54 km | RUNNING (R25, GPU) | Next if k matters: refit gamma (n50) jointly with k |
| A12 | Innovations on ARRIVALS' settlement success at the Allee knife's edge (user's preferred option 2): juvenile arrivals x exp(sigma u D), D = absolute Allee mate-finding deficit; per cell and per cell-year; K real | RUNNING (R24): per cell (200 it, uncapped rerun in R25) moved offsets -27/-34/-44/-32 -> -13/-15/-13/-16 for gap .22, prior 995, East u mean -.28; cell-year rerun --tight | User: fractions reach everywhere, so this is strong and weak at once; expect a stiff fit at the Allee knee. Follow-ups: Allee SHAPE (smoothness, aggregation k) x arrivals; absence / timing supervision (weak timing incentive, see below) |
| A8 | [audit] Exchangeable (old separate correlated fields) partial refit | BUILT, PARKED | overlay_step43_exchangeable.json; recreates old structure, not the establishment decomposition. Rank-2 was dropped historically because one field decoupled and K became an absence switch (age_priors._exchangeable_w_env docstring) |

## B. Dispersal ("non-dumb" movement) on Z_disp

Inventory (SBX "Dispersal splits"):
- 4x3 production split; 4x4 (b44) built, null result; 8 directions never built.
- Within-band directional choice from Z_disp (step 27) failed its null and water cross-leak guards.
- Old 27 km settlement tests were judged on scores only.
- Pitfall: every planted front mechanism was absorbed by fecundity / founders / survival; impose rules or pin
  compensators.

| # | Branch | Status | Notes |
|---|---|---|---|
| B1 | Habitat-informed kernel choice across bands (and within band) | DONE (R17) | The requirement stays x6-10; stronger choice shifts it to the West |
| B2 | Philopatry: departure set by origin habitat | DONE (R18) | Worse: raises the requirement (x16-22). The model wants MORE eastern dispersal |
| B3 | Settlement set by DESTINATION habitat on [cur] (forward's own ideal-free settlement, 3 x 3 blocks, dispersers only; r0 and capacity weights; imposed) | DONE (B3): full-strength sorting wrecks the native West (West -log-lik x4-6) and raises the eastern requirement (x15-16) | Rejected at full strength. OPEN: weak (settle_tau) or East-only version. Codex's step 43 / 46 placement fits (full refits) are complementary |
| B4 | 8 directions / sectors | OPEN, low | b44 null; N_DIRECTIONS = 4 hard-coded; about 4.4 GB Z_disp |
| B5 | Path segments (z_disp_seg4) for choice | OPEN | Step 30 cross-leak pending |
| B6 | Step 25 native departure re-read on fronts | OPEN | Did not reduce inflation; fronts never read |
| B7 | East departure increase over time (partial-migration evolution) | DONE (R19) | East x9.2 -> x6.4, West x2 -> x7-8; not the lever |
| B8 | Sedentary founders -> evolving dispersal | DONE (R20) | Early core r .126 -> .15-.16 (CBC .24); requirement unchanged |
| B9 | [audit] Departure FLOOR from near-empty cells (Veit min(N/K, 1) sends nobody from empty cells; R19 / R20 keep p(0) = 0) | DONE (B9, B9b, B9c): with frozen readouts a fitted floor (.43) halves the eastern requirement; jointly with the field the floor collapses to .07, with no gain and unchanged inflation | A substitute for the frozen field, not an improvement. Closed for this structure |
| B10 | [audit] Plains barrier through LANDSCAPE, not time: Z_disp-dependent establishment of arrivals, Keitt Allee pinning in patchy habitat, sink edges limiting supply | NEXT, PROMISED-NOT-DONE | Targets: T5 Plains longitude map; fronts met near 95W about 1995 (Hochachka & Dhondt 2000) |

Branch B summary: R17-R20 tested forms do not remove the requirement. Branch NOT closed (B3, B9, B10).

## C. Carrying capacity and density regulation

| # | Branch | Status | Notes |
|---|---|---|---|
| C1 | Factor decomposition on [cur] | DONE | See 0.6. Caveat: [cur] initialises in 1951, so its "1941-55" era really covers 1951-55 |
| C2 | Survival-side density dependence: low-density survival anchored (survanchor_forward.py, joint refit) | DONE (C2 / C2b): after refitting the field, a FIXED anchor costs little (+51 at .45, +502 at .6) with timing unchanged; a fittable anchor is escaped (hi -> .22) | Low survival at the FRONT is not what times the invasion; the field's survival inflation (.13 -> .50) serves settled abundance / fill-in. Winter-capacity version still OPEN (A6) |
| C3 | Emergent capacity (site-limited Veit reproduction) on the 54 km absolute-Allee yearling base | OPEN | Tested only at 27 km, on levels (E053) |
| C4 | Territorial floaters | OPEN | Links to B9 |
| C5 | Learned crowding shape | see A2 | |
| C6 | [audit] Capacity rise concentrated in occupied / settled cells, with the West calibrated separately | PROMISED-NOT-DONE | Said "next" after R3; R11 is a different test (BUI -> K) |
| C7 | [audit] Codex's step 44 kprivate (small private capacity component) | PENDING (Codex) | Read when done |

## D. Covariate drivers

| # | Branch | Status | Notes |
|---|---|---|---|
| D1 | BUI / HYDE as drivers (R8-R13) | DONE | See 0.9. BUI: CONUS only; HISDAC survivorship bias grows back in time |
| D2 | Winter minimum temperature x feeder / urban proxy on winter survival (Zuckerberg 2011; Brittingham & Temple 1988) | OPEN | Survival bound about 2x in severe winters |
| D3 | BUI survivorship sensitivity before 1940 | DONE for the K route; OPEN for the Fmax route | |
| D4 | [audit] Climate warming after 1970 as a candidate | OPEN | Literature: mostly after the acceleration |
| D5 | [audit] Urban-tolerance test of the dims the finch fit leans on (urban_tolerance score in species_similarity.csv; suggested in GRID54_Z_EAST_ATTRIBUTION, never run) | OPEN | On the native Z the relevant dims must be re-identified |
| D6 | [audit] Answer to "is BUI captured in DESK, coincidence or nonlinear exception?" | DONE (answered) | BUI and the fitted change are substitutes, not the same spatial pattern (R9d); no evidence of a nonlinear BUI role (E039); encoder-level attribution (E2) is the decisive test |

## E. Encoder-level attribution (native field)

| # | Branch | Status | Notes |
|---|---|---|---|
| E1 | Split native Z's change into the observation-fitted regional field R(s, t) vs the covariate-driven A . dD | BLOCKED (TACC 1-2) | Z = M + A . (D - Dbar) + R (z_assim.py). R fitted on 1996+ sources; earlier, R is extrapolation. The readout's eastern change is mostly after 1980 (R16 by year) |
| E2 | Family swaps (BUI, HYDE, LUH classes, LUH management "clock", temperature, moisture) on the finch's fitted H | BLOCKED (TACC 1-5) | backcast_attribution.py template; report the interaction residual |
| E3 | Does an establishment readout load on R or on covariates? | after E1 / A3 | |
| E4 | [audit] Hold R's time part at its 1996 value and re-run the key arms | BLOCKED (TACC 1) | Raised as item 2 of "other areas"; never listed |

## F. Diagnostics and anomalies

| # | Item | Status |
|---|---|---|
| F1 | [cur] 1951-62 stall near the release (r .01 vs CBC about .24) | DIAGNOSED (likely: dispersal drain into Allee-dead neighbours); R20 did not fix it; B9 / B10 are the follow-ups |
| F2 | Why step 39 (yearling) stays 17-26 yr early under low F | DONE: it doesn't simply stay early. Frozen modern habitat leaves the East at replacement (lambda 1.07), so uniform F cuts make 37-70% of eastern cells never reached; offsets were computed on reached cells only. The knife edge in the real landscape. LESSON: report "never reached" with every offset |
| F3 | Modern zero share too high in every fit (East .71 predicted vs .49 observed, 2010-25) | OPEN (occupancy / observation variants exist in code) |
| F4 | Stochastic twin on the real checkpoint | PROMISED-NOT-DONE |
| F5 | Re-run [s3] / [s3n] reads on [cur] (R7, R8 / R10, R14) | PARTLY (R15 and realised r replicated) |
| F6 | [audit] Neubert & Caswell linear spread-speed screen (c*) | PROMISED-NOT-DONE (original plan A2) |
| F7 | [audit] Frozen-habitat fronts of the step 26 short-kernel fits; kernel distance x establishment cross test | DONE for the fronts (release-only frozen -25 to -32 yr for MDD 200 and 110: kernel length is not the lever); the cross test is still open |
| F8 | [audit] Abundance / fill-in against target T4 | PROMISED-NOT-DONE |
| F9 | [audit] Rerun invalid arms: R14c / R14d (BUI spline low-density) and the native single-bt BUI split | DONE on [cur]: the BUI spline low-density boost fails (gap .22, fronts 60-80 yr early); the BUI split on Fmax closes 90% (bt 5.4). R14d (West-ID spline) not rerun |
| F10 | [audit] Pushed vs pulled front test (growth-rate decomposition) | OPEN (front fullness gave "likely") |

## G. Literature and considerations

| # | Item | Status |
|---|---|---|
| G1 | Low-density growth factors synthesis | DONE |
| G2 | Verify: Badyaev "new populations start as sinks"; Hosseini 2006; Weiss 1997 numbers; Bock & Lepthien 1976 party-hour series; US feeding prevalence before 1980 | OPEN (user downloads) |
| G3 | Acquisition list (Wikle 2003, Mundinger & Hope 1982, Belthoff & Gauthreaux 1991, ...) | OPEN |
| G4 | [audit] CONSIDERATIONS G14 / G15: NOT WRITTEN, yet SBX cites "G15" seven times (read those as the PROPOSED item). Proposed G14: invasion shape must come from population processes or documented change; report alpha* (share of fitted habitat change still needed). Proposed G15: the western front also stalled then accelerated in the 1960s-80s (Tyler 1992; Podrebarac & Finck 1991), so some calendar-time change may be real | PROPOSED (the user's document; propose, do not write unasked) |
| G5 | [audit] Skeptic pass on the leading mechanism before any Phase C promotion (original plan) | PROMISED-NOT-DONE |
| G6 | [audit] Literature-prototype ideas not yet tried: Keitt et al. 2001 Allee pinning (in B10); Kirkpatrick-Barton adaptation clock (R6b tested a local clock; not the trait-field form); Shigesada stratified diffusion; Clark, Lewis & Horvath establishment by population size | OPEN |

## H. TACC transfer list (user action; run `du -sh` first)

Run root: `/work/07980/bwa386/ls6/houfin/research/field54/runs/field54_prod`.
1. `.../assim/production/field_d_prod_ema.pt` (+ .log): A, u, R. A few MB. **E1, E4.**
2. `.../cubes/production/desk_interp_s1_ema/` (+ cube_meta.json, EMA_DONE.json). About 480 MB. **E1, E2.**
3. `/work/07980/bwa386/ls6/houfin/processed/field54/states/` (yearly_states 1902-2025 + state_schema.json, v1).
   About 0.5-1.2 GB. **E2.**
4. `.../models/production/interp_s1/` (env_model_semisup.pth, desk_meta.npz). **E2.**
5. `.../cubes/production/desk_interp_s1/` (raw cube; reproduction check). About 480 MB. **E2.**
6. `.../cubes/production/field_d_prod_ema_s1/` (128-d field incl. the WINTER block). About 480 MB. **A6, E1
   cross-check.**

## I. Promotion path (Phase C, after the sandbox)

Candidates: none established yet. A3 only if the homotopy ends competitive.
1. Matched full fits: local float32 MAP, then TACC float64 refine, two seeds, matched controls.
2. Scored on the pathology checklist + lambda change + frozen fronts + realised-r anchors + region-tiered abundance.
3. New structure off by default, with tests that the off setting reproduces the current model; skeptic pass (G5)
   first.

## J. Corrections log (fact, then lesson)

1. Veit & Lewis 1996 called "the only fitted model". It is a parameterised simulation compared with CBC, and many
   fits exist. LESSON: do the literature work before characterising the field.
2. Unit conversion: model units -> "birds" is a project convention (E063: 400 m count radius, ASSUMED detection .5,
   unvalidated extrapolation). "Front carried by fractions of a bird" withdrawn. LESSON: claims that depend on the
   conversion must say so; prefer unit-free readouts.
3. BBS vs CBC: model "birds" were compared with raw winter CBC counts ("180 vs 4,079"); withdrawn. The early-East
   "deep sink" claim was reversed in BBS units. Still conversion-dependent in SBX: R0's "223 km vs 267 observed" CBC
   radius, R1 / R3 CBC-radius comparisons, R5's "fitted model behind early" side finding, and R5b's 1970s trough
   (an artefact of the CBC term). LESSON: compare only in observation units; CBC only via a CBC observation model.
4. BUI is NOT static (whole-grid medians were dominated by fill values and unbuilt cells). LESSON: check full arrays
   within coverage before calling a covariate static.
5. Efficiency: runs were left to converge for hours on CPU; the GPU went unused. LESSON: short trajectory screens on
   the GPU (free; Codex has no GPU workers); converge only the winners.
6. Monitoring / autonomy: completions were missed; the assistant idled and asked the user to choose diagnostics
   (four user prompts). LESSON: see working rules; arm a monitor for every job.
7. Averaged vs native Z: early rounds used the averaged 27 km Z ([s3]). LESSON: production-facing reads on native
   field54 Z.
8. "What constrains the early East?" was already answered (step 5). LESSON: read the records first.
9. Allee form / base configuration: R13 and the [s3n] growth-gap numbers are relative-Allee only; [cur] was
   misdescribed as "Veit reproduction" with an "exemption ending". LESSON: verify the overlay before running.
10. Middle ground: proposed free functions of time / density instead of Z-built readouts, repeatedly. LESSON: start
    from Z / Z_disp.
11. Single shared field presented as a finding; it is known history (ladder step 2). LESSON: do not present
    history as discovery.
12. A3 mis-worded ("beside Sa / Sj / Fmax / K"); the plan was then not followed (an exchangeable refit was built).
    LESSON: re-read the row and the user's words before building.
13. "10x potential -> 2x realised = dissipation" was partly arithmetic (r = log lambda).
14. "Options and history are in the plan" was stated before they were written. LESSON: write first, then claim.
15. Withdrawn claims: the "static 78%" establishment headline (unstable controls); the "eastern-specific x9 vs x2"
    clue (split not identified); R6's fixed-point design (oscillating; replaced by R6b).
16. Optimizer stalls / invalid arms (not results): R8 first static attempt; the native BUI split single-bt arm
    (N_step3_native; mislabelled as "R13" before); R14c BUI spline; R16 first lowdens-time attempt; R20 .3 -> 1 first
    attempt (non-finite gradient; fixed by clamping 1 - p).
17. The joint-refit parameter file was overwritten by the control (naming bug; summaries intact; fixed).
18. Offset-only "timing matches" hid never-reached cells: R2 (F x.4: never .25), R3 (never .24-.34), R1b (up to .48)
22. Round-2 review (2026-11-04): the Z_disp freeze was not removed from the decoupled family despite J21 (so every "inflation is worth ~X" is an upper bound); R40's G was called "establishment" though it is a density bonus regulating abundance where c = 0 (J19 again, via gate width scaling with n50); realised-growth anchors compared with the wrong region; unconverged runs called plateaus/verdicts. Lessons: for EVERY edit, print the list of inputs it touches into the summary; before naming a readout, compute where it acts (share of cell-years inside its gate, c = 0 share); compare with anchors on their own spatial support.
23. Dead sites (2026-11-04): on the veit-departure base [cur], dispersal_logit_intercept / slope never reach the likelihood (departure p = N/K, no parameters); R80/R81d/R85 "dispersal refits" refitted nothing (values exactly at prior means). Lesson: before refitting a site, check its gradient is non-zero at the start (print |grad| per target in the start record) and that the base config actually reads it.
24. Wrong support for the banding comparison (2026-11-04): R81-R83/BBL-ERA compared BBL adult survival (banded, dense core cells) with the OCCUPIED-cell mean Sa (~.2-.35, marginal cells); at the banded cells [cur] Sa is .72-.75, ABOVE banding. Lesson (J22 again): every external anchor is compared on its own cells/years, weighted as its data are -- compute the readout at the anchor's support before naming a discrepancy.
21. Review of R22-R29 (2026-11-04): R27's alpha also scaled Z_disp (mislabelled); a 2-parameter sandbox residual was called a "needed" growth-rate time change; A9/R28b's G is N/K-gated (J19 recurring); T0's West got WORSE (native range) while I reported fronts; the fit's never-reached baseline was cited inconsistently (.047 vs .068); unconverged runs were called optima; BBS-only objective presented without the NestWatch caveat. Lessons: list which inputs an edit touches; score every solution on the full posterior; report regional splits before calling something "substitutable"; one never-reached baseline.
20. GPU over-subscription (2026-11-03 22:2x): concurrent SBX_MEM_FRACTIONs summed > 1 (.45 + .3 + .3 + a .2 smoke) on a card that also hosts the Windows desktop; six arms died mid-run (R25 k3_u at eval 997). Lesson: at most 2 GPU jobs, fractions summing <= .7; smoke tests on spare CPU cores.
19. R22/R23 (and Codex's process_noise low_density target) defined "front / low density" by N/K, carried over from the relative-Allee era; [cur] has ABSOLUTE Allee. Lesson: gate on the base's own Allee form at absolute density (R24 does).
    vs .04 fitted. R2 / R3 headlines are overstated. LESSON: report "never reached" beside every offset (now stored
    in all summaries).

### K-timing. Timing terms that count per cell, not per bird (user question 2026-11-03)
Why: the NB zero costs ~mu, so fractional-bird fronts decades early are nearly free (R22: within ~1,800 of the fit with fronts
33-55 yr early). step40 rangev2 added PRESENCE constraints only (unobserved cells unconstrained); East lambda rise .94 vs .98.
Forms: (1) hurdle presence with a log-density link (replaces the NB zero part); (2) interval-censored, effort-aware
first-detection hazard per cell (BBS 1966+, CBC pre-1966, literature state dates); (3) absence-before-arrival (the
missing half of step40; needs (1) or a threshold to bite); (4) summary-curve synthetic likelihood (T2/T3/T5, contact
year); (5) re-weighting (equal-cell or tempered zeros; least principled); (6) establishment-probability observation
layer P = 1 - exp(-b N) (E063 as observation, not dynamics; estimates the bird conversion b).
Sandbox test RUNNING as R26 (forms 1+2 combined: log-link first detection per cell; arms T0/T0w no corrections, T1/T1w per-cell establishment; Allee n50 free); (6) still to do; does timing
then come from arrivals, and does the needed habitat-change share / lambda rise fall? Diagnostic only.

## K. Sandbox infrastructure and pitfalls

- **Code:** `research/exp/invasion_sandbox/`.
  - `harness.py` (replay, readouts, observation-unit `observe`, habitat_keys).
  - `r7_shared_calendar.py` (frozen-habitat inverse arms).
  - `zreadout.py` (Z readouts, kernel choice, philopatry, departure ramps).
  - `refit_fields.py` (partial refit of habitat sites on the full posterior; --estab, --field_alpha homotopy).
  - Forks, each with asserted anchors and identity checks: `clock_forward.py`, `allee_forward.py`,
    `lowdens_forward.py`, `zcrowd_forward.py`, `dep_forward.py`.
  - Diagnostics: `realized_r.py`, `growth_gap.py`, `growth_gap_factors.py` (power-aware), `lambda_decomp.py`,
    `r9_describe.py`, `readout_overlap.py`, `release_area.py`.
- **Runs:**
  - GPU via `sbx_env.sh` (SBX_DEVICE=gpu, SBX_PRECISION, SBX_OVERLAY).
  - Sequential GPU queues on E: (one GPU job at a time; concurrent jobs thrash memory). CPU arms in parallel.
  - float32 about 33x faster than CPU, fine for screening. float64 for anything starting at an optimum.
  - About 40 iterations capture most of the gain. About 1,000-unit optimizer noise between capped runs.
- **Pitfalls hit:**
  - `pkill -f` self-kill (use `sbx_stop.sh`).
  - Heredoc / escape mangling (write edit scripts with the Write tool; MSYS_NO_PATHCONV=1).
  - Git Bash has no /mnt/e.
  - Objective rows must match the readout's 25-49N band.
  - Scale objectives (/1e4) and coefficients (b = .1 u).
  - Clamp 1 - p >= 1e-9 in powered departure.
  - Clear stale per-step tracers after scans (zcrowd fork).
  - `grep -c` exit 1 when there are 0 segfaults is not a failure.
- **Host:** BIOS 4601 / microcode 0x133 / RAM 4000 MT/s since 2026-11-03 09:18. Stress test clean; canaries
  bit-identical all day.

## L. Path to a non-pathological L-BFGS mode (user request 2026-11-04 evening): structure, priors, loss
Principle: no fixed values in production; make the plausible answer the MODE by (i) giving the data that can pin each rate a
likelihood, (ii) priors that price the substitution ridges, (iii) structure that does not force K and the vital rates to
co-move. Sandbox anchors / pins / fixed weights were DIAGNOSTICS only.

Pathologies located (sandbox, [cur]):
- P1 early eastern adult survival ~.15 (BBL .56) carries the deep early sink; raising it costs ~160 training and nothing
  in the invasion era (R81b).
- P2 juvenile survival carries ~half the potential-lambda rise but little forecast value (R69, R72, R74).
- P3 one shared habitat field: K carries the abundance trend; the vital rates' time change rides along (R53, R60, R82).
- P4 modern level weakly identified; clean tests conflict (R79); evidence partly observer-conditioned (R59, R66).
- P5 substitution ridges: Allee strength <-> habitat change (R26-R30); low-density brake <-> swing (R73 / R77);
  weights <-> time change (round-7 review).

(a) STRUCTURE
- A1 Separate the K field from the demographic field: K reads Z with its own weights; Sa / Sj / Fmax read a demographic factor
  with their own coupling (manifold_prior rank2 grouping K vs vital rates, coupling estimated). Sandbox: decoupled
  structures are within ~400 of the fit (A9 / R42).
- A2 (option) Vital rates read static Z plus an explicit per-region linear time term with its own prior (R61: the forecast
  information is "each region's trend continues"), so priors act on the size of the time change directly.
- A3 Allee: keep absolute; replace the hard n50 pin with a soft literature prior scaled to [cur]'s own K* (R44a: pin is
  V&L-consistent in the East). No N/K-gated low-density readouts (J19).
- A4 Disease already acts on survival; keep its time path separable from habitat-driven Sa (identified post-1993 only).

(b) PRIORS
- B1 Price the REALISED time change of the vital rates (per region, per decade, on logit Sa / logit Sj / log Fmax), not the
  weights -- weight priors are compensated (round-7 review). Width chosen so a ~.75 swing in potential lambda is cheap.
- B2 Juvenile / adult survival ratio: tighten toward the literature and give its TIME change its own (narrower) prior, so Sj
  cannot be the convenient lever (P2).
- B3 Observer-route effects: check the sigma prior; the co-adaptation with time-varying habitat (R59) argues for a modest
  prior or era-blocked structure (diagnostic first).

(c) LOSS / OBSERVATION
- C1 (highest) BBL dead-recovery likelihood for adult survival by region x era (SURVIVAL_BBL_SCOPING_2026-10-31.md: ~2,600
  recoveries; crude S .51-.56). Replaces any Sa anchor with data; addresses P1 directly.
- C2 BBL natal-dispersal distances (n 316) on the kernel (dispersal trades with growth, R80).
- C3 Timing / presence terms (K-timing forms): R26's log-link first-detection term was inert WHILE a structural lever exists;
  once C1 removes the Sa lever, re-test (i) log-link first detection, (ii) CBC early presence + absence-before-arrival,
  (iii) hurdle zeros -- opt-in, judged on clean rows.
- C4 Validation protocol as the selection criterion: score the observation_mask rows (2021-25 + 31 blocks), time-split refits
  with common / refitted observers, never-reached share, occupied-cell swing, plausibility panel (BBL Sa, NestWatch F).

SEQUENCE
1. Sandbox (now): implement C1 as a likelihood term in the partial refit (replaces --sa_anchor) + B2 ratio prior + B1 realised-
   change prior; score on clean rows and the plausibility panel. Replicate at two splits.
2. Local full L-BFGS (all latents, GPU float32 -> float64 refine) from [cur] with C1 + A1 (+ B1/B2): check the mode
   (Sa plausible, swing ~.75 not ~1.1, clean held-out within ~500 of [cur], fronts / never-reached). Multi-start (R30 / R36
   showed basins).
3. Then HMC (TACC; user transfers) once an L-BFGS mode is plausible and stable across starts.

### L addendum (user, 2026-11-04 evening)
- L-RIDGE: multi-modality and ridges are model-revision diagnostics, not just optimiser nuisances. Broad, bounded regions are
  acceptable; ENDLESS ridges (unbounded by data or prior) or genuinely different modes signal model revision. Tests per
  candidate structure: profile likelihoods along the known ridges (n50 / Allee strength, early Sa level, swing size a, disease
  magnitude, low-density G), all else refit, reporting whether each profile is bounded and by what (data term vs prior); and
  multi-start spread of the L-BFGS mode (>= 3 dispersed starts). Known so far: n50 is a flat ridge when growth can change
  (R28a / R43: worth ~3) -- bounded only by the pin; Allee strength <-> habitat change and G <-> swing are long ridges.
- R84b: the fit's post-1990 rise in eastern POTENTIAL lambda (.94 -> 1.56) is entirely cancelled by disease (EFFECTIVE stays
  ~.9-.96): the modern "paradise" is a rates <-> disease ridge, bounded only through disease priors. Effective-lambda swing:
  .31 (1940) / .48 (1966) -> ~.94 (1990), then flat.
- L-DISEASE (missing so far): disease is a ~30% eastern survival toll from 2001 (R84) that offsets the modern rise in potential
  lambda. Plan: (i) report EFFECTIVE (disease-inclusive) lambda in every swing readout; (ii) disease x swing ridge -- refit
  disease alone (vs dispersal alone) at a in {.5, .75, 1} (R80 refit both together); (iii) bring post-1994 BBL recoveries in
  with disease-reduced survival (needs the realised m from the forward); (iv) consider literature on Mycoplasma survival
  effects as a prior on the disease magnitude.
- A5 (user, 2026-11-04 evening) DENSITY REGULATION. Verified: [cur] disease acts on SURVIVAL (overlay target = survival;
  the code default and earlier lineages used target = capacity). But crowding acts ONLY on fecundity (F_eff = Fmax / (1 +
  c (N/K)^2)), with c solved from the DISEASE-FREE rates so the equilibrium equals K: K is a parameter, not emergent; every
  density effect (food, feeders, disease transmission) must pass through fecundity or K; disease mortality pulls abundance
  below K with no compensation. This produces the rates <-> disease <-> K ridge (R84b) and decouples low-density growth from
  abundance. Options: S1 emergent equilibrium -- crowding on survival or split survival / fecundity with an estimated share,
  so disease deaths partly substitute for crowding deaths (compensation); S2 density-dependent juvenile survival
  (recruitment) alongside fecundity crowding (smaller change). Test S2 first as a sandbox fork: does it bound the ridge
  (smaller fitted disease toll and smaller modern potential-lambda rise at equal clean-row fit)?
- L-DISEASE (R87 addendum): the disease FORM is intact (documented arrival map + lag fields; severity modifier with late /
  native terms and a 24-coefficient spatial field; gate; host resistance rec / tau_rec; hill on current density). Arrival
  timing follows the map (~1-1.5 yr) -- not a lever. Two likely levers: (a) resistance collapsed to ~4% recovered vs a prior
  median 38% -> a permanent ~30% eastern survival tax that cancels the modern potential-lambda rise; (b) the spatial severity
  field (regional absorber). Tests R88 (resistance at prior median; severity field off). Production candidates: a literature
  prior on resistance / recovery (host resistance and tolerance evolution in eastern House Finches and population
  stabilisation after the 1994-98 decline -- to be checked in the vault / Zotero, not from memory), post-1994 BBL recoveries
  with disease-reduced survival as a likelihood, and a smaller severity-field budget.

### L addendum (2026-11-04, after review 8; record SANDBOX_INVASION_SHAPE "REVIEW 8")
- L-RIDGE: parameter-fixing profiles are OBSOLETE (they escape: rec -> tau_rec R88, mu_sev -> b_late R89). Profile on
  OUTPUT caps (disease mortality ceiling R90; realised toll; realised Sa) and converge each point.
- L-DISEASE: form is done (resistance not identified, severity field ~270, onset ramp ~300). Re-centred on disease's
  DENSITY DEPENDENCE (hill): disease is the model's only post-1994 survival density dependence, so the rates <-> disease
  ridge is a regulation (A5) problem. R94 tests density-independent disease.
- "Effective lambda" readouts are fit-density, no-crowding numbers; report LOW-DENSITY lambda (= potential) and REALISED
  growth instead (r84_disease corrected).
- A5: S2 (juvenile DD) done, no help (R86); S1 (adult-survival DD) untested.
- A1 (separate K / rate weights): untested -- the decoupled a = 1 arms share weights.
- C1: BBL likelihood alone cannot hold Sa (R83, R92); demoted behind the observation-model fixes; ADD a banding-informed
  prior on Sa's TIME CHANGE (banding: Sa ~flat .575 -> .525; fit 2.4x rise). R95 tests time-static Sa.
- C2 (dispersal trade): VOID on [cur] (veit departure, no parameters).
- PROMOTED to the top: NB variance form (kappa too tight at high counts, constant phi too loose at low: R93) and route-effect
  structure / held-out level-vs-pattern scoring (NB-CAL) -- they set every cost scale.
- Sequence step 2 (local full L-BFGS) now requires: NB form fix, route-effect structure, a TRANSIENT handle on the toll
  (post-arrival decline), converged multistart from >= 3 starts ([cur], R81b end, R90 c = .25 end); distinct end points
  > ~1,000 apart with different tolls = distinct modes -> model revision (user's criterion).
- Start-noise calibration (~750 between near-identical twins) required before any "within noise / affordable" claim.
