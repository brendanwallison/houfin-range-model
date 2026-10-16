# Multi-source representation survey (BBS + CBC + eBird): a rapid GPU screen, then confirmation

Written 2026-10-16 at the user's request ("formulate a series of tests ... prioritize rapid GPU testing for the early
survey of the performance landscape"). Builds on E060/E061 (auxiliary heads), E069b-E082 (CBC supervision harms
out-of-community sedentary breeding change; calibration cannot remove the seasonal gap), E083 (a winter block adds at
most a little for sedentary readouts), and the design discussion of 2026-10-16 (shared trunk, separate heads,
sum-of-Ruzickas kernel). Experiment numbers are claimed per stage with the peer session (E084+).

## The question
Which way of bringing CBC (winter) and eBird (dense, modern, breeding / non-breeding) into DESK improves -- or at least
does not harm -- what the downstream needs: breeding-season change of SEDENTARY OUT-OF-COMMUNITY species (the House
Finch analogue), with spatial skill kept, and an interrogable kernel (each block a named Ruzicka similarity)?

## Primary yardstick (every arm, sedentary first)
1. Dev SEDENTARY breeding-season change, slope k and corr, all held-out + buffer cells and the top cover-change tercile
   (e077_bbs_change_k.py --similarity; E077b design). Residents, migrants reported.
2. Spatial level on the community target (fair_rivals, community) -- must not fall by > 0.01.
3. Community-species change (e077b --species-set comm).
4. For arms with CBC: held-out early CBC lists (e069b_heldout_lists.py) -- does the winter information generalize?
5. Backcast z-change size (e078_zchange.py).

## Stage 1 -- rapid screen (150-epoch schedule)
Why 150 epochs: E055b / E059b -- the short schedule removes the selected-vs-late lottery and costs nothing on
out-of-community species (+0.007 / -0.005), though 0.02-0.04 on community change. ~25 min per run (vs ~75).
CAVEAT (decisive): E081's harm appeared at the LATE (epoch-490) state and was mixed at selected states. A 150-epoch run
sits closer to a selected state, so the screen may be BLIND to that failure mode. Arm S1 calibrates this.
Arms (T0 1996, seeds 1 and 2; each vs its own 150-epoch control):
- S0  controls: 150-epoch seed 1 (exists: desk_tempho_1995_s1_ep150), seeds 2 and 3 (seed noise reference)
- S1  CALIBRATION: CBC Ruzicka supervising z directly (E081-B form) -- the screen must reproduce E081's sedentary harm;
      if it does not, the screen cannot detect this failure and Stage 2 must use full-length runs
- S2  CBC METRIC HEAD: a linear head (64 -> 24) on z, its own contract dot(head_i, head_j) ~ Ruzicka_CBC(i, j)
      (no calibration needed -- winter has its own space); CBC 1966-2024 incl. withheld years
- S3  eBIRD METRIC HEAD: same, eBird Status & Trends breeding-season communities (reference species; static modern
      map) -- dense spatial coverage
- S4  CBC + eBIRD heads (three signals)
- S5  CBC PRESENCE via a LEARNED head (the E069b loss on a trainable linear map from z) -- does freeing the readout
      remove the harm E069b had with a fixed one?
Screen rule (pre-registered per stage): PROMISING = sedentary dev change dk > -0.05 (all cells and top tercile, both
seeds) AND community space not worse by > 0.01 AND (CBC arms) held-out list gain > 0. HARMFUL = sedentary dk < -0.12
both seeds. Ranked by sedentary dk.
Cost: 10 arm runs + 2 controls, ~5 h GPU; scoring ~2 h CPU (parallel).

## Stage 2 -- confirmation of promising arms (full schedule)
500 epochs, seeds 1-2 (+3 if needed), selected AND late states, the standard yardstick on GPU (fair_rivals --target dev
with species strata, sedentary primary), both T0s (1986, 1996). Plus:
- PRIVATE CAPACITY: if heads on z are neutral but weak, give each head its own capacity (extra latent dims read only by
  the head, or a head on the trunk's hidden layer) -- the shared/private version, trained wherever each source exists.
- KERNEL FORM for the downstream: the sum-of-Ruzickas readout (block weights by marginal likelihood) vs breeding-only
  on the House Finch-analogue readout (fast_arm-style, per-block priors).

## Stage 3 -- production candidates (the user's decision)
Only if Stage 2 shows a gain for sedentary out-of-community species without spatial loss: the representation, its
kernel definition and the age-model readout change.

## Not in this survey (separate threads)
Resident-only community (peer E067 / E071); LUH3 irrigation replacement (M31, E080); early-CBC detection indices (E070).
