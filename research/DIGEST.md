# DIGEST (newest first)

## 2026-10-09 (data landed)
- TACC artifacts transferred and verified: 880 processed files (27 GB on TACC; history_vectors excluded)
  and 93 raw files (BBS 2026 release, AVONET, masks), every one matching its TACC sha256.
- The concerning TACC numbers, for the record (old suite, A1-diluted): out-of-community DESK-vs-no-change
  change skill +0.0002 (base), -0.0007 / -0.0037 / -0.0100 (tempho 1975/1985/1995); even the 96 community
  species score +0.009 / +0.011 / +0.002 / -0.004. Those reports were written by ef7b6a2, which is what
  the reproduction runs.
- 18 jobs queued: caches, ESK projections, reproductions, the first experiments (atlas, readout, planted,
  route ceiling), the corrected suite.

## 2026-10-08 (setup, no data yet)
- Local loop infrastructure is up in WSL2: systemd-managed pueue queue (jobs survive sessions; a 15-min
  canary ran to completion across wsl.exe exits), immutable code snapshots, append-only registry, tick
  journal/LOCK/STOP, generated STATE.md. Full test suite: 938 pass; 2 fail -- one needs data (ref grid),
  one was a real optimizer fragility, now fixed.
- GP-species suite audited and fixed (A1-A8): change skill now only over species whose change is
  resolvable above split-half noise; in-sample rows get exact leave-one-out (local GPs were reproducing
  their own modern noise); noise-only back-transform (local GPs predicted spurious declines); baseline
  scales from every training row; a persistent+changing spacetime kernel as the honest temporal rival;
  persistence baseline; block-safe bootstrap; oracle no longer reads withheld decades.
- Found while testing: the per-species amplitude likelihood is multimodal -- a single-start fit can
  "collapse" a species' amplitude where a far better optimum exists. Every scale fit now multi-starts.
  This also touches DESK's own BLR fits; how much it changes past "rare species collapse" readings is an
  open question for the corrected-suite run (E006).
- DESK training can resume from checkpoints; tempho epoch selection no longer reads withheld years (A9).
- Pre-registered before any data: E004 (atlas) and E005 (downstream-matched readout + kernel variants).
