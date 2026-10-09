# DIGEST (newest first)

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
