# STATE -- generated 2026-10-08T21:59:45-05:00 by research/tick.py; do not edit (edit intent.md)
phase 0 | eval pre-v1 | research/desk-temporal @ 68ce0d8

## Jobs

## Recent events
- 10-08T20:25 done     E000/canary-999da1905e3e 
- 10-08T20:42 failed   E000/pytest-baseline-d1aebbdd90de 
- 10-08T21:26 failed   E000/pytest-suite-51727b29ecda 
- 10-08T21:41 failed   E000/pytest-suite-455bb115ec4f 
- 10-08T21:49 failed   E000/pytest-suite-a3fabd63d2f1 
- 10-08T21:59 failed   E000/pytest-suite-3319b009ee87 

## Hypotheses
open 32
- open, priority 1: E1, T0, O3, C1, E4, P1, R1, B1

## Findings
0 verified findings

## Intent (research/intent.md)
Phase 0 (local port). Built and tested: env, queue, runner, tick, protocol, register, audited GP suite
(A1-A8 + multi-start/bounds), DESK resume + A9, cache, ESK projection, fast arm, atlas, planted generator.
WAITING ON USER: /work symlink (sudo) + TACC transfer (plan Phase 0.2).
When data lands (DUE: immediately): sha256 -c, cold copy to /mnt/e/Datasets/houfin, then enqueue in order:
  E001 repro (base, tempho1995) | E002 caches (4) -> E003 esk -> E004 atlas | E005 fast arm | E007 planted
  E006 corrected suite (after E001) | E008 route ceiling (O3). Gate 0 report when E001 + E002 finish.
Live questions: does the local run reproduce TACC? where does temporal signal leave (E004)? is the
downstream readout itself a lever (E005)?
Phase 0.4 DESK rerun: reproduce sweep_t0_f100_base from its generated overlay (scripts/sweep/generate_overlays.py make_overlay; look in the transferred sweeps/desk_hp/ for the overlay/manifest), output to a NEW dir, 2 seeds; must land within the 6.6% seed floor of the TACC run.
