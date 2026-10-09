Phase 0 (local port). Built and tested: env, queue, runner, tick, protocol, register, audited GP suite
(A1-A8 + multi-start/bounds), DESK resume + A9, cache, ESK projection, fast arm, atlas, planted generator.
WAITING ON USER: /work symlink (sudo) + TACC transfer (plan Phase 0.2).
When data lands (DUE: immediately): sha256 -c, cold copy to /mnt/e/Datasets/houfin, then enqueue in order:
  E001 repro (base, tempho1995) | E002 caches (4) -> E003 esk -> E004 atlas | E005 fast arm | E007 planted
  E006 corrected suite (after E001). Gate 0 report when E001 + E002 finish.
Live questions: does the local run reproduce TACC? where does temporal signal leave (E004)? is the
downstream readout itself a lever (E005)?
