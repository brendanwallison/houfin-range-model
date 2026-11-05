# Local source-control handoff, 2026-11-05

The local backlog is preserved on the existing `research/local-takeover-foundation`
branch. No additional branches were created, no branch was merged into main, and
no experiment was started, stopped, or reconfigured for this cleanup.

Preservation commits, in dependency order:

- `bf105f7`: model/input/encoder changes, overlays, visualization scripts, and tests (74 paths).
- `3042633`: experiment implementations, local/HPC dispatchers, age handoff, and R scripts (402 paths).
- `58ce553`: historical notes, compact evidence, figures, registry, and read provenance (368 paths).

These preserve work already present locally. They do not constitute a scientific
review, successful test-suite run, or promotion of any candidate model. Historical
status files and agent summaries retain their original claims and timestamps;
they should be checked against actual artifacts before being relied upon.

The five existing focused branches remain separate and unchanged:

- `research/local-baseline-reconstruction` (`1127a8c`): reconstructed baseline and input audit.
- `research/habitat-field-geometry` (`8a9a1af`): structural model experiments.
- `research/structural-readouts` (`642404c`): exact-stage diagnostic readouts.
- `research/local-loop-control` (`b65fa65`): detached execution and numerical controls.
- `research/likelihood-profiles` (`9bc5f7f`): conditional profile implementation and registered queue.

Generated arrays, checkpoints, downloaded R libraries, local worktrees, and
`.research_runtime/` remain ignored. Added exclusions cover scratch code, raw
logs, completion flags, and ten bulky generated JSON payloads (180,172,299 bytes).
Their original paths, sizes, and SHA256 values are recorded in
`LOCAL_GENERATED_ARTIFACT_MANIFEST_20261105.json`. Those payloads remain on this
machine; the manifest is not an off-machine backup. Compact companion reports
and the generating analysis scripts are retained in Git.

An existing process rewrote `OVERFLOW_EARLY_LIVE_20261102.md` during this cleanup.
It is excluded as live output; its earlier snapshot remains in the preservation
commit. No file was moved or deleted from disk.

Verification: 341 Python files parsed with `ast.parse`; 171 JSON files parsed
successfully. This check imported no model and used no GPU. A credential-pattern
scan found no private-key blocks or standard AWS/GitHub token patterns in the
source directories. The initial inventory and verification output are retained
locally under `.research_runtime/takeover/source_control_*20261105.json`.

The user ended Codex's experiment work and transferred responsibility to Claude.
`LOCAL_TAKEOVER_NEXT_20261105.json` is an older checkpoint, not current ownership
or process status. Existing detached runs were left alone; their live artifacts
are the source for current execution status.
