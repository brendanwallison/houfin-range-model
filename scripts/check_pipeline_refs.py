#!/usr/bin/env python
"""Static consistency gate for pipeline wiring. No data, no cluster, no imports.

Two checks that no test covered, each of which had already let a real defect ship:

1. **Every STAGES token resolves to a stage function.** ``pipeline.sh`` validates inside
   its loop (``unknown stage: X; exit 2``), so a typo or a removed stage is only caught
   at runtime, on the cluster, partway through a submitted job. ``04_states.slurm`` and
   ``submit_preprocess_all.sh`` both shipped a nonexistent ``amplitude`` stage this way.

   What this canNOT catch: a token that resolves but is the WRONG stage.
   ``submit_encoder.sh`` defaulted to the retired eBird-only ``esk`` instead of the active
   ``spacetime-esk``; both are real functions, so no static check distinguishes them, and
   that failure is silent -- ``desk`` then trains against a stale basis. The only structural
   fix for that class is deleting the retired stage so the name cannot be typed.

2. **Nothing references a module that no longer exists.** Deleting a module is only safe
   if no launcher, config, doc, or test still names it.

    python scripts/check_pipeline_refs.py
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TACC = REPO / "scripts" / "tacc"
PIPELINE = TACC / "pipeline.sh"

# A STAGES assignment is either quoted (a list) or a single unquoted token. Matching
# `[\w -]+` unquoted swallows the rest of the line, so the two forms are separate patterns:
# the quoted one reads inside the quotes, the bare one stops at the first whitespace.
_STAGES_QUOTED = re.compile(r'STAGES(?:=|:-)\s*"([^"]*)"')
_STAGES_BARE = re.compile(r"STAGES(?:=|:-)\s*([A-Za-z0-9_-]+)")
# Stage names are lowercase identifiers with - or _; anything else in a STAGES assignment
# is prose or shell syntax (a trailing `}` from `${STAGES:-...}` is stripped before this).
_IS_STAGE_SHAPED = re.compile(r"^[a-z][a-z0-9_-]*$")


def defined_stages():
    """Stage names declared in pipeline.sh, plus the hyphen spellings callers may use.

    Returns ``(accepted_tokens, n_functions)``: pipeline.sh maps hyphens to underscores
    (``fn="stage_${s//-/_}"``), so both spellings are legal for the same stage.
    """
    text = PIPELINE.read_text()
    fns = set(re.findall(r"^stage_([a-z0-9_]+)\s*\(\)", text, re.MULTILINE))
    return fns | {f.replace("_", "-") for f in fns}, len(fns)


def stage_tokens():
    """Every (file, lineno, token) appearing in a STAGES= assignment under scripts/tacc/."""
    out = []
    for path in sorted(TACC.iterdir()):
        if path.suffix not in (".sh", ".slurm"):
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            groups = [m.group(1) for m in _STAGES_QUOTED.finditer(line)]
            if not groups:
                groups = [m.group(1) for m in _STAGES_BARE.finditer(line)]
            for group in groups:
                for tok in group.split():
                    tok = tok.rstrip("}")               # ${STAGES:-a b c} default expansion
                    if _IS_STAGE_SHAPED.match(tok):
                        out.append((path.relative_to(REPO), i, tok))
    return out


def dangling_module_refs():
    """`python -m src.x.y` / `python path/to/z.py` invocations whose target does not exist."""
    bad = []
    # Require a real dotted path (no trailing dot, so prose like "src.data..." is not a target).
    pat_mod = re.compile(r"python3?\s+-m\s+((?:src|scripts)(?:\.[A-Za-z0-9_]+)+)(?![.\w])")
    pat_file = re.compile(r"python3?\s+((?:src|scripts)/[A-Za-z0-9_/.-]+\.py)")
    for path in sorted(REPO.rglob("*")):
        if path.suffix not in (".sh", ".slurm", ".py") or ".venv" in path.parts:
            continue
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue                                    # this file's own examples
        for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            for m in pat_mod.finditer(line):
                target = REPO / (m.group(1).replace(".", "/") + ".py")
                if not target.exists() and not target.with_suffix("").is_dir():
                    bad.append((path.relative_to(REPO), i, m.group(1)))
            for m in pat_file.finditer(line):
                if not (REPO / m.group(1)).exists():
                    bad.append((path.relative_to(REPO), i, m.group(1)))
    return bad


def main():
    failures = 0
    known, n_stages = defined_stages()

    unknown = [(f, i, t) for f, i, t in stage_tokens() if t not in known]
    if unknown:
        failures += 1
        print(f"FAIL: {len(unknown)} STAGES token(s) with no stage_* function in pipeline.sh:")
        for f, i, t in unknown:
            print(f"  {f}:{i}: {t!r}")
    else:
        print(f"ok: every STAGES token resolves ({n_stages} stages defined)")

    dangling = dangling_module_refs()
    if dangling:
        failures += 1
        print(f"FAIL: {len(dangling)} invocation(s) of a module/script that does not exist:")
        for f, i, t in dangling:
            print(f"  {f}:{i}: {t}")
    else:
        print("ok: every `python -m ...` / `python ....py` target exists")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
