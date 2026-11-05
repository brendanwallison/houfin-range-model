"""Read-only source/provenance audit; writes only to the explicit output directory.

No model imports, checkpoint unpickling, data mutation, job submission or process
termination. Source snapshots and current code are compared byte-for-byte; AST
indexes help audit dependencies without trusting stale comments.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_index(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    definitions, imports = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append({"name": node.name, "line": node.lineno,
                                "end_line": node.end_lineno,
                                "kind": type(node).__name__})
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append("." * node.level + (node.module or ""))
    return {"definitions": sorted(definitions, key=lambda item: item["line"]),
            "imports": sorted(set(imports))}


def compare_sources(frozen: Path, current: Path) -> list[dict]:
    result = []
    for path in sorted((frozen / "src").rglob("*.py")):
        rel = path.relative_to(frozen)
        other = current / rel
        digest = sha256(path)
        row = {"path": rel.as_posix(), "bytes": path.stat().st_size,
               "frozen_sha256": digest,
               "current_sha256": sha256(other) if other.is_file() else None,
               **source_index(path)}
        row["state"] = ("missing" if not other.is_file() else
                        "identical" if row["current_sha256"] == digest else
                        "newline_only" if path.read_bytes().replace(b"\r\n", b"\n") ==
                        other.read_bytes().replace(b"\r\n", b"\n") else "different")
        result.append(row)
    return result


def git_command(root: Path, args: tuple, autocrlf: str | None = None) -> list[str]:
    return ["git", *(["-c", f"core.autocrlf={autocrlf}"] if autocrlf else []),
            "-C", str(root), *args]


def git(root: Path, *args: str, autocrlf: str | None = None) -> str:
    proc = subprocess.run(git_command(root, args, autocrlf), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=True)
    return proc.stdout


def resolved_overlay(path: Path) -> dict:
    """Show executable values without inherited explanatory comments."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(val) for key, val in value.items() if not key.startswith("_")}
        if isinstance(value, list):
            return [clean(val) for val in value]
        return value
    return clean(json.loads(path.read_text(encoding="utf-8-sig")))


def artifact_inventory(root: Path) -> list[dict]:
    """Small scientific summaries only; large arrays/checkpoints stay external."""
    rows = []
    for path in sorted(root.glob("**/*summary.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            rows.append({"path": path.relative_to(root).as_posix(),
                         "sha256": sha256(path), "bytes": path.stat().st_size,
                         "converged": value.get("converged"),
                         "objective": value.get("end_neg_logp"),
                         "n_eval": value.get("n_eval"),
                         "message": value.get("message"),
                         "modified_utc": datetime.fromtimestamp(
                             path.stat().st_mtime, timezone.utc).isoformat()})
        except (ValueError, OSError) as exc:
            rows.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--git-autocrlf", choices=("true", "false", "input"),
                        help="Match the owning Git client's line-ending configuration")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = compare_sources(args.frozen, args.current)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "git_autocrlf": args.git_autocrlf,
                "head": git(args.current, "rev-parse", "HEAD", autocrlf=args.git_autocrlf).strip(),
                "branch": git(args.current, "branch", "--show-current", autocrlf=args.git_autocrlf).strip(),
                "tracked_status": git(args.current, "status", "--porcelain=v1", "-uno", autocrlf=args.git_autocrlf).splitlines(),
                "untracked_paths": git(args.current, "ls-files", "--others", "--exclude-standard", autocrlf=args.git_autocrlf).splitlines(),
                "roots": {key: str(getattr(args, key)) for key in ("current", "frozen", "sandbox")},
                "source_counts": {state: sum(row["state"] == state for row in rows)
                                  for state in ("identical", "newline_only", "different", "missing")},
                "sources": rows, "overlay_sha256": sha256(args.overlay),
                "overlay": resolved_overlay(args.overlay),
                "artifacts": artifact_inventory(args.sandbox)}
    # Preserve the exact existing tracked patch separately; never stage it automatically.
    patch = subprocess.run(git_command(args.current, ("diff", "--binary", "HEAD"), args.git_autocrlf),
                           capture_output=True, check=True).stdout
    (args.out / "pre_takeover_tracked.patch").write_bytes(patch)
    target = args.out / "inventory.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"manifest": str(target), "source_counts": manifest["source_counts"],
                      "artifacts": len(manifest["artifacts"]),
                      "untracked": len(manifest["untracked_paths"])}))


if __name__ == "__main__":
    main()
