"""Recover explicitly selected model dependencies in a clean isolated worktree.

Never changes the original checkout or frozen run. No data/checkpoints are copied.
The byte-identical recovery precedes any scientific changes or diagnostic fixes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--supplement-tests", action="store_true",
                        help="Add referenced synthetic helpers/overlay dependencies to a verified recovery")
    args = parser.parse_args()
    root, target = args.workspace.resolve(), args.target.resolve()
    if not target.is_relative_to(root / ".worktrees") or not (target / ".git").is_file():
        raise ValueError("Target must be a registered worktree below workspace/.worktrees")
    if args.supplement_tests:
        manifest_path = target / "research/experiments/TAKEOVER_BASELINE_SOURCE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["files"]:
            if digest(target / record["path"]) != record["sha256"]:
                raise ValueError(f"Recovered source changed: {record['path']}")
        candidates = [args.frozen / "tests/test_full_model_smoke.py",
                      args.frozen / "tests/test_disease_depression.py",
                      args.frozen / "scripts/viz/map_diagnostics.py",
                      args.frozen / "src/vis/age_model_math.py",
                      args.frozen / "src/vis/visualize_age_model.py",
                      args.frozen / "src/data/preprocess/bbs.py",
                      args.frozen / "src/data/combine/model_inputs.py"]
        for record in manifest["files"]:
            if record["path"].startswith("tests/"):
                text = (target / record["path"]).read_text()
                candidates.extend(args.frozen / "config/overlays" / name
                                  for name in re.findall(r'map_run[\w]+\.json', text))
        seen = {record["path"] for record in manifest["files"]}
        while candidates:
            source = candidates.pop()
            rel = source.relative_to(args.frozen)
            if rel.as_posix() in seen:
                continue
            if source.suffix == ".json":
                extends = json.loads(source.read_text()).get("_extends", [])
                for parent in ([extends] if isinstance(extends, str) else extends):
                    candidates.append((source.parent / parent).resolve())
            dest = target / rel
            if dest.exists() and digest(dest) != digest(source):
                original = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{rel.as_posix()}"],
                                          capture_output=True, check=True).stdout
                if original.replace(b"\r\n", b"\n") != dest.read_bytes().replace(b"\r\n", b"\n"):
                    raise ValueError(f"Refuse to overwrite modified dependency: {rel}")
            previous = digest(dest) if dest.exists() else None
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            manifest["files"].append({"path": rel.as_posix(), "origin": "frozen_step43_test_dependency",
                                      "sha256": digest(dest), "previous_sha256": previous,
                                      "bytes": dest.stat().st_size})
            seen.add(rel.as_posix())
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"files": len(manifest["files"]), "action": "supplemented test dependencies"}))
        return
    gitdir = (target / ".git").read_text().strip().removeprefix("gitdir: ")
    # Windows-created worktrees store a Windows gitdir; Linux Git cannot resolve
    # that spelling. Supply the translated gitdir explicitly, without rewriting it.
    if len(gitdir) > 2 and gitdir[1] == ":" and str(target).startswith("/mnt/"):
        gitdir = "/mnt/" + gitdir[0].lower() + gitdir[2:].replace("\\", "/")
    status = subprocess.run(["git", "-c", "core.autocrlf=true", "--git-dir", gitdir,
                             "--work-tree", str(target), "status", "--porcelain=v1"],
                            capture_output=True, text=True, check=True)
    if status.stdout.strip():
        raise ValueError("Refuse to overwrite a dirty worktree")
    selected = [(path, args.frozen) for path in sorted((args.frozen / "src/model").glob("*.py"))]
    selected += [(args.frozen / path, args.frozen) for path in (
        "src/config_utils.py", "src/community_encoder/cube_contract.py",
        "config/age_model_config.json", "config/data_config.json",
        "config/overlays/grid54_step43_factor_p2_veit_s41.json",
        "config/overlays/grid54_data.json")]
    selected += [(path, root) for path in sorted((root / "research/exp/invasion_sandbox").glob("*.py"))]
    selected += [(root / "research/exp/invasion_sandbox/sbx_env.sh", root)]
    # Small mechanism regression tests, not upstream encoder tests or generated results.
    test_names = ("test_reproduction_forms.py", "test_disease_mortality.py", "test_allee.py",
                  "test_density_regulation.py", "test_priority_overflow.py", "test_split_readout.py",
                  "test_tied_readout.py", "test_observer_effect.py", "test_observer_holdout.py",
                  "test_juvenile_ratio.py", "test_numerics_floor.py", "test_occupancy.py")
    selected += [(args.frozen / "tests" / name, args.frozen) for name in test_names
                 if (args.frozen / "tests" / name).is_file()]
    selected += [(root / "tests/test_invasion_sandbox.py", root)]
    records = []
    for source, base in selected:
        rel = source.relative_to(base)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        previous = digest(dest) if dest.is_file() else None
        shutil.copyfile(source, dest)
        records.append({"path": rel.as_posix(), "origin": "frozen_step43" if base == args.frozen else "local_sandbox",
                        "sha256": digest(dest), "previous_sha256": previous,
                        "bytes": dest.stat().st_size})
    manifest = {"frozen_base_commit_label": (args.frozen / "BASE_COMMIT.txt").read_text().strip(),
                "scope": "model, required config and cube contract, local sandbox, mechanism tests",
                "files": records}
    path = target / "research/experiments/TAKEOVER_BASELINE_SOURCE_MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(records), "bytes": sum(row["bytes"] for row in records),
                      "target": str(target), "manifest": str(path)}))


if __name__ == "__main__":
    main()
