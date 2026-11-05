import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "takeover_audit", Path(__file__).parents[1] / "research/exp/local_takeover_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class SourceAuditTests(unittest.TestCase):
    def test_missing_changed_and_identical_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen, current = root / "frozen", root / "current"
            (frozen / "src").mkdir(parents=True)
            (current / "src").mkdir(parents=True)
            for name in ("same", "changed", "missing"):
                (frozen / f"src/{name}.py").write_text("import math\ndef f():\n    return 1\n")
            (current / "src/same.py").write_bytes((frozen / "src/same.py").read_bytes())
            (current / "src/changed.py").write_text("def f():\n    return 2\n")
            rows = {row["path"]: row for row in audit.compare_sources(frozen, current)}
            self.assertEqual(rows["src/same.py"]["state"], "identical")
            self.assertEqual(rows["src/changed.py"]["state"], "different")
            self.assertEqual(rows["src/missing.py"]["state"], "missing")
            self.assertEqual(rows["src/same.py"]["imports"], ["math"])

    def test_newline_only_change_does_not_look_like_model_evolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("frozen", "current"):
                (root / name / "src").mkdir(parents=True)
            (root / "frozen/src/a.py").write_bytes(b"x = 1\n")
            (root / "current/src/a.py").write_bytes(b"x = 1\r\n")
            self.assertEqual(audit.compare_sources(root / "frozen", root / "current")[0]["state"],
                             "newline_only")

    def test_overlay_keeps_switches_but_not_stale_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.json"
            path.write_text(json.dumps({"_comment": "old", "population_model": {
                "_disease_comment": "capacity", "disease": {"target": "survival"}}}))
            self.assertEqual(audit.resolved_overlay(path), {
                "population_model": {"disease": {"target": "survival"}}})

    def test_incomplete_summary_is_recorded_as_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "partial_summary.json").write_text('{"unfinished":')
            self.assertIn("error", audit.artifact_inventory(root)[0])


if __name__ == "__main__":
    unittest.main()
