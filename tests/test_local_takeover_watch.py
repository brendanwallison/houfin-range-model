import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "takeover_watch", Path(__file__).parents[1] / "research/exp/local_takeover_watch.py")
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


class ProcessIdentityTests(unittest.TestCase):
    def test_pid_reuse_is_not_continuation(self):
        first = {"start_ticks": 10, "state": "R"}
        self.assertEqual(watch.classify(first, {"start_ticks": 11, "state": "R"}), "pid_reused")
        self.assertEqual(watch.classify(first, {"start_ticks": 10, "state": "S"}), "live")
        self.assertEqual(watch.classify(first, {"start_ticks": 10, "state": "Z"}), "zombie")
        self.assertEqual(watch.classify(first, None), "process_missing")

    def test_stat_parser_handles_spaces_and_parentheses_in_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "42").mkdir()
            fields = ["S"] + ["0"] * 21
            fields[11], fields[12], fields[19] = "7", "9", "1234"
            (root / "42/stat").write_text("42 (a tricky (process)) " + " ".join(fields))
            self.assertEqual(watch.identity(42, root), {"state": "S", "cpu_ticks": 16, "start_ticks": 1234})


if __name__ == "__main__":
    unittest.main()
