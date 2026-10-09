"""Run a script or module inside WSL with the research job environment, from the LIVE checkout.

For interactive checks only. Anything whose result is recorded goes through ``jobs.py``, which runs a
code snapshot of a committed SHA.

    python research/lib/launch.py research/exp/foo.py --arg      # a script
    python research/lib/launch.py -m pytest -q tests/test_x.py   # a module
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

os.environ.update(paths.job_env("default", code_dir=paths.REPO))
sys.path[:0] = [paths.REPO, paths.REPO + "/src"]
os.chdir(paths.REPO)
if len(sys.argv) < 2:
    sys.exit(__doc__)
if sys.argv[1] == "-m":
    mod = sys.argv[2]
    sys.argv = [mod] + sys.argv[3:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
else:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
