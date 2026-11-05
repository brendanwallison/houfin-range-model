"""Exercise the actual registered plan generator CLI without launching fits."""
import json
from pathlib import Path
import subprocess
import sys

from research.exp.local_loop_supervisor import Supervisor


def test_generated_day_graph_parses_and_has_consistent_dependencies(tmp_path):
    root = tmp_path / "code"
    (root / "research/bin").mkdir(parents=True)
    (root / "research/bin/local_takeover_run.sh").write_text("#!/bin/bash\n")
    artifacts = tmp_path / "artifacts"
    (artifacts / "campaign_03").mkdir(parents=True)
    (artifacts / "campaign_03/campaign_manifest.json").write_text(json.dumps({"cases":[["example","overlay.json"]]}))
    out = artifacts / "day"
    script = Path(__file__).resolve().parents[1] / "research/exp/local_day_plan.py"
    result = subprocess.run([sys.executable,str(script),"--root",str(root),"--artifacts",str(artifacts),
                             "--out",str(out),"--code-commit","test"],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)["tasks"] == 14
    supervisor = Supervisor(out / "plan.json",out / "control")
    try:
        ids = [task["id"] for task in supervisor.tasks]
        assert "fit_duration2" in ids and "fit_compact_best_refine" in ids
        assert not next(task for task in supervisor.tasks if task["id"] == "fit_duration2")["needs"]
        assert all(task.get("guards") for task in supervisor.tasks if not task.get("adopt"))
    finally:
        supervisor.close()
