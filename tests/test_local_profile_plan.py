import json
from pathlib import Path
import subprocess
import sys

from research.exp.local_loop_supervisor import Supervisor


def test_actual_profile_plan_cli_has_ten_points_and_valid_dependencies(tmp_path):
    root = tmp_path / "code"
    (root / "research/bin").mkdir(parents=True)
    (root / "research/bin/local_takeover_run.sh").write_text("#!/bin/bash\n")
    artifacts = tmp_path / "artifacts"
    owner = artifacts / "conditional_profiles/free_compact_owner"
    owner.mkdir(parents=True)
    (owner / "worker.json").write_text(json.dumps({"worker":{"pid":0,"start_ticks":0}}))
    script = Path(__file__).resolve().parents[1] / "research/exp/local_profile_plan.py"
    out = artifacts / "grid"
    result = subprocess.run([sys.executable,str(script),"--root",str(root),"--artifacts",str(artifacts),
                             "--out",str(out),"--code-commit","test"],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)["tasks"] == 20
    supervisor = Supervisor(out / "plan.json",out / "control")
    try:
        assert sum(task["lane"] == "gpu" for task in supervisor.tasks) == 9
        assert len(supervisor.tasks) == 20
        assert any(task["id"] == "original_rms15_nuisance_sensitivity" for task in supervisor.tasks)
    finally:
        supervisor.close()
