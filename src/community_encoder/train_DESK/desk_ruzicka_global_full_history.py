"""Compatibility wrapper for the modular DESK implementation."""
import sys
from pathlib import Path

try:
    from .desk_training import run_desk_experiment
except ImportError:  # pragma: no cover - allows direct script execution
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from community_encoder.train_DESK.desk_training import run_desk_experiment


if __name__ == "__main__":
    run_desk_experiment()
