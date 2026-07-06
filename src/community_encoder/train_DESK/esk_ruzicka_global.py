"""Compatibility wrapper for the modular ESK implementation."""
import sys
from pathlib import Path

try:
    from .esk_kernel import run_esk_experiment
except ImportError:  # pragma: no cover - allows direct script execution
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from community_encoder.train_DESK.esk_kernel import run_esk_experiment


if __name__ == "__main__":
    run_esk_experiment()
