#!/usr/bin/env python
"""Single entrypoint for the community-encoder stages (one subcommand per process).

Centralizes the encoder's dual import-root quirk: the ``train_DESK`` modules use
``src.``-style imports while ``build_final_z_cube`` uses the ``community_encoder``
top-level root (+ a ``src.config_utils`` loader). Putting both the repo root and
``src/`` on ``sys.path`` here lets every stage resolve regardless of style. Each
stage is meant to run as its own process (the TACC pipeline calls this once per
stage), so the two roots never collide within one interpreter.

    python scripts/run_encoder.py {ebird-cache|amplitude|esk|desk|cube|validate}
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ebird-cache":
        from src.community_encoder.train_DESK.ebird_cache import build_ebird_cache
        build_ebird_cache()
    elif cmd == "amplitude":
        from src.community_encoder.train_DESK.spacetime_community import build_amplitude_points
        build_amplitude_points()
    elif cmd == "esk":
        from src.community_encoder.train_DESK.esk_kernel import run_esk_experiment
        run_esk_experiment()
    elif cmd == "desk":
        from src.community_encoder.train_DESK.desk_training import run_desk_experiment
        run_desk_experiment()
    elif cmd == "cube":
        from community_encoder.build_final_z_cube import build_spacetime_cube
        build_spacetime_cube()
    elif cmd == "validate":
        from src.community_encoder.train_DESK.validate_spacetime import run_validate
        run_validate()
    else:
        sys.exit(f"unknown encoder stage: {cmd!r} "
                 "(ebird-cache|amplitude|esk|desk|cube|validate)")


if __name__ == "__main__":
    main()
