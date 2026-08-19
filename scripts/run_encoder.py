#!/usr/bin/env python
"""Single entrypoint for the community-encoder stages (one subcommand per process).

Centralizes the encoder's dual import-root quirk: the ``train_DESK`` modules use
``src.``-style imports while ``build_final_z_cube`` uses the ``community_encoder``
top-level root (+ a ``src.config_utils`` loader). Putting both the repo root and
``src/`` on ``sys.path`` here lets every stage resolve regardless of style. Each
stage is meant to run as its own process (the TACC pipeline calls this once per
stage), so the two roots never collide within one interpreter.

    python scripts/run_encoder.py
        {ebird-cache|bbs-points|trend-points|trend-reference|esk|spacetime-esk|desk|
         cube|validate|validate-reference|bbs-route-validate|single-year-analysis}
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
    elif cmd == "trend-points":
        from src.community_encoder.train_DESK.trend_community import build_trend_points
        build_trend_points()
    elif cmd == "esk":
        from src.community_encoder.train_DESK.esk_kernel import run_esk_experiment
        run_esk_experiment()
    elif cmd == "spacetime-esk":
        from src.community_encoder.train_DESK.esk_kernel import run_spacetime_esk
        run_spacetime_esk()
    elif cmd == "desk":
        from src.community_encoder.train_DESK.desk_training import run_desk_experiment
        run_desk_experiment()
    elif cmd == "cube":
        from community_encoder.build_final_z_cube import build_spacetime_cube
        build_spacetime_cube()
    elif cmd == "validate":
        from src.community_encoder.train_DESK.validate_spacetime import run_validate
        run_validate()
    elif cmd == "single-year-analysis":
        from community_encoder.analysis_2023.single_year_analysis import run_single_year_analysis
        from community_encoder.analysis_2023.compare_esk_desk import compare_esk_desk
        run_single_year_analysis()
        compare_esk_desk()
    elif cmd == "bbs-points":
        # The raw-BBS + eBird-window training target: what the surveyors counted, rather than a
        # reconstruction from published trend rates.
        from src.community_encoder.train_DESK.bbs_community_points import main as bbs_points_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        bbs_points_main()
    elif cmd == "trend-reference":
        # The trend products rebuilt WITHOUT our spatial blur, for use as a sanity-check
        # reference rather than as a target.
        from src.community_encoder.train_DESK.validate_trend_reference import build_reference_points
        build_reference_points()
    elif cmd == "validate-reference":
        # Five comparisons against that reference: direction and rank (clean), species trend
        # sign, and the full and spatial similarity structures (the spatial one grades the axis
        # the products' own smoothing acts on). Needs a GPU queue -- one encode pass over the
        # EMA span, same cost as bbs-route-validate.
        from src.community_encoder.train_DESK.validate_trend_reference import run_panel
        run_panel()
    elif cmd == "bbs-route-validate":
        # Route-level BBS validation: grades DESK against a no-change null on GENUINELY
        # SURVEYED cell-years, escaping the IDW-interpolated target every other metric uses.
        # Needs a GPU queue (one whole-grid forward per year of the EMA warmup span).
        from src.community_encoder.train_DESK.validate_bbs_routes import run
        run()
    else:
        sys.exit(f"unknown encoder stage: {cmd!r} (ebird-cache|bbs-points|trend-points|"
                 "trend-reference|esk|spacetime-esk|desk|cube|validate|validate-reference|"
                 "bbs-route-validate|"
                 "single-year-analysis)")


if __name__ == "__main__":
    main()
