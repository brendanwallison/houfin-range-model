#!/bin/bash
# Prepend to every queue task (research/age_model/bin/worker.slurm runs tasks as `bash <task>`). All workers share
# one checkout; without this a push mid-task changed model sources between a task's MAP and refine and the drift
# guard refused the refine (E058a, E062a/b, 2026-10-18). Each task now runs from its own detached worktree.
# --- per-task code snapshot: a push to the shared checkout can never change code under this task ---
REPO=/work/07980/bwa386/ls6/houfin/age-research
TWT=$SCRATCH/age_wt/$(basename "$0" .sh)
git -C $REPO pull -q --ff-only 2>/dev/null; git -C $REPO worktree prune
rm -rf "$TWT"; git -C $REPO worktree add -f --detach "$TWT" HEAD >/dev/null 2>&1 || exit 1
cd "$TWT" && export PYTHONPATH="$TWT:$TWT/src" && git log --oneline -1
