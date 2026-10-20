#!/bin/bash
# Prepend to every queue task (research/age_model/bin/worker.slurm runs tasks as `bash <task>`). All workers share
# one checkout; without this a push mid-task changed model sources between a task's MAP and refine and the drift
# guard refused the refine (E058a, E062a/b, 2026-10-18). Each task now runs from its own detached worktree.
# --- per-task code snapshot: a push to the shared checkout can never change code under this task ---
REPO=/work/07980/bwa386/ls6/houfin/age-research
TWT=$SCRATCH/age_wt/$(basename "$0" .sh)
# Several workers often claim tasks in the same second. Git's worktree metadata is
# shared, so concurrent prune/add operations race and one task can exit before it even
# writes a log. Serialize only this short setup section. The main checkout is deployed
# explicitly by the monitor; a per-task network pull is both unnecessary and another
# shared-repository race.
mkdir -p "$SCRATCH/age_wt"
exec 9>"$SCRATCH/age_wt/.setup.lock"
flock 9
git -C "$REPO" worktree prune
rm -rf "$TWT"
git -C "$REPO" worktree add -f --detach "$TWT" HEAD >/dev/null 2>&1 || exit 1
flock -u 9
cd "$TWT" && export PYTHONPATH="$TWT:$TWT/src" && git log --oneline -1
