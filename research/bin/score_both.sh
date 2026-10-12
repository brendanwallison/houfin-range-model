#!/usr/bin/env bash
# Score one DESK cache on BOTH targets with the standard yardstick (the user, 2026-10-12: record both categories for
# every model -- training is the expensive part, scoring is cheap):
#   fair_rivals_incr_p96.json      the community's ESK coordinates (the 96 reference species)
#   fair_rivals_incr_p96dev.json   the 273 OUT-OF-COMMUNITY dev species (the production-relevant case)
# The species target fits 273 per-species BLRs per arm, so it runs --lean (optional arms skipped; every headline
# number kept) with 2 placebo draws instead of 5.
# usage: score_both.sh <python> <out dir> <fair_rivals args: --cache ... [--ref-cache ...] [--eval-set ...] ...>
#        SCORE_TARGETS="dev" score_both.sh ...   to run one target only (backfills of caches already scored on esk)
set -euo pipefail
PY="$1"; OUT="$2"; shift 2
for tgt in ${SCORE_TARGETS:-esk dev}; do
  if [ "$tgt" = esk ]; then
    "$PY" research/exp/fair_rivals.py "$@" --out "$OUT" --incremental --n-pos 96 --pos-seeds 1 2 3 4 5 \
      --strong-base --tag _p96
  else
    "$PY" research/exp/fair_rivals.py "$@" --out "$OUT" --incremental --n-pos 96 --pos-seeds 1 2 --strong-base \
      --lean --tag _p96dev --target dev
  fi
done
