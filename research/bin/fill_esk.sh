#!/usr/bin/env bash
# Copy the observed-community ESK files (esk_epochs.npz, esk_annual.npy) from a control cache into research caches
# that lack them. They are projections of the OBSERVED communities, identical for every model on the same split
# (checked: seeds 1 and 2 byte-identical), and fair_rivals needs them; new caches (notably epoch_variant's late
# caches) are written without them. Existing files are never overwritten.
# usage: fill_esk.sh <control cache> <cache> [<cache> ...]
set -euo pipefail
CTRL="$1"; shift
for d in "$@"; do
  for f in esk_epochs.npz esk_annual.npy; do
    [ -f "$d/$f" ] || { cp "$CTRL/$f" "$d/$f"; echo "[fill_esk] $f -> $(basename "$d")"; }
  done
done
