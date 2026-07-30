#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="logs/rsl_rl/parahand_only_grasp_object"
RUN_DIR="$(
  find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' \
    | sort \
    | tail -n 1
)"

if [[ -z "$RUN_DIR" ]]; then
  echo "No training run found in $LOG_ROOT" >&2
  exit 1
fi

CHECKPOINT_FILE="$(
  find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' -printf '%p\n' \
    | sort -V \
    | tail -n 1
)"

if [[ -z "$CHECKPOINT_FILE" ]]; then
  echo "No checkpoint found in $RUN_DIR" >&2
  exit 1
fi

_VISER_PORT_OVERRIDE=18701 \
uv run play Mjlab-Grasp-Object-ParaHand-Only \
  --viewer viser \
  --device cuda:1 \
  --num-envs 100 \
  --episode-length-s 20 \
  --checkpoint-file "$CHECKPOINT_FILE" \
  --curriculum-stage 0
