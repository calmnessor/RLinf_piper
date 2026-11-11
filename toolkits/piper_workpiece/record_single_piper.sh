#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Record externally teleoperated feedback; this process never enables or moves the arm.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/lerobot_piper/bin/python}"
COLOR="${COLOR:-blue}"
TASK="${TASK:-Pick up the $COLOR workpiece and put it into the box.}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/Data/Industrial_sorting/rlinf_recordings/workpiece_${COLOR}_$(date +%Y%m%d_%H%M%S)}"
args=(--dataset_root "$DATASET_ROOT" --can "${CAN:-can0}"
      --recording_mode "${RECORDING_MODE:-eef_joint}"
      --task_instruction "$TASK" --fps "${FPS:-30}" --episodes "${EPISODES:-10}"
      --episode_time "${EPISODE_TIME:-300}" --reset_time "${RESET_TIME:-200}"
      --cameras "${CAMERAS:-${CAMERAS_FILE:-$SCRIPT_DIR/cameras.json}}"
      --log_dir "${LOG_DIR:-$ROOT/logs/piper_workpiece}")
[[ -z "${DATASET_REPO:-}" ]] || args+=(--repo_id "$DATASET_REPO")
case "${RESUME:-false}" in 1|true) args+=(--resume) ;; 0|false) ;; *) echo "RESUME must be true/false or 1/0" >&2; exit 2 ;; esac
case "${LABEL_OUTCOMES:-false}" in 1|true) args+=(--label_outcomes) ;; 0|false) ;; *) echo "LABEL_OUTCOMES must be true/false or 1/0" >&2; exit 2 ;; esac
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON_BIN" -u -m toolkits.piper_workpiece.record "${args[@]}" "$@"
