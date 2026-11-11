#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Run the three-camera absolute-joint bridge against the local GR00T model server.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/lerobot_piper/bin/python}"
MODE="${1:-safe}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in safe|real) ;; *) echo "Usage: $0 {safe|real}" >&2; exit 2 ;; esac
COLOR="${COLOR:-blue}"
TASK="${TASK:-Pick up the $COLOR workpiece and put it into the box.}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON_BIN" -u -m toolkits.piper_workpiece.groot_bridge \
  --mode "$MODE" --host "${HOST:-127.0.0.1}" --port "${PORT:-5566}" --can "${CAN:-can0}" \
  --task_instruction "$TASK" --fps "${FPS:-30}" --can_hz "${CAN_HZ:-100}" \
  --speed_pct "${SPEED_PCT:-100}" --gripper_effort "${GRIPPER_EFFORT:-1000}" \
  --max_steps "${MAX_STEPS:-700}" --prefetch_steps "${PREFETCH_STEPS:-15}" --blend_steps "${BLEND_STEPS:-4}" \
  --rtc "${RTC:-1}" --rtc_delay_steps "${RTC_DELAY_STEPS:-0}" --dry_run_steps "${DRY_RUN_STEPS:-0}" \
  --cameras "${CAMERAS:-${CAMERAS_FILE:-$SCRIPT_DIR/cameras.json}}" \
  --log_dir "${LOG_DIR:-$ROOT/logs/piper_workpiece}" "$@"
