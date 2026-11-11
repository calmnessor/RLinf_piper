#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$HOME/workspace/lerobot-groot17-inference}"
PYTHON_BIN="${PYTHON_BIN:-$RUNTIME_ROOT/.venv/bin/python}"
POLICY_PATH="${POLICY_PATH:-$HOME/workspace/lerobot/checkpoint/workpiece_groot17_480ep_3cam_joint7/pretrained_model}"
export PYTHONPATH="$ROOT:$RUNTIME_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
exec "$PYTHON_BIN" -u -m toolkits.piper_workpiece.groot_server \
  --policy_path "$POLICY_PATH" --runtime_root "$RUNTIME_ROOT" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-5566}" --device "${DEVICE:-cuda}" \
  --rtc "${RTC:-1}" --rtc_ramp_rate "${RTC_RAMP_RATE:-6}" \
  --weights_dtype "${WEIGHTS_DTYPE:-bfloat16}" --log_dir "${LOG_DIR:-$ROOT/logs/piper_workpiece}" "$@"
