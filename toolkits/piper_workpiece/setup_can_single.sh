#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
CAN="${CAN:-can0}"
BITRATE="${BITRATE:-1000000}"
sudo ip link set dev "$CAN" down
sudo ip link set dev "$CAN" type can bitrate "$BITRATE"
sudo ip link set dev "$CAN" up
ip -details link show dev "$CAN"
