# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR00T-only bridge, reusing the existing CAN publisher and cleanup unchanged."""

import argparse
import json
import time

import numpy as np

from ._logging import get_logger
from .common import (
    ACTION_MODE,
    ACTION_NAMES,
    DEFAULT_CAMERAS,
    load_camera_config,
    read_joint_state,
    send_joint_target,
    wait_for_feedback,
)
from .controller import (
    JointPublisher,
    TrajectoryTrace,
    close_cameras,
    enable_arm,
    make_arm,
    observation,
    open_cameras,
    timed_observation,
)
from .groot_rtc import RTCRequestClient
from .trajectory import AsyncInference, execute_pipeline
from .transport import connect_with_retry, recv_obj, send_obj, setup_logging

logger = get_logger()


def run(args: argparse.Namespace) -> None:
    """Run observation/inference separately from the 100 Hz joint publisher."""
    setup_logging("workpiece_groot_bridge_" + args.mode, args.log_dir)
    requester = RTCRequestClient(bool(args.rtc), args.rtc_delay_steps, args.fps)
    arm = None
    cameras = {}
    connection = None
    publisher = None
    inference = None
    trace = None
    enabled = False
    try:
        # No PiperInit queries, mode changes, enables, or commands in safe mode.
        arm = make_arm(args.can)
        arm.ConnectPort(piper_init=False)
        current = wait_for_feedback(arm)
        logger.info("%s", f"[state raw] {current.tolist()}")
        cameras = open_cameras(load_camera_config(args.cameras))
        connection = connect_with_retry(args.host, args.port)
        connection.settimeout(120)
        send_obj(connection, {"type": "ping"})
        meta = recv_obj(connection)
        if (
            not meta.get("ok")
            or meta.get("action_mode") != ACTION_MODE
            or meta.get("action_names") != ACTION_NAMES
        ):
            raise RuntimeError(f"Wrong policy server: {meta}")
        if args.rtc and not (meta.get("rtc_supported") and meta.get("rtc_enabled")):
            raise RuntimeError(
                "Restart the GR00T server with RTC enabled, or set RTC=0"
            )
        logger.info("%s", f"[task] {args.task_instruction}")
        actions = requester(
            connection, observation(arm, cameras, args.task_instruction)
        )
        if args.mode == "safe":
            logger.info(
                "%s",
                "[safe] CAN read-only; predicted raw absolute joint/gripper targets:",
            )
            for index, target in enumerate(actions):
                logger.info("%s", f"  {index:02d}: {target.round(2).tolist()}")
            logger.info("%s", f"[safe] OK: {len(actions)} x 7, no motion commands sent")
            if not getattr(args, "dry_run_steps", 0):
                return

        # Start from the measured pose, not a preset HOME or dataset first frame.
        current = read_joint_state(arm)
        prefetch_steps = args.prefetch_steps
        if not 0 < prefetch_steps < len(actions):
            raise ValueError(
                "prefetch_steps must be positive and less than the prediction chunk size"
            )
        if args.mode == "real":
            # Cleanup also holds measured joints if enabling only partially succeeds.
            enabled = True
            enable_arm(arm)
        trace = TrajectoryTrace(args.log_dir, args.mode, arm)
        publisher = JointPublisher(
            arm,
            current,
            hz=args.can_hz,
            fps=args.fps,
            speed_pct=args.speed_pct,
            gripper_effort=args.gripper_effort,
            read_only=args.mode == "safe",
            trace=trace,
        )
        # Bootstrap after enable, from the measured pose, without replaying startup latency.
        publisher.started_at = time.monotonic()
        publisher.install_prediction(actions, publisher.started_at, blend_steps=0)

        def observe_for_rtc(stopping):
            obs, at = timed_observation(arm, cameras, args.task_instruction, stopping)
            return requester.attach(obs, at, publisher), at

        inference = AsyncInference(connection, observe_for_rtc, requester)
        logger.info(
            "%s",
            f"[pipeline] rtc={bool(args.rtc)} async prefetch={prefetch_steps} blend={args.blend_steps} "
            f"fps={args.fps} can_hz={args.can_hz} read_only={publisher.read_only}",
        )
        logger.info("%s", f"[trace] {trace.path}")
        publisher.start()
        execute_pipeline(
            publisher,
            inference,
            fps=args.fps,
            max_steps=args.dry_run_steps if args.mode == "safe" else args.max_steps,
            prefetch_steps=prefetch_steps,
            blend_steps=args.blend_steps,
        )
    except KeyboardInterrupt:
        logger.info("%s", "[bridge] interrupted")
    finally:
        if publisher is not None:
            publisher.stop()
            logger.info(
                "%s",
                f"[publisher] samples={publisher.samples} overruns={publisher.overruns} "
                f"max_interval_ms={publisher.max_interval_s * 1000:.2f}",
            )
        if enabled:
            try:
                # Stop at measured joints on exit; retain the last gripper target.
                hold = read_joint_state(arm).copy()
                if publisher is not None:
                    hold[6] = publisher.last_target[6]
                send_joint_target(arm, hold, args.speed_pct, args.gripper_effort)
                logger.info("%s", "[bridge] hold current joints")
            except Exception as exc:
                logger.info("%s", f"[bridge] exit hold failed: {exc}")
        if inference is not None:
            inference.stop()
        if trace is not None:
            trace.close()
        if connection is not None:
            try:
                send_obj(connection, {"type": "close"})
            except (OSError, ConnectionError):
                pass
            connection.close()
        close_cameras(cameras)
        if arm is not None:
            arm.DisconnectPort()


def get_args() -> argparse.Namespace:
    """Parse bridge settings without opening hardware or loading a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["safe", "real"], default="safe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5566)
    parser.add_argument("--can", default="can0")
    parser.add_argument(
        "--task_instruction",
        default="Pick up the blue workpiece and put it into the box.",
    )
    parser.add_argument("--cameras", default=json.dumps(DEFAULT_CAMERAS))
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--can_hz", type=float, default=100)
    parser.add_argument("--speed_pct", type=int, default=100)
    parser.add_argument("--gripper_effort", type=int, default=1000)
    parser.add_argument(
        "--max_steps", type=int, default=700, help="0: run until Ctrl+C"
    )
    parser.add_argument(
        "--prefetch_steps",
        type=int,
        default=15,
        help="Request a new chunk when this many old steps remain",
    )
    parser.add_argument(
        "--blend_steps",
        type=int,
        default=4,
        help="Joint-only transition duration in model steps; 0 disables blending",
    )
    parser.add_argument(
        "--dry_run_steps",
        type=int,
        default=0,
        help="safe only: exercise asynchronous playback without any CAN writes",
    )
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--rtc", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--rtc_delay_steps",
        type=int,
        default=0,
        help="0: estimate capture-to-result delay; positive: fixed frozen-prefix steps",
    )
    args = parser.parse_args()
    if (
        not np.isfinite([args.fps, args.can_hz]).all()
        or args.fps <= 0
        or args.can_hz <= 0
        or args.max_steps < 0
        or args.prefetch_steps <= 0
        or args.blend_steps < 0
        or args.dry_run_steps < 0
        or args.rtc_delay_steps < 0
    ):
        parser.error("Invalid rate, step budget, prefetch_steps or blend_steps")
    if not 1 <= args.speed_pct <= 100 or not 0 <= args.gripper_effort <= 5000:
        parser.error("speed_pct must be 1..100 and gripper_effort must be 0..5000")
    return args


if __name__ == "__main__":
    run(get_args())
