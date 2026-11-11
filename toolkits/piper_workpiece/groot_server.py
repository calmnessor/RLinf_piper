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

"""GR00T N1.7 server using the existing workpiece absolute-joint TCP contract."""

import argparse
import errno
import socket
import time
import traceback

from ._logging import get_logger
from .common import ACTION_MODE, ACTION_NAMES, action_chunk, observation_batch
from .groot_model import load_groot_model
from .groot_rtc import NativeGrootRTC
from .transport import recv_obj, send_obj, setup_logging

logger = get_logger()


def run(args: argparse.Namespace) -> None:
    """Serve raw absolute action chunks with the saved checkpoint processors."""
    setup_logging("workpiece_groot_server", args.log_dir)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((args.host, args.port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise SystemExit(
                    f"[server] {args.host}:{args.port} is already in use. "
                    "If GR00T is already running, use run_bridge_workpiece_groot.sh. "
                    "Stop the existing server before starting a replacement."
                ) from None
            raise
        server.listen(5)
        policy, pre, post, torch = load_groot_model(
            args.policy_path, args.runtime_root, args.device, args.weights_dtype
        )
        rtc = NativeGrootRTC(policy, pre, post, torch, args.rtc_ramp_rate)
        metadata = {
            "policy_type": "groot",
            "model_version": "N1.7",
            "checkpoint": args.policy_path,
            "embodiment_id": policy.deployment_embodiment_id,
            "action_mode": ACTION_MODE,
            "action_names": ACTION_NAMES,
            "chunk_size": policy.config.chunk_size,
            "fps": 30,
            "joint_unit": "0.001 degree",
            "gripper_unit": "0.001 mm",
            "rtc_supported": True,
            "rtc_enabled": bool(args.rtc),
            "rtc_method": "native_gr00t_overlap",
            "rtc_ramp_rate": args.rtc_ramp_rate,
        }
        server.settimeout(1)
        logger.info("%s", f"[server] listening {args.host}:{args.port}")
        try:
            while True:
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                logger.info("%s", f"[server] connected {address}")
                with connection:
                    policy.reset()
                    try:
                        while True:
                            request = recv_obj(connection)
                            if request.get("type") == "close":
                                break
                            if request.get("type") == "ping":
                                send_obj(connection, {"ok": True, **metadata})
                                continue
                            batch = observation_batch(
                                request["observation"], args.task_instruction, torch
                            )
                            start = time.perf_counter()
                            with torch.inference_mode():
                                processed = pre(batch)
                                rtc_request = request.get("rtc")
                                if rtc_request is not None and not args.rtc:
                                    raise ValueError("Server RTC is disabled")
                                restored, rtc_metrics = rtc.predict(
                                    processed, rtc_request
                                )
                            actions = action_chunk(
                                restored.squeeze(0).detach().float().cpu().numpy()
                            )
                            elapsed = time.perf_counter() - start
                            logger.info(
                                "%s",
                                f"[infer] {elapsed * 1000:.0f} ms shape={actions.shape} "
                                f"rtc={rtc_metrics} first_raw={actions[0].round(1).tolist()}",
                            )
                            send_obj(
                                connection,
                                {
                                    "ok": True,
                                    "actions": actions,
                                    "inference_ms": elapsed * 1000,
                                    "rtc": rtc_metrics,
                                    **metadata,
                                },
                            )
                    except ConnectionError:
                        logger.info("%s", "[server] disconnected")
                    except Exception as exc:
                        traceback.print_exc()
                        try:
                            send_obj(connection, {"ok": False, "error": repr(exc)})
                        except OSError:
                            pass
        except KeyboardInterrupt:
            logger.info("%s", "[server] stopped")


def get_args() -> argparse.Namespace:
    """Parse server settings before importing model-specific dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--runtime_root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5566)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weights_dtype", choices=["float32", "bfloat16"], default="float32"
    )
    parser.add_argument(
        "--task_instruction",
        default="Pick up the blue workpiece and put it into the box.",
    )
    parser.add_argument("--log_dir", default="logs/workpiece_groot")
    parser.add_argument("--rtc", type=int, choices=[0, 1], default=1)
    parser.add_argument("--rtc_ramp_rate", type=float, default=6.0)
    args = parser.parse_args()
    if not 0 < args.rtc_ramp_rate < float("inf"):
        parser.error("rtc_ramp_rate must be finite and positive")
    return args


if __name__ == "__main__":
    run(get_args())
