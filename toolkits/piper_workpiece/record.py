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

"""Record externally teleoperated Piper demonstrations as LeRobot episodes."""

import argparse
import inspect
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from ._logging import get_logger
from .common import ACTION_NAMES, CAMERA_NAMES, load_camera_config, read_joint_state
from .controller import close_cameras, make_arm, open_cameras
from .controls import RecordingControls
from .transport import setup_logging

logger = get_logger()

EEF_NAMES = (
    ["eef_x", "eef_y", "eef_z"] + [f"eef_rot_{i}" for i in range(6)] + ["gripper.pos"]
)


def recording_names(mode: str) -> list[str]:
    """Return the feature order used by the existing Piper recording datasets."""
    if mode == "joint":
        return list(ACTION_NAMES)
    if mode == "eef":
        return list(EEF_NAMES)
    if mode == "eef_joint":
        return EEF_NAMES + ACTION_NAMES[:6]
    raise ValueError(f"Expected eef, joint, or eef_joint; got {mode!r}")


def read_recording_state(arm: Any, mode: str) -> np.ndarray:
    """Read raw absolute joints, metre XYZ, and interleaved rotation-6D."""
    joints = read_joint_state(arm)
    if mode == "joint":
        return joints
    if getattr(arm, "eef_feedback_ready", False) is False:
        raise RuntimeError("Waiting for all three end-effector CAN feedback frames")
    message = arm.GetArmEndPoseMsgs()
    if message.time_stamp <= 0:
        raise RuntimeError("No end-effector feedback has arrived")
    pose = message.end_pose
    roll, pitch, yaw = np.deg2rad(
        [pose.RX_axis / 1000, pose.RY_axis / 1000, pose.RZ_axis / 1000]
    )
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    eef = [
        pose.X_axis / 1e6,
        pose.Y_axis / 1e6,
        pose.Z_axis / 1e6,
        cp * cy,
        sr * sp * cy - cr * sy,
        cp * sy,
        sr * sp * sy + cr * cy,
        -sp,
        sr * cp,
        joints[6],
    ]
    value = np.asarray(
        eef + list(joints[:6]) if mode == "eef_joint" else eef, dtype=np.float32
    )
    if not np.isfinite(value).all():
        raise ValueError("Piper recording feedback contains NaN or infinity")
    return value


def dataset_features(mode: str, cameras: dict, label_outcomes: bool) -> dict:
    """Describe the numeric and RGB video features without changing state units."""
    names = recording_names(mode)
    features = {
        key: {"dtype": "float32", "shape": (len(names),), "names": names}
        for key in ("observation.state", "action")
    }
    for name in CAMERA_NAMES:
        config = cameras[name]
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (config["height"], config["width"], 3),
            "names": ["height", "width", "channels"],
        }
    if label_outcomes:
        features.update(
            {
                key: {"dtype": "bool", "shape": (1,), "names": None}
                for key in ("is_success", "done")
            }
        )
    return features


def transition_frame(
    previous: dict, following: dict, task: str, label_outcomes: bool = False
) -> dict:
    """Pair observation[t] with measured absolute state[t+1] as the label."""
    frame = {
        "observation.state": previous["observation.state"].copy(),
        "action": following["observation.state"].copy(),
        "task": task,
    }
    for name in CAMERA_NAMES:
        key = f"observation.images.{name}"
        frame[key] = previous[key]
    if label_outcomes:
        frame.update(is_success=np.array([False]), done=np.array([False]))
    return frame


def episode_buffer(dataset: Any) -> dict:
    """Locate the pending episode on supported LeRobot writer layouts."""
    writer = getattr(dataset, "writer", dataset)
    buffer = getattr(writer, "episode_buffer", None)
    if buffer is None:
        raise RuntimeError("LeRobot dataset has no active episode buffer")
    return buffer


def label_episode(dataset: Any, success: bool) -> None:
    """Set the human-verified outcome on every frame before committing."""
    buffer = episode_buffer(dataset)
    count = int(buffer["size"])
    if count <= 0:
        raise ValueError("Cannot label an empty episode")
    buffer["is_success"] = [np.array([success], dtype=bool) for _ in range(count)]
    buffer["done"] = [np.array([i == count - 1], dtype=bool) for i in range(count)]


def open_dataset(args: argparse.Namespace, features: dict) -> Any:
    """Create or resume a compatible LeRobot dataset without overwriting one."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.dataset_root).expanduser()
    repo_id = args.repo_id or f"local/{root.name}"
    if args.resume:
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        if info["fps"] != args.fps:
            raise ValueError(
                f"Resume FPS mismatch: saved={info['fps']}, requested={args.fps}"
            )
        for key, expected in features.items():
            actual = info["features"].get(key, {})
            if any(
                actual.get(field) != value
                for field, value in expected.items()
                if field != "shape"
            ) or list(actual.get("shape", [])) != list(expected["shape"]):
                raise ValueError(
                    f"Resume feature mismatch for {key}: {actual} vs {expected}"
                )
        if ("is_success" in info["features"]) != args.label_outcomes:
            raise ValueError("Resume label_outcomes must match the existing dataset")
        if hasattr(LeRobotDataset, "resume"):
            return LeRobotDataset.resume(
                repo_id=repo_id, root=root, video_backend="pyav", image_writer_threads=4
            )
        dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
        dataset.start_image_writer(num_processes=0, num_threads=4)
        return dataset
    if root.exists():
        raise FileExistsError(
            f"Dataset root already exists: {root}; select a new root or --resume"
        )
    options = {
        "repo_id": repo_id,
        "root": root,
        "fps": args.fps,
        "robot_type": "piper",
        "features": features,
        "use_videos": True,
        "video_backend": "pyav",
        "image_writer_threads": 4,
    }
    if "streaming_encoding" in inspect.signature(LeRobotDataset.create).parameters:
        options["streaming_encoding"] = False
    return LeRobotDataset.create(**options)


def finalize_dataset(dataset: Any) -> None:
    """Discard an uncommitted episode and flush all saved metadata footers."""
    try:
        dataset.clear_episode_buffer()
    finally:
        dataset.finalize()


def commit_episode(dataset: Any) -> None:
    """Encode without forking a process that owns live camera and CAN threads."""
    if "parallel_encoding" in inspect.signature(dataset.save_episode).parameters:
        dataset.save_episode(parallel_encoding=False)
    else:
        dataset.save_episode()


def capture_observation(arm: Any, cameras: dict, mode: str) -> dict:
    """Capture RGB frames and feedback without querying or commanding the arm."""
    state = read_recording_state(arm, mode)
    at = time.monotonic()
    return {
        "observation.state": state,
        "captured_at": at,
        **{
            f"observation.images.{name}": cameras[name]
            .async_read(timeout_ms=5000)
            .copy()
            for name in CAMERA_NAMES
        },
    }


def wait_decision(
    controls: RecordingControls, seconds: float | None, labeling: bool = False
) -> str:
    """Wait for reset completion, an outcome label, discard, or stop."""
    deadline = math.inf if seconds is None else time.monotonic() + seconds
    while time.monotonic() < deadline:
        events = controls.poll()
        keys = (
            ("stop", "discard", "success", "failure")
            if labeling
            else ("stop", "discard", "finish")
        )
        for key in keys:
            if key in events:
                return key
        time.sleep(0.01)
    return "finish"


def discard_and_reset(
    dataset: Any, controls: RecordingControls, seconds: float
) -> bool:
    """Discard a pending episode, then give the operator time to restore the scene."""
    dataset.clear_episode_buffer()
    logger.info(
        "[reset] pending episode discarded; restore scene, right=start rerecording, Esc=stop"
    )
    while True:
        decision = wait_decision(controls, seconds)
        if decision != "discard":
            return decision == "stop"


def record_episode(
    dataset: Any,
    arm: Any,
    cameras: dict,
    args: argparse.Namespace,
    controls: RecordingControls,
) -> tuple[str, int, list[float]]:
    """Record one pending episode, with no catch-up bursts after slow reads."""
    previous = capture_observation(arm, cameras, args.recording_mode)
    started = previous["captured_at"]
    deadline = started + 1 / args.fps
    captures = [started]
    count = 0
    while time.monotonic() - started < args.episode_time:
        events = controls.poll()
        for key in ("stop", "discard", "success", "failure", "finish"):
            if key in events and (
                args.label_outcomes or key not in {"success", "failure"}
            ):
                return key, count, captures
        if time.monotonic() < deadline:
            time.sleep(max(0, min(0.005, deadline - time.monotonic())))
            continue
        following = capture_observation(arm, cameras, args.recording_mode)
        dataset.add_frame(
            transition_frame(
                previous, following, args.task_instruction, args.label_outcomes
            )
        )
        captures.append(following["captured_at"])
        count += 1
        previous = following
        deadline = max(deadline + 1 / args.fps, time.monotonic())
    return "finish", count, captures


def run(args: argparse.Namespace) -> None:
    """Collect follower feedback while an external leader controls the arm."""
    setup_logging("piper_record", args.log_dir)
    camera_config = load_camera_config(args.cameras)
    features = dataset_features(args.recording_mode, camera_config, args.label_outcomes)
    logger.info(
        "[record] mode=%s names=%s fps=%s; CAN read-only",
        args.recording_mode,
        recording_names(args.recording_mode),
        args.fps,
    )
    if args.describe:
        logger.info(
            "%s", json.dumps({"features": features, "cameras": camera_config}, indent=2)
        )
        return
    with RecordingControls() as controls, ExitStack() as resources:
        arm = make_arm(args.can)
        resources.callback(arm.DisconnectPort)
        arm.ConnectPort(piper_init=False)
        deadline = time.monotonic() + 10
        while True:
            try:
                read_recording_state(arm, args.recording_mode)
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        cameras = open_cameras(camera_config)
        resources.callback(close_cameras, cameras)
        dataset = open_dataset(args, features)
        resources.callback(finalize_dataset, dataset)
        root = Path(args.dataset_root).expanduser()
        metadata_path = root / "meta/piper_recording.json"
        contract = {
            "recording_mode": args.recording_mode,
            "names": recording_names(args.recording_mode),
            "fps": args.fps,
            "joint_unit": "0.001 degree",
            "gripper_unit": "0.001 mm",
            "eef_position_unit": "metre",
            "eef_rotation": "interleaved first two rotation-matrix columns",
            "action_semantics": "observation[t] -> measured absolute state[t+1]",
            "cameras": camera_config,
        }
        metadata_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        saved = 0
        while saved < args.episodes:
            logger.info(
                "[record] episode=%s/%s: right=finish, left=discard, Esc=stop%s",
                saved + 1,
                args.episodes,
                "; s=success, f=failure" if args.label_outcomes else "",
            )
            decision, count, captures = record_episode(
                dataset, arm, cameras, args, controls
            )
            if decision == "stop":
                break
            if decision == "discard" or count == 0:
                if discard_and_reset(dataset, controls, args.reset_time):
                    break
                continue
            outcome = None
            if args.label_outcomes:
                if decision not in {"success", "failure"}:
                    logger.info(
                        "[label] s=success, f=failure, left/d=discard, Esc=stop"
                    )
                    decision = wait_decision(controls, None, labeling=True)
                if decision == "stop":
                    break
                if decision == "discard":
                    if discard_and_reset(dataset, controls, args.reset_time):
                        break
                    continue
                outcome = decision == "success"
                label_episode(dataset, outcome)
            logger.info(
                "[reset] restore the scene; right=save/continue, left=discard/rerecord; timeout=%ss",
                args.reset_time,
            )
            decision = wait_decision(controls, args.reset_time)
            if decision == "stop":
                break
            if decision == "discard":
                if discard_and_reset(dataset, controls, args.reset_time):
                    break
                continue
            episode_index = int(dataset.meta.total_episodes)
            commit_episode(dataset)
            saved += 1
            intervals = np.diff(captures)
            note = {
                "episode_index": episode_index,
                "task": args.task_instruction,
                "is_success": outcome,
                "frames": count,
                "measured_seconds": captures[-1] - captures[0],
                "max_capture_interval_s": float(intervals.max()),
                "capture_times_relative_s": [value - captures[0] for value in captures],
            }
            with (root / "meta/piper_episode_notes.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(note) + "\n")
            logger.info(
                "[saved] episode=%s frames=%s success=%s; max interval=%.3fs",
                episode_index,
                count,
                outcome,
                intervals.max(),
            )


def get_args() -> argparse.Namespace:
    """Parse portable recording settings without importing hardware packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--repo_id")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--cameras", help="Camera JSON text or JSON file")
    parser.add_argument(
        "--recording_mode", choices=["eef_joint", "joint", "eef"], default="eef_joint"
    )
    parser.add_argument(
        "--task_instruction",
        default="Pick up the blue workpiece and put it into the box.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode_time", type=float, default=300)
    parser.add_argument("--reset_time", type=float, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--label_outcomes", action="store_true")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Show the recording schema without opening cameras or CAN",
    )
    parser.add_argument("--log_dir", default="logs/piper_workpiece")
    args = parser.parse_args()
    if (
        args.fps <= 0
        or args.episodes <= 0
        or not np.isfinite([args.episode_time, args.reset_time]).all()
        or args.episode_time <= 0
        or args.reset_time < 0
    ):
        parser.error(
            "FPS, episodes, and episode_time must be positive; reset_time must be nonnegative"
        )
    return args


if __name__ == "__main__":
    try:
        run(get_args())
    except KeyboardInterrupt:
        logger.info(
            "[record] stopped; pending episode discarded and saved episodes finalized"
        )
