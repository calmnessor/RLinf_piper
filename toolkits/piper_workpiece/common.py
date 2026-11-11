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

"""Wire contract and raw SDK joint targets for the workpiece checkpoint."""

import json
import threading
import time
from pathlib import Path

import numpy as np

ACTION_NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
CAMERA_NAMES = ("front", "left", "wrist")
ACTION_MODE = "absolute_joint_raw"
DEFAULT_CAMERAS = {
    "front": {
        "type": "opencv",
        "index_or_path": "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC1N16200F0-video-index0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "fourcc": "MJPG",
    },
    "left": {
        "type": "opencv",
        "index_or_path": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.4.2:1.3-video-index0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "fourcc": "YUYV",
    },
    "wrist": {
        "type": "opencv",
        "index_or_path": "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC1WC5200NZ-video-index0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "fourcc": "MJPG",
    },
}


class JointFeedbackTrackerMixin:
    """A joint-state timestamp alone only proves that ONE joint pair arrived."""

    def __init__(self, *args, **kwargs):
        self._joint_feedback_seen = set()
        self._joint_feedback_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def ParseCANFrame(self, message):
        result = super().ParseCANFrame(message)
        if message.arbitration_id in {0x2A2, 0x2A3, 0x2A4, 0x2A5, 0x2A6, 0x2A7, 0x2A8}:
            with self._joint_feedback_lock:
                self._joint_feedback_seen.add(message.arbitration_id)
        return result

    @property
    def feedback_ready(self):
        with self._joint_feedback_lock:
            return self._joint_feedback_seen.issuperset({0x2A5, 0x2A6, 0x2A7, 0x2A8})

    @property
    def eef_feedback_ready(self) -> bool:
        with self._joint_feedback_lock:
            return self._joint_feedback_seen.issuperset({0x2A2, 0x2A3, 0x2A4})


def load_camera_config(value: str | None = None) -> dict:
    """Load the three named camera configurations from JSON text or a file."""
    if value is None:
        return {name: dict(config) for name, config in DEFAULT_CAMERAS.items()}
    config = (
        json.loads(value)
        if value.lstrip().startswith("{")
        else json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
    )
    if set(config) != set(CAMERA_NAMES):
        raise ValueError(f"Expected camera keys {CAMERA_NAMES}, got {tuple(config)}")
    for name, settings in config.items():
        if settings.get("type", "opencv") != "opencv":
            raise ValueError(f"Camera {name} requires an OpenCV configuration")
        for key in ("width", "height", "fps"):
            if not isinstance(settings.get(key), int) or settings[key] <= 0:
                raise ValueError(f"Camera {name}.{key} must be a positive integer")
        if "index_or_path" not in settings:
            raise ValueError(f"Camera {name} requires index_or_path")
        if settings.get("color_mode", "rgb").lower() != "rgb":
            raise ValueError(f"Camera {name} must return RGB for the saved checkpoint")
    return config


def state_vector(value):
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError(f"Expected seven finite raw joint/gripper values, got {state}")
    return state


def action_chunk(value):
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions shaped (T, 7), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Model returned NaN or infinity")
    return actions


def read_joint_state(arm):
    if getattr(arm, "feedback_ready", True) is False:
        raise RuntimeError(
            "Waiting for all three joint-pair frames and gripper feedback"
        )
    joints = arm.GetArmJointMsgs()
    gripper = arm.GetArmGripperMsgs()
    # SDK caches are zero before their first received CAN feedback frame.
    if joints.time_stamp <= 0 or gripper.time_stamp <= 0:
        raise RuntimeError(
            "No joint/gripper CAN feedback yet; check can0 and arm power"
        )
    return state_vector(
        [
            *(getattr(joints.joint_state, f"joint_{i}") for i in range(1, 7)),
            gripper.gripper_state.grippers_angle,
        ]
    )


def wait_for_feedback(arm, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return read_joint_state(arm)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def raw_target(value):
    """Only round to CAN integer units; never clip, scale, or integrate."""
    return tuple(int(round(float(x))) for x in state_vector(value))


def send_joint_target(arm, value, speed_pct=100, gripper_effort=1000):
    target = raw_target(value)
    arm.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)  # CAN / MOVE J
    arm.JointCtrl(*target[:6])
    arm.GripperCtrl(target[6], gripper_effort, 0x01, 0x00)


def execute_chunk(
    actions,
    publish,
    *,
    fps=30,
    max_steps=0,
    sent=0,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Consume every predicted target in order, with no inference catch-up burst."""
    for target in action_chunk(actions):
        if max_steps and sent >= max_steps:
            break
        start = clock()
        publish(target)
        sent += 1
        sleep(max(0.0, start + 1.0 / fps - clock()))
    return sent


def observation_batch(observation, task, torch):
    # Apply the saved seven-dimensional statistics before GR00T pads the state.
    batch = {
        "observation.state": torch.from_numpy(
            state_vector(observation["observation.state"]).copy()
        ),
        "task": observation.get("task", task),
    }
    for name in CAMERA_NAMES:
        key = f"observation.images.{name}"
        frame = np.asarray(observation[key])
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"{key} must be RGB HWC uint8, got {frame.shape} {frame.dtype}"
            )
        batch[key] = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
    return batch
