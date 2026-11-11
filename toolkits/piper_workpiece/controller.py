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

"""Piper single-arm absolute-joint bridge. safe is strictly CAN read-only."""

import csv
import threading
import time
from pathlib import Path

from ._logging import get_logger
from .common import (
    ACTION_MODE,
    ACTION_NAMES,
    CAMERA_NAMES,
    JointFeedbackTrackerMixin,
    action_chunk,
    read_joint_state,
    send_joint_target,
    state_vector,
)
from .trajectory import TimedTrajectory, splice_trajectory
from .transport import recv_obj, send_obj

logger = get_logger()


class JointPublisher:
    """Sample a continuous, time-aligned joint trajectory at CAN rate."""

    def __init__(
        self,
        arm,
        initial,
        *,
        hz=100,
        fps=30,
        speed_pct=100,
        gripper_effort=1000,
        read_only=False,
        trace=None,
        clock=time.monotonic,
    ):
        self.arm = arm
        self.clock = clock
        self.started_at = clock()
        self.trajectory = TimedTrajectory.hold(initial, self.started_at)
        self.last_target = state_vector(initial).copy()
        self.hz = hz
        self.fps = fps
        self.speed_pct = speed_pct
        self.gripper_effort = gripper_effort
        self.read_only = read_only
        self.trace = trace
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = None
        self.thread = None
        self.samples = 0
        self.overruns = 0
        self.max_interval_s = 0.0
        self._last_sample_at = None

    @property
    def target(self):
        with self.lock:
            return self.trajectory.sample(self.clock())

    @property
    def plan_end(self):
        with self.lock:
            return self.trajectory.end

    def install_prediction(self, actions, observed_at, *, blend_steps=4):
        if self.error is not None:
            raise RuntimeError("CAN publisher failed") from self.error
        prediction = TimedTrajectory.prediction(actions, observed_at, self.fps)
        with self.lock:
            plan, metrics = splice_trajectory(
                self.trajectory, prediction, self.clock(), self.fps, blend_steps
            )
            if plan is not None:
                self.trajectory = plan
        return metrics

    def send_once(self):
        now = self.clock()
        with self.lock:
            target = self.trajectory.sample(now)
        if not self.read_only:
            send_joint_target(self.arm, target, self.speed_pct, self.gripper_effort)
        self.last_target = target.copy()
        if self.trace is not None:
            self.trace(now, target)
        if self._last_sample_at is not None:
            self.max_interval_s = max(self.max_interval_s, now - self._last_sample_at)
        self._last_sample_at = now
        self.samples += 1

    def start(self):
        self.thread = threading.Thread(
            target=self._run, name="joint-publisher", daemon=True
        )
        self.thread.start()

    def _run(self):
        deadline = self.clock()
        try:
            while not self.stop_event.is_set():
                self.send_once()
                deadline += 1 / self.hz
                now = self.clock()
                if deadline < now:
                    self.overruns += 1
                    deadline = now + 1 / self.hz
                self.stop_event.wait(max(0, deadline - now))
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


class TrajectoryTrace:
    """Record 100-Hz command and measured positions, in original SDK units."""

    def __init__(self, directory, mode, arm):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = (
            Path(directory) / f"workpiece_joint_trace_{mode}_{time.time_ns()}.csv"
        )
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["monotonic_s"]
            + [f"command_{x}" for x in ACTION_NAMES]
            + [f"measured_{x}" for x in ACTION_NAMES]
        )
        self.arm = arm

    def __call__(self, now, target):
        self.writer.writerow([f"{now:.9f}", *target, *read_joint_state(self.arm)])

    def close(self):
        self.file.close()


def open_cameras(config):
    from .camera import create_camera

    if set(config) != set(CAMERA_NAMES):
        raise ValueError("Camera keys must be front, left, wrist, matching training")
    cameras = {}
    try:
        for name in CAMERA_NAMES:
            values = dict(config[name])
            values.pop("type", None)
            for attempt in range(1, 4):
                camera = create_camera(values)
                cameras[name] = camera
                try:
                    # Avoid the shared driver's one-second first-frame timeout.
                    camera.connect(warmup=False)
                    frame = camera.async_read(timeout_ms=5000)
                    break
                except Exception as exc:
                    logger.info(
                        "%s", f"[camera] {name} start attempt {attempt}/3: {exc}"
                    )
                    if (
                        camera.is_connected
                        or getattr(camera, "thread", None) is not None
                    ):
                        camera.disconnect()
                    del cameras[name]
                    if attempt == 3:
                        raise
                    time.sleep(1)
            logger.info(
                "%s", f"[camera] {name} {frame.shape} {values['index_or_path']}"
            )
        return cameras
    except BaseException:
        close_cameras(cameras)
        raise


def close_cameras(cameras):
    for camera in cameras.values():
        try:
            if camera.is_connected or getattr(camera, "thread", None) is not None:
                camera.disconnect()
        except Exception as exc:
            logger.info("%s", f"[camera] cleanup failed: {exc}")


def make_arm(can):
    from piper_sdk import C_PiperInterface_V2

    class DeploymentPiper(JointFeedbackTrackerMixin, C_PiperInterface_V2):
        pass

    return DeploymentPiper(
        can, start_sdk_joint_limit=False, start_sdk_gripper_limit=False
    )


def observation(arm, cameras, task):
    return {
        "observation.state": read_joint_state(arm),
        "task": task,
        **{
            f"observation.images.{name}": cameras[name]
            .async_read(timeout_ms=5000)
            .copy()
            for name in CAMERA_NAMES
        },
    }


def timed_observation(arm, cameras, task, stopping):
    state = read_joint_state(arm)
    observed_at = time.monotonic()
    obs = {"observation.state": state, "task": task}
    for name in CAMERA_NAMES:
        if stopping.is_set():
            raise RuntimeError("Observation capture cancelled")
        obs[f"observation.images.{name}"] = (
            cameras[name].async_read(timeout_ms=5000).copy()
        )
    return obs, observed_at


def request_actions(connection, obs):
    send_obj(connection, {"type": "infer", "observation": obs})
    response = recv_obj(connection)
    if not response.get("ok"):
        raise RuntimeError(f"Policy server: {response.get('error', response)}")
    if response.get("action_mode") != ACTION_MODE:
        raise ValueError("Server is not the workpiece absolute-joint server")
    return action_chunk(response["actions"])


def enable_arm(arm):
    deadline = time.monotonic() + 10
    while not arm.EnablePiper():
        if time.monotonic() >= deadline:
            raise RuntimeError("Piper did not enable within 10 seconds")
        time.sleep(0.02)
