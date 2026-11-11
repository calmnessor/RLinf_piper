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

"""Piper SDK and camera fakes for the standalone workpiece tools."""

from types import SimpleNamespace

import numpy as np


class FakePiperInterface:
    """Expose cached feedback and record every CAN-writing SDK operation."""

    instances: list = []

    def __init__(self, channel: str, **kwargs) -> None:
        self.channel = channel
        self.commands: list[tuple] = []
        self.connected = False
        self.joints = [30000, 1000, -20000, 4000, 5000, 6000]
        self.gripper = 35000
        self.instances.append(self)

    def ParseCANFrame(self, message) -> None:
        """Let the production feedback tracker observe each incoming identifier."""

    def ConnectPort(self, piper_init: bool = True) -> None:
        self.connected = True
        if piper_init:
            self.commands.append(("init",))
        for identifier in range(0x2A2, 0x2A9):
            self.ParseCANFrame(SimpleNamespace(arbitration_id=identifier))

    def DisconnectPort(self) -> None:
        self.connected = False

    def GetArmJointMsgs(self):
        return SimpleNamespace(
            time_stamp=1,
            joint_state=SimpleNamespace(
                **{f"joint_{i}": value for i, value in enumerate(self.joints, 1)}
            ),
        )

    def GetArmGripperMsgs(self):
        return SimpleNamespace(
            time_stamp=1, gripper_state=SimpleNamespace(grippers_angle=self.gripper)
        )

    def GetArmEndPoseMsgs(self):
        return SimpleNamespace(
            time_stamp=1,
            end_pose=SimpleNamespace(
                X_axis=100000,
                Y_axis=200000,
                Z_axis=300000,
                RX_axis=0,
                RY_axis=0,
                RZ_axis=0,
            ),
        )

    def EnablePiper(self) -> bool:
        self.commands.append(("enable",))
        return True

    def MotionCtrl_2(self, *args) -> None:
        self.commands.append(("motion", *args))

    def JointCtrl(self, *args) -> None:
        self.commands.append(("joint", *args))

    def GripperCtrl(self, *args) -> None:
        self.commands.append(("gripper", *args))


class FakeOpenCVCamera:
    """Return stable RGB frames and expose the capture lifecycle."""

    def __init__(self, values: dict) -> None:
        self.is_connected = False
        self.thread = None
        self.image = np.full(
            (values["height"], values["width"], 3), [25, 80, 140], dtype=np.uint8
        )

    def connect(self, warmup: bool = False) -> None:
        self.is_connected = True

    def async_read(self, timeout_ms: int = 5000) -> np.ndarray:
        if not self.is_connected:
            raise RuntimeError("Camera is disconnected")
        return self.image

    def disconnect(self) -> None:
        self.is_connected = False
