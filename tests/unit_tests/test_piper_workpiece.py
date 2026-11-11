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

"""Collection semantics, deployment lifecycle, and RTC without robot hardware."""

import json
import socket
import sys
import threading
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from robot_mocks.piper_workpiece import FakeOpenCVCamera, FakePiperInterface

from toolkits.piper_workpiece import camera, groot_bridge, record
from toolkits.piper_workpiece.common import (
    ACTION_MODE,
    ACTION_NAMES,
    load_camera_config,
)
from toolkits.piper_workpiece.controller import JointPublisher, make_arm
from toolkits.piper_workpiece.controls import RecordingControls
from toolkits.piper_workpiece.groot_rtc import NativeGrootRTC, trajectory_prefix
from toolkits.piper_workpiece.trajectory import TimedTrajectory
from toolkits.piper_workpiece.transport import recv_obj, send_obj


@pytest.fixture
def hardware(monkeypatch):
    FakePiperInterface.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "piper_sdk",
        SimpleNamespace(C_PiperInterface_V2=FakePiperInterface),
    )
    monkeypatch.setattr(camera, "create_camera", FakeOpenCVCamera)
    return FakePiperInterface.instances


@contextmanager
def model_service(fail_after_first=False):
    """Speak the actual wire protocol from a fake external inference service."""
    errors = []
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(5)
    port = server.getsockname()[1]

    def serve():
        try:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(5)
                predictions = 0
                while True:
                    request = recv_obj(connection)
                    if request["type"] == "close":
                        return
                    response = {
                        "ok": True,
                        "action_mode": ACTION_MODE,
                        "action_names": ACTION_NAMES,
                        "rtc_supported": True,
                        "rtc_enabled": True,
                    }
                    if request["type"] == "infer":
                        predictions += 1
                        if fail_after_first and predictions > 1:
                            send_obj(
                                connection, {"ok": False, "error": "inference failed"}
                            )
                            return
                        response.update(
                            actions=np.full((4, 7), 8000, dtype=np.float32),
                            rtc={"active": bool(request.get("rtc"))},
                        )
                    send_obj(connection, response)
        except (ConnectionError, OSError):
            pass
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.close()
        thread.join(timeout=6)
        assert not thread.is_alive(), "Fake model service did not stop"
        assert not errors


def bridge_args(tmp_path, port, mode="safe", dry_run_steps=0):
    return Namespace(
        mode=mode,
        host="127.0.0.1",
        port=port,
        can="can0",
        cameras=json.dumps(load_camera_config()),
        task_instruction="Pick up the blue workpiece and put it into the box.",
        fps=30,
        can_hz=100,
        speed_pct=50,
        gripper_effort=1000,
        max_steps=8,
        prefetch_steps=2,
        blend_steps=1,
        dry_run_steps=dry_run_steps,
        rtc=1,
        rtc_delay_steps=0,
        log_dir=str(tmp_path),
    )


@pytest.mark.parametrize("dry_run_steps", [0, 5])
def test_safe_bridge_never_writes_can(tmp_path, hardware, dry_run_steps):
    with model_service() as port:
        groot_bridge.run(bridge_args(tmp_path, port, dry_run_steps=dry_run_steps))
    arm = hardware[-1]
    assert arm.commands == []
    assert not arm.connected


@pytest.mark.parametrize("fail", [False, True])
def test_real_bridge_holds_measured_joints_after_completion_or_error(
    tmp_path, hardware, fail
):
    with model_service(fail_after_first=fail) as port:
        args = bridge_args(tmp_path, port, mode="real")
        if fail:
            with pytest.raises(RuntimeError, match="Asynchronous"):
                groot_bridge.run(args)
        else:
            groot_bridge.run(args)
    arm = hardware[-1]
    assert ("enable",) in arm.commands
    joint_commands = [command[1:] for command in arm.commands if command[0] == "joint"]
    assert len(joint_commands) > 1
    assert joint_commands[-1] == tuple(arm.joints)
    assert not arm.connected


def test_recorded_state_order_and_units_are_preserved(hardware):
    arm = make_arm("can0")
    arm.ConnectPort(piper_init=False)
    joint = record.read_recording_state(arm, "joint")
    combined = record.read_recording_state(arm, "eef_joint")
    assert record.recording_names("joint") == ACTION_NAMES
    assert record.recording_names("eef_joint")[9] == "gripper.pos"
    np.testing.assert_allclose(combined[:3], [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(combined[3:9], [1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(combined[10:], joint[:6])
    assert combined[9] == joint[6] == 35000
    assert arm.commands == []


def test_transition_labels_next_state_and_keeps_previous_images():
    previous = {"observation.state": np.arange(16, dtype=np.float32)}
    following = {"observation.state": np.arange(16, dtype=np.float32) + 100}
    for name in ("front", "left", "wrist"):
        previous[f"observation.images.{name}"] = np.full((2, 3, 3), 10, dtype=np.uint8)
        following[f"observation.images.{name}"] = np.full((2, 3, 3), 20, dtype=np.uint8)
    frame = record.transition_frame(previous, following, "task", True)
    np.testing.assert_array_equal(frame["observation.state"], np.arange(16))
    np.testing.assert_array_equal(frame["action"], np.arange(16) + 100)
    assert frame["observation.images.front"].max() == 10
    assert not frame["is_success"][0]


def test_manual_outcome_applies_to_episode_and_marks_last_frame():
    dataset = SimpleNamespace(writer=SimpleNamespace(episode_buffer={"size": 3}))
    record.label_episode(dataset, False)
    buffer = record.episode_buffer(dataset)
    assert [bool(x[0]) for x in buffer["is_success"]] == [False, False, False]
    assert [bool(x[0]) for x in buffer["done"]] == [False, False, True]


def test_finalize_flushes_saved_episodes_even_when_discard_fails():
    calls = []

    class ExternalDataset:
        def clear_episode_buffer(self):
            calls.append("clear")
            raise RuntimeError("image cleanup failed")

        def finalize(self):
            calls.append("finalize")

    with pytest.raises(RuntimeError, match="image cleanup"):
        record.finalize_dataset(ExternalDataset())
    assert calls == ["clear", "finalize"]


def test_discard_waits_for_operator_before_rerecording():
    calls = []
    events = iter([set(), {"discard"}, {"finish"}])
    dataset = SimpleNamespace(clear_episode_buffer=lambda: calls.append("clear"))
    controls = SimpleNamespace(poll=lambda: next(events))
    assert record.discard_and_reset(dataset, controls, 1) is False
    assert calls == ["clear"]


@pytest.mark.parametrize("mode", ["joint", "eef_joint"])
def test_lerobot_roundtrip_discard_resume_and_outcomes(tmp_path, mode):
    lerobot = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    config = load_camera_config()
    for settings in config.values():
        settings.update(width=64, height=64)
    features = record.dataset_features(mode, config, True)
    args = Namespace(
        dataset_root=str(tmp_path / "dataset"),
        repo_id="local/piper_toolkit_test",
        fps=30,
        resume=False,
        label_outcomes=True,
    )
    dataset = record.open_dataset(args, features)
    size = len(record.recording_names(mode))

    def add(offset):
        for i in range(4):
            previous = {
                "observation.state": np.arange(size, dtype=np.float32) + offset + i
            }
            following = {"observation.state": previous["observation.state"] + 1}
            for name in ("front", "left", "wrist"):
                previous[f"observation.images.{name}"] = np.full(
                    (64, 64, 3), [20, 80, 140], dtype=np.uint8
                )
            dataset.add_frame(
                record.transition_frame(
                    previous,
                    following,
                    "Pick up the blue workpiece and put it into the box.",
                    True,
                )
            )

    try:
        add(0)
        record.label_episode(dataset, True)
        record.commit_episode(dataset)
        add(100)
        dataset.clear_episode_buffer()
        add(200)
        record.label_episode(dataset, False)
        record.commit_episode(dataset)
    finally:
        record.finalize_dataset(dataset)
    args.resume = True
    dataset = record.open_dataset(args, features)
    try:
        add(300)
        record.label_episode(dataset, True)
        record.commit_episode(dataset)
    finally:
        record.finalize_dataset(dataset)
    root = Path(args.dataset_root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    assert (info["total_episodes"], info["total_frames"]) == (3, 12)
    table = pa.concat_tables(
        [pq.read_table(path) for path in sorted((root / "data").rglob("*.parquet"))]
    )
    states = np.asarray(table["observation.state"].to_pylist())
    np.testing.assert_array_equal(np.asarray(table["action"].to_pylist()), states + 1)
    assert list(states[:, 0]) == [0, 1, 2, 3, 200, 201, 202, 203, 300, 301, 302, 303]
    assert table["is_success"].to_pylist() == [True] * 4 + [False] * 4 + [True] * 4
    reader = lerobot.LeRobotDataset(
        repo_id=args.repo_id, root=root, video_backend="pyav"
    )
    frame = reader[0]
    assert tuple(frame["observation.state"].shape) == (size,)
    for name in ("front", "left", "wrist"):
        assert tuple(frame[f"observation.images.{name}"].shape) == (3, 64, 64)


def test_camera_file_and_json_share_configuration():
    path = Path(record.__file__).with_name("cameras.json")
    assert (
        load_camera_config(str(path))
        == load_camera_config(path.read_text(encoding="utf-8"))
        == load_camera_config()
    )
    assert load_camera_config()["left"]["fourcc"] == "YUYV"


def test_keyboard_handles_split_arrows_and_ignores_vertical_arrows():
    controls = RecordingControls()
    assert controls.feed(b"\x1b[") == set()
    assert controls.feed(b"C") == {"finish"}
    assert controls.feed(b"\x1b[D") == {"discard"}
    assert controls.feed(b"\x1b[A\x1b[B") == set()
    assert controls.feed(b"sf") == {"success", "failure"}


def test_publisher_interpolates_joints_without_interpolating_gripper(hardware):
    arm = make_arm("can0")
    arm.ConnectPort(piper_init=False)
    initial = np.zeros(7, dtype=np.float32)
    clock = [0.0]
    publisher = JointPublisher(arm, initial, fps=30, clock=lambda: clock[0])
    actions = np.full((3, 7), 10000, dtype=np.float32)
    publisher.install_prediction(actions, 0.0, blend_steps=0)
    clock[0] = 1 / 60
    publisher.send_once()
    joint = [x for x in arm.commands if x[0] == "joint"][-1]
    grip = [x for x in arm.commands if x[0] == "gripper"][-1]
    assert joint[1:] == (5000,) * 6
    assert grip[1] == 0


def test_rtc_prefix_uses_future_model_times():
    trajectory = TimedTrajectory.prediction(
        np.arange(28, dtype=np.float32).reshape(4, 7), 0, 30
    )
    prefix = trajectory_prefix(trajectory, 1 / 30, 30)
    assert prefix.shape == (3, 7)
    np.testing.assert_array_equal(prefix[0], np.arange(7, 14))


def test_native_rtc_preserves_zero_normalized_rows_and_exact_raw_frozen_prefix():
    torch = pytest.importorskip("torch")
    stats = {"min": torch.full((7,), -1000.0), "max": torch.full((7,), 1000.0)}
    pack = type("GrootN17PackInputsStep", (), {})()
    pack.raw_stats = {}
    pack.modality_config = {}
    pack.normalize_min_max = True
    pack.clip_outliers = True
    pack.stats = {"action": stats}

    class GrootActionUnpackUnnormalizeStep:
        normalize_min_max = True
        libero_gripper_action = False

        def __init__(self):
            self.stats = {"action": stats}

        def __call__(self, value):
            return (value + 1) / 2 * (stats["max"] - stats["min"]) + stats["min"]

    decode = GrootActionUnpackUnnormalizeStep()
    options = []

    class NativeModel:
        def get_action(self, inputs, options):
            assert inputs["action"].shape == (1, 3, 132)
            assert torch.count_nonzero(inputs["action"][0, 0]) == 0
            options_seen.append(options)
            return {"action_pred": torch.full((1, 40, 132), 0.25)}

    options_seen = options
    policy = SimpleNamespace(
        config=SimpleNamespace(
            use_relative_actions=False,
            max_action_dim=132,
            chunk_size=40,
            use_bf16=False,
        ),
        _groot_model=NativeModel(),
        _filter_groot_inputs=lambda *args, **kwargs: {
            "state": torch.zeros((1, 1, 132))
        },
    )
    rtc = NativeGrootRTC(
        policy,
        SimpleNamespace(steps=[pack]),
        SimpleNamespace(steps=[decode], __call__=decode),
        torch,
    )
    rtc.post = decode
    raw = np.array([[0] * 7, [123.456] * 7, [200] * 7], dtype=np.float32)
    restored, metrics = rtc.predict(
        {}, {"prefix_actions": raw, "inference_delay_steps": 2}
    )
    np.testing.assert_array_equal(restored[0, :2].numpy(), raw[:2])
    assert options[0]["action_horizon"] == 3
    assert metrics["frozen_steps"] == 2
