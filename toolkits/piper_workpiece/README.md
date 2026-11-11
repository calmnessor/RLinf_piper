# Piper Collection and GR00T Deployment

Run Piper demonstration collection and absolute-joint GR00T N1.7 deployment from an RLinf checkout on the Linux robot computer. Record the existing 16-dimensional `eef_joint` representation or 7-dimensional `joint` representation. Deploy a trained LeRobot GR00T N1.7 checkpoint with three cameras, asynchronous prefetch, 100 Hz CAN publication, and Native RTC.

These are standalone hardware tools within the repository. They write LeRobot datasets and run inference and robot control in separate processes. GR00T CFG training for RECAP, EnvWorker integration, and automatic success detection require further integration. Read the [Chinese instructions](README.zh-CN.md) for the same workflow.

## Prepare the Runtime

Copy this directory and its parent `toolkits/__init__.py` into the RLinf checkout on the robot computer. Run the following commands from the checkout root. Windows supports editing and CPU tests; SocketCAN and camera capture run on the connected Linux computer.

Reuse the existing environments on computer 203:

| Process | Default Python | Dependencies |
|---|---|---|
| Recording and bridge | `$HOME/miniconda3/envs/lerobot_piper/bin/python` | The current LeRobot runtime, NumPy, OpenCV, `piper_sdk.C_PiperInterface_V2`, PyAV, video encoders |
| GR00T server | `$HOME/workspace/lerobot-groot17-inference/.venv/bin/python` | The current LeRobot N1.7 implementation, PyTorch, Accelerate, safetensors |

Override `PYTHON_BIN` per script. Set `RUNTIME_ROOT` to the GR00T runtime containing `src`, `assets/base_model/config.json`, and `assets/Cosmos-Reason2-2B`. This checkpoint uses that LeRobot N1.7 implementation; the native NVIDIA GR00T environment installed by RLinf has a different loading contract.

The scripts import this toolkit's modules instead of scripts from `piper-lerobot-vla-client`. They own their logging and run without starting Ray.

Inspect the existing CAN interface:

```bash
ip -details link show can0
```

Expect an UP interface at 1 Mbit/s with robot feedback. When reconfiguration is needed, stop processes using that interface first, then run:

```bash
CAN=can0 bash toolkits/piper_workpiece/setup_can_single.sh
```

This reconfigures SocketCAN without sending robot position targets. It temporarily takes the interface down and brings it back up.

## Configure Three Cameras

Edit `toolkits/piper_workpiece/cameras.json` to map `front`, `left`, and `wrist` to the physical cameras used in training. The supplied file contains computer 203's device paths: front/wrist use MJPG, and the left RealSense RGB device uses YUYV. All three capture 640×480 RGB at 30 FPS.

Use `CAMERAS_FILE=/path/to/cameras.json` for another configuration or `CAMERAS='{"front": ...}'` for complete inline JSON. Recording and deployment share this configuration. Preserve the checkpoint's view identities, order, and image processing.

## Record Demonstrations

Inspect the recording schema before connecting hardware:

```bash
bash toolkits/piper_workpiece/record_single_piper.sh --describe
```

This displays the schema without opening CAN, cameras, or a dataset. During recording, the existing external leader/follower connection drives the arm. The recorder reads follower feedback without enabling motors, resetting, or sending targets.

```bash
DATASET_ROOT=/home/pai/Data/Industrial_sorting/rlinf_recordings/workpiece_blue_01 \
COLOR=blue RECORDING_MODE=eef_joint EPISODES=10 \
bash toolkits/piper_workpiece/record_single_piper.sh
```

This creates ten demonstrations at 30 FPS, with a maximum episode duration of 300 seconds and a scene-reset wait of 200 seconds. Override the complete instruction with `TASK`; select the other tasks with `COLOR=red` or `yellow`. Use a new `DATASET_ROOT`, or set `RESUME=true` to append to a compatible dataset. Resume validates FPS, dimensions, feature order, and outcome-label settings.

| Key | While recording | During the scene-reset wait |
|---|---|---|
| Right arrow | Finish the current recording | Save it and begin the next episode |
| Left arrow | Discard the current recording | Discard the pending recording |
| Escape / Ctrl+C | Discard pending data and exit | Discard pending data and exit |

Each stage also ends when its timer expires. After discarding, restore the scene and press Right arrow to rerecord, or wait for `RESET_TIME` to expire. Saved episodes are retained and finalized. Left arrow does not delete a previously committed episode. Keep an interactive terminal focused; use `ssh -t` when connecting through SSH.

`eef_joint` orders XYZ in metres, the interleaved first two rotation-matrix columns, the gripper, and six joints: 16 values. `joint` orders six joints followed by the gripper: seven values. Joint and gripper units remain the SDK's `0.001 degree` and `0.001 mm`. `eef` keeps the first ten values. Every demonstration label pairs `observation[t]` with the measured absolute `state[t+1]`; actions remain absolute.

To record human-verified outcomes for later learning, create a labeled dataset:

```bash
DATASET_ROOT=/home/pai/Data/Industrial_sorting/rlinf_recordings/workpiece_blue_labeled_01 \
RECORDING_MODE=joint LABEL_OUTCOMES=true \
bash toolkits/piper_workpiece/record_single_piper.sh
```

Press `s` or `f` while recording to finish and label success or failure. After finishing with Right arrow, choose `s` or `f` at the outcome prompt. `is_success` is stored on every frame, and `done=true` marks the last frame. The operator supplies the result; filenames do not determine success. Unlabeled recording preserves the original feature structure. Keep the label setting unchanged when resuming.

When using a dataset containing failures with RECAP, select a non-SFT data category that reads `is_success`. The current return script treats every `type: sft` episode as successful; saved labels must be paired with the correct data configuration.

The dataset contains `meta/info.json`, Parquet data, and three video streams. `meta/piper_recording.json` records units, features, and camera settings. `meta/piper_episode_notes.jsonl` records actual capture times and operator outcomes to help identify capture delays. The dataset's nominal FPS remains the configured rate.

## Deploy a Trained GR00T Checkpoint

Start the model process in one terminal. The default checkpoint is the 480-episode model on computer 203; you can select it explicitly:

```bash
POLICY_PATH=/home/pai/workspace/lerobot/checkpoint/workpiece_groot17_480ep_3cam_joint7/pretrained_model \
PORT=5566 bash toolkits/piper_workpiece/start_server_workpiece_groot.sh
```

The server defaults to BF16 parameters, preserves the saved normalization and embodiment ID, and returns 40×7 absolute SDK targets. Wait for `[server] listening`. Connect to an existing compatible server or select another free port when needed. The launcher leaves existing processes running.

In another terminal, check cameras, feedback, and inference:

```bash
PORT=5566 COLOR=blue bash toolkits/piper_workpiece/run_bridge_workpiece_groot.sh safe
```

`safe` sends no CAN initialization queries, enables, or motion commands. Add `DRY_RUN_STEPS=120` to exercise asynchronous playback with CAN still read-only. Before real execution, give the bridge exclusive ownership of follower motion and stop the external leader/follower command sender.

```bash
PORT=5566 COLOR=blue SPEED_PCT=50 MAX_STEPS=700 \
bash toolkits/piper_workpiece/run_bridge_workpiece_groot.sh real
```

Execution starts from measured joints. Joint targets interpolate on a 30 Hz model timeline published at 100 Hz; the gripper uses zero-order hold. Defaults are 15-step asynchronous prefetch, four-step transition blending, and `RTC=1`. `SPEED_PCT` sets the firmware speed percentage without rescaling the model timeline. `MAX_STEPS=700` is approximately 23.3 seconds, including holds during late inference; zero runs until Ctrl+C.

On completion, Ctrl+C, or a Python inference exception, the bridge requests a hold at measured joints with the last gripper target, retains motor enable, and closes resources. This cleanup requires working CAN and servos; power loss or a forced process kill prevents it.

Set `RTC=0` to disable Native RTC or `BLEND_STEPS=0` to disable transition blending. Match the server and bridge `PORT` and `RTC` settings. Logs and 100 Hz target/feedback CSV files default to `logs/piper_workpiece` in the checkout. The CSV is a control diagnostic; complete autonomous rollout recording also needs synchronized images and task outcomes.

## Check the Tools

Run the CPU tests from the checkout root:

```bash
python -m pytest tests/unit_tests/test_piper_workpiece.py -q
```

The tests use fake hardware SDKs and a model service to check collection semantics, discard/outcome handling, read-only operation, interpolation, RTC, and exit holding. Validate physical cameras, GPU checkpoint loading, and motion separately on the target computer.
