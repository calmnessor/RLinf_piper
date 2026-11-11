# Piper 真机采集与 GR00T 部署

在连接 Piper 的 Linux 电脑上，从 RLinf 仓库运行示范采集和 GR00T N1.7 绝对关节部署。采集支持原有 16 维 `eef_joint` 格式及 7 维 `joint` 格式；部署支持已训练的 LeRobot GR00T N1.7 三相机、7 维 checkpoint，并保留异步预取、100 Hz CAN 发布和 Native RTC。

这些是仓库内独立运行的硬件工具。采集结果是 LeRobot 数据集，模型服务与机械臂 bridge 分进程运行。RECAP 的 GR00T CFG 训练、RLinf EnvWorker 和自动成功判断仍需后续接入。

## 准备运行环境

将本目录和上一级的 `toolkits/__init__.py` 放进机械臂电脑上的 RLinf checkout，后续命令均在该 checkout 根目录执行。Windows 上可以编辑代码和运行 CPU 测试；SocketCAN 和相机采集在连接硬件的 Linux 电脑运行。

现有 203 电脑可继续使用以下环境，不需要覆盖已工作的环境：

| 进程 | 默认 Python | 依赖 |
|---|---|---|
| 采集、bridge | `$HOME/miniconda3/envs/lerobot_piper/bin/python` | 当前 LeRobot、NumPy、OpenCV、`piper_sdk.C_PiperInterface_V2`、PyAV、视频编码器 |
| GR00T server | `$HOME/workspace/lerobot-groot17-inference/.venv/bin/python` | 当前 LeRobot N1.7 实现、PyTorch、Accelerate、safetensors |

用 `PYTHON_BIN` 覆盖某个脚本的解释器。模型服务的 `RUNTIME_ROOT` 指向已有 GR00T runtime，里面要有 `src`、`assets/base_model/config.json` 和 `assets/Cosmos-Reason2-2B`。这是当前 checkpoint 配套的 LeRobot N1.7 实现；RLinf 原生 NVIDIA GR00T 安装环境不能直接替代它。

脚本调用本目录的 Python 模块，不调用旧 `piper-lerobot-vla-client` 目录的脚本。独立工具进程使用自己的日志，不启动 Ray。

先检查 CAN 的现有状态：

```bash
ip -details link show can0
```

应看到接口 UP、1 Mbit/s，并有机械臂反馈。如果需要重新配置，在没有采集或执行进程使用该接口时运行：

```bash
CAN=can0 bash toolkits/piper_workpiece/setup_can_single.sh
```

该命令重新配置 SocketCAN；不会给机械臂发送位置目标。它会暂时将 CAN 接口关闭再开启。

## 配置三路相机

编辑 `toolkits/piper_workpiece/cameras.json`，使 `front`、`left`、`wrist` 对应训练时的物理相机。文件已带入 203 当前设备路径：front/wrist 使用 MJPG，left 的 RealSense RGB 使用 YUYV，均为 640×480、30 FPS、RGB 输出。

用 `CAMERAS_FILE=/path/to/cameras.json` 选择其他配置文件，或用 `CAMERAS='{"front": ...}'` 传入完整 JSON。采集和部署共用这个配置；相机字段、顺序和图像处理应与 checkpoint 一致。

## 采集示范

先只检查字段和相机配置：

```bash
bash toolkits/piper_workpiece/record_single_piper.sh --describe
```

该命令不连接 CAN、相机或创建数据集。开始实际采集时，由原来的外部主臂/主从链路控制从臂；采集程序只读取从臂反馈，不使能、不复位、不下发动作。

```bash
DATASET_ROOT=/home/pai/Data/Industrial_sorting/rlinf_recordings/workpiece_blue_01 \
COLOR=blue RECORDING_MODE=eef_joint EPISODES=10 \
bash toolkits/piper_workpiece/record_single_piper.sh
```

这会新建 10 条示范，默认 30 FPS、每条最长 300 秒、场景恢复等待 200 秒。`TASK` 可覆盖完整语言指令；`COLOR=red` 或 `yellow` 可切换原来的另外两个任务。`DATASET_ROOT` 必须是新目录；续录时使用同一目录并设置 `RESUME=true`。续录会检查 FPS、特征维度、字段顺序和结果标签配置。

| 控制 | 当前录制中 | 录制结束后的场景恢复阶段 |
|---|---|---|
| `→` | 结束当前条 | 保存当前条并开始下一条 |
| `←` | 丢弃当前条 | 丢弃尚未保存的当前条 |
| Esc / Ctrl+C | 丢弃未保存部分并退出 | 丢弃未保存部分并退出 |

计时结束也会结束当前阶段。丢弃后会再次等待场景恢复，按 `→` 开始重录，或等待 `RESET_TIME` 结束。已经保存的 episode 会保留并 finalize；`←` 不删除已提交到磁盘的上一条。使用 SSH 时保持交互式 TTY，例如 `ssh -t`，并让采集终端获得键盘焦点。

`eef_joint` 的顺序为 XYZ（米）、交错排列的旋转矩阵前两列（6 维）、夹爪、六关节，共 16 维。`joint` 为六关节、夹爪，共 7 维。关节保持 SDK 的 `0.001 degree`，夹爪保持 `0.001 mm`。`eef` 可保存前 10 维。示范标签始终是 `observation[t] → measured absolute state[t+1]`，不会改成增量动作。

为了后续使用真实任务结果，另建带结果标签的数据集：

```bash
DATASET_ROOT=/home/pai/Data/Industrial_sorting/rlinf_recordings/workpiece_blue_labeled_01 \
RECORDING_MODE=joint LABEL_OUTCOMES=true \
bash toolkits/piper_workpiece/record_single_piper.sh
```

录制中按 `s` 或 `f` 可以结束当前条并标为成功或失败；按 `→` 结束后也可以再选择 `s` / `f`。标签写入每帧 `is_success`，最后一帧 `done=true`。标签由人判断，不根据文件名自动推断。未开启标签时保留原来的特征结构；同一数据集的续录不能切换标签模式。

后续将含失败的数据交给 RECAP 时，使用读取 `is_success` 的非 SFT 数据类别。当前 RECAP return 脚本会将 `type: sft` 数据全部视为成功；仅保存标签还不能替代正确的数据配置。

数据集位于 `DATASET_ROOT`，包括 `meta/info.json`、Parquet 和三路视频。`meta/piper_recording.json` 记录单位、特征和相机配置，`meta/piper_episode_notes.jsonl` 记录实际采样时间及人工结果；这有助于发现相机读取造成的录制降帧。标称 FPS 仍为设定值。

## 部署已训练的 GR00T

先在模型服务终端加载 checkpoint。默认路径是 203 上的 480 条数据模型，可以按下面方式显式指定：

```bash
POLICY_PATH=/home/pai/workspace/lerobot/checkpoint/workpiece_groot17_480ep_3cam_joint7/pretrained_model \
PORT=5566 bash toolkits/piper_workpiece/start_server_workpiece_groot.sh
```

服务默认使用 BF16 参数，保留 checkpoint 保存的归一化和 embodiment ID，输出 40×7 维原始 SDK 绝对目标。等待日志出现 `[server] listening`。端口已有服务时，可以直接连接对应的兼容服务，或使用另一个空闲端口；脚本不会停止现有进程。

在另一个终端先检查图像、状态和推理输出：

```bash
PORT=5566 COLOR=blue bash toolkits/piper_workpiece/run_bridge_workpiece_groot.sh safe
```

`safe` 不发送 CAN 初始化查询、使能或动作指令。设置 `DRY_RUN_STEPS=120` 可在同样只读的条件下检查异步时间线。实际执行前，让部署 bridge 独占从臂的运动控制权，停止外部主从运动发送端。

```bash
PORT=5566 COLOR=blue SPEED_PCT=50 MAX_STEPS=700 \
bash toolkits/piper_workpiece/run_bridge_workpiece_groot.sh real
```

真实执行从实测关节开始，六关节按时间线插值，夹爪采用零阶保持。默认动作时间线为 30 Hz，CAN 发布 100 Hz，提前 15 步异步推理，衔接平滑 4 步，`RTC=1`。`SPEED_PCT` 调整固件速度百分比，不自动改变 30 Hz 时间线。`MAX_STEPS=700` 约 23.3 秒，包含推理迟到时的保持时间；0 表示持续执行，直到 Ctrl+C。

正常结束、Ctrl+C 或 Python 推理异常后，bridge 请求保持退出时的实测关节和最后夹爪目标，保留使能，再关闭资源。这依赖 CAN 和伺服仍可工作；断电或进程被强制杀死时无法执行这段清理。

`RTC=0` 可关闭 Native RTC，`BLEND_STEPS=0` 可关闭衔接平滑。两端 `PORT`、`RTC` 设置需相匹配。日志和 100 Hz 目标/反馈 CSV 默认放在 checkout 的 `logs/piper_workpiece`。部署 CSV 是控制诊断记录；后续完整自主 rollout 记录还需要同步图像和任务结果。

## 检查代码

在仓库根目录运行 CPU 测试：

```bash
python -m pytest tests/unit_tests/test_piper_workpiece.py -q
```

测试使用假的硬件 SDK 和模型服务，检查采集语义、撤销/结果标记、只读模式、插值、RTC 和退出保持。真实相机、GPU 模型加载及真机动作需要在目标电脑另行验证。
