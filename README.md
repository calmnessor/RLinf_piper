# RLinf_piper

基于 [RLinf](https://github.com/RLinf/RLinf) 的 Piper 单臂真机项目，加入示范数据采集和 LeRobot GR00T N1.7 绝对关节部署工具，作为后续真机强化学习实验的基础。

当前新增功能集中在 [`toolkits/piper_workpiece`](toolkits/piper_workpiece)。采集端支持末端位姿与关节角联合记录，部署端保留异步推理、Native GR00T RTC、关节插值和退出保持。原有 RLinf 训练代码、文档和 Apache-2.0 许可证保留。

## 当前进度

| 能力 | 状态 | 说明 |
|---|---|---|
| Piper 真机示范采集 | 已实现 | 外部主臂遥操作，从臂只读记录，支持丢弃、重录和续录 |
| 三相机数据记录 | 已实现 | `front`、`left`、`wrist`，默认 640×480、30 FPS |
| 16 维联合数据 | 已实现 | XYZ、旋转 6D、夹爪、六关节 |
| 7 维绝对关节数据 | 已实现 | 六关节与夹爪，保持 SDK 原始单位 |
| 人工成功/失败标记 | 已实现 | 可选 `is_success` 与末帧 `done` |
| 本地 GR00T N1.7 checkpoint 部署 | 已实现 | 适配当前 LeRobot 格式、三相机、7 维 checkpoint |
| 异步推理、插值和 Native RTC | 已实现 | 30 Hz 动作时间线、100 Hz CAN 发布 |
| RLinf 原生 EnvWorker / 训练 actor 接入 | 待适配 | 当前新增工具以独立进程运行 |
| 项目专用 GR00T SFT、云端部署和 DAgger | 后续计划 | 上游通用代码保留，尚未接成此项目的完整流程 |
| GR00T + RECAP 训练闭环 | 后续计划 | 需要补齐经验记录、价值模型及 CFG 策略训练与推理 |

## 目录

```text
toolkits/piper_workpiece/
├── record_single_piper.sh          # 采集入口
├── setup_can_single.sh             # SocketCAN 配置
├── start_server_workpiece_groot.sh # GR00T 模型服务
├── run_bridge_workpiece_groot.sh   # safe / real 执行入口
├── cameras.json                    # 三路相机配置
├── record.py                       # LeRobot 数据写入与录制控制
├── controller.py                   # CAN 发布、反馈和相机管理
├── trajectory.py                   # 时间线、插值与异步预取
├── groot_model.py                  # checkpoint 严格加载
├── groot_server.py                 # 模型请求处理
├── groot_bridge.py                 # 真机执行流程与退出保持
├── groot_rtc.py                    # Native GR00T RTC
└── README.zh-CN.md                 # 完整操作说明

tests/unit_tests/test_piper_workpiece.py
tests/robot_mocks/piper_workpiece.py
```

## 运行环境

代码可在 Windows 上编辑和运行 CPU 测试。连接机械臂、SocketCAN 和摄像头的进程在 Linux 电脑运行。

新增工具复用现有的两个环境：

| 进程 | 默认位置 | 主要依赖 |
|---|---|---|
| 采集与 bridge | `$HOME/miniconda3/envs/lerobot_piper/bin/python` | 当前 LeRobot、NumPy、OpenCV、`piper_sdk.C_PiperInterface_V2`、PyAV、视频编码器 |
| GR00T 模型服务 | `$HOME/workspace/lerobot-groot17-inference/.venv/bin/python` | 配套 LeRobot N1.7 实现、PyTorch、Accelerate、safetensors |

使用 `PYTHON_BIN` 覆盖解释器路径。模型服务的 `RUNTIME_ROOT` 目录需要包含：

```text
lerobot-groot17-inference/
├── .venv/bin/python
├── src/
└── assets/
    ├── base_model/config.json
    └── Cosmos-Reason2-2B/
```

**当前部署适配的是配套 LeRobot N1.7 runtime。** 上游 RLinf 使用的 NVIDIA 官方 GR00T 模型加载接口具有不同的 checkpoint 与 processor 契约。

数据集、模型权重和完整运行环境在本机准备，Git 仓库保存源代码与配置。仓库中的解释器、checkpoint 和相机默认值对应开发环境，换机器时请通过环境变量和 `cameras.json` 调整。

## 快速开始

### 1. 获取代码与配置硬件

```bash
git clone --branch main --single-branch https://github.com/calmnessor/RLinf_piper.git
cd RLinf_piper
ip -details link show can0
```

确认 CAN 接口为 UP、比特率为 1 Mbit/s，并能收到机械臂反馈。需要重新配置时，先停止使用该接口的进程，再运行：

```bash
CAN=can0 bash toolkits/piper_workpiece/setup_can_single.sh
```

该命令会关闭并重新启用 CAN 接口，不下发机械臂位置目标。

编辑 [`cameras.json`](toolkits/piper_workpiece/cameras.json)，保持 `front`、`left`、`wrist` 与训练视角一致。默认 front/wrist 使用 MJPG，left 的 RealSense RGB 使用 YUYV。也可通过 `CAMERAS_FILE=/path/to/cameras.json` 指定配置。

### 2. 检查采集配置

```bash
bash toolkits/piper_workpiece/record_single_piper.sh --describe
```

该命令只显示特征结构和相机配置，不连接 CAN、相机或创建数据集。

### 3. 采集真机示范

由外部主臂或现有主从链路控制从臂，采集进程只读取反馈，不使能、不复位、不发送运动指令。

```bash
DATASET_ROOT="$HOME/Data/Industrial_sorting/rlinf_recordings/workpiece_blue_01" \
COLOR=blue RECORDING_MODE=eef_joint EPISODES=10 \
bash toolkits/piper_workpiece/record_single_piper.sh
```

默认 30 FPS、每条最长 300 秒，场景恢复等待 200 秒。`COLOR=red` / `yellow` 切换任务，`TASK` 覆盖完整语言指令。

| 参数 | 含义 |
|---|---|
| `RECORDING_MODE=eef_joint` | 16 维：XYZ、旋转 6D、夹爪、六关节 |
| `RECORDING_MODE=joint` | 7 维：六关节、夹爪 |
| `RECORDING_MODE=eef` | 10 维：XYZ、旋转 6D、夹爪 |
| `RESUME=true` | 在兼容的已有数据集上续录 |
| `LABEL_OUTCOMES=true` | 加入人工成功/失败标记 |

关节保持 SDK 的 `0.001 degree`，夹爪保持 `0.001 mm`；XYZ 为米，旋转使用交错排列的旋转矩阵前两列。示范标签为 **当前观测 → 下一时刻实测绝对状态**。

录制中按 `→` 结束当前条；在场景恢复阶段再按 `→` 保存并继续。按 `←` 丢弃尚未保存的当前条，恢复场景后按 `→` 重录。Esc / Ctrl+C 退出，未保存部分丢弃，已保存 episode 会 finalize。SSH 连接请使用交互式 TTY。

如需结果标签，在新数据集上启用：

```bash
DATASET_ROOT="$HOME/Data/Industrial_sorting/rlinf_recordings/workpiece_blue_labeled_01" \
RECORDING_MODE=joint LABEL_OUTCOMES=true \
bash toolkits/piper_workpiece/record_single_piper.sh
```

用 `s` / `f` 标记成功或失败，结果写入每帧 `is_success`，末帧 `done=true`。后续 RECAP 对含失败的数据应使用读取这些标签的非 SFT 数据类别；当前 return 脚本会将 `type: sft` 全部视为成功。

### 4. 启动 GR00T 模型服务

在终端 1 指定已训练的 LeRobot checkpoint：

```bash
POLICY_PATH="$HOME/workspace/lerobot/checkpoint/workpiece_groot17_480ep_3cam_joint7/pretrained_model" \
PORT=5566 bash toolkits/piper_workpiece/start_server_workpiece_groot.sh
```

默认使用 BF16 参数，并保留保存的归一化和 embodiment ID。服务返回 40×7 维原始 SDK 绝对关节/夹爪目标。等待日志出现 `[server] listening` 后运行 bridge。

### 5. 验证并执行

在终端 2 先检查状态、图像和推理输出：

```bash
PORT=5566 COLOR=blue bash toolkits/piper_workpiece/run_bridge_workpiece_groot.sh safe
```

`safe` 保持 CAN 只读。加入 `DRY_RUN_STEPS=120` 可检查异步时间线，仍不发送运动命令。

真机执行前，停止外部主从运动发送端，让 bridge 独占从臂运动控制：

```bash
PORT=5566 COLOR=blue SPEED_PCT=50 MAX_STEPS=700 \
bash toolkits/piper_workpiece/run_bridge_workpiece_groot.sh real
```

默认执行参数：

| 参数 | 默认值 | 行为 |
|---|---:|---|
| `FPS` | 30 | 模型动作时间线频率 |
| `CAN_HZ` | 100 | CAN 发布频率 |
| `PREFETCH_STEPS` | 15 | 提前异步请求下一块动作 |
| `BLEND_STEPS` | 4 | 关节衔接平滑步数，0 关闭 |
| `RTC` | 1 | Native GR00T overlap / frozen-prefix |
| `MAX_STEPS` | 700 | 约 23.3 秒，包含迟到推理期间的保持；0 持续运行 |

六关节按时间插值，夹爪采用零阶保持。`SPEED_PCT` 调整固件速度百分比，不自动缩放 30 Hz 时间线。正常结束、Ctrl+C 或 Python 推理异常后，bridge 请求保持实测关节和最后夹爪目标，保留使能，再关闭资源；这段清理依赖 CAN 与伺服正常工作。

## 数据与日志

采集数据保存在 `DATASET_ROOT`，包含 LeRobot 元数据、Parquet 与三路视频。`meta/piper_recording.json` 保存单位和配置，`meta/piper_episode_notes.jsonl` 保存实际采样时间及人工结果。

日志与 100 Hz 目标/反馈 CSV 默认位于 `logs/piper_workpiece`。部署 CSV 用于诊断控制；完整自主 rollout 还需要同步图像、任务结果和接管记录。

## 验证情况

```bash
python -m pytest tests/unit_tests/test_piper_workpiece.py -q
```

截至 2026-11-11，机械臂电脑现有 Python 环境通过 **16 项 CPU 测试**，覆盖 7/16 维数据写入、丢弃、续录、三路视频解码、结果标签、只读部署、插值、RTC 固定前缀和退出保持。部署生命周期测试使用模拟 SDK 与模型服务，数据读写测试使用实际 LeRobot。

真实相机、GPU checkpoint 加载和机械臂动作仍需在目标环境验证。完整参数与录制控制见 [操作说明](toolkits/piper_workpiece/README.zh-CN.md)。

## 后续方向

后续将独立工具接入 Piper 工件任务环境与 RLinf EnvWorker，再完善自主经验记录和人工接管。在此基础上评估 DAgger，以及 GR00T 的 advantage 条件训练、CFG 推理和 RECAP 闭环。

## 来源与许可证

框架来源：[RLinf/RLinf](https://github.com/RLinf/RLinf)，本地基线提交为 [`58d1b9ab`](https://github.com/RLinf/RLinf/commit/58d1b9abd6c65dd2dce7fdf57afedfdd9273ef9c)。本仓库保留上游提交历史；Piper 工具是本项目新增部分。

上游框架的完整介绍与引用信息见 [原始 README](https://github.com/RLinf/RLinf/blob/58d1b9abd6c65dd2dce7fdf57afedfdd9273ef9c/README.md)。许可证为 [Apache-2.0](LICENSE)；模型、数据与第三方 SDK 遵循各自的许可证。
