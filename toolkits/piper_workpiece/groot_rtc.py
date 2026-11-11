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

"""Time-aligned raw action prefixes and native GR00T N1.7 RTC decoding."""

import math
import time
from collections import deque

import numpy as np

from ._logging import get_logger
from .common import ACTION_MODE, action_chunk
from .transport import recv_obj, send_obj

logger = get_logger()


def trajectory_prefix(trajectory, observed_at, fps, horizon=40):
    """Sample the committed trajectory at the NEW observation's future action times."""
    count = max(
        0, min(horizon, math.floor((trajectory.end - observed_at) * fps + 1e-7))
    )
    if not count:
        return np.empty((0, 7), dtype=np.float32)
    return action_chunk(
        [trajectory.sample(observed_at + (i + 1) / fps) for i in range(count)]
    )


class RTCRequestClient:
    def __init__(self, enabled=True, delay_steps=0, fps=30):
        self.enabled = enabled
        self.fixed_delay_steps = delay_steps
        self.fps = fps
        # Cold bootstrap is excluded. Until warm timings exist, budget eight model steps.
        self.delays = deque([8 / fps], maxlen=4)

    def attach(self, obs, observed_at, publisher):
        obs["_rtc_observed_at"] = observed_at
        if self.enabled:
            with publisher.lock:
                prefix = trajectory_prefix(publisher.trajectory, observed_at, self.fps)
            if len(prefix):
                # Cover the control-thread poll and first CAN publication after arrival.
                delay = self.fixed_delay_steps or (
                    math.ceil(max(self.delays) * self.fps) + 1
                )
                obs["_rtc_request"] = {
                    "prefix_actions": prefix,
                    "inference_delay_steps": min(delay, len(prefix)),
                }
        return obs

    def __call__(self, connection, obs):
        rtc = obs.get("_rtc_request")
        send_obj(connection, {"type": "infer", "observation": obs, "rtc": rtc})
        response = recv_obj(connection)
        if not response.get("ok"):
            raise RuntimeError(f"Policy server: {response.get('error', response)}")
        if response.get("action_mode") != ACTION_MODE:
            raise ValueError("Server is not an absolute-joint server")
        metrics = response.get("rtc", {})
        if rtc is not None and not metrics.get("active"):
            raise RuntimeError(
                "RTC prefix was sent but the server did not apply native RTC"
            )
        if "_rtc_observed_at" in obs:
            latency = time.monotonic() - obs["_rtc_observed_at"]
            self.delays.append(latency)
            if self.enabled:
                logger.info(
                    "%s",
                    f"[rtc] active={metrics.get('active', False)} "
                    f"overlap={metrics.get('overlap_steps', 0)} "
                    f"frozen={metrics.get('frozen_steps', 0)} "
                    f"capture_to_result_ms={latency * 1000:.1f}",
                )
        return action_chunk(response["actions"])


class NativeGrootRTC:
    """Use the checkpoint's exact flat min/max representation and native velocity ramp.

    Construct native options explicitly: unlike the generic RTC adapter this preserves
    valid all-zero normalized prefix rows (the center of the dataset range).
    """

    def __init__(self, policy, pre, post, torch, ramp_rate=6.0):
        self.policy, self.post, self.torch = policy, post, torch
        self.ramp_rate = ramp_rate
        pack = next(
            s for s in pre.steps if type(s).__name__ == "GrootN17PackInputsStep"
        )
        decode = next(
            s
            for s in post.steps
            if type(s).__name__ == "GrootActionUnpackUnnormalizeStep"
        )
        if (
            policy.config.use_relative_actions
            or pack.raw_stats
            or pack.modality_config
            or not pack.normalize_min_max
            or not decode.normalize_min_max
            or decode.libero_gripper_action
        ):
            raise ValueError(
                "RTC adapter requires this checkpoint's flat absolute min/max representation"
            )
        self.minimum = torch.as_tensor(
            pack.stats["action"]["min"], dtype=torch.float32
        ).cpu()
        self.maximum = torch.as_tensor(
            pack.stats["action"]["max"], dtype=torch.float32
        ).cpu()
        if self.minimum.shape != (7,) or self.maximum.shape != (7,):
            raise ValueError("RTC action statistics must have seven dimensions")
        for name, expected in [("min", self.minimum), ("max", self.maximum)]:
            if not torch.equal(
                expected, torch.as_tensor(decode.stats["action"][name]).float().cpu()
            ):
                raise ValueError("RTC input and output normalization statistics differ")
        self.clip_outliers = pack.clip_outliers

    def normalize(self, raw, device):
        torch = self.torch
        value = torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)
        lo, hi = self.minimum.to(device), self.maximum.to(device)
        delta = hi - lo
        scaled = (
            2 * (value - lo) / torch.where(delta != 0, delta, torch.ones_like(delta))
            - 1
        )
        scaled = torch.where(delta != 0, scaled, torch.zeros_like(scaled))
        if self.clip_outliers:
            scaled = scaled.clamp(
                -1, 1
            )  # Same normalization as the saved preprocessor.
        return torch.nn.functional.pad(
            scaled, (0, self.policy.config.max_action_dim - 7)
        )

    def predict(self, processed, request=None):
        torch, policy = self.torch, self.policy
        metrics = {"active": False, "overlap_steps": 0, "frozen_steps": 0}
        if request is None:
            normalized = policy.predict_action_chunk(processed)
            return self.post(normalized.float()), metrics
        raw = action_chunk(request["prefix_actions"])
        if len(raw) > policy.config.chunk_size:
            raise ValueError("RTC prefix exceeds model horizon")
        delay = request["inference_delay_steps"]
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise ValueError("RTC inference_delay_steps must be a non-negative integer")
        frozen = min(delay, len(raw))
        inputs = policy._filter_groot_inputs(processed, include_action=False)
        inputs = dict(inputs)
        device = inputs["state"].device
        inputs["action"] = self.normalize(raw, device)
        options = {
            "action_horizon": len(raw),
            "rtc_overlap_steps": len(raw),
            "rtc_frozen_steps": frozen,
            "rtc_ramp_rate": self.ramp_rate,
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=policy.config.use_bf16,
        ):
            outputs = policy._groot_model.get_action(inputs, options=options)
        normalized = outputs["action_pred"][:, : policy.config.chunk_size, :7]
        restored = self.post(normalized.float())
        # BF16 denoising quantizes the fixed prefix internally. Keep the already
        # committed raw targets exact at the wire boundary; never re-quantize them.
        if frozen:
            restored = restored.clone()
            restored[:, :frozen, :] = torch.as_tensor(
                raw[:frozen], device=restored.device
            )
        metrics.update(
            active=True,
            overlap_steps=len(raw),
            frozen_steps=frozen,
            ramp_rate=self.ramp_rate,
        )
        return restored, metrics
