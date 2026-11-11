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

"""Time-aligned absolute-joint trajectories; no delta integration or IK.

Action i describes the state at observation_time + (i + 1) / fps.
Joint positions interpolate in time. Gripper commands remain discrete values.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np

from ._logging import get_logger
from .common import action_chunk, state_vector

logger = get_logger()


class TimedTrajectory:
    def __init__(self, times, values):
        self.times = np.array(times, dtype=np.float64, copy=True)
        self.values = np.array(action_chunk(values), dtype=np.float64, copy=True)
        if (
            self.times.shape != (len(self.values),)
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError(
                "Trajectory timestamps must be finite and strictly increasing"
            )
        self.times.flags.writeable = False
        self.values.flags.writeable = False

    @classmethod
    def hold(cls, value, at):
        return cls([at], [state_vector(value)])

    @classmethod
    def prediction(cls, actions, observed_at, fps):
        actions = action_chunk(actions)
        return cls(observed_at + (np.arange(len(actions)) + 1) / fps, actions)

    @property
    def end(self):
        return float(self.times[-1])

    def sample(self, at):
        target = np.empty(7, dtype=np.float64)
        for joint in range(6):
            target[joint] = np.interp(at, self.times, self.values[:, joint])
        # Zero-order hold: never average grip widths or execute a future grip early.
        index = max(
            0,
            min(len(self.times) - 1, np.searchsorted(self.times, at, side="right") - 1),
        )
        target[6] = self.values[index, 6]
        return target


def splice_trajectory(old, prediction, now, fps, blend_steps):
    """Keep the current command continuous; discard elapsed prediction nodes.

    Blending compares OLD and NEW predictions at the SAME future timestamps.
    A late result never replays its expired prefix or extends its original end.
    """
    future = prediction.times > now + 1e-9
    if not future.any():
        return None, {"discarded": len(prediction.times), "stale": True}
    times = prediction.times[future]
    new_values = prediction.values[future].copy()
    anchor = old.sample(now)
    raw_jump = prediction.sample(now)[:6] - anchor[:6]
    if blend_steps:
        for i, at in enumerate(times):
            fraction = min(1.0, max(0.0, (at - now) * fps / blend_steps))
            weight = fraction * fraction * (3.0 - 2.0 * fraction)
            new_values[i, :6] = (1 - weight) * old.sample(at)[:6] + weight * new_values[
                i, :6
            ]
    # The anchor also retains the old grip until the first FUTURE prediction node.
    result = TimedTrajectory(np.r_[now, times], np.vstack([anchor, new_values]))
    return result, {
        "discarded": int((~future).sum()),
        "stale": False,
        "raw_seam_deg": float(np.max(np.abs(raw_jump)) / 1000),
        "remaining_ms": (prediction.end - now) * 1000,
    }


@dataclass
class InferenceResult:
    actions: np.ndarray
    observed_at: float
    completed_at: float


class AsyncInference:
    """Exactly one outstanding request. Only this worker uses the TCP socket.

    Observation capture is also off the control thread. Closing the worker first
    interrupts socket I/O, then joins it before cameras or CAN are disconnected.
    """

    def __init__(self, connection, observe, infer, clock=time.monotonic):
        self.connection = connection
        self.observe = observe
        self.infer = infer
        self.clock = clock
        self.thread = None
        self.done = threading.Event()
        self.stopping = threading.Event()
        self.result = None
        self.error = None

    @property
    def busy(self):
        return self.thread is not None

    def submit(self):
        if self.busy or self.stopping.is_set():
            return False
        self.done.clear()
        self.result = None
        self.error = None
        self.thread = threading.Thread(
            target=self._work, name="joint-inference", daemon=True
        )
        self.thread.start()
        return True

    def _work(self):
        try:
            obs, observed_at = self.observe(self.stopping)
            if not self.stopping.is_set():
                actions = action_chunk(self.infer(self.connection, obs))
                self.result = InferenceResult(actions.copy(), observed_at, self.clock())
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()

    def poll(self):
        if not self.busy or not self.done.is_set():
            return None
        self.thread.join()
        self.thread = None
        if self.error is not None:
            raise RuntimeError(
                "Asynchronous observation/inference failed"
            ) from self.error
        return self.result

    def stop(self):
        self.stopping.set()
        if self.thread is not None:
            if self.thread.is_alive():
                import socket

                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                # Also release the descriptor: shutdown alone does not wake every
                # Windows socketpair/transport implementation's blocked recv.
                self.connection.close()
            self.thread.join()
            self.thread = None


def _report(message, **kwargs):
    logger.info("%s", message)


def execute_pipeline(
    publisher,
    inference,
    *,
    fps,
    max_steps,
    prefetch_steps,
    blend_steps,
    clock=time.monotonic,
    sleep=time.sleep,
    report=_report,
):
    """Run a fixed-rate trajectory timeline, with no control-thread inference.

    max_steps counts 30-Hz control intervals, including a hold if a request is
    exceptionally late. No burst of expired targets is sent to catch up.
    """
    start = publisher.started_at
    deadline = start + max_steps / fps if max_steps else float("inf")
    next_report = 30
    underflow = False
    waits = 0
    installed = 0
    stale = 0
    while True:
        now = clock()
        if publisher.error is not None:
            raise RuntimeError("CAN publisher failed") from publisher.error
        steps = int((now - start) * fps + 1e-7)
        if now >= deadline - 1e-9:
            break
        result = inference.poll()
        if result is not None:
            metrics = publisher.install_prediction(
                result.actions, result.observed_at, blend_steps=blend_steps
            )
            if metrics["stale"]:
                stale += 1
            else:
                installed += 1
            report(
                f"[async] latency_ms={(result.completed_at - result.observed_at) * 1000:.1f} "
                f"discarded={metrics['discarded']} stale={metrics['stale']} "
                f"raw_seam_deg={metrics.get('raw_seam_deg', 0):.3f} "
                f"remaining_ms={metrics.get('remaining_ms', 0):.1f}",
                flush=True,
            )
        now = clock()
        end = publisher.plan_end
        empty = now > end + 1e-9
        if empty and not underflow:
            waits += 1
            report(
                "[async] buffer exhausted; holding final target while inference finishes",
                flush=True,
            )
        underflow = empty
        if (
            not inference.busy
            and end - now <= prefetch_steps / fps
            and end < deadline - 1e-9
        ):
            inference.submit()
        if steps >= next_report:
            report(
                f"[pipeline] steps={steps} remaining_ms={max(0, end - now) * 1000:.1f} "
                f"last_raw={publisher.target.round(1).tolist()}",
                flush=True,
            )
            next_report = (steps // 30 + 1) * 30
        sleep(max(0, min(0.005, deadline - clock())))
    summary = {
        "steps": max_steps,
        "installed_chunks": installed,
        "stale_chunks": stale,
        "buffer_exhaustions": waits,
        "elapsed_s": clock() - start,
    }
    report(f"[pipeline] complete {summary}", flush=True)
    return summary
