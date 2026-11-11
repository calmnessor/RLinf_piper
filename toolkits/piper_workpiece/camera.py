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

"""Deployment-local camera startup/recovery, without changing shared drivers."""

import logging
import time

logger = logging.getLogger(__name__)


class CameraReadLoopMixin:
    def _read_loop(self):
        # Keep this Event alive locally even if teardown clears self.stop_event.
        stop = self.stop_event
        if stop is None:
            raise RuntimeError("Camera stop event was not initialized")
        failures = 0
        while not stop.is_set():
            try:
                raw = self._read_from_hardware()
                if stop.is_set():
                    break
                frame = self._postprocess_image(raw)
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = time.perf_counter()
                self.new_frame_event.set()
                failures = 0
            except Exception as exc:
                if stop.is_set():
                    break
                failures += 1
                if failures == 1 or failures % 20 == 0:
                    logger.warning(
                        "%s frame not ready (%s); waiting for recovery", self, exc
                    )
                stop.wait(0.05)

    def _stop_read_thread(self):
        stop, thread = self.stop_event, self.thread
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            # V4L2 may still be inside a blocking read (normally <=10 seconds).
            # Do not release its VideoCapture or overwrite thread state mid-read.
            thread.join(timeout=12)
            if thread.is_alive():
                raise RuntimeError(
                    f"{self}: camera hardware read did not stop; reconnect this camera"
                )
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()


def create_camera(values):
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    class DeploymentCamera(CameraReadLoopMixin, OpenCVCamera):
        pass

    config = dict(values)
    config.setdefault(
        "backend", 200
    )  # cv2.CAP_V4L2: use the actual Linux capture device.
    return DeploymentCamera(OpenCVCameraConfig(**config))
