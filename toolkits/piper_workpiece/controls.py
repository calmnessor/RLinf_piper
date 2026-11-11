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

"""Terminal controls for recording over a local terminal or an SSH TTY."""

import os
import select
import sys
import time
from types import TracebackType


class RecordingControls:
    """Read arrows without a desktop keyboard hook; restore terminal on exit."""

    def __init__(self) -> None:
        self._fd = None
        self._settings = None
        self._pending = b""
        self._escape_at = 0.0

    def __enter__(self) -> "RecordingControls":
        import termios
        import tty

        if not sys.stdin.isatty():
            raise RuntimeError(
                "Recording requires an interactive terminal; connect using ssh -t"
            )
        self._fd = sys.stdin.fileno()
        self._settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def feed(self, data: bytes) -> set[str]:
        """Decode complete keys, retaining a split arrow escape sequence."""
        self._pending += data
        events = set()
        while self._pending:
            if self._pending.startswith((b"\x1b[C", b"\x1bOC")):
                events.add("finish")
                self._pending = self._pending[3:]
            elif self._pending.startswith((b"\x1b[D", b"\x1bOD")):
                events.add("discard")
                self._pending = self._pending[3:]
            elif self._pending in (b"\x1b", b"\x1b[", b"\x1bO"):
                if not self._escape_at:
                    self._escape_at = time.monotonic()
                break
            elif (
                self._pending.startswith((b"\x1b[", b"\x1bO"))
                and len(self._pending) >= 3
            ):
                self._pending = self._pending[3:]
            else:
                key, self._pending = self._pending[:1], self._pending[1:]
                events.update(
                    {
                        b"s": {"success"},
                        b"f": {"failure"},
                        b"d": {"discard"},
                        b"\x1b": {"stop"},
                    }.get(key, set())
                )
            self._escape_at = 0.0
        return events

    def poll(self) -> set[str]:
        """Return key events available now, including a standalone Escape."""
        events = set()
        if self._fd is not None and select.select([self._fd], [], [], 0)[0]:
            data = os.read(self._fd, 128)
            if not data:
                return {"stop"}
            events = self.feed(data)
        if self._escape_at and time.monotonic() - self._escape_at >= 0.1:
            self._pending = b""
            self._escape_at = 0.0
            events.add("stop")
        return events

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._settings is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)
