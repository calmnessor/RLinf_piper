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

"""The existing length-prefixed pickle protocol for trusted model servers."""

import logging
import pickle
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

from ._logging import get_logger

logger = get_logger()

_HEADER = struct.Struct("!Q")


def send_obj(sock: socket.socket, obj: Any) -> None:
    """Send one object using the deployed server's eight-byte framing."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive a complete message segment, raising when the peer disconnects."""
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("Socket closed while receiving a model response")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def recv_obj(sock: socket.socket) -> Any:
    """Receive an object from a trusted peer; pickle executes peer-provided code."""
    (size,) = _HEADER.unpack(recv_exact(sock, _HEADER.size))
    return pickle.loads(recv_exact(sock, size))


def connect_with_retry(
    host: str,
    port: int,
    timeout_s: float = 5.0,
    retry_times: int = 30,
    retry_sleep: float = 2.0,
) -> socket.socket:
    """Connect to the selected inference process with bounded retries."""
    last_error = None
    for attempt in range(retry_times):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout_s)
            sock.connect((host, port))
            return sock
        except OSError as exc:
            sock.close()
            last_error = exc
            if attempt + 1 < retry_times:
                logger.info(
                    "Connecting to %s:%s: attempt %s/%s",
                    host,
                    port,
                    attempt + 1,
                    retry_times,
                )
                time.sleep(retry_sleep)
    raise ConnectionError(
        f"Failed to connect to {host}:{port} after {retry_times} attempts"
    ) from last_error


def setup_logging(script_name: str, log_dir: str = ".") -> str:
    """Configure console and file logging for one standalone process."""
    root = Path(log_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{script_name}_{time.time_ns()}.log"
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(path, encoding="utf-8"),
    ):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return str(path)
