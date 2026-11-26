"""Length-prefixed JSON wire protocol shared with the C++ Source Engine plugin.

Frame layout:
    [4 bytes little-endian uint32 payload_length][payload_length bytes UTF-8 JSON]

Messages (Python -> Source plugin):
    {"op": "ping"}
    {"op": "reset", "chamber_id": 1}
    {"op": "step", "command": {"cmd": "move_to", "entity_id": "c1_exit_0"}}
    {"op": "shutdown"}

Responses (Source plugin -> Python):
    {"ok": true, "op": "pong"}
    {"ok": true, "op": "reset", "snapshot": { ... }}
    {"ok": true, "op": "step", "snapshot": { ... }, "reward": -0.05, "done": false, "success": false, "milestones": []}
    {"ok": false, "error": "..."}
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 27500
DEFAULT_PIPE_NAME = r"\\.\pipe\portal_ml_bridge"
MAX_FRAME_BYTES = 4 * 1024 * 1024  # 4 MiB safety cap

_HEADER_STRUCT = struct.Struct("<I")


class BridgeProtocolError(RuntimeError):
    """Raised on framing, serialization, or remote error responses."""


def encode_message(payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_FRAME_BYTES:
        raise BridgeProtocolError(f"Outgoing payload too large ({len(raw)} bytes)")
    return _HEADER_STRUCT.pack(len(raw)) + raw


def decode_message(raw_json: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(raw_json.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeProtocolError(f"Malformed JSON frame: {exc}") from exc
    if not isinstance(obj, dict):
        raise BridgeProtocolError("Expected JSON object at top level of frame")
    return obj


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise BridgeProtocolError("Peer closed IPC connection unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(encode_message(payload))


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, _HEADER_STRUCT.size)
    (length,) = _HEADER_STRUCT.unpack(header)
    if length == 0 or length > MAX_FRAME_BYTES:
        raise BridgeProtocolError(f"Invalid IPC frame length: {length}")
    body = _recv_exact(sock, length)
    return decode_message(body)
