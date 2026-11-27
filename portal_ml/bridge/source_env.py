"""Source Engine IPC environment client and loopback mock server.

`SourcePortalEnv` implements the exact same `PortalEnv` interface as
`MockPortalEnv`, so the evolutionary trainer can switch seamlessly between
the in-memory Python mock simulator and a live Source Engine server plugin.
"""

from __future__ import annotations

import socket
import threading
from typing import Any

from portal_ml.bridge.protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    BridgeProtocolError,
    recv_frame,
    send_frame,
)
from portal_ml.motor.primitives import MotorCommand
from portal_ml.schema.blackboard import BlackboardSnapshot
from portal_ml.sim.mock_env import MockPortalEnv, StepResult


class SourcePortalEnv:
    """Gym-style Portal environment communicating with the C++ Source server plugin."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> SourcePortalEnv:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        send_frame(self._sock, payload)
        resp = recv_frame(self._sock)
        if not resp.get("ok", False):
            raise BridgeProtocolError(resp.get("error", "Unknown Source plugin error"))
        return resp

    def ping(self) -> bool:
        resp = self._request({"op": "ping"})
        return resp.get("op") == "pong"

    def map_info(self) -> dict[str, Any]:
        return self._request({"op": "map_info"})

    def reset(self, chamber_id: int) -> BlackboardSnapshot:
        resp = self._request({"op": "reset", "chamber_id": int(chamber_id)})
        return BlackboardSnapshot.from_dict(resp["snapshot"])

    def step(self, command: MotorCommand) -> StepResult:
        resp = self._request({"op": "step", "command": command.to_dict()})
        snap = BlackboardSnapshot.from_dict(resp["snapshot"])
        return StepResult(
            snapshot=snap,
            reward=float(resp.get("reward", 0.0)),
            done=bool(resp.get("done", snap.done)),
            success=bool(resp.get("success", snap.success)),
            milestones=list(resp.get("milestones", [])),
        )


class MockBridgeServer:
    """In-process TCP bridge server backed by `MockPortalEnv` for protocol integration tests."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = 0) -> None:
        self.host = host
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((host, port))
        self.port: int = self._server_sock.getsockname()[1]
        self._server_sock.listen(4)
        self._server_sock.settimeout(0.2)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._env = MockPortalEnv()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server_sock.close()

    def __enter__(self) -> MockBridgeServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _serve_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                client, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with client:
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._handle_client(client)

    def _handle_client(self, client: socket.socket) -> None:
        while not self._stop_event.is_set():
            try:
                req = recv_frame(client)
            except BridgeProtocolError:
                break
            op = req.get("op")
            try:
                if op == "ping":
                    send_frame(client, {"ok": True, "op": "pong"})
                elif op == "reset":
                    snap = self._env.reset(int(req["chamber_id"]))
                    send_frame(
                        client,
                        {"ok": True, "op": "reset", "snapshot": snap.to_dict()},
                    )
                elif op == "step":
                    cmd = MotorCommand.from_dict(req["command"])
                    res = self._env.step(cmd)
                    send_frame(
                        client,
                        {
                            "ok": True,
                            "op": "step",
                            "snapshot": res.snapshot.to_dict(),
                            "reward": res.reward,
                            "done": res.done,
                            "success": res.success,
                            "milestones": res.milestones,
                        },
                    )
                elif op == "shutdown":
                    send_frame(client, {"ok": True, "op": "shutdown"})
                    self._stop_event.set()
                    break
                else:
                    send_frame(
                        client, {"ok": False, "error": f"Unknown op: {op!r}"}
                    )
            except Exception as exc:  # noqa: BLE001
                send_frame(client, {"ok": False, "error": str(exc)})
