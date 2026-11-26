"""Minimal OBS WebSocket v5 client (stdlib only), enough to start and stop recordings.

Protocol: https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import uuid
from typing import Any


class ObsError(RuntimeError):
    pass


class ObsWebSocket:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4455,
        password: str | None = None,
        timeout: float = 10.0,
    ):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._handshake(host, port)
        hello = self._recv_json()
        if hello.get("op") != 0:
            raise ObsError(f"expected Hello, got {hello}")
        identify: dict[str, Any] = {"rpcVersion": 1, "eventSubscriptions": 0}
        auth = hello["d"].get("authentication")
        if auth:
            if password is None:
                raise ObsError("OBS websocket requires a password")
            secret = base64.b64encode(hashlib.sha256((password + auth["salt"]).encode()).digest())
            identify["authentication"] = base64.b64encode(
                hashlib.sha256(secret + auth["challenge"].encode()).digest()
            ).decode()
        self._send_json({"op": 1, "d": identify})
        identified = self._recv_json()
        if identified.get("op") != 2:
            raise ObsError(f"identify failed: {identified}")

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"")
        finally:
            self._sock.close()

    def __enter__(self) -> ObsWebSocket:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def call(self, request_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        payload: dict[str, Any] = {"requestType": request_type, "requestId": request_id}
        if data:
            payload["requestData"] = data
        self._send_json({"op": 6, "d": payload})
        while True:
            message = self._recv_json()
            if message.get("op") == 7 and message["d"].get("requestId") == request_id:
                status = message["d"]["requestStatus"]
                if not status.get("result"):
                    raise ObsError(
                        f"{request_type} failed: {status.get('code')} {status.get('comment', '')}"
                    )
                return message["d"].get("responseData", {}) or {}

    # --- WebSocket framing (RFC 6455, client side) ---

    def _handshake(self, host: str, port: int) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            "Sec-WebSocket-Protocol: obswebsocket.json\r\n\r\n"
        )
        self._sock.sendall(request.encode())
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ObsError("connection closed during handshake")
            response += chunk
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise ObsError(f"handshake refused: {response[:200]!r}")

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 1 << 16:
            header.append(0x80 | 126)
            header += struct.pack("!H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", length)
        mask = os.urandom(4)
        header += mask
        self._sock.sendall(bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def _recv_exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ObsError("connection closed")
            data += chunk
        return data

    def _recv_frame(self) -> tuple[int, bytes]:
        first, second = self._recv_exact(2)
        opcode = first & 0x0F
        length = second & 0x7F
        if length == 126:
            (length,) = struct.unpack("!H", self._recv_exact(2))
        elif length == 127:
            (length,) = struct.unpack("!Q", self._recv_exact(8))
        mask = self._recv_exact(4) if second & 0x80 else None
        payload = self._recv_exact(length)
        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    def _send_json(self, message: dict[str, Any]) -> None:
        self._send_frame(0x1, json.dumps(message).encode())

    def _recv_json(self) -> dict[str, Any]:
        buffer = b""
        while True:
            opcode, payload = self._recv_frame()
            if opcode == 0x9:  # ping
                self._send_frame(0xA, payload)
                continue
            if opcode == 0x8:
                raise ObsError("OBS closed the connection")
            if opcode in (0x1, 0x0):
                buffer += payload
                # Text frames from OBS are never fragmented in practice; FIN is checked implicitly
                # by json parsing succeeding.
                try:
                    return json.loads(buffer)
                except json.JSONDecodeError:
                    continue
