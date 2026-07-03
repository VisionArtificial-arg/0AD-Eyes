#!/usr/bin/env python
"""Standalone Wayland ScreenCast capture helper (EPIC A1 — portal backend).

Run under the **system** Python (which has PyGObject + GStreamer + the pipewiresrc
plugin), NOT the project venv. It is deliberately dependency-light and imports
nothing from ``zero_ad_eyes`` so it can execute under a different interpreter.

It negotiates an ``xdg-desktop-portal`` ScreenCast session over D-Bus (the same path
Discord/OBS use), opens the resulting PipeWire stream through a GStreamer pipeline,
and writes raw frames to **stdout** as a simple length-prefixed stream that the
in-venv ``PortalPipeWireGrabber`` reads:

    per frame:  <width:u32 LE><height:u32 LE><stride:u32 LE> then height*stride bytes
                of BGR pixels (stride may include row padding; the reader de-pads).

The portal ``restore_token`` is persisted (``--restore-token-file``) so the consent
picker only appears the first time; later runs restore the same source silently.
"""

from __future__ import annotations

import argparse
import struct
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gio, GLib, Gst  # noqa: E402

PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_OBJ = "/org/freedesktop/portal/desktop"
SC_IFACE = "org.freedesktop.portal.ScreenCast"
REQ_IFACE = "org.freedesktop.portal.Request"

# SelectSources source-type bitmask and cursor-mode enum (portal spec).
SOURCE_MONITOR, SOURCE_WINDOW = 1, 2
CURSOR_HIDDEN, CURSOR_EMBEDDED = 1, 2
PERSIST_UNTIL_REVOKED = 2


def log(msg: str) -> None:
    print(f"[portal-helper] {msg}", file=sys.stderr, flush=True)


class PortalCapture:
    def __init__(self, source_type: int, cursor_mode: int, token_path: str | None) -> None:
        self._loop = GLib.MainLoop()
        self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self._sender = self._bus.get_unique_name()[1:].replace(".", "_")
        self._counter = 0
        self._source_type = source_type
        self._cursor_mode = cursor_mode
        self._token_path = token_path
        self._session: str | None = None

    # --- D-Bus portal request/response plumbing ------------------------------ #

    def _token(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def _call(self, method: str, build_params, on_ok) -> None:
        """Invoke a portal method that returns a Request; await its Response signal."""

        handle_token = self._token("t")
        req_path = f"{PORTAL_OBJ}/request/{self._sender}/{handle_token}"
        state: dict[str, int] = {}

        def on_response(_c, _s, _p, _i, _sig, params):
            self._bus.signal_unsubscribe(state["sub"])
            code, results = params.unpack()
            if code != 0:
                log(f"{method} failed (portal response code {code})")
                self._loop.quit()
                return
            on_ok(results)

        state["sub"] = self._bus.signal_subscribe(
            PORTAL_BUS, REQ_IFACE, "Response", req_path, None, Gio.DBusSignalFlags.NONE, on_response
        )
        self._bus.call_sync(
            PORTAL_BUS,
            PORTAL_OBJ,
            SC_IFACE,
            method,
            build_params(handle_token),
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        )

    # --- ScreenCast negotiation --------------------------------------------- #

    def start(self) -> None:
        self._create_session()
        self._loop.run()

    def _create_session(self) -> None:
        def params(handle_token):
            opts = {
                "handle_token": GLib.Variant("s", handle_token),
                "session_handle_token": GLib.Variant("s", self._token("s")),
            }
            return GLib.Variant("(a{sv})", (opts,))

        self._call("CreateSession", params, self._on_session)

    def _on_session(self, results) -> None:
        self._session = results["session_handle"]
        log("session created; selecting sources")
        self._select_sources()

    def _select_sources(self) -> None:
        def params(handle_token):
            opts = {
                "handle_token": GLib.Variant("s", handle_token),
                "types": GLib.Variant("u", self._source_type),
                "multiple": GLib.Variant("b", False),
                "cursor_mode": GLib.Variant("u", self._cursor_mode),
                "persist_mode": GLib.Variant("u", PERSIST_UNTIL_REVOKED),
            }
            saved = self._read_token()
            if saved:
                opts["restore_token"] = GLib.Variant("s", saved)
            return GLib.Variant("(oa{sv})", (self._session, opts))

        self._call("SelectSources", params, lambda _r: self._start_stream())

    def _start_stream(self) -> None:
        def params(handle_token):
            opts = {"handle_token": GLib.Variant("s", handle_token)}
            return GLib.Variant("(osa{sv})", (self._session, "", opts))

        self._call("Start", params, self._on_started)

    def _on_started(self, results) -> None:
        if "restore_token" in results:
            self._write_token(results["restore_token"])
        streams = results["streams"]
        if not streams:
            log("no streams returned")
            self._loop.quit()
            return
        node_id = streams[0][0]
        log(f"stream started, pipewire node {node_id}; opening remote")
        self._open_remote(node_id)

    def _open_remote(self, node_id: int) -> None:
        variant, fd_list = self._bus.call_with_unix_fd_list_sync(
            PORTAL_BUS,
            PORTAL_OBJ,
            SC_IFACE,
            "OpenPipeWireRemote",
            GLib.Variant("(oa{sv})", (self._session, {})),
            GLib.VariantType("(h)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            None,
        )
        fd = fd_list.get(variant.unpack()[0])
        self._build_pipeline(fd, node_id)

    # --- GStreamer: pipewire node -> BGR frames on stdout -------------------- #

    def _build_pipeline(self, fd: int, node_id: int) -> None:
        desc = (
            f"pipewiresrc fd={fd} path={node_id} ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink name=sink emit-signals=true "
            f"max-buffers=1 drop=true sync=false"
        )
        pipeline = Gst.parse_launch(desc)
        sink = pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_sample)
        pipeline.set_state(Gst.State.PLAYING)
        log("pipeline PLAYING")

    def _on_sample(self, sink) -> int:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        stride = buf.get_size() // height
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                out = sys.stdout.buffer
                out.write(struct.pack("<III", width, height, stride))
                out.write(mapinfo.data)
                out.flush()
            except (BrokenPipeError, ValueError):
                # Consumer closed the pipe (e.g. the grabber terminated us) — stop cleanly.
                self._loop.quit()
            finally:
                buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    # --- restore-token persistence ------------------------------------------ #

    def _read_token(self) -> str | None:
        if not self._token_path:
            return None
        try:
            with open(self._token_path, encoding="utf-8") as fh:
                return fh.read().strip() or None
        except FileNotFoundError:
            return None

    def _write_token(self, token: str) -> None:
        if self._token_path and token:
            with open(self._token_path, "w", encoding="utf-8") as fh:
                fh.write(token)


def main() -> int:
    ap = argparse.ArgumentParser(description="xdg-desktop-portal ScreenCast -> stdout BGR frames")
    ap.add_argument("--source-type", choices=("window", "monitor"), default="window")
    ap.add_argument("--cursor", choices=("hidden", "embedded"), default="embedded")
    ap.add_argument("--restore-token-file", default=None)
    args = ap.parse_args()

    Gst.init(None)
    capture = PortalCapture(
        source_type=SOURCE_WINDOW if args.source_type == "window" else SOURCE_MONITOR,
        cursor_mode=CURSOR_HIDDEN if args.cursor == "hidden" else CURSOR_EMBEDDED,
        token_path=args.restore_token_file,
    )
    try:
        capture.start()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
