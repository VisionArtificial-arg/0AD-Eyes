#!/usr/bin/env python3
"""Record real human 0 A.D. games from their replays with OBS (CAP-2, issue #30).

The replays are Release 28 (0.28.0) games shared on the Wildfire Games forums. They only
re-simulate on 0.28.0, so they play in the official AppImage, not in the 0.29.0-dev fork.
The replaycam mod (next to this file) takes the human player's perspective and moves the
camera like the fork's director; it fast-forwards between segments and prints markers this
script turns into OBS StartRecord/StopRecord calls, one .mkv per segment.

Everything lives under one root (default ~/.temp/0ad-r28):
  0ad-0.28.0-x86_64.AppImage   the official build
  home/{data,config,cache}     XDG dirs for the game, so the user's own 0 A.D. is untouched
  replays-src/<name>/commands.txt (+ metadata.json)
  obs-websocket-password       created by `setup`

Usage:
  record_replays.py setup             write OBS websocket config, start OBS, add the screen source
  record_replays.py plan              print the segments that would be recorded
  record_replays.py record [--only NAME ...]
  record_replays.py test NAME         run one replay with the camera but without OBS (1 segment)

Stdlib only; runs on the host (OBS is a flatpak there).
"""

from __future__ import annotations

import argparse
import configparser
import datetime as dt
import json
import os
import pty
import random
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obsws import ObsError, ObsWebSocket  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_ROOT = Path.home() / ".temp" / "0ad-r28"
OBS_APP = "com.obsproject.Studio"
OBS_CONFIG = Path.home() / ".var" / "app" / OBS_APP / "config" / "obs-studio"
OBS_PROFILE = "0AD-Eyes"
SCREEN_INPUT = "Screen"
SCREEN_KIND = "pipewire-screen-capture-source"


@dataclass
class Replay:
    name: str
    commands: Path
    settings: dict
    duration: float  # sim seconds
    mods: list[str]
    humans: list[int]  # player ids (1-based)
    segments: list[float] = field(default_factory=list)
    player: int = 1

    @property
    def map_name(self) -> str:
        return self.settings.get("mapName", "?")

    def player_data(self, pid: int) -> dict:
        return self.settings["PlayerData"][pid - 1] or {}


def load_replay(directory: Path) -> Replay:
    commands = directory / "commands.txt"
    with commands.open(encoding="utf-8") as f:
        start = json.loads(f.readline()[len("start ") :])
        duration_ms = 0
        for line in f:
            if line.startswith("turn "):
                duration_ms += int(line.split()[2])
    settings = start["settings"]
    humans = [i + 1 for i, p in enumerate(settings["PlayerData"]) if p and not p.get("AI")]
    mods = [m["name"] for m in start.get("mods", []) if m["name"] != "0ad"]
    return Replay(directory.name, commands, settings, duration_ms / 1000.0, mods, humans)


def plan_segments(replay: Replay, count: int, length: float, rng: random.Random) -> None:
    """Spread `count` segments over the game (early, middle, late), never past its end."""
    last_start = replay.duration - length - 5
    if last_start < 30:
        replay.segments = []
        return
    fractions = [(i + 0.5) / count for i in range(count)]
    starts: list[float] = []
    for frac in fractions:
        start = min(max(60.0, frac * replay.duration - length / 2), last_start)
        if not starts or start >= starts[-1] + length:
            starts.append(round(start))
    replay.segments = starts
    replay.player = rng.choice(replay.humans) if replay.humans else 1


def available_mods(root: Path) -> set[str]:
    mods_dir = root / "home" / "data" / "0ad" / "mods"
    return {p.name for p in mods_dir.iterdir()} if mods_dir.is_dir() else set()


def game_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key, sub in (
        ("XDG_DATA_HOME", "data"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        path = root / "home" / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def install_mod(root: Path) -> None:
    """Symlink the replaycam mod into the game's mods folder so edits here apply directly."""
    target = root / "home" / "data" / "0ad" / "mods" / "replaycam"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() == (HERE / "replaycam").resolve():
            return
        raise SystemExit(f"{target} exists and is not our symlink; remove it first")
    target.symlink_to(HERE / "replaycam")


def game_command(
    root: Path, replay: Replay, args: argparse.Namespace, segments: list[float]
) -> list[str]:
    have = available_mods(root)
    missing = [m for m in replay.mods if m not in have]
    # GUI-only mods (autociv, ModernGUI, ...) don't change the simulation; only map mods matter.
    needed = [m for m in replay.mods if m in have]
    cmd = [str(root / args.appimage), "-mod=public"]
    cmd += [f"-mod={m}" for m in needed]
    cmd += [
        "-mod=replaycam",
        f"-replay-visual={replay.commands}",
        f"-xres={args.width}",
        f"-yres={args.height}",
        "-nosound",
        "-conf=pauseonfocusloss:false",
        "-conf=locale:en",  # same HUD language as the fork's training frames
        "-conf=replaycam.enabled:true",
        f"-conf=replaycam.player:{replay.player}",
        f"-conf=replaycam.segments:{','.join(str(int(s)) for s in segments)}",
        f"-conf=replaycam.length:{int(args.length)}",
        f"-conf=replaycam.shot:{int(args.shot)}",
        f"-conf=replaycam.ffrate:{int(args.ffrate)}",
    ]
    if missing:
        print(f"    note: mods not installed (assumed GUI-only): {', '.join(missing)}")
    return cmd


# --- OBS ---


def obs_password(root: Path) -> str:
    return (root / "obs-websocket-password").read_text().strip()


def write_obs_config(root: Path) -> None:
    password_file = root / "obs-websocket-password"
    if not password_file.exists():
        password_file.write_text(secrets.token_urlsafe(24) + "\n")
        password_file.chmod(0o600)
    ws = OBS_CONFIG / "plugin_config" / "obs-websocket" / "config.json"
    ws.parent.mkdir(parents=True, exist_ok=True)
    ws.write_text(
        json.dumps(
            {
                "alerts_enabled": False,
                "auth_required": True,
                "first_load": False,
                "server_enabled": True,
                "server_password": obs_password(root),
                "server_port": 4455,
            },
            indent=2,
        )
    )
    user_ini = OBS_CONFIG / "user.ini"
    if not user_ini.exists():
        # Skip the auto-config wizard and open on our profile (OBS 31+ keeps this in user.ini).
        ini = configparser.ConfigParser()
        ini.optionxform = str  # type: ignore[assignment]
        ini["General"] = {"FirstRun": "true"}
        ini["Basic"] = {
            "Profile": OBS_PROFILE,
            "ProfileDir": OBS_PROFILE,
            "SceneCollection": OBS_PROFILE,
            "SceneCollectionFile": OBS_PROFILE,
        }
        with user_ini.open("w") as f:
            ini.write(f, space_around_delimiters=False)


def obs_running() -> bool:
    return (
        subprocess.run(
            ["flatpak", "ps", "--columns=application"], capture_output=True, text=True
        ).stdout.find(OBS_APP)
        != -1
    )


def connect_obs(root: Path, wait: float = 40.0) -> ObsWebSocket:
    if not obs_running():
        # A killed OBS leaves run_* sentinels, and the next start then blocks on a "crash
        # detected, start in safe mode?" prompt (OBS 33 ignores --disable-shutdown-check).
        for sentinel in (OBS_CONFIG / ".sentinel").glob("run_*"):
            sentinel.unlink()
        subprocess.Popen(
            ["flatpak", "run", OBS_APP, "--profile", OBS_PROFILE, "--disable-shutdown-check"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.time() + wait
    while True:
        try:
            return ObsWebSocket(password=obs_password(root))
        except (OSError, ObsError):
            if time.time() > deadline:
                raise
            time.sleep(1.0)


def ensure_screen_source(obs: ObsWebSocket) -> bool:
    """Adds the PipeWire screen capture to the current scene; True if it was just created."""
    inputs = obs.call("GetInputList", {"inputKind": SCREEN_KIND}).get("inputs", [])
    if inputs:
        return False
    scene = obs.call("GetSceneList")["currentProgramSceneName"]
    obs.call(
        "CreateInput", {"sceneName": scene, "inputName": SCREEN_INPUT, "inputKind": SCREEN_KIND}
    )
    return True


def source_thumbnail(obs: ObsWebSocket) -> str:
    """A tiny screenshot of the screen source, to tell a live capture from a frozen one."""
    return obs.call(
        "GetSourceScreenshot", {"sourceName": SCREEN_INPUT, "imageFormat": "png", "imageWidth": 64}
    )["imageData"]


def wait_file_settled(path: Path, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        size = path.stat().st_size if path.exists() else -1
        if size == last and size > 0:
            return
        last = size
        time.sleep(1.0)


# --- LOG.md ---

LOG_HEADER = (
    "| File | Date | Map (biome) | My civ | Opponent(s)/AI difficulty | Length | Zoom/camera | "
    "Phases reached | Battle? | Notes |"
)


def log_row(replay: Replay, video: Path, start: float, end: float) -> str:
    me = replay.player_data(replay.player)
    others = []
    for pid in range(1, len(replay.settings["PlayerData"]) + 1):
        if pid == replay.player:
            continue
        p = replay.player_data(pid)
        who = f"{p.get('AI')} d{p.get('AIDiff')}" if p.get("AI") else "human"
        others.append(f"{p.get('Civ')} ({who})")
    biome = replay.settings.get("Biome", "")
    minutes = (end - start) / 60
    return (
        f"| {video.name} | {dt.date.today().isoformat()} | {replay.map_name} ({biome}) | "
        f"{me.get('Civ')} | "
        f"{', '.join(others)} | {minutes:.1f} min | replaycam director, default zoom | - | - | "
        f"replay {replay.name}, human player {replay.player} ({me.get('Name', '?')}), "
        f"sim {start / 60:.1f}-{end / 60:.1f} min of {replay.duration / 60:.0f} |"
    )


def append_log(log: Path, row: str) -> None:
    text = log.read_text() if log.exists() else ""
    if LOG_HEADER not in text:
        text += (
            ("\n" if text and not text.endswith("\n") else "")
            + LOG_HEADER
            + "\n"
            + "|---" * 10
            + "|\n"
        )
    log.write_text(text + row + "\n")


# --- Running ---


def pty_lines(fd: int):
    """Lines read from a pty master until the child closes it (EIO on Linux)."""
    buffer = b""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            chunk = b""
        if not chunk:
            break
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            yield line.decode("utf-8", "replace").rstrip("\r") + "\n"
    if buffer:
        yield buffer.decode("utf-8", "replace")
    os.close(fd)


def run_replay(
    root: Path, replay: Replay, args: argparse.Namespace, obs: ObsWebSocket | None
) -> int:
    segments = replay.segments[:1] if obs is None else replay.segments
    cmd = game_command(root, replay, args, segments)
    skip = (max(segments) if segments else 0) / args.ffrate
    timeout = skip + len(segments) * args.length * 1.5 + 300
    print(f"    {' '.join(cmd[:2])} ... (timeout {timeout / 60:.0f} min)")
    # A pseudo-terminal, so the game line-buffers stdout and markers arrive as they are printed.
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        cmd, env=game_env(root), stdin=subprocess.DEVNULL, stdout=slave, stderr=slave
    )
    os.close(slave)
    log_path = root / "recordlogs" / f"{replay.name}.log"
    log_path.parent.mkdir(exist_ok=True)
    recorded = 0
    seg_start = 0.0
    start_thumb = ""
    deadline = time.time() + timeout
    try:
        with log_path.open("w") as game_log:
            for line in pty_lines(master):
                game_log.write(line)
                if time.time() > deadline:
                    print("    timeout")
                    break
                if not line.startswith("REPLAYCAM "):
                    continue
                parts = line.split()
                event = parts[1]
                print(f"    {line.strip()}")
                if event == "SEGMENT_START":
                    seg_start = float(parts[3])
                    if obs:
                        obs.call("StartRecord")
                        start_thumb = source_thumbnail(obs)
                elif event == "SEGMENT_END":
                    if obs:
                        # The camera moves every few seconds, so an unchanged picture means the
                        # capture died (e.g. a window was shared instead of the whole screen).
                        frozen = source_thumbnail(obs) == start_thumb
                        out = Path(obs.call("StopRecord")["outputPath"])
                        wait_file_settled(out)
                        if frozen:
                            out.unlink()
                            raise SystemExit(
                                "screen capture is frozen; reselect the whole screen "
                                "in OBS (Screen > Reload)"
                            )
                        video = args.out / f"{replay.name}_p{replay.player}_s{parts[2]}{out.suffix}"
                        shutil.move(out, video)
                        append_log(
                            args.out / "LOG.md", log_row(replay, video, seg_start, float(parts[3]))
                        )
                        print(f"    -> {video}")
                    recorded += 1
                elif event == "DONE":
                    break
    finally:
        if obs and obs.call("GetRecordStatus").get("outputActive"):
            obs.call("StopRecord")
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
    return recorded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=("setup", "plan", "record", "test"))
    parser.add_argument(
        "names", nargs="*", help="replay folder names (test: exactly one; record: optional filter)"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--appimage", default="0ad-0.28.0-x86_64.AppImage")
    parser.add_argument("--out", type=Path, default=REPO / "recordings")
    parser.add_argument("--segments", type=int, default=2, help="segments per game (default 2)")
    parser.add_argument(
        "--length", type=float, default=180, help="sim seconds per segment (default 180)"
    )
    parser.add_argument(
        "--shot", type=float, default=6, help="seconds per camera target (default 6)"
    )
    parser.add_argument(
        "--ffrate", type=float, default=10, help="sim rate between segments (default 10)"
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=30, help="seed for which human is followed")
    args = parser.parse_args()
    root: Path = args.root

    if args.action == "setup":
        write_obs_config(root)
        with connect_obs(root) as obs:
            created = ensure_screen_source(obs)
            print(
                "OBS websocket OK;",
                "screen source created: pick the monitor in the portal dialog "
                "and allow restoring it"
                if created
                else "screen source already present",
            )
        return

    rng = random.Random(args.seed)
    replays = [
        load_replay(d)
        for d in sorted((root / "replays-src").iterdir())
        if (d / "commands.txt").exists()
    ]
    if args.names:
        replays = [r for r in replays if r.name in args.names]
    for r in replays:
        plan_segments(r, args.segments, args.length, rng)

    if args.action == "plan":
        total = 0.0
        for r in replays:
            segs = ", ".join(f"{s / 60:.1f}" for s in r.segments) or "(too short)"
            print(
                f"{r.name:24} {r.map_name:24} {r.duration / 60:5.1f} min  "
                f"player {r.player}  segments at {segs} min"
            )
            total += len(r.segments) * args.length
        print(f"{len(replays)} replays, {total / 3600:.1f} h of footage")
        return

    install_mod(root)
    if args.action == "test":
        if len(replays) != 1:
            raise SystemExit("test needs exactly one replay name")
        run_replay(root, replays[0], args, None)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    with connect_obs(root) as obs:
        if ensure_screen_source(obs):
            raise SystemExit(
                "screen source was missing; pick the monitor in the portal, then rerun"
            )
        for i, r in enumerate(replays, 1):
            if not r.segments:
                print(f"[{i}/{len(replays)}] {r.name}: too short, skipped")
                continue
            print(f"[{i}/{len(replays)}] {r.name} ({r.map_name}, player {r.player})")
            n = run_replay(root, r, args, obs)
            print(f"    {n}/{len(r.segments)} segments recorded")


if __name__ == "__main__":
    main()
