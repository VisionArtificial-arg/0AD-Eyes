# PC setup (Bazzite) — story CAP-1

The PC runs everything: 0 A.D., OBS, the engine fork, dataset generation and the live
demo. Bazzite is an atomic (read-only) Fedora, so everything except OBS runs inside **one
distrobox (`zad`)**. That includes the Python environment, the engine build and the game.

The workspace directory (`$WS` below) holds the repos, the box's home and the podman
storage. Deleting `$WS` and removing the box (`distrobox rm -f zad`) undoes the whole
setup. Any path works. The first PC setup used `~/.temp`.

Each step ends with a check. A step is done only when its check passes; paste the output
into the session if it fails.

## 1. The box

Keep podman's image storage inside the workspace, then create the box with its home
there too:

```bash
export WS=~/.temp            # or any other directory
mkdir -p $WS/containers/storage $WS/box-home
cat > $WS/containers/storage.conf <<EOF
[storage]
driver = "overlay"
graphroot = "$WS/containers/storage"
runroot = "/run/user/$(id -u)/ws-containers"
EOF
cat > $WS/env.sh <<EOF
export CONTAINERS_STORAGE_CONF="$WS/containers/storage.conf"
EOF
source $WS/env.sh            # before every distrobox/podman command
distrobox create --name zad --image registry.fedoraproject.org/fedora:latest --home $WS/box-home
```

Inside the box `$HOME` is `$WS/box-home`. That is where gh's token, uv's cache, rustup
and the 0 A.D. user data dir all end up. Every command below runs inside the box
(`distrobox enter zad`) unless it says "host".

**Check:** `distrobox enter zad -- bash -c 'echo $HOME'` prints `$WS/box-home`.

**After a reboot** the box can refuse to start with `creating resolv.conf ... userdata/resolv.conf:
no such file or directory`, then `cannot stat .../userdata/.containerenv`. The runroot
under `/run/user` lost part of the container's state. Recreate it and start again:

```bash
U=/run/user/$(id -u)/ws-containers/overlay-containers/$(podman inspect -f '{{.Id}}' zad)/userdata
mkdir -p $U && touch $U/.containerenv && podman start zad
```

## 2. Tools, login and workspace layout

```bash
sudo dnf install -y gh git git-lfs tesseract uv just python3.12 \
  mesa-libGL glib2 libxcb libstdc++ zlib-ng-compat
gh auth login -p https -w && gh auth setup-git    # the sibling repos are private
cd $WS
git clone https://github.com/VisionArtificial-arg/0AD-Eyes.git
git clone https://github.com/VisionArtificial-arg/TpFinalCaptura.git
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/VisionArtificial-arg/0A.DDataset.git
```

- Fedora's `uv` won't download Python, so `python3.12` comes from dnf.
- OpenCV needs `mesa-libGL` and `glib2`. Without them every test fails to import with
  `libGL.so.1: cannot open shared object file`.

**Check:** `ls $WS` lists all three repos.

## 3. The 0AD-Eyes environment

```bash
cd $WS/0AD-Eyes && uv sync --extra cpu && just setup
```

`just setup` and `uv run` sync without extras, so torch disappears again. That is what CI
does: `just validate` skips the one torch test.

**Check:** `just validate` is green.

## 4. Portal capture dependencies (host, for the live demo, CAP-4)

```bash
/usr/bin/python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; print('ok')"
gst-inspect-1.0 pipewiresrc | head -3
echo $XDG_CURRENT_DESKTOP
flatpak install -y flathub com.obsproject.Studio   # recording (CAP-3)
```

**Check:** the first command prints `ok` and `pipewiresrc` is found. If not, note it in
the session. It doesn't block W0.

## 5. GPU

The card is an **RX 5500 (Navi 14, gfx1012, RDNA1)**, which ROCm/PyTorch doesn't
support. It renders the game; it doesn't train.
- Training runs on **Kaggle or Colab** (MOD-3).
- Inference uses the CPU, or optionally ncnn + Vulkan (MOD-7).

Spend no time trying to get ROCm working.

## 6. Build the engine fork (ENG-0)

Build dependencies, a Rust pinned for SpiderMonkey, then the art and the build:

```bash
sudo dnf install -y cmake gcc-c++ make patch subversion pkgconf m4 clang llvm-devel \
  which xz bzip2 curl perl rustup cbindgen nodejs libuuid-devel \
  boost-devel enet-devel fmt-devel freetype-devel gloox-devel libcurl-devel \
  libicu-devel libidn-devel libpng-devel libsodium-devel libvorbis-devel libogg-devel \
  libxml2-devel miniupnpc-devel openal-soft-devel SDL2-devel wxGTK-devel \
  zlib-ng-compat-devel libX11-devel libXcursor-devel mesa-libGL-devel \
  mesa-libEGL-devel mesa-dri-drivers vulkan-loader-devel
rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.86.0
export PATH=$HOME/.cargo/bin:$PATH

cd $WS/0A.DDataset
git lfs install --local && git lfs pull   # 6.5 GB of art from upstream LFS
git lfs dedup                             # btrfs: share blocks between .git/lfs and the checkout
libraries/build-source-libs.sh -j"$(nproc)"
build/workspaces/update-workspaces.sh -j"$(nproc)"
make -C build/workspaces/gcc -j"$(nproc)"
```

What each build failure looked like:

| Symptom | Fix |
|---|---|
| Premake: `uuid/uuid.h: No such file` | `libuuid-devel` |
| SpiderMonkey: `Don't know how to translate x86_64-pc-linux-gnu for rustc` | mozjs 128 can't parse new rustc (Fedora ships 1.100): use rustup 1.86 on `PATH` and delete `libraries/source/spidermonkey/mozjs-*/build-*` before retrying |
| SpiderMonkey: `Cannot find cbindgen` | `cbindgen` |

The build needs about 15 GB free: 6.5 GB of art (twice, until `git lfs dedup` runs),
about 3 GB for SpiderMonkey and the engine objects. It takes about 30 min on 16 threads.
Once `build-source-libs.sh` has run, Rust is only needed to rebuild those libraries, so
`rm -rf ~/.rustup ~/.cargo` frees 1.2 GB; rebuilding the engine itself doesn't use it.

**Check:** `binaries/system/pyrogenesis` exists and opens the main menu with art.

## 7. First dataset frame (the W0 gate)

Pick a 2-player map from `binaries/data/mods/public/maps/skirmishes/` (the first run
used `acropolis_bay_2p`), then run:

```bash
binaries/system/pyrogenesis -xres=1920 -yres=1080 \
  -autostart="skirmishes/<map>" -autostart-ai=1:petra -autostart-ai=2:petra \
  -autostart-player=-1 -generate-dataset -dataset-interval=30
```

The game doesn't quit on its own; `timeout 150` in front gives about 180 pairs. A warning
about a missing `~/.local/share/0ad/mods` directory is harmless.

**Check (W0 gate):** under the box's 0 A.D. user data dir
(`$WS/box-home/.local/share/0ad/`), a `dataset/<split>/rgb/` and `mask/` PNG pair
exists, and the mask shows flat class colours that line up with the RGB.

## 8. Real-play recordings from replays (CAP-2)

The real-play test set is OBS footage of **real human games**, replayed from their
replay files. They are public Release 28 (0.28.0) games: the
[Spring 2026 tournament](https://wildfiregames.com/forum/topic/162666-spring-2026-tournament-replays/)
1v1s and the [comp stomps](https://wildfiregames.com/forum/topic/141266-comp-stomps-release-28/)
thread (humans vs Petra). A replay only re-simulates on the version it was played on, so
they run in the official 0.28.0 AppImage, not in the fork (0.29.0-dev).

Everything lives in `$WS/0ad-r28` (host, no box needed):

```bash
cd $WS/0ad-r28
curl -LO https://releases.wildfiregames.com/0ad-0.28.0-x86_64.AppImage \
     -LO https://releases.wildfiregames.com/0ad-0.28.0-x86_64.AppImage.sha256sum
sha256sum -c *.sha256sum && chmod +x 0ad-0.28.0-x86_64.AppImage
# replays-src/<name>/commands.txt (+ metadata.json): unzip the forum attachments here;
# replays-src/SOURCES.tsv lists where each one came from.
# Mainland Balanced games need the feldmap 3.0.1 mod (feldmap.zip in the feldmap forum
# thread), unzipped to home/data/0ad/mods/feldmap/.
```

`tools/replay_recording/record_replays.py` drives everything, stdlib only:
- It runs the AppImage with its own XDG dirs under `$WS/0ad-r28/home`, so a personal
  0 A.D. install is untouched.
- It loads the `replaycam` GUI mod, which takes the human player's perspective (their
  fog, HUD and panels) and moves the camera like the fork's director. It
  fast-forwards between segments and prints markers.
- On each marker it starts or stops OBS over its websocket. Each segment becomes
  `recordings/<replay>_p<player>_s<n>.mkv`, plus a row in `recordings/LOG.md`.

```bash
python3 tools/replay_recording/record_replays.py setup   # once: websocket config, OBS, screen source
python3 tools/replay_recording/record_replays.py plan    # 2 x 3 min per game by default
python3 tools/replay_recording/record_replays.py record  # about 10 min per game; leave the PC alone
```

- **Setup:** in the portal dialog that `setup` opens, share the **entire screen, not a
  window**. Each replay opens a new game window, so a window capture freezes on the
  last frame once its window closes. The recorder stops if the picture doesn't change
  during a segment. If the dialog doesn't show, it is probably behind a fullscreen game
  (Alt+Tab). Otherwise use OBS > Screen > Reload.
- **Bitrate:** the OBS profile records at **6 Mbps VBR**
  (`recordEncoder.json`: `{"rate_control":"VBR","bitrate":6000,"maxrate":9000,...}`),
  about 135 MB per 3-minute segment. That's a normal 1080p30 stream rate. CQP 16 looked
  fine on a quiet screen but reached 30–90 Mbps in busy late games, and 19 games filled
  the disk. `tools/replay_recording/shrink_recordings.sh` re-encodes old CQP files (in
  the box, `sudo dnf install ffmpeg-free`). It uses CPU `libopenh264`, because Fedora's
  Mesa has no H.264 VAAPI encoding.
- **OBS crash prompt:** killing OBS (`flatpak kill`) leaves `run_*` files in
  `~/.var/app/com.obsproject.Studio/config/obs-studio/.sentinel/`. The next start then
  waits on a "crash detected, safe mode?" prompt with the websocket off. The recorder
  deletes them before starting OBS. OBS 33 ignores `--disable-shutdown-check`.
- **Language:** the game runs with `-conf=locale:en`, so its HUD matches the fork's
  training frames.

## 9. Claude Code

Install Claude Code on the PC, then start it in `$WS/0AD-Eyes`. It picks up `CLAUDE.md`,
which points to this file and to the roadmap in GitHub issues. `gh` lives in the box, so
run it as `distrobox enter zad -- gh …` from the host.
