# PC setup (Bazzite) — story CAP-1

The PC runs everything: 0 A.D., OBS, the engine fork, dataset generation and the live
demo. Bazzite is an atomic (read-only) Fedora, so host tools come from Flatpak,
Homebrew or user-level installs, and the engine fork builds inside a distrobox.

Each step ends with a check. A step is done only when its check passes; paste the output
into the session if it fails.

## 1. Workspace layout

Clone the three repos as siblings:

```bash
mkdir -p ~/dev/va-final && cd ~/dev/va-final
git clone git@github.com:VisionArtificial-arg/0AD-Eyes.git
git clone git@github.com:VisionArtificial-arg/TpFinalCaptura.git
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:VisionArtificial-arg/0A.DDataset.git
```

**Check:** `ls ~/dev/va-final` lists all three.

## 2. Host tools and the 0AD-Eyes environment

```bash
flatpak install -y flathub com.play0ad.zeroad com.obsproject.Studio
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install just tesseract
cd ~/dev/va-final/0AD-Eyes && uv sync --extra cpu && just setup
```

**Check:** `just validate` is green.

## 3. Portal capture dependencies (for the live demo, CAP-4)

```bash
/usr/bin/python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; print('ok')"
gst-inspect-1.0 pipewiresrc | head -3
echo $XDG_CURRENT_DESKTOP
```

**Check:** the first command prints `ok` and `pipewiresrc` is found. If not, note it in
the session. It doesn't block W0.

## 4. GPU

The card is an **RX 5500 (Navi 14, gfx1012, RDNA1)**, which ROCm/PyTorch doesn't
support. It renders the game; it doesn't train.
- Training runs on **Kaggle or Colab** (MOD-3).
- Inference uses the CPU, or optionally ncnn + Vulkan (MOD-7).

Spend no time trying to get ROCm working.

## 5. Build the engine fork (ENG-0)

```bash
distrobox create --name zad --image registry.fedoraproject.org/fedora:latest
distrobox enter zad
# inside the box: install the build deps listed for Fedora in
# https://gitea.wildfiregames.com/0ad/0ad/wiki/BuildInstructions
cd ~/dev/va-final/0A.DDataset
git lfs pull                       # game art from upstream LFS; multi-GB
libraries/build-source-libs.sh -j"$(nproc)"
build/workspaces/update-workspaces.sh -j"$(nproc)"
make -C build/workspaces/gcc -j"$(nproc)"
```

**Check:** `binaries/system/pyrogenesis` exists and opens the main menu with art.

## 6. First dataset frame (the W0 gate)

Pick a 2-player map from `binaries/data/mods/public/maps/skirmishes/`, then run:

```bash
binaries/system/pyrogenesis -xres=1920 -yres=1080 \
  -autostart="skirmishes/<map>" -autostart-ai=1:petra -autostart-ai=2:petra \
  -autostart-player=-1 -generate-dataset -dataset-interval=30
```

**Check (W0 gate):** under the 0 A.D. user data dir (usually `~/.local/share/0ad/`), a
`dataset/<split>/rgb/` and `mask/` PNG pair exists, and the mask shows flat class
colours that line up with the RGB.

## 7. Claude Code

Install Claude Code on the PC, then start it in `~/dev/va-final/0AD-Eyes`. It picks up
`CLAUDE.md`, which points to this file and to the roadmap in GitHub issues.

Suggested first prompt: *"We're in milestone W0 Spike. Walk me through
docs/pc-setup.md (CAP-1), then do ENG-0."*
