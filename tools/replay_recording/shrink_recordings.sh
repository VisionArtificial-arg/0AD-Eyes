#!/usr/bin/env bash
# Re-encode OBS recordings to ~6 Mbps H.264 (a typical 1080p30 stream bitrate) and replace
# the originals once the new file's duration matches. CPU encoder (libopenh264) because
# Fedora's Mesa has no H.264 VAAPI encoding. Run inside the box: needs ffmpeg-free.
# Usage: shrink_recordings.sh DIR [JOBS]
set -euo pipefail
dir=${1:?recordings dir}; jobs=${2:-6}
shrink() {
  f=$1; tmp="${f%.mkv}.shrink.tmp.mkv"
  # Already small enough: skip (OBS CQP files are 3-10x larger than 6 Mbps).
  br=$(ffprobe -v error -show_entries format=bit_rate -of csv=p=0 "$f")
  if [ "${br:-0}" -lt 8000000 ]; then echo "skip $f ($br bps)"; return; fi
  ffmpeg -hide_banner -v error -y -i "$f" -c:v libopenh264 -b:v 6M -maxrate 9M -g 60 -an "$tmp"
  a=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  b=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$tmp")
  if python3 -c "import sys; sys.exit(abs($a-$b) > 1.0)"; then mv "$tmp" "$f"; echo "ok $f"; else rm -f "$tmp"; echo "MISMATCH $f $a vs $b"; fi
}
export -f shrink
find "$dir" -maxdepth 1 -name '*.mkv' ! -name '*.shrink.tmp.mkv' -print0 | sort -z | xargs -0 -n1 -P "$jobs" bash -c 'shrink "$0"'
