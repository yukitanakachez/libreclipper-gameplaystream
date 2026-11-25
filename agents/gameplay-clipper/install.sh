#!/usr/bin/env bash
# The workspace machine's build (agent.yaml: workspace.install). Runs once per version, as root,
# on Debian 13, in the agent folder, after Starwish's own `apt-get update`. Jobs then start from
# what it leaves behind, so nothing here is fetched again by a job.
set -euo pipefail

# --- system packages ----------------------------------------------------------------------------
# libgles2 and libegl1: MediaPipe's sound classifier and face model open libGLESv2 when they
# load, even on a machine with no graphics card (the first dev run failed without them).
apt-get install -y ffmpeg libgl1 libglib2.0-0 libgles2 libegl1

# --- the engine ------------------------------------------------------------------------------
wheels=(vendor/*.whl)
if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
  echo "install.sh: expected one wheel in vendor/ (run build.sh before uploading)" >&2
  exit 1
fi
# --ignore-installed: Debian owns click 8.1.8 and pip cannot uninstall it.
pip install --ignore-installed "./${wheels[0]}[stream]"

# Every engine command reads /etc/streamclip-env.sh first (see streamclip-env.sh and bin/streamclip).
install -m 644 streamclip-env.sh /etc/streamclip-env.sh
install -m 755 bin/streamclip /usr/local/bin/streamclip
# prompt.md runs each stage through this: started in the background, then polled.
install -m 755 bin/clipstage /usr/local/bin/clipstage
found="$(command -v streamclip)"
if [ "$found" != /usr/local/bin/streamclip ]; then
  echo "install.sh: streamclip resolves to $found, not the agent's /usr/local/bin/streamclip" >&2
  exit 1
fi
# shellcheck source=streamclip-env.sh
. /etc/streamclip-env.sh
mkdir -p "$CLIPPER_HOME/assets" "$STREAMCLIP_MODELS" "$HF_HOME" "$(dirname "$STREAMCLIP_TWITCHDOWNLOADER")"

# --- Twitch chat (optional) ------------------------------------------------------------------
# TwitchDownloaderCLI, a fixed release checked against its SHA-256. Chat is an extra: when this
# fails the build goes on, and the engine notes chat as missing in each job's stream.json.
CHAT_URL=https://github.com/lay295/TwitchDownloader/releases/download/1.56.5/TwitchDownloaderCLI-1.56.5-Linux-x64.zip
CHAT_SHA256=f2bcee00d0e6da759d97c23767345634cb9cdb10cdff96cfa96f97983d51bf55

install_chat_tool() {
  # Called after `||`, where bash ignores `set -e`: every step returns on failure itself.
  local work
  work="$(mktemp -d)" || return 1
  curl -fsSL --retry 3 -o "$work/cli.zip" "$CHAT_URL" || { rm -rf "$work"; return 1; }
  (cd "$work" && echo "$CHAT_SHA256  cli.zip" | sha256sum -c --quiet -) || { rm -rf "$work"; return 1; }
  python3 -m zipfile -e "$work/cli.zip" "$work/x" || { rm -rf "$work"; return 1; }
  local binary
  binary="$(find "$work/x" -type f -name TwitchDownloaderCLI | head -n 1)"
  [ -n "$binary" ] || { echo "install.sh: TwitchDownloaderCLI is not in the zip" >&2; rm -rf "$work"; return 1; }
  install -m 755 "$binary" "$STREAMCLIP_TWITCHDOWNLOADER" || { rm -rf "$work"; return 1; }
  rm -rf "$work"
  # It must start here, or the engine would try it on every job. `--version` exits 1 even when
  # it works (the tool exits 1 for anything but a verb), so look at what it printed instead.
  local said
  said="$("$STREAMCLIP_TWITCHDOWNLOADER" --version 2>&1)" || true
  case "$said" in
    *1.56.5*) ;;
    *) rm -f "$STREAMCLIP_TWITCHDOWNLOADER"
       echo "install.sh: TwitchDownloaderCLI does not start: ${said:0:300}" >&2
       return 1 ;;
  esac
  echo "install.sh: TwitchDownloaderCLI 1.56.5 installed"
}
install_chat_tool || echo "install.sh: Twitch chat tool not installed; jobs will go on without chat" >&2

# --- models ----------------------------------------------------------------------------------
# The sound classifier, both speech-to-text sizes and the face model, so no job downloads one.
streamclip warm

# The ffmpeg every engine test runs against (static 9.0.2 with libass, checked by SHA-256),
# not Debian's 7.1, which crashed in the render ("malloc(): corrupted top size") on the
# second dev run. It goes to $CLIPPER_HOME/.tools/bin, where the engine looks first, and the
# engine puts its folder first on PATH for yt-dlp too. Debian's stays for anything else.
python3 - <<'PY'
from clipper import provision, shell

# Not provision.ensure: it keeps any ffmpeg on the PATH that has the filters, and Debian's does.
provision._install_ffmpeg()
chosen = shell.ffmpeg()
assert chosen.startswith("/opt/streamclip/.tools/bin/"), chosen
assert provision.ffmpeg_ok(), "the downloaded ffmpeg lacks a filter the engine needs"
print("install.sh: the engine uses", chosen)
PY

# Load both MediaPipe models once, so a missing system library fails this build, not a job.
python3 - <<'PY'
import numpy as np

from clipper.reframe import faces
from streamclip.signals import events

events.default_classifier()(np.zeros(16000, dtype=np.float32))
faces._make_landmarker()
print("install.sh: the sound classifier and the face model load")
PY

# A job's commands may not run as root: let them read the models and write their lock files.
chmod -R a+rwX /opt/streamclip
