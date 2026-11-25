# Settings every engine command on the workspace machine runs with. install.sh copies this file
# to /etc/streamclip-env.sh, and /usr/local/bin/streamclip (bin/streamclip here) reads it before
# each command, so the install's `streamclip warm` and every job agree on where things are.

# Measured by the Starwish check: only ultrafast encodes a 40 s clip in under a minute on the
# slower of the machines Starwish hands out.
export STREAMCLIP_X264_PRESET=ultrafast

# One place for every model and download, filled once by `streamclip warm` at install time.
export CLIPPER_HOME=/opt/streamclip                 # the face model goes in its assets/
export STREAMCLIP_MODELS=/opt/streamclip/models     # the sound classifier
export HF_HOME=/opt/streamclip/huggingface          # the speech-to-text models

# Twitch chat. When install.sh could not install it, the file is missing and the engine
# carries on without chat. The tool is a .NET program: this lets it run without ICU.
export STREAMCLIP_TWITCHDOWNLOADER=/opt/streamclip/bin/TwitchDownloaderCLI
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

# MediaPipe (glog) and TensorFlow Lite: errors only. Their warnings filled the job's log and,
# once, the customer's live progress line.
export GLOG_minloglevel=2
export TF_CPP_MIN_LOG_LEVEL=2
