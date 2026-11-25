"""The gameplay-clipper Starwish agent's folder: what gets uploaded, and the small programs in it.

Nothing here talks to Starwish. `bash agents/gameplay-clipper/build.sh` refreshes the copies
these tests compare against the engine.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
AGENT = ROOT / "agents" / "gameplay-clipper"
JUDGE_FILES = ("call.py", "prompt.md", "answer_schema.json")


def _yaml() -> dict:
    return yaml.safe_load((AGENT / "agent.yaml").read_text())


# --- the judge tool --------------------------------------------------------------------------

def test_the_judge_tool_is_the_engines_own_file():
    built = (AGENT / "tools" / "judge_call.py").read_text()
    assert built == (ROOT / "streamclip" / "judge" / "call.py").read_text()
    for name in JUDGE_FILES[1:]:
        assert (AGENT / "tools" / name).read_text() == (ROOT / "streamclip" / "judge" / name).read_text()


def test_the_judge_tool_imports_only_the_standard_library():
    tool = AGENT / "tools" / "judge"
    assert os.access(tool, os.X_OK)
    assert tool.read_text().startswith("#!/usr/bin/env python3\n")
    for path in (tool, AGENT / "tools" / "judge_call.py"):
        tree = ast.parse(path.read_text())
        names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
        names |= {node.module.split(".")[0] for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom) and node.module and node.level == 0}
        assert names - {"judge_call"} <= set(sys.stdlib_module_names), path


def _run_judge(env: dict) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if k not in ("JUDGE_ANTHROPIC_KEY", "ANTHROPIC_API_KEY",
                                                              "STARWISH_OUTPUT_DIR")}
    return subprocess.run([sys.executable, str(AGENT / "tools" / "judge")], env={**clean, **env},
                          capture_output=True, text=True, timeout=60)


def test_the_judge_tool_without_an_output_folder_says_so(tmp_path):
    done = _run_judge({})
    assert done.returncode == 2
    assert "STARWISH_OUTPUT_DIR" in done.stderr and "Traceback" not in done.stderr


def test_the_judge_tool_with_nothing_to_judge_exits_2(tmp_path):
    done = _run_judge({"STARWISH_OUTPUT_DIR": str(tmp_path)})    # there is no /job/files/work here
    assert done.returncode == 2
    assert "no candidates" in done.stderr and "Traceback" not in done.stderr
    assert not list(tmp_path.iterdir())


# --- agent.yaml --------------------------------------------------------------------------------

def test_agent_yaml_says_what_the_spec_says():
    y = _yaml()
    assert y["name"] == "gameplay-clipper" and y["display_name"] == "Gameplay Clipper"
    assert y["price"] == 1000 and "workspace" in y and "credentials" not in y["workspace"]
    (tool,) = y["tools"]
    assert tool["name"] == "judge" and list(tool["credentials"]) == ["JUDGE_ANTHROPIC_KEY"]
    assert tool["credentials"]["JUDGE_ANTHROPIC_KEY"]["hosts"] == ["api.anthropic.com"]
    assert tool["credentials"]["JUDGE_ANTHROPIC_KEY"]["header"] == "x-api-key: {value}"
    assert "args" not in tool and len(tool["actions"]) == 1          # the model cannot pass its own arguments
    assert tool["actions"][0]["args"] == []
    for phrase in ("English", "own or have permission", "6 hours", "past Twitch broadcast"):
        assert phrase in y["description"]
    assert len(y.get("tests") or []) <= 10


def test_agent_yaml_keeps_the_values_only_yuki_sets():
    y = _yaml()
    assert y["max_response_time"] == "1h30m"
    assert y["harness"] == {"preset": "claude-code", "model": "claude-sonnet-5-5"}
    assert y["workspace"]["install"] == "./install.sh"


def test_agent_yaml_inputs_and_outputs():
    y = _yaml()
    params = y["inputs"]["params"]["properties"]
    assert params["clips"] == {"type": "integer", "minimum": 1, "maximum": 10, "default": 6,
                               "description": params["clips"]["description"]}
    assert params["creator"]["maxLength"] == 60 and params["url"]["type"] == "string"
    assert not y["inputs"]["params"].get("required")
    (files,) = y["inputs"]["files"]
    assert files["max_count"] == 1 and files["required"] is False
    assert set(files["accept"]) == {"video/mp4", "video/quicktime", "video/x-matroska", "video/webm"}
    out = {f["name"]: f for f in y["outputs"]["files"]}
    # Optional: a refusal or a request with no stream writes nothing, and a job missing a
    # required output fails review, which refunds the customer and lowers the agent's rating.
    assert not any(f.get("required") for f in out.values())
    assert {"*.mp4", "*.txt"} <= set(out)


# --- what is uploaded --------------------------------------------------------------------------

def test_nothing_secret_or_heavy_is_in_the_folder():
    files = [p for p in AGENT.rglob("*") if p.is_file() and "vendor" not in p.parts]
    assert all(p.stat().st_size < 30_000_000 for p in files)
    assert not any(p.suffix in (".env", ".pem", ".key") or p.name.startswith(".env") for p in files)


def test_the_check_agent_is_gone():
    assert not (AGENT / "probe").exists() and not (AGENT / "tools" / "probe").exists()


def test_the_wheel_is_uploaded_though_git_ignores_it():
    lines = (AGENT / ".starwishignore").read_text().splitlines()
    assert "!vendor/" in lines and "!vendor/*.whl" in lines
    assert "agents/gameplay-clipper/vendor/" in (ROOT / ".gitignore").read_text().splitlines()


# --- the workspace machine's install -------------------------------------------------------------

def test_build_and_install_scripts_are_executable_shell():
    for name in ("build.sh", "install.sh", "bin/streamclip"):
        path = AGENT / name
        assert os.access(path, os.X_OK), name
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_install_follows_what_the_starwish_check_found():
    text = (AGENT / "install.sh").read_text()
    assert "--ignore-installed" in text                      # Debian owns click
    assert re.search(r"^streamclip warm$", text, re.M)       # no job downloads a model
    assert "/usr/local/bin/streamclip" in text               # every engine command gets the settings
    assert "provision._install_ffmpeg()" in text            # the ffmpeg the tests run on
    assert "libgles2" in text                                # MediaPipe opens libGLESv2 when it loads
    assert "events.default_classifier()(" in text and "faces._make_landmarker()" in text


def test_install_pins_the_chat_tool_and_never_fails_over_it():
    text = (AGENT / "install.sh").read_text()
    assert "TwitchDownloaderCLI-1.56.5-Linux-x64.zip" in text
    assert re.search(r"\b[0-9a-f]{64}\b", text)
    assert "sha256sum -c" in text
    assert re.search(r"^install_chat_tool \|\| ", text, re.M)


def test_every_engine_command_gets_the_agents_settings():
    env = (AGENT / "streamclip-env.sh").read_text()
    assert re.search(r"^export STREAMCLIP_X264_PRESET=ultrafast$", env, re.M)
    for model in ("STREAMCLIP_WHISPER", "STREAMCLIP_WINDOW_MODEL", "STREAMCLIP_CAPTION_MODEL"):
        assert model not in env                               # the engine's defaults stand
    wrapper = (AGENT / "bin" / "streamclip").read_text()
    assert ". /etc/streamclip-env.sh" in wrapper and "exec python3 -m streamclip" in wrapper


# --- clipstage: long stages in the background, polled in short calls ---------------------------
# The harness gives up on a command that is silent for 300 s, so the agent starts each stage with
# `clipstage start` and checks it with `clipstage poll`. A fake `streamclip` stands in here.

FAKE_STREAMCLIP = r"""#!/usr/bin/env bash
root="$CLIPSTAGE_ROOT"
if [ "$1" = status ]; then
  cat "$root/fake/status" 2>/dev/null || echo "new"
  exit 0
fi
printf '%s\n' "$@" > "$root/fake/args.$1"
echo "Fetching the stream's sound…"
if [ -f "$root/fake/noise" ]; then cat "$root/fake/noise"; fi
sleep "$(cat "$root/fake/sleep" 2>/dev/null || echo 0)"
echo "Scoring every second of the sound…"
[ -f "$root/fake/status.after" ] && cp "$root/fake/status.after" "$root/fake/status"
exit "$(cat "$root/fake/exit" 2>/dev/null || echo 0)"
"""

FAKE_PROGRESS = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CLIPSTAGE_ROOT/fake/progress"
"""


def _stage_env(tmp_path: Path) -> dict:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    for name, text in (("streamclip", FAKE_STREAMCLIP), ("starwish-job", FAKE_PROGRESS)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    (tmp_path / "fake").mkdir()
    (tmp_path / "run").mkdir()
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CLIPSTAGE_ROOT": str(tmp_path), "CLIPSTAGE_WAIT": "6"}


def _clipstage(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(AGENT / "bin" / "clipstage"), *args], env=env,
                          capture_output=True, text=True, timeout=60)


def _finish(env: dict, stage: str) -> subprocess.CompletedProcess:
    for _ in range(5):
        done = _clipstage(env, "poll", stage)
        if not done.stdout.startswith("running"):
            return done
    raise AssertionError("the stage never finished")


def test_clipstage_passes_customer_text_as_single_arguments(tmp_path):
    env = _stage_env(tmp_path)
    source = "/job/input/my clip's \"best\" $(touch pwned).mp4"
    (tmp_path / "run" / "source").write_text(source + "\n")
    (tmp_path / "run" / "clips").write_text("2\n")
    (tmp_path / "run" / "creator").write_text("Big Streamer `whoami`")
    started = _clipstage(env, "start", "find")
    assert started.returncode == 0 and started.stdout.startswith("started"), started
    assert "machine:" in started.stdout
    done = _finish(env, "find")
    assert done.stdout.startswith("done") and "took " in done.stdout
    assert re.search(r"^\d\d:\d\d:\d\d Scoring every second of the sound…$", done.stdout, re.M)
    args = (tmp_path / "fake" / "args.find").read_text().splitlines()
    assert args == ["find", source, "--job", str(tmp_path / "work"), "--clips", "2",
                    "--creator", "Big Streamer `whoami`"]
    assert not list(tmp_path.rglob("pwned"))


def test_clipstage_find_without_creator_or_clips(tmp_path):
    env = _stage_env(tmp_path)
    (tmp_path / "run" / "source").write_text("https://www.twitch.tv/videos/123")
    assert _clipstage(env, "start", "find").returncode == 0
    _finish(env, "find")
    assert (tmp_path / "fake" / "args.find").read_text().splitlines() == [
        "find", "https://www.twitch.tv/videos/123", "--job", str(tmp_path / "work"), "--clips", "6"]


def test_clipstage_refuses_to_start_without_a_source_or_with_a_bad_count(tmp_path):
    env = _stage_env(tmp_path)
    missing = _clipstage(env, "start", "find")
    assert missing.returncode != 0 and "source" in missing.stdout
    (tmp_path / "run" / "source").write_text("https://www.twitch.tv/videos/123")
    (tmp_path / "run" / "clips").write_text("11")
    bad = _clipstage(env, "start", "find")
    assert bad.returncode != 0 and "1 to 10" in bad.stdout
    assert not (tmp_path / "fake" / "args.find").exists()


def test_clipstage_judge_and_make_arguments(tmp_path):
    env = _stage_env(tmp_path)
    raw = tmp_path / "tool-output" / "c3" / "judge_raw.json"
    (tmp_path / "run" / "raw").write_text(str(raw))
    _clipstage(env, "start", "judge")
    _finish(env, "judge")
    assert (tmp_path / "fake" / "args.judge").read_text().splitlines() == [
        "judge", str(tmp_path / "work"), "--raw", str(raw)]
    _clipstage(env, "start", "make")
    _finish(env, "make")
    assert (tmp_path / "fake" / "args.make").read_text().splitlines() == [
        "make", str(tmp_path / "work"), "--out", str(tmp_path / "output")]


def test_clipstage_judge_without_an_answer_ranks_by_excitement(tmp_path):
    env = _stage_env(tmp_path)
    _clipstage(env, "start", "judge")
    _finish(env, "judge")
    assert (tmp_path / "fake" / "args.judge").read_text().splitlines() == ["judge", str(tmp_path / "work")]


def test_clipstage_poll_reports_progress_once_per_new_line(tmp_path):
    env = {**_stage_env(tmp_path), "CLIPSTAGE_WAIT": "1"}
    (tmp_path / "run" / "source").write_text("https://www.twitch.tv/videos/123")
    (tmp_path / "fake" / "sleep").write_text("4")
    (tmp_path / "fake" / "status").write_text("find sound")
    _clipstage(env, "start", "find")
    first = _clipstage(env, "poll", "find")
    second = _clipstage(env, "poll", "find")
    assert first.returncode == 0 and first.stdout.startswith("running: find sound")
    assert second.stdout.startswith("running")
    assert (tmp_path / "fake" / "progress").read_text().splitlines() == [
        "progress Fetching the stream's sound…"]
    assert _finish({**env, "CLIPSTAGE_WAIT": "6"}, "find").stdout.startswith("done")


MEDIAPIPE_NOISE = """INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1795477651.008062    1699 inference_feedback_manager.cc:114] Feedback manager requires a model
Exception ignored in: <function AudioClassifier.__del__ at 0x7fd9788391c0>
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/mediapipe/tasks/python/audio/audio_classifier.py", line 456
TypeError: 'NoneType' object is not callable
"""


def test_clipstage_passes_on_only_the_engines_own_lines(tmp_path):
    env = {**_stage_env(tmp_path), "CLIPSTAGE_WAIT": "1"}
    (tmp_path / "run" / "source").write_text("https://www.twitch.tv/videos/123")
    (tmp_path / "fake" / "sleep").write_text("4")
    (tmp_path / "fake" / "noise").write_text(MEDIAPIPE_NOISE)
    _clipstage(env, "start", "find")
    _clipstage(env, "poll", "find")
    assert (tmp_path / "fake" / "progress").read_text().splitlines() == [
        "progress Fetching the stream's sound…"]
    done = _finish({**env, "CLIPSTAGE_WAIT": "6"}, "find")
    assert "W0000" not in done.stdout and "Traceback" not in done.stdout      # the timeline too


def test_the_camera_lines_always_reach_the_job_s_log(tmp_path):
    # The timeline keeps its last 40 lines, and make prints more than that for five clips:
    # run 10 lost its "Camera:" line, the one that says whether the streamer's camera was found.
    env = _stage_env(tmp_path)
    many = "".join(f"Clip {n // 9 + 1} of 5: step {n}\n" for n in range(60))
    (tmp_path / "fake" / "noise").write_text(
        "Looking for the streamer's camera…\nCamera: found.\n" + many)
    _clipstage(env, "start", "make")
    done = _finish(env, "make")
    lines = done.stdout.splitlines()
    assert any(line.endswith(" Looking for the streamer's camera…") for line in lines)
    assert any(line.endswith(" Camera: found.") for line in lines)
    assert lines[-1].endswith(" Scoring every second of the sound…")          # the tail is still there
    assert not any(line.endswith(" step 5") for line in lines)                 # and still only a tail


def test_the_engines_libraries_log_only_errors():
    env = (AGENT / "streamclip-env.sh").read_text()
    assert re.search(r"^export GLOG_minloglevel=2$", env, re.M)
    assert re.search(r"^export TF_CPP_MIN_LOG_LEVEL=2$", env, re.M)


def test_clipstage_a_refusal_carries_the_engines_reason(tmp_path):
    env = _stage_env(tmp_path)
    (tmp_path / "run" / "source").write_text("https://www.twitch.tv/twitch")
    (tmp_path / "fake" / "exit").write_text("2")
    (tmp_path / "fake" / "status.after").write_text(
        "failed: That is a channel, not a past broadcast. Send the link of one finished video.\n")
    _clipstage(env, "start", "find")
    done = _finish(env, "find")
    assert done.returncode == 2
    assert done.stdout.splitlines()[0] == (
        "refused: That is a channel, not a past broadcast. Send the link of one finished video.")


def test_clipstage_a_failure_shows_the_reason_and_the_log(tmp_path):
    env = _stage_env(tmp_path)
    (tmp_path / "fake" / "exit").write_text("1")
    (tmp_path / "fake" / "status.after").write_text("failed: No clip could be made: no video\n")
    _clipstage(env, "start", "make")
    done = _finish(env, "make")
    assert done.returncode == 1
    assert done.stdout.startswith("failed: No clip could be made: no video")
    assert "Scoring every second of the sound…" in done.stdout


def test_clipstage_does_not_start_a_stage_twice(tmp_path):
    env = _stage_env(tmp_path)
    (tmp_path / "fake" / "sleep").write_text("2")
    _clipstage(env, "start", "make")
    again = _clipstage(env, "start", "make")
    assert again.returncode == 0 and again.stdout.startswith("already started")
    _finish(env, "make")
    assert _clipstage(env, "start", "make").stdout.startswith("already started")


def test_clipstage_rejects_unknown_commands(tmp_path):
    env = _stage_env(tmp_path)
    assert _clipstage(env, "start", "run").returncode == 64
    assert _clipstage(env, "wait", "find").returncode == 64


def test_install_puts_clipstage_on_the_path():
    assert "install -m 755 bin/clipstage /usr/local/bin/clipstage" in (AGENT / "install.sh").read_text()
    assert os.access(AGENT / "bin" / "clipstage", os.X_OK)


# --- prompt.md -------------------------------------------------------------------------------------

def test_prompt_runs_every_stage_through_clipstage():
    text = (AGENT / "prompt.md").read_text()
    for stage in ("find", "judge", "make"):
        assert f"clipstage start {stage}" in text and f"clipstage poll {stage}" in text
    assert not re.search(r"`streamclip (find|judge|make|run)\b", text)     # never in the foreground
    for word in ("running:", "done:", "refused:", "failed:"):
        assert word in text
    for name in ("/job/run/source", "/job/run/clips", "/job/run/creator", "/job/run/raw"):
        assert name in text
    clipstage = (AGENT / "bin" / "clipstage").read_text()
    assert all(f'"$RUN/{name}"' in clipstage for name in ("source", "clips", "creator", "raw"))


def test_prompt_finds_the_judges_answer_by_looking():
    text = (AGENT / "prompt.md").read_text()
    assert "find /job/tool-output -name judge_raw.json" in text
    assert "/c1/" not in text
    assert "find /job/input -type f" in text


def test_prompt_says_what_it_needs_when_there_is_no_input():
    text = (AGENT / "prompt.md").read_text()
    needed = re.search(r'"(I need the link of one past Twitch broadcast[^"]*)"', text)
    assert needed and "video file" in needed.group(1)
    assert "Ask no questions" in text and "clips.json" in text and "notes" in text


def test_agent_yaml_has_its_three_starwish_tests():
    tests = {t["name"]: t for t in _yaml()["tests"]}
    assert set(tests) == {"sample-two-clips", "refuses-a-channel-link", "asks-for-input-when-there-is-none"}
    sample = tests["sample-two-clips"]
    assert sample["params"] == {"clips": 2} and (AGENT / sample["input"]).is_file()
    assert sample["input"] == "./tests/sample-stream.mp4"
    assert "twitch.tv/twitch" in tests["refuses-a-channel-link"]["request"]
    assert "input" not in tests["asks-for-input-when-there-is-none"]
    for test in tests.values():
        assert test["must"].strip()
