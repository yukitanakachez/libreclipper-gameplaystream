#!/usr/bin/env bash
# Makes agents/gameplay-clipper ready to upload. Run it from anywhere before every
# `starwish dev`, `test` or `publish`:
#
#   bash agents/gameplay-clipper/build.sh
#
# 1. Builds the engine's wheel into vendor/ (git ignores it; .starwishignore uploads it).
# 2. Copies the judge's three files from streamclip/judge/ into tools/, where the `judge` tool
#    runs them on the harness machine. Those copies are committed, and
#    tests/stream/test_stream_agent_folder.py fails when they drift from the engine's.
#
# Uses $UV when it is set (this repo's own uv), else `uv` on the PATH.
set -euo pipefail

AGENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$AGENT/../.." && pwd)"
UV="${UV:-uv}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# uv writes a .gitignore into its output folder, so build elsewhere and move only the wheel.
(cd "$ROOT" && "$UV" build --wheel --out-dir "$work" >&2)
wheels=("$work"/*.whl)
[ "${#wheels[@]}" -eq 1 ] && [ -f "${wheels[0]}" ] || { echo "build.sh: expected one wheel, got: ${wheels[*]}" >&2; exit 1; }

mkdir -p "$AGENT/vendor"
rm -f "$AGENT"/vendor/*.whl
cp "${wheels[0]}" "$AGENT/vendor/"

for name in call.py prompt.md answer_schema.json; do
  target="$name"
  [ "$name" = call.py ] && target=judge_call.py
  cp "$ROOT/streamclip/judge/$name" "$AGENT/tools/$target"
done

echo "build.sh: vendor/$(basename "${wheels[0]}") and tools/judge_call.py, prompt.md, answer_schema.json are ready"
