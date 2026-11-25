# LibreClipper Gameplay Stream

Turn a past Twitch gameplay broadcast into finished vertical clips for TikTok, Reels and
Shorts, on your own computer, with nobody reviewing in between.

Give it a link to a finished broadcast (or a video file) and it returns 1080×1920 clips of the
stream's loudest, funniest moments: the streamer's face above the game, a title, big
word-by-word captions, a creator tag and sound effects on the beats. Each clip comes with a
caption to paste and hashtags, and `clips.json` lists them best first, with a score and the
reason each was picked.

This is the gameplay-stream engine from [LibreClipper](https://github.com/yukitanakachez/libreclipper),
the open-source podcast clipper, released on its own.

## How it works

1. **Find.** It downloads the broadcast with yt-dlp, scans the sound for spikes, laughs and
   shouting, and shortlists the hot spots. Each one gets a transcript (Whisper) and a sheet of
   frames.
2. **Judge.** One call to Claude reads the shortlist and picks the moments worth posting, with
   where each clip should start and end, a title and a caption.
3. **Make.** It finds the streamer's camera box, lays the face over the gameplay, cuts dead
   air, burns in the captions and sound effects, and renders each clip with FFmpeg.

## Get started

You need Python 3.11+, [uv](https://docs.astral.sh/uv/) and an
[Anthropic API key](https://console.anthropic.com/). It runs on macOS and Linux. If your
FFmpeg can't burn in captions (it needs libass), `streamclip warm` downloads a pinned build
into the project folder.

```bash
git clone https://github.com/yukitanakachez/libreclipper-gameplaystream.git
cd libreclipper-gameplaystream
uv sync --extra stream
export ANTHROPIC_API_KEY=sk-ant-...
uv run streamclip warm            # downloads the models (and FFmpeg if needed) once
```

Clip a broadcast from the command line:

```bash
uv run streamclip run https://www.twitch.tv/videos/<id> --clips 5 --creator "@streamer"
```

Finished clips land in `stream-jobs/<name>/out/`. Or use the local test page, which shows
each job's progress and plays the clips:

```bash
uv run streamclip serve           # http://127.0.0.1:8765
```

### Commands

| Command | What it does |
|---|---|
| `run <link or file>` | find, judge and make in one go |
| `find <link or file> --job DIR` | shortlist the hot spots and prepare them for the judge |
| `judge DIR` | ask the judge, and turn its answer into the chosen moments |
| `make DIR [--only N]` | render the chosen moments (or just clip N again) |
| `status DIR` | where a job is, in one line |
| `warm` | download the models once, so no job has to |
| `serve` | start the local test page |

Useful options: `--clips N` (1 to 10, default 6), `--creator NAME`, `--no-sounds`,
`--speed`, `--dead-air SECONDS`. Broadcasts must be finished (not live) and at most 6 hours.

## Starwish agent

`agents/gameplay-clipper/` packages the engine as an agent for
[Starwish](https://starwish.ing): run `bash agents/gameplay-clipper/build.sh` before uploading.

## Tests

```bash
uv sync --extra stream
uv run pytest                     # the network and paid tests are deselected by default
```

The test stream in `agents/gameplay-clipper/tests/sample-stream.mp4` is built by
`scripts/make_stream_sample.py` from public-domain NASA footage and a game drawn by the script;
see [its sources](agents/gameplay-clipper/tests/SOURCES.md).

## Privacy

Everything heavy (download, speech-to-text, face tracking, rendering) runs on your computer.
Only the shortlisted moments' transcripts and frame sheets are sent to Anthropic, to pick the
clips. Nothing is ever posted for you. The test page listens on `127.0.0.1` only and has no
sign-in: don't put it on a public address.

## Credits

- Titles use [Anton](https://fonts.google.com/specimen/Anton) (SIL Open Font License,
  `streamclip/assets/fonts/OFL-Anton.txt`).
- Downloads by [yt-dlp](https://github.com/yt-dlp/yt-dlp), video by [FFmpeg](https://ffmpeg.org),
  speech-to-text by [Whisper](https://github.com/openai/whisper) (mlx-whisper or
  faster-whisper), faces by Google's [MediaPipe](https://developers.google.com/mediapipe).

## License and legal

MIT: see [LICENSE](LICENSE).

**Clip only streams you own or have the streamer's permission to clip**, and follow Twitch's
terms and each streamer's clipping policy. This project is not affiliated with Twitch or any
streamer.
