# Stream clipper: tuning notes (Lane N)

Tuning ran on Yuki's Mac on 2026-11-23, not on Starwish. Footage lives in `~/stream-tuning/`,
outside the repo. Only measurements, hot-spot times, scores and these notes are kept here.

## The set

Six past Twitch broadcasts from five channels. Each was public and finished on 2026-11-23, and
each was confirmed with the engine's describe step (the downloader's metadata call) before it
was chosen.

| Name | Channel | Broadcast | Date | Length | Game (from the chapters) | Why it is in the set |
|---|---|---|---|---|---|---|
| `speed` | IShowSpeed | <https://www.twitch.tv/videos/2934967396> | 2026-11-20 | 4 h 04 m | Grand Theft Auto VI | Yuki's pick. Loud, camera over a game, over four hours. |
| `kai` | Kai Cenat | <https://www.twitch.tv/videos/2936683310> | 2026-11-21 | 6 h 45 m | Grand Theft Auto VI, with 44 minutes of IRL in the middle | Yuki's pick. The longest in the set, so it is the timing stream for the "six hours in under an hour" target. |
| `jynxzi` | Jynxzi | <https://www.twitch.tv/videos/2937733476> | 2026-11-22 | 5 h 14 m | Just Chatting, Chess, Rainbow Six Siege | Loud gameplay channel. The same broadcast the Starwish check used, so times can be compared. |
| `caseoh` | CaseOh | <https://www.twitch.tv/videos/2932003293> | 2026-11-18 | 4 h 01 m | Family Feud, Trees Hate You, Suzy Says!, Minecraft | Loud gameplay channel, several games in one stream. |
| `nl` | Northernlion | <https://www.twitch.tv/videos/2932471430> | 2026-11-18 | 4 h 53 m | Slay the Spire II, Super Mario Party Jamboree | The calmer channel: talk over a game, few screams. The check against fitting only the loudest streamers. |
| `speedcam` | IShowSpeed | <https://www.twitch.tv/videos/2908755591> | 2026-10-23 | 2 h 35 m | Just Chatting, then IRL | Camera only, no game on screen, to exercise that layout. |

Not in the set: xQc. Every recent broadcast of his is between 8 and 23 hours, and the long
timing stream is already covered by `kai`. Kai Cenat's and IShowSpeed's first GTA 6 broadcasts
run over 47 hours and are refused by the 12-hour limit.

## How it was measured

`scripts/stream_tune.py` reads the job folders and prints the tables below. Its columns:

- **Hit rate.** Of the stream's viewer clips, weighted by views, the share whose middle falls
  inside a shortlisted window.
- **Hit rate, clips withheld.** The same, with the shortlist built again from the saved sound
  and chat but with viewer clips left out of the curve. Viewer clips are both a signal and the
  answer key, so this is the only fair test of the sound and chat signals. Tuning went by it.
- **Judge keep rate.** The share of the shortlist kept with a score of 7 or more.
- **Busiest 30 min.** The share of the shortlist that falls in its busiest half hour.

Three limits on these numbers, all found while measuring:

1. **The answer key is thin.** Twitch serves only the first page of each clip order without a
   browser token, so a stream has 2 to 50 clips, and weighted by views one or two of them
   decide a stream's whole rate. A 24-window shortlist also covers 9% to 23% of a stream by
   chance alone. Differences of a few points mean nothing.
2. **Twitch mutes parts of past broadcasts.** 15% of `kai`, 11% of `speed` and 7% of
   `speedcam` is silent in the download. The most-viewed clip of `speed` (8,219 views, the
   game's ending) sits in a muted stretch, so no setting can find it.
3. **Who judged.** There is no model key meant for tuning on this machine. The first run
   (`speedcam`) found one in the shell's environment and the engine made one real judge call
   (`claude-opus-5-5`, 24 candidates) before it was noticed; every later run had the key names
   unset. `nl` and `speed` were judged by the tuning agent (Claude Fable 5.2) acting as the
   judge: it read each candidate's transcript and contact sheets, followed
   `streamclip/judge/prompt.md`, and its answer went in through `streamclip judge --raw`. For
   `nl` it read all 24 transcripts and 12 of the 24 sheets; for `speed` all 24 transcripts and
   16 of the 24 sheets (every candidate it kept, and the ones with little speech). So the keep
   rates of `nl` and `speed` are the tuning agent's, not the production judge's. The other
   three streams have no keep rate.

## Before

The engine as merged (Lanes A to L), run once per stream with `scripts/stream_tune.py run`,
one stream at a time on this Mac (Apple M4 Pro, 14 processors).

| stream | length | shortlist | viewer clips | hit rate | hit rate, clips withheld | judge keep rate | busiest 30 min | extras | dead air (clip: seconds cut/cuts) |
|---|---|---|---|---|---|---|---|---|---|
| caseoh | 4.01 h | 24 | 5 | 6% | 6% | | 50% | chat timed out at 300 s, viewer clips yes, picture yes | 01: 0.0 s/0; 02: 1.1 s/1; 03: 1.6 s/1 |
| jynxzi | 5.23 h | 24 | 28 | 14% | 0% | | 54% | chat cut short (see below), viewer clips yes, picture yes, chat fetch 43 s | 01: 0.0 s/0; 02: 9.6 s/8; 03: 2.3 s/3; 04: 12.9 s/7; 05: 6.8 s/4 |
| kai | 6.75 h | 24 | 50 | 9% | 0% | | 29% | chat yes, viewer clips yes, picture yes, chat fetch 176 s | 01: 12.8 s/5; 02: 0.0 s/0; 03: 0.0 s/0; 04: 0.0 s/0; 05: 1.9 s/1 |
| nl | 4.89 h | 24 | 13 | 73% | 7% | 21% (tuning agent) | 29% | chat yes, viewer clips yes, picture yes, chat fetch 64 s | 01: 3.8 s/4; 02: 3.9 s/3; 03: 0.5 s/1; 04: 1.6 s/1; 05: 1.3 s/2 |
| speed | 4.07 h | 24 | 47 | 19% | 19% | 21% (tuning agent) | 46% | chat timed out at 300 s, viewer clips yes, picture yes | 01: 0.8 s/1; 02: 2.9 s/4; 03: 3.4 s/5; 04: 0.0 s/0; 05: 0.0 s/0 |
| speedcam | 2.59 h | 24 | 2 | 50% | 0% | 4% (production judge) | 50% | chat yes, viewer clips yes, picture yes, chat fetch 96 s | 01 to 05: 0.0 s/0 |
| total | 27.54 h | 144 | 145 | 31% | 3% | 15% | | | 67.2 s in 51 cuts, 28 clips |

The clips in this table were made from picks ranked by excitement alone (no judge), except
`speedcam`. The withheld column was worked out afterwards with every stream's chat fetched in
full.

### Stage times on this Mac, before

Seconds. Starwish's machines are 3 to 10 times slower for speech-to-text and encoding (see
the check report), so these are a floor, not a forecast.

| stream | length | find | of which sound | scoring | windows | judge | make | of which video fetch | second listen | render | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| caseoh | 4.01 h | 571 | 300 | 69 | 201 | 0 | 206 | 66 | 35 | 77 | 777 |
| jynxzi | 5.23 h | 547 | 273 | 46 | 227 | 0 | 226 | 51 | 92 | 48 | 773 |
| kai | 6.75 h | 365 | 176 | 60 | 128 | 0 | 218 | 69 | 24 | 99 | 583 |
| nl | 4.89 h | 456 | 106 | 44 | 305 | 0 | 175 | 58 | 36 | 54 | 631 |
| speed | 4.07 h | 456 | 300 | 35 | 119 | 0 | 249 | 66 | 45 | 107 | 705 |
| speedcam | 2.59 h | 455 | 96 | 28 | 329 | 52 | 230 | 78 | 30 | 94 | 737 |

- "Sound" is the wait for the sound and the extras together. On `caseoh` and `speed` the
  sound was in after 204 s and 101 s and the job then sat until chat gave up at 300 s.
- The judge's 52 s on `speedcam` is the one real call. Everywhere else there was no judge.
- `make` produced 5 clips per stream (3 on `caseoh`), not 6: with no judge the engine makes
  only the floor.
- The longest stream, `kai` (6 h 45 m), took 9 m 43 s end to end without a judge.

### Chat on big channels, before

Three things went wrong with chat, none of them a constant:

- The chat fetch is cut off at 300 s. `caseoh` (115,238 messages) takes 322 s on this Mac and
  `speed` (101,429 messages) takes 299 s, so both lost their chat, and the job waited the full
  300 s for nothing after the sound was already in.
- The chat tool sometimes returns a cut-off chat and reports success. `jynxzi` came back with
  14,266 messages covering the first 36 minutes of a 5 h 14 m stream, and the engine wrote
  "chat ok". Fetched again on its own it gave 56,150 messages.
- For the tuning that follows, every stream's chat was fetched again in full, one at a time,
  with no time limit, and written into the job by hand.

## Tuning the shortlist

Method: the sound of each stream was scored once and saved; each trial changed constants,
built the curve and the shortlist again from the saved sound and the full chat with viewer
clips withheld, and counted the view-weighted hit rate over all six streams. Two more numbers
were watched with it: the same rate with each clip counted by the logarithm of its views (so
one clip cannot decide a stream), and how many of the moments the tuning agent had judged
good on `nl` and `speed` stayed on the shortlist.

### What the signals look like on real streams

- **The voice signal is almost silent.** The sound classifier's scores for shouting, laughter
  and the like are small: the strongest second in a thousand scores 0.07 to 0.41, and almost
  nothing passes 0.5. Loudness, chat and viewer clips all reach 1. So with the starting
  weights the "voice" third of the curve added next to nothing.
- **Chat bursts rarely reach the range on the biggest chats.** A burst starts counting at 2×
  the usual rate and is full at 5×. On `caseoh` the busiest second in a thousand is 2.3× and
  on `speed` the busiest in a hundred is 3.2×: a chat that is always fast has no room to
  double. On `nl` (smaller chat) bursts of 5× to 9× are common.
- **Muted stretches make false hot spots.** Next to a stretch Twitch has muted, the ten-minute
  baseline falls to its floor of -60 dB, so ordinary sound reads as "+40 dB over" and "jumped
  45 dB". Four of the 24 hot spots on `speed` and four on `speedcam` were of this kind before
  tuning. The floor (`BASELINE_FLOOR_DB`) is the only constant that touches this; raising it
  to -30 removed most of them but put 98% of `caseoh`, a quiet stream, under the floor, so it
  was not changed. This needs a change of method (see "Outside this lane").

### The weights (kept)

| trial | loudness | surge | voice | chat | clips | hit rate, clips withheld | by log of views | `nl` | kept |
|---|---|---|---|---|---|---|---|---|---|
| start | 0.30 | 0.15 | 0.35 | 0.10 | 0.10 | 3% | 12% | 7% | |
| more voice | 0.30 | 0.15 | 1.00 | 0.10 | | 14% | 16% | 0% | no: `nl` falls |
| more chat | 0.30 | 0.15 | 0.35 | 0.40 | | 7% | 21% | 73% | part |
| more of both | 0.30 | 0.15 | 1.00 | 0.40 | | 18% | 26% | 73% | yes |
| the same, voice 0.6 | 0.30 | 0.15 | 0.60 | 0.40 | | 18% | 26% | 73% | |
| the same, chat 0.25 | 0.30 | 0.15 | 0.60 | 0.25 | | 15% | 24% | 73% | no |
| no surge | 0.30 | 0 | 1.00 | 0.40 | | 17% | 24% | 73% | no |

(The trial weights are relative; the curve divides by their sum.) The kept set, scaled to add
up to one and with viewer clips given 0.14:

`WEIGHTS = {"loudness": 0.17, "surge": 0.08, "voice": 0.39, "chat": 0.22, "clips": 0.14}`

Total hit rate with clips withheld: **3% to 18%**. `nl`: 7% to 73%. The voice weight looks
large only because the classifier's scores are small.

The viewer-clip weight cannot be tuned this way: viewer clips are the answer key, so any
higher weight "wins". At 0.14 the most-viewed clip of `jynxzi` is not on the shortlist; at
0.25 nearly every stream's top clips are (88% overall). Whether a viewer clip should all but
guarantee a place is a product decision, left for Yuki.

### Loudness and surge ranges (no change)

Tried on top of the kept weights: `LOUD_FULL_DB` 8, 10, 16, 20; `SURGE_MIN_DB`/`SURGE_FULL_DB`
4/14, 8/22, 10/26; `SURGE_LOOKBACK` 2, 5; `BASELINE_WINDOW` 300, 1200; `BASELINE_FLOOR_DB`
-27, -30, -35; `MUTED_AFTER` 10. Every one landed between 17% and 18%, and between 21% and
26% by log of views: inside the noise. None was kept.

### Chat lag and ranges (no change)

`CHAT_LAG` 3, 9, 12, 15 all lose `nl`'s main clip (73% to 8%, 8%, 66%, 0%); 6 stays.
`CHAT_MIN_RATIO`/`CHAT_FULL_RATIO` 1.5/4, 1.5/3, 1.3/2.5, 2/4, 2/8, 3/8 all score lower
(13% to 17%). `CHAT_SMOOTH` 3 and 9, `CHAT_MIN_RATE` 0.2 and 1.0, `CHAT_BASELINE` 300: no
gain. The starting values hold.

### Shortlist constants (no change)

`SMOOTH` 1, 3, 9, 15: 17%, 17%, 17%, 8%. `SPREAD_SHARE` 2, 4, 6: 18%, 18%, 19%, with fewer of
the judged-good moments of `speed` kept at 4 and 6. `MUSIC_MIN` 0.4 and `MUSIC_RUN` 10: no
change. The starting values hold.

### A short scream against a longer, quieter loud stretch (Lane C's question)

Measured on the six streams: every run of seconds 6 dB or more over the baseline was sorted
into "short and sharp" (3 s or less, peaking 15 dB or more over) and "long and milder" (6 s
or more, peaking under 15 dB over), and each was checked against the shortlist.

| | runs | on the shortlist | median curve height |
|---|---|---|---|
| short and sharp, starting weights | 320 | 33 (10%) | 0.29 |
| long and milder, starting weights | 184 | 25 (14%) | 0.28 to 0.36 by stream |
| short and sharp, tuned weights | 320 | 14 (4%) | 0.16 |
| long and milder, tuned weights | 184 | 15 (8%) | 0.15 to 0.25 by stream |

So yes: a short scream is less likely to be shortlisted than a longer, milder stretch,
although it is louder at its peak. The cause is the five-second smoothing before the peaks
are picked: a two-second burst keeps about 0.56 of its height and a stretch of five seconds
or more keeps all of it. This is a matter of method, not of a constant (`SMOOTH` 1 and 3 were
tried above and score no better), and it was not patched. How much it matters is not known:
the answer key cannot say whether the 306 short screams left out were good clips.

## Tuning dead air (no change)

Every wordless gap of half a second or more was listed for the 26 clips made after tuning,
with how far under the clip's speech level it is and how much the picture moves. Then each
setting was counted:

| `QUIET_BELOW_DB` | `STILL_BELOW` | cut, 26 clips |
|---|---|---|
| 20 | 1.5 (start) | 30.9 s in 20 cuts |
| 17 | 1.5 | 33.5 s in 23 cuts |
| 15 | 1.5 | 35.4 s in 25 cuts |
| 12 | 1.5 | 50.3 s in 42 cuts |
| 10 | 1.5 | 57.4 s in 43 cuts |
| 20 | 1.0 | 26.9 s in 17 cuts |
| 20 | 2.5 | 33.9 s in 21 cuts |
| 20 | 4.0 | 36.5 s in 23 cuts |

Frames were pulled from the gaps to see what each setting cuts:

- What 20 / 1.5 cuts now is right: `kai` clip 01 loses 12.8 s of a map screen in silence;
  `jynxzi` clips 03 and 05 lose stretches where the stream's sound is 50 dB down.
- What 12 would add is wrong: 6.5 s of `speed` clip 02, which is the pool cue being lined up
  before the one shot (the build-up of the clip), and the answers turning over on the board
  in `caseoh` clip 05.
- 15 and 17 add a second or two across all 26 clips; not worth the risk.

Most wordless gaps on real streams are not quiet at all (game sound, music, a second voice
the transcript missed), so dead air is rare: 7 of the 26 clips have any cut. The starting
values stay. This was judged from numbers and still frames; nobody listened to the clips.

One thing found here that is not a constant: `jynxzi` clip 01 was made from a stretch where
the stream's sound is nearly gone (its words sit at -57 dB; the clip's loud parts are at
-19 dB). The speech-to-text still found words, so the clip has captions and almost no voice.
`SPEECH_FLOOR_DB` (-60) only affects what counts as quiet inside such a clip; the clip should
not have been picked.

## The camera, after the small-face fix

The integration branch was merged before any clip was made again. Each job's `camera.json`
was deleted and the clips re-made.

| stream | where the camera is | before the fix | after the fix | confidence | what the face panel shows |
|---|---|---|---|---|---|
| nl | top right, small | none | found | 0.96 | The streamer, well framed, in 5 of 6 clips. Clip 06 is over an intermission card with no camera: the panel shows a cartoon face from the card. |
| speed | left middle, small | none | found | 0.94 | The streamer and whoever sits beside him, in 4 of 5 clips. Clip 03 (a cutscene) shows the game character's face in the panel. |
| caseoh | top left, small, dark room | none | found | 0.93 | The streamer in 4 of 5 clips (dark, as the camera is). Clip 03 shows the game's cartoon contestant in the panel. |
| kai | left middle, small | none | safe box, then found (see below) | 0.30, then 0.91 | With the safe box, a tight crop of his face. With the box found, the whole camera picture. |
| jynxzi | bottom right, large; chess, browser and desktop screens | found | none | | No camera for the stream. Each clip is searched on its own: clips 01 and 03 get a face panel (01 with a strip of the chess move list on its left), clips 02, 04 and 05 are the screen alone. |
| speedcam | the whole picture | none | none | | Right: there is no box. Clips switch between the whole picture, a face shot and a split. |

- **`kai`: one constant.** The edge finder tried 8 lines on each side of the face and the
  camera's border was further out than the eighth. `TRIES = 16` in
  `streamclip/framing/camera.py` finds it (x 0.00, y 0.33, 0.21 × 0.23, right by eye); 12 is
  enough, 16 and 24 give the same box, and the other five streams' results do not change.
  Kept.
- **`jynxzi`: written down, not fixed.** This is the same broadcast as the coordinator's
  example. With the tuned shortlist only 7 of the 12 searched windows show the camera, and the
  camera's spot holds 42% of the faces seen (chess avatars and web pages supply the rest), so
  it fails the 60% rule and no camera is found. Lowering `STEADY_SHARE` to 0.4 or 0.5 finds
  the spot but then `box_from_edges` finds no box and the safe box (0.61, 0.54, 0.39 × 0.39,
  confidence 0.3) takes in the move list, as the coordinator saw. On the other still screens
  in the set (the Family Feud board on `caseoh`, card and board games on `nl`) the box is
  found at 0.93 and 0.96, so the still-screen miss is this stream only. Left alone.
- **A face that is not the streamer's.** In three clips the panel shows a drawn or game face
  (`nl` 06, `speed` 03, `caseoh` 03). In each the camera box showed no face at that moment
  and the clip was searched on its own.
- Time: looking for the camera took 16 to 18 s per stream here with the search, against 10 to
  13 s before.

## After

Shortlists built again with the tuned weights from the saved sound and the full chat; clips
made again on the merged engine.

| stream | hit rate | hit rate, clips withheld | judge keep rate | busiest 30 min | camera | dead air (clip: seconds cut/cuts) |
|---|---|---|---|---|---|---|
| caseoh | 6% → 96% | 6% → 78% | | 50% → 38% | none → found | 05: 1.1 s/1, others 0 |
| jynxzi | 14% → 2% | 0% → 2% | | 54% → 33% | found → none | 02: 2.3 s/3; 03: 6.5 s/5; 04: 0.5 s/1; 05: 6.8 s/4 |
| kai | 9% → 22% | 0% → 22% | | 29% → 21% | none → found | 01: 12.8 s/5, others 0 |
| nl | 73% → 73% | 7% → 73% | 21% → 21% (tuning agent) | 29% → 21% | none → found | none |
| speed | 19% → 19% | 19% → 19% | 21% → 17% (tuning agent) | 46% → 42% | none → found | 01: 0.8 s/1, others 0 |
| speedcam | 50% → 100% | 0% → 0% | 4% (production judge) → not judged | 50% → 38% | none → none | none |
| total | **31% → 68%** | **3% → 18%** | | | | 30.9 s in 20 cuts, 31 clips |

What the numbers do and do not show:

- The gain is real but narrow: the withheld hit rate rose on `caseoh`, `kai` and `nl`, each
  through one or two heavily viewed clips now on the shortlist. With clips counted by the
  logarithm of their views the total went from 12% to 26%.
- **The judge keep rate did not rise.** `nl` stayed at 5 of 24. `speed` went from 5 to 4 of
  24: the tuned shortlist dropped one moment the agent had scored 7 and its four new windows
  scored 6, 3, 5 and 5. On `nl` the tuned shortlist dropped four kept moments (three scored
  7, one scored 6) and brought in three new ones scored 7, 8 and 7, two more kept at 6, and
  one complete story that was dropped for its content.
- `jynxzi` fell with clips counted (14% to 2%): its most-viewed clip is no longer shortlisted
  at a clip weight of 0.14. See the note on that weight above.
- `speed` did not move: its most-viewed clip is in a muted stretch, and 16 of its 24 hot spots
  are still in the last hour (the spread rule allows 8 per half hour).
- The spread improved everywhere.
- Stage times were not measured again: the shortlists were rebuilt from saved sound, so only
  the windows and `make` ran. The times under "Before" stand, plus about 6 s for the camera.

### The clips Yuki will watch

`nl` (6 clips) and `speed` (5 clips), both judged by the tuning agent and not by the
production judge. For the tuned shortlists the agent kept its earlier answer for every window
that had not moved (times shifted with the window) and read the transcript and contact sheet
of every new window: 12 new on `nl` (all 12 transcripts, 6 sheets) and 4 new on `speed` plus
one that had moved 13 s (all 5 transcripts and sheets).

| stream | clip | score | title |
|---|---|---|---|
| nl | 01 | 9 | Her Dad Opened The Closet Door |
| nl | 02 | 8 | Finally Somebody Recognised Him |
| nl | 03 | 7 | He Told The Discord His Age |
| nl | 04 | 7 | He Waited For A Sandwich He Never Ordered |
| nl | 05 | 7 | The Worst Thing He Ever Said To His Grandfather |
| nl | 06 | 6 | The 4,300 Dollar Twitch Button |
| speed | 01 | 9 | He Pressed The Wrong Save |
| speed | 02 | 8 | You Only Get One Shot |
| speed | 03 | 7 | I Did Not Work This Hard For This |
| speed | 04 | 7 | The First GTA 6 Cheat Code |
| speed | 05 | 6 | Ban The Word Sleep |

The other four streams' clips were made without a judge (picked by excitement alone, titled
with the stream's title) and are there for the camera and dead-air checks only.

The footage is still in `~/stream-tuning/`. Task 3 Step 3 (Yuki watches) and Task 4 (the
footage is deleted) are not done.

## Outside this lane

Found on real streams, not changed here because they are not constants or not this lane's
files:

1. **Voice scores need a range** (`streamclip/signals/events.py` or `curve.py`). Map the
   classifier's score onto 0 to 1 the way loudness and chat are mapped, for example against
   the stream's own high percentile. The weight of 0.39 is a stand-in for that.
2. **Leave muted seconds out of the loudness baseline** (`streamclip/signals/loudness.py`).
   Simulated with the tuned weights: on `speed` the shortlist then holds 5 instead of 8 of the
   windows the agent had judged poor (and 3 instead of 4 of the good ones); the hit rate of
   the other streams does not change.
3. **A clip can be picked where the stream's sound is nearly gone** (`jynxzi` clip 01). A
   stretch tens of dB under the stream's level is not "muted" by the -70 dB rule.
4. **The five-second smoothing favours long, milder stretches over short screams** (the
   table above). A change of method in `streamclip/signals/shortlist.py`, for example picking
   peaks on the raw curve and ranking by the smoothed one.
5. **The spread rule lets one hour take most of the shortlist** (`speed`: 16 of 24).
6. **Chat fetch** (`streamclip/source/extras.py`): 300 s is too short for chats over 100,000
   messages; the job should not wait for chat once the sound is in and scored; and a chat
   that ends long before the stream does should be reported as cut short, not "ok".
7. **Chat bursts on very fast chats** never double. A range against the chat's own spread
   (a percentile) would fit both a 300-message-a-minute chat and a 30-message one.
8. **The viewer-clip list is one page per order** (2 to 50 clips a stream), which is thin as a
   signal and as an answer key.
9. **Without a judge** the engine made 5 clips, or 3 (`caseoh`), not the 6 asked for, and
   titled each with the stream's title, emoji included.
10. **A drawn or game face can become the face panel** when the camera shows no face (three
    clips above), and on `jynxzi` no camera is found for the stream at all.
11. `tests/stream/test_stream_tune.py` is a new test file for the script; the lane's file list
    names only the script. Three expectations in `tests/stream/test_stream_signals_curve.py`
    were numbers worked out from the old weights; they now read the weights.
