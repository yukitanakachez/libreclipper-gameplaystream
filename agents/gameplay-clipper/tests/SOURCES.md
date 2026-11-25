# Where the sample stream comes from

`sample-stream.mp4` is a stand-in gameplay stream for the tests: five minutes of a game
filling the frame, with a talking face in a camera box in the top-left corner. It is built
by `scripts/make_stream_sample.py`, which downloads the one outside part below into a
scratch folder and makes the rest itself. The parts are not committed, only the script and
the finished file.

## The parts

### 1. Face and speech: NASA Artemis interview soundbites

| | |
|---|---|
| What | "Artemis Media Resource-Interview Soundbites": short on-camera answers about the Artemis programme, each speaker framed alone, facing the camera, speaking English. Used: NASA Administrator Jim Bridenstine and NASA astronaut Reid Wiseman. |
| Address | https://images.nasa.gov/details/Artemis%20Media%20Resource-Interview%20Soundbites |
| File fetched | https://images-assets.nasa.gov/video/Artemis%20Media%20Resource-Interview%20Soundbites/Artemis%20Media%20Resource-Interview%20Soundbites~large.mp4 (1920×1080, 388.3 s, 252,227,642 bytes) |
| Author | NASA Headquarters (NASA ID "Artemis Media Resource-Interview Soundbites", created 2019-12-09) |
| Licence | Public domain. Work of the US federal government (17 U.S.C. § 105); NASA material is not protected by copyright unless noted, and this item carries no copyright notice. See NASA's media usage guidelines: https://www.nasa.gov/nasa-brand-center/images-and-media/ |
| Credit line | None required. Courtesy line if wanted: "Footage: NASA". |
| Stretches used (source seconds) | Bridenstine: 16.32–36.04, 37.92–56.99, 58.74–69.28, 70.91–89.19, 90.84–111.90, 113.45–129.99, 131.46–161.06, 163.05–181.04. Wiseman: 194.64–222.35, 223.83–250.32, 252.02–263.39, 264.83–287.82, 289.35–306.55. Then Bridenstine again: 16.32–36.04, 37.92–56.99, 58.74–61.39. Each is one answer with the pause around it, skipping the title cards and black frames between answers. |
| Fetched | 2026-11-23 |

NASA's guidelines add two conditions that are not copyright: the material may not be used
to suggest that NASA endorses a product, and people who appear in it may have rights of
publicity of their own. The sample is a test fixture, and nothing in it claims NASA's
involvement. It should stay that: not used in marketing, and not shown as an example of
the product's output.

### 2. The game picture: drawn by our script

| | |
|---|---|
| What | A side-scrolling space shooter: a starfield in three layers, two ridges of hills, the player's ship, bullets, enemy waves, explosions, a lives and score display, and a large blast at each loud moment. Drawn frame by frame with Pillow at 640×360 from a fixed seed, then scaled to 1280×720. No outside art, fonts or code. |
| Address | `scripts/make_stream_sample.py` (`Game`) in this repository |
| Author | The LibreClipper builders |
| Licence | Ours; same terms as this repository. |
| Credit line | None |
| Stretch used | All 300 s |
| Made | 2026-11-23 |

### 3. The game sound: made by our script

| | |
|---|---|
| What | An engine drone (two low tones and filtered noise), a short chirp per shot, a noise pop per enemy hit, and a 2 s boom at each loud moment. Synthesised with numpy. |
| Address | `scripts/make_stream_sample.py` (`game_sound`) in this repository |
| Author | The LibreClipper builders |
| Licence | Ours; same terms as this repository. |
| Credit line | None |
| Stretch used | All 300 s |
| Made | 2026-11-23 |

Every part may be redistributed and used commercially: the face footage is in the public
domain and the game picture and sound are our own.

## What is where

Times are in seconds of `sample-stream.mp4`.

| What | Where |
|---|---|
| Length | 300.0 s, 1280×720, 30 fps, H.264 High, a keyframe every second; AAC-LC stereo, 48 kHz; about 20 MB |
| Camera box | Picture 320×256 at x=28, y=28 (fraction of the frame: x 0.022, y 0.039, w 0.25, h 0.356). With its 4 px light grey border: 328×264 at x=24, y=24. Same place throughout. |
| Who is on camera | Bridenstine 0–152.8, Wiseman 152.8–258.6, Bridenstine 258.6–300 |
| Speech | English, from the person in the camera box, about 240 s in all, at about −20 LUFS. Short pauses of 3 to 4 s fall between answers, where only the game sound plays. |
| Game sound | About 18 dB under the speech |
| Loud moments | 60–62, 135–137, 250–252: a boom in the game sound, the speech raised 10 dB and a blast on screen. Measured 16.6, 15.4 and 11.9 dB above the 20 s before each. |
| Dead air | 200–204: digital silence and a frozen picture (measured −68 dB; the picture does not change). |
| Motion | The game moves the whole time apart from the dead air: mean frame difference about 4.7 (0–255, frames a tenth of a second apart, measured at 320×180). |
