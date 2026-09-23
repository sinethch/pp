# StickDancer 🕺

Fully automated content pipeline that turns TikTok dance videos into **funny stick-figure YouTube Shorts** — with every pose and movement copied exactly from the original dancer. Designed to run entirely on **GitHub Actions** with zero manual involvement once configured.

```
TikTok account  ──▶  yt-dlp download  ──▶  MediaPipe pose extraction
                                              │  (33 landmarks/frame, exact positions)
                                              ▼
                        Funny stick-figure renderer (1080x1920 vertical)
                                              │
                                              ▼
                     ffmpeg (add original dance audio) ──▶ YouTube Shorts upload
```

Runs on a schedule (GitHub Actions cron), detects brand-new videos, and posts only the ones it hasn't posted before. State is committed back to the repo automatically.

---

## 1. Project layout

```
config.json                 # accounts + brand/title/schedule settings
state.json                  # auto-managed: which TikTok videos were already posted
src/
  fetch.py                  # list & download newest TikTok videos (yt-dlp)
  pose.py                   # MediaPipe pose extraction per frame
  render.py                 # draw the funny stick figure + dynamic background
  assemble.py               # ffmpeg: frames + original audio -> shorts MP4
  upload.py                 # YouTube Data API v3 upload
  main.py                   # orchestrator
tests/
  smoke_render.py           # local render test (no network/GPU needed)
setup_oauth.py              # one-time OAuth: get a YouTube refresh token
.github/workflows/pipeline.yml
```

---

## 2. One-time setup

### 2.1 Google Cloud + YouTube API

1. Go to https://console.cloud.google.com and create a project.
2. Enable **YouTube Data API v3**.
3. Create OAuth credentials of type **Desktop app**.
4. Download the credentials and save them as `client_secret.json` in this folder.

### 2.2 Get a YouTube refresh token (run once, locally)

```bash
pip install -r requirements.txt
python setup_oauth.py
```

A browser window opens; log in with the **YouTube channel** that will post the Shorts. The script prints three values — those become GitHub secrets.

### 2.3 Configure your TikTok accounts

Edit `config.json`:

```json
"tiktok_accounts": [
  { "url": "https://www.tiktok.com/@creator1", "tag": "Creator Name" },
  { "url": "https://www.tiktok.com/@creator2", "tag": "Another Name" }
]
```

You can also tune:
- `lookback_hours` — only consider videos this recent
- `max_new_per_account` — max clips posted per run, per account
- `min_duration_sec` / `max_duration_sec` — YouTube Shorts safety window (6–55 s)
- `youtube.title_template` / `description_template` — supports `{tag}` and `{title}`
- `render.*` — colors, line width, confetti on/off, and `background`:
  - `"source"` *(default, recommended)* — the **original video IS the
    background**: the dancer is removed (replaced with a median
    reconstruction of the background) and the stick figure dances in their
    place. Most authentic look.
  - `"neon"` — dark gradient, glow, colorful music bars
  - `"sketch"` — hand-drawn sketchbook style: paper background, doodle sun,
    clouds, bouncing notes, squiggly ground with hatching, ink equalizer

### 2.4 Add GitHub secrets

Repo → *Settings → Secrets and variables → Actions*:

| Secret               | Value                                        |
| -------------------- | -------------------------------------------- |
| `YT_CLIENT_ID`       | from `python setup_oauth.py` output          |
| `YT_CLIENT_SECRET`   | from `python setup_oauth.py` output          |
| `YT_REFRESH_TOKEN`   | from `python setup_oauth.py` output          |
| `TIKTOK_COOKIES`     | *optional* Netscape cookie file for TikTok login (see below) |

**Optional TikTok cookies:** if a creator needs login to download, export your TikTok
session cookies in Netscape format and put the whole file contents into the
`TIKTOK_COOKIES` secret.

### 2.5 Push to GitHub

```bash
git init
git add .
git commit -m "init stickdancer pipeline"
git remote add origin git@github.com:you/repo.git
git push -u origin main
```

---

## 3. Scheduling

The default workflow posts **every 2 hours** (`cron: "0 */2 * * *"` in
`.github/workflows/pipeline.yml`, UTC). Change it to match your rhythm:

| Wanted | cron (UTC) |
| ------ | ---------- |
| Every hour     | `0 * * * *` |
| Every 2 hours  | `0 */2 * * *` |
| Every 6 hours  | `0 */6 * * *` |
| Every day 20:00 | `0 20 * * *` |

You can also trigger a run manually: repo → **Actions** → *StickDancer* → **Run workflow**.

---

## 4. Local run (testing)

```bash
pip install -r requirements.txt
python setup_oauth.py            # once, see 2.2

# test the renderer (no TikTok/YouTube needed):
python tests/smoke_render.py     # writes work/preview.jpg

# full dry run against a real account (will upload!):
python -m src.main --config config.json
```

Set the same env vars locally (`YT_CLIENT_ID`, `YT_CLIENT_SECRET`,
`YT_REFRESH_TOKEN`, optional `TIKTOK_COOKIES`) when running `src/main.py`.

---

## 5. How it copies movements exactly

1. **MediaPipe Pose** extracts **33 body landmarks for every sampled frame**
   (nose, shoulders, elbows, wrists, hips, knees, ankles, …).
2. **MediaPipe Face Mesh** extracts **facial metrics** on the same frames:
   eye openness (left & right, from eyelid positions), mouth openness (lip
   separation) and smile/frown curve — so the face is not static.
3. Landmarks are median-smoothed to remove jitter **without wiping out sharp
   movements** (blinks, lip flaps, fast hand hits).
4. The renderer maps every landmark to the 1080×1920 canvas with auto-fit +
   centering, so the figure fills the frame. The stick body is drawn
   joint-by-joint, frame-by-frame — same poses, same limb angles, same timing.
5. The goofy face is driven by the source in real time:
   - real eye closure → the figure's eyes close
   - source talks/raps → the figure's mouth **opens and closes in sync**
   - smile / frown expressions → mouth curve matches
   - head tilt is copied from the nose-shoulder angle
6. The original dance audio is kept, so moves, lip sync and music stay aligned.
7. With `render.background: "source"` (the default) the **original clip is the
   background**: the dancer is erased from each frame (replaced by a median
   background reconstruction) and the stick figure dances in exactly their
   place — same location, same size, same background as the source.

Fun extras (tunable in `config.json`): animated music bars, confetti rain,
googly eye motion, spinning sparkles on fast hands, pulsing spotlight,
drop-shadow.

---

## 6. Troubleshooting

- **`YT_CLIENT_ID` not found in environment** → you must add the secrets (2.4).
- **Upload fails with "access_denied"** → refresh token has the wrong scope;
  re-run `python setup_oauth.py`.
- **No new videos found** → account URL typo, or all recent videos were already
  posted/outside the duration/lookback window.
- **Downloads need login** → add the `TIKTOK_COOKIES` secret.
- **Posting the same clip twice** → check `state.json`; it is committed back to
  the repo after every run.

---

## ⚠️ Content rights

You are posting derivative clips drawn from third-party dance videos. Usage of
those creator's audio/moves may be copyrighted. Posting transformative
stick-figure renderings is generally fine for personal/entertainment
experimentation, but **only launch with accounts you have permission to copy**
and consider crediting the source creator in the description.