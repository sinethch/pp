import time
from pathlib import Path

import yt_dlp

from .util import LOG, retry, write_cookies_from_env


def _ydl_opts(cookies_file: str | None, outtmpl: str | None = None) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "noplaylist": False,
        "socket_timeout": 30,
        "retries": 5,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if outtmpl:
        opts.update(
            {
                "outtmpl": outtmpl,
                "format": "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/bv*[height<=1080]+ba/b",
                "merge_output_format": "mp4",
                "fragment_retries": 5,
                "noprogress": True,
            }
        )
    return opts


class TikTokFetcher:
    def __init__(self, config: dict, work_dirs: dict):
        self.tiktok_cfg = config["tiktok"]
        self.work_dirs = work_dirs
        self.cookies_file = write_cookies_from_env(str(work_dirs["raw"] / "cookies.txt"))

    def list_recent(self, account_url: str, limit: int = 10) -> list[dict]:
        """List the most recent videos from an account without downloading them."""

        def _run():
            opts = {**_ydl_opts(self.cookies_file), "extract_flat": "in_playlist", "skip_download": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(account_url, download=False)
            return info

        info = retry(_run, label=f"list {account_url}")
        entries = [e for e in (info or {}).get("entries") or [] if e]
        if not entries:
            LOG.warning("No entries found for %s", account_url)
            return []
        return entries[:limit]

    def find_new_videos(self, account_url: str, already_processed: set) -> list[dict]:
        """Return recent entries that are new (not already processed) and meet time/duration rules."""
        now = time.time()
        lookback = self.tiktok_cfg["lookback_hours"] * 3600
        candidates = []
        for entry in self.list_recent(account_url):
            video_id = entry.get("id")
            if not video_id or video_id in already_processed:
                continue
            ts = entry.get("timestamp") or 0
            if ts and (now - ts) > lookback:
                continue
            duration = entry.get("duration") or 0
            if duration:
                if duration < self.tiktok_cfg["min_duration_sec"]:
                    LOG.info("Skipping %s: too short (%.0fs)", video_id, duration)
                    continue
                if duration > self.tiktok_cfg["max_duration_sec"]:
                    LOG.info("Skipping %s: too long (%.0fs)", video_id, duration)
                    continue
            candidates.append(entry)
            if len(candidates) >= self.tiktok_cfg["max_new_per_account"]:
                break
        return candidates

    def download(self, entry: dict) -> Path:
        video_id = entry.get("id")
        raw_dir = self.work_dirs["raw"]
        outtmpl = str(raw_dir / f"{video_id}.%(ext)s")
        url = entry.get("url") or f"https://www.tiktok.com/video/{video_id}"
        LOG.info("Downloading video %s (%s)", video_id, url)

        def _run():
            opts = _ydl_opts(self.cookies_file, outtmpl=outtmpl)
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

        retry(_run, label=f"download {video_id}")
        matches = list(raw_dir.glob(f"{video_id}.*"))
        matches = [m for m in matches if m.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")]
        if not matches:
            raise RuntimeError(f"Download of {video_id} produced no file")
        return sorted(matches)[0]