import argparse
import time

from .assemble import assemble
from .config import ensure_work_dirs, load_config
from .fetch import TikTokFetcher
from .pose import extract_poses
from .render import FunnyStickRenderer
from .state import load_state, mark_posted, posted_ids
from .upload import YouTubeUploader
from .util import LOG, setup_logging


def build_title(cfg: dict, entry: dict, tag: str) -> str:
    template = cfg["youtube"]["title_template"]
    title = entry.get("title") or "dance"
    return template.format(tag=tag, title=title)


def build_description(cfg: dict, entry: dict, tag: str) -> str:
    template = cfg["youtube"]["description_template"]
    return template.format(tag=tag, title=entry.get("title") or "dance")


def build_tags(tag: str) -> list[str]:
    return ["stickfigure", "stickman", "dance", "shorts", "funny", "trending"] + [tag]


def process_video(fetcher, renderer, uploader, config, dirs, entry, tag):
    video_id = entry["id"]
    LOG.info("=== Processing new video %s (tag=%s) ===", video_id, tag)

    source = fetcher.download(entry)

    target_fps = config["render"]["fps"]
    max_dur = config["tiktok"]["max_duration_sec"]
    npz = extract_poses(source, dirs, video_id, target_fps, max_dur)

    title = build_title(config, entry, tag)
    description = build_description(config, entry, tag)
    overlay_text = tag if tag else "Stick Dancer"
    frame_dir, fps, _ = renderer.render(npz, video_id, dirs, overlay_text, tag, source_video=source)
    out = assemble(frame_dir, video_id, source, fps, dirs["out"])

    url = uploader.upload(config["youtube"], out, title, description, build_tags(tag))
    mark_posted(config_state(), video_id, url, title)
    LOG.info("POSTED %s -> %s", video_id, url)
    return url


_state = {}


def config_state():
    return _state["state"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TikTok dance -> YouTube Shorts stick figure pipeline")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    config = load_config(args.config)
    dirs = ensure_work_dirs(config)
    state = load_state()
    _state["state"] = state

    fetcher = TikTokFetcher(config, dirs)
    renderer = FunnyStickRenderer(config)
    uploader = YouTubeUploader()

    already_posted = posted_ids(state)
    wait = config["tiktok"].get("wait_seconds_between_downloads", 20)
    posted_any = False

    for account in config["tiktok_accounts"]:
        url = account["url"]
        tag = account.get("tag", "")
        username = url.rstrip("/").split("/")[-1]
        LOG.info("Checking account %s", username)
        try:
            new_videos = fetcher.find_new_videos(url, already_posted)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Failed to list %s: %s", username, exc)
            continue
        if not new_videos:
            LOG.info("  no new videos")
            continue
        for entry in new_videos:
            if entry["id"] in already_posted:
                continue
            try:
                process_video(fetcher, renderer, uploader, config, dirs, entry, tag)
                posted_any = True
            except Exception as exc:  # noqa: BLE001
                LOG.exception("Failed to process %s: %s", entry.get("id"), exc)
            time.sleep(wait)

    if not posted_any:
        LOG.info("Nothing new to post this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())