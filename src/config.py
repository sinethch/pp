import json
from pathlib import Path

from .util import LOG

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "tiktok_accounts": [],
    "tiktok": {
        "lookback_hours": 72,
        "max_new_per_account": 3,
        "min_duration_sec": 6,
        "max_duration_sec": 55,
        "wait_seconds_between_downloads": 20,
    },
    "render": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "line_width": 16,
        "stick_color": "#22d3ee",
        "head_color": "#fef08a",
        "background": "neon",
        "confetti": True,
    },
    "youtube": {
        "title_template": "This stick dancer has the moves! {tag} #shorts #dance",
        "description_template": "Follow for daily stick figure dance videos! #shorts #dance #stickfigure",
        "privacy_status": "public",
        "category_id": "22",
        "language": "en",
        "made_for_kids": False,
    },
    "work": {"dir": "work", "raw_dir": "raw", "poses_dir": "poses", "frames_dir": "frames", "out_dir": "out"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    config = _deep_merge(DEFAULTS, json.loads(cfg_path.read_text(encoding="utf-8")))
    if not config["tiktok_accounts"]:
        LOG.warning("No TikTok accounts configured in config.json - add accounts to get started.")
    return config


def ensure_work_dirs(config: dict) -> dict:
    base = PROJECT_ROOT / config["work"]["dir"]
    raw = base / config["work"]["raw_dir"]
    poses = base / config["work"]["poses_dir"]
    frames = base / config["work"]["frames_dir"]
    out = base / config["work"]["out_dir"]
    for path in (base, raw, poses, frames, out):
        path.mkdir(parents=True, exist_ok=True)
    return {"base": base, "raw": raw, "poses": poses, "frames": frames, "out": out}