import json
from pathlib import Path

from .config import PROJECT_ROOT
from .util import LOG

STATE_FILE = PROJECT_ROOT / "state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return {"posted": data.get("posted", [])}
        except json.JSONDecodeError:
            LOG.warning("state.json corrupt, starting fresh")
    return {"posted": []}


def save_state(state: dict) -> None:
    payload = {"posted": state.get("posted", [])}
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mark_posted(state: dict, video_id: str, url: str, title: str) -> None:
    entry = {"id": video_id, "url": url, "title": title}
    posted = state.setdefault("posted", [])
    posted[:] = [p for p in posted if p["id"] != video_id] + [entry]
    posted[:] = posted[-200:]
    save_state(state)


def posted_ids(state: dict) -> set:
    return {p["id"] for p in state.get("posted", [])}