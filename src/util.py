import logging
import os
import sys
import time

LOG = logging.getLogger("stickdancer")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    LOG.setLevel(level)
    LOG.addHandler(handler)
    LOG.propagate = False


def retry(fn, attempts: int = 4, base_delay: float = 5.0, label: str = "op"):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = base_delay * (2 ** i)
            LOG.warning("%s failed (%s), retrying in %.0fs (%s/%s)", label, exc, wait, i + 1, attempts)
            time.sleep(wait)
    raise last


def write_cookies_from_env(path: str) -> str | None:
    payload = os.environ.get("TIKTOK_COOKIES", "").strip()
    if not payload:
        return None
    if payload.lower().startswith("# http") or "\n# http" in payload[:4096]:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        LOG.info("Wrote TikTok cookies file from TIKTOK_COOKIES secret")
        return path
    LOG.warning("TIKTOK_COOKIES secret present but does not look like a Netscape cookies file; ignoring.")
    return None