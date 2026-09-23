import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from .util import LOG, retry

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


class YouTubeUploader:
    def __init__(self):
        self.client_id = os.environ["YT_CLIENT_ID"]
        self.client_secret = os.environ["YT_CLIENT_SECRET"]
        self.refresh_token = os.environ["YT_REFRESH_TOKEN"]

    def _creds(self):
        creds = Credentials(
            token=None,
            refresh_token=self.refresh_token,
            token_uri=TOKEN_URI,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=SCOPES,
        )
        creds.refresh(Request())
        return creds

    def upload(self, cfg: dict, video_path: Path, title: str, description: str, tags: list[str]) -> str:
        yt = build("youtube", "v3", credentials=self._creds(), cache_discovery=False)
        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:4900],
                "tags": tags[:400],
                "categoryId": str(cfg.get("category_id", "22")),
                "defaultLanguage": cfg.get("language", "en"),
            },
            "status": {
                "privacyStatus": cfg.get("privacy_status", "public"),
                "selfDeclaredMadeForKids": bool(cfg.get("made_for_kids", False)),
            },
        }
        media = MediaFileUpload(str(video_path), chunksize=8 * 1024 * 1024, resumable=True)
        request = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        LOG.info("Uploading %s as '%s'", video_path.name, title)

        def _run():
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status and status.total_size:
                    pct = status.resumable_progress / status.total_size * 100
                    LOG.info("  upload %3.0f%%", pct)
            return response

        resp = retry(_run, attempts=3, label="youtube upload")
        video_id = resp["id"]
        url = f"https://youtu.be/{video_id}"
        LOG.info("Uploaded -> %s", url)
        return url