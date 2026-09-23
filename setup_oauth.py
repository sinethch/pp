"""One-time OAuth setup: obtains a long-lived refresh token for YouTube uploads.

Run locally:  python setup_oauth.py
You must first create a Google Cloud OAuth client ID (type "Desktop app").
Prints a refresh token -> save it as the YT_REFRESH_TOKEN GitHub secret.
"""

import os
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET_FILE = Path("client_secret.json")


def main() -> None:
    client_id = os.environ.get("YT_CLIENT_ID", "")
    client_secret = os.environ.get("YT_CLIENT_SECRET", "")

    if not CLIENT_SECRET_FILE.exists() and (not client_id or not client_secret):
        print("client_secret.json not found and YT_CLIENT_ID/YT_CLIENT_SECRET not set.")
        print("Steps:")
        print("  1. https://console.cloud.google.com -> create a project.")
        print("  2. Enable 'YouTube Data API v3'.")
        print("  3. Create OAuth client ID of type 'Desktop app'.")
        print("  4. Download the credentials -> save as client_secret.json, or")
        print("     export YT_CLIENT_ID and YT_CLIENT_SECRET.")
        raise SystemExit(1)

    if CLIENT_SECRET_FILE.exists():
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
    else:
        client_config = {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            }
        }
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

    creds = flow.run_local_server(port=0, prompt="consent")
    with open("oauth_token.json", "w", encoding="utf-8") as fh:
        fh.write(creds.to_json())
    print()
    print("SUCCESS! Add these to your GitHub repo secrets:")
    print(f"  YT_CLIENT_ID     = {creds.client_id}")
    print(f"  YT_CLIENT_SECRET = {creds.client_secret}")
    print()
    print("  YT_REFRESH_TOKEN = " + creds.refresh_token)
    print()
    print("A copy was also saved to oauth_token.json for local runs.")


if __name__ == "__main__":
    main()