"""
One-Click Spotify OAuth Authorization Helper for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Starts a temporary loopback HTTP listener on 127.0.0.1:8888, opens the Spotify authorization
page in your default browser, captures the returned OAuth authorization code,
exchanges it for access + refresh tokens, and saves them permanently in M.A.D.D.Y. memory.
"""
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.spotify import SpotifyManager

def authenticate():
    manager = SpotifyManager()
    if not manager.client_id or not manager.client_secret:
        print("[ERROR] Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in .env.")
        print("Please ensure both are set in your project .env file before running.")
        sys.exit(1)

    auth_code_holder = {}

    class OAuthCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/callback":
                params = parse_qs(parsed.query)
                if "code" in params:
                    code = params["code"][0]
                    auth_code_holder["code"] = code
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=utf-8")
                    self.end_headers()
                    html = """
                    <html>
                    <head><title>M.A.D.D.Y. - Spotify Linked</title></head>
                    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; text-align: center; padding: 50px; background: #121212; color: #ffffff;">
                        <h1 style="color: #1DB954; font-size: 32px;">Authorization Successful!</h1>
                        <p style="font-size: 18px; color: #b3b3b3;">Spotify is now officially connected to M.A.D.D.Y.</p>
                        <p style="font-size: 14px; color: #777;">You can safely close this browser tab and return to M.A.D.D.Y.</p>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode("utf-8"))
                else:
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<h1>Authorization Failed: Missing 'code' parameter</h1>")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress normal HTTP logging
            return

    server_address = ("127.0.0.1", 8888)
    try:
        httpd = HTTPServer(server_address, OAuthCallbackHandler)
    except OSError as err:
        print(f"[ERROR] Could not bind to 127.0.0.1:8888: {err}")
        print("Make sure port 8888 is not already in use by another application.")
        sys.exit(1)

    auth_url = manager.get_auth_url()
    print("=" * 70)
    print("               M.A.D.D.Y. SPOTIFY AUTHENTICATION HELPER")
    print("=" * 70)
    print("1. Opening your browser to Spotify authorization page...")
    print(f"2. URL: {auth_url}")
    print("3. Waiting for authorization callback on http://127.0.0.1:8888/callback ...")
    print("=" * 70)

    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    # Handle single callback request
    while "code" not in auth_code_holder:
        httpd.handle_request()

    code = auth_code_holder["code"]
    print("\n[OK] Received authorization code from Spotify.")
    print("Exchanging authorization code for permanent tokens...")

    res = manager.exchange_code_for_tokens(code)
    if res.get("success"):
        token_data = res.get("token_data", {})
        has_refresh = bool(token_data.get("refresh_token"))
        print("\n" + "=" * 70)
        print("          SPOTIFY OAUTH AUTHENTICATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"  Access Token Received : True (Expires in 3600s)")
        print(f"  Permanent Refresh Token: {has_refresh}")
        print(f"  Token Auto-Refresh     : Enabled (Background renewal active)")
        print(f"  Credentials Stored in  : M.A.D.D.Y. Unified User Preferences")
        print("=" * 70)
        print("You can now ask M.A.D.D.Y. to list playlists and control playback!\n")
    else:
        print(f"\n[ERROR] Failed to exchange code for tokens: {res.get('error')}")

if __name__ == "__main__":
    authenticate()
