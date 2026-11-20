"""
Structured Semantic Spotify Tools for M.A.D.D.Y. ACI.
Provides official Spotify Web API integration, token management, playlist discovery,
and playback control without exposing secret tokens to the LLM.
"""
import base64
import json
import os
import re
import sys
import time
import urllib.parse
from typing import Any, Dict, List, Optional
import httpx
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tools.base import RiskLevel, Tool
from memory.user_preferences import UserPreferences

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

class SpotifyManager:
    """
    Manages Spotify OAuth token lifecycle, API communications, and playback control.
    """

    def __init__(self, user_preferences: Optional[UserPreferences] = None):
        self.user_preferences = user_preferences or UserPreferences()
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self.redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

    def get_auth_url(self) -> str:
        """Generate Spotify OAuth authorization URL."""
        if not self.client_id:
            return "https://developer.spotify.com/dashboard"

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": "user-read-playback-state user-modify-playback-state playlist-read-private playlist-read-collaborative",
        }
        return f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens."""
        if not self.client_id or not self.client_secret:
            return {"success": False, "error": "Missing Spotify client credentials"}

        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
        }
        try:
            resp = httpx.post(SPOTIFY_TOKEN_URL, data=data, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                token_data = resp.json()
                token_data["expires_at"] = time.time() + token_data.get("expires_in", 3600) - 60
                self.user_preferences.set_preference("spotify_token_data", token_data)
                return {"success": True, "token_data": token_data}
            return {"success": False, "error": f"Token exchange failed: {resp.status_code} {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def refresh_access_token(self) -> Optional[str]:
        """Use refresh_token to obtain a fresh access token seamlessly."""
        token_data = self.user_preferences.get_preference("spotify_token_data", {})
        refresh_token = token_data.get("refresh_token") if isinstance(token_data, dict) else None
        if not refresh_token or not self.client_id or not self.client_secret:
            return None

        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        try:
            resp = httpx.post(SPOTIFY_TOKEN_URL, data=data, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                new_data = resp.json()
                token_data["access_token"] = new_data.get("access_token")
                token_data["expires_at"] = time.time() + new_data.get("expires_in", 3600) - 60
                if "refresh_token" in new_data:
                    token_data["refresh_token"] = new_data["refresh_token"]
                self.user_preferences.set_preference("spotify_token_data", token_data)
                return token_data.get("access_token")
        except Exception:
            pass
        return None

    def get_access_token(self) -> Optional[str]:
        """Retrieve active access token from preferences or environment, auto-refreshing if expired."""
        # 1. Check environment
        env_token = os.getenv("SPOTIFY_ACCESS_TOKEN")
        if env_token:
            return env_token

        # 2. Check stored preference
        token_data = self.user_preferences.get_preference("spotify_token_data", {})
        if token_data and isinstance(token_data, dict):
            expires_at = token_data.get("expires_at", 0)
            # If expired or expiring in under 60 seconds, auto-refresh
            if time.time() >= expires_at and token_data.get("refresh_token"):
                refreshed = self.refresh_access_token()
                if refreshed:
                    return refreshed
            return token_data.get("access_token")

        return None

    def is_authenticated(self) -> bool:
        """Check whether valid Spotify credentials or access tokens are present."""
        return bool(self.get_access_token() or (self.client_id and self.client_secret))

    def _require_auth(self) -> Dict[str, Any]:
        """Standardized structured response when Spotify authorization is required."""
        msg = (
            "Spotify is not yet authorized. To enable playback and playlist controls, "
            "please provide Spotify Developer API credentials (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)."
        )
        return {
            "success": False,
            "authenticated": False,
            "error": {
                "type": "SPOTIFY_AUTH_REQUIRED",
                "message": msg,
            },
            "message": msg,
            "auth_url": self.get_auth_url(),
            "instructions": [
                "1. Go to https://developer.spotify.com/dashboard and create a free App.",
                f"2. Add {self.redirect_uri} to your Spotify App's Redirect URIs.",
                "3. Set environment variables SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.",
            ],
        }

    def get_playlists(self, limit: int = 20) -> Dict[str, Any]:
        """Fetch the authenticated user's actual playlists."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = httpx.get(f"{SPOTIFY_API_BASE}/me/playlists?limit={limit}", headers=headers, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                playlists = []
                for p in items:
                    playlists.append({
                        "id": p.get("id"),
                        "name": p.get("name"),
                        "tracks_count": p.get("tracks", {}).get("total", 0),
                        "uri": p.get("uri"),
                    })
                return {
                    "success": True,
                    "authenticated": True,
                    "total": len(playlists),
                    "playlists": playlists,
                }
            elif resp.status_code == 401:
                return self._require_auth()
            else:
                return {
                    "success": False,
                    "error": f"Spotify API returned status {resp.status_code}: {resp.text}",
                }
        except Exception as e:
            return {"success": False, "error": f"Network error connecting to Spotify: {str(e)}"}

    def get_devices(self) -> List[Dict[str, Any]]:
        """Retrieve available Spotify playback devices."""
        token = self.get_access_token()
        if not token:
            return []
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = httpx.get(f"{SPOTIFY_API_BASE}/me/player/devices", headers=headers, timeout=5.0)
            if resp.status_code == 200:
                return resp.json().get("devices", [])
        except Exception:
            pass
        return []

    def ensure_active_device(self, play: bool = False, auto_launch: bool = True) -> Optional[str]:
        """Ensure there is an active device. If none is active, activate the first available device.
        If no device exists at all, launch Spotify and wait for it to register."""
        devices = self.get_devices()
        for d in devices:
            if d.get("is_active"):
                return d.get("id")

        token = self.get_access_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        if devices and token:
            target_id = devices[0].get("id")
            try:
                httpx.put(
                    f"{SPOTIFY_API_BASE}/me/player",
                    headers=headers,
                    json={"device_ids": [target_id], "play": play},
                    timeout=6.0,
                )
                time.sleep(0.3)
                return target_id
            except Exception:
                return target_id

        if auto_launch:
            try:
                from platform_layer import get_platform
                get_platform().open_application("spotify")
            except Exception:
                pass

            for _ in range(6):
                time.sleep(0.5)
                devices = self.get_devices()
                if devices:
                    target_id = devices[0].get("id")
                    if token:
                        try:
                            httpx.put(
                                f"{SPOTIFY_API_BASE}/me/player",
                                headers=headers,
                                json={"device_ids": [target_id], "play": play},
                                timeout=6.0,
                            )
                        except Exception:
                            pass
                    return target_id

            if sys.platform.startswith("win"):
                try:
                    import ctypes
                    ctypes.windll.user32.keybd_event(0xB3, 0, 0, 0)
                    time.sleep(0.05)
                    ctypes.windll.user32.keybd_event(0xB3, 0, 2, 0)
                except Exception:
                    pass

        return None

    def search_and_play(self, query: str, position_ms: Optional[int] = None) -> Dict[str, Any]:
        """Search Spotify for a track and play it immediately, optionally at a specific timestamp."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        clean_q = query.strip()
        clean_q = re.sub(r"\b(?:on|in|from|at|via|through|with|or|using)?\s*spotify\b", "", clean_q, flags=re.IGNORECASE).strip()
        clean_q = re.sub(r"\b(?:the\s+song|a\s+song|some\s+song|track|music)\b", "", clean_q, flags=re.IGNORECASE).strip()
        if not clean_q or clean_q.lower() in ("a song", "some song", "music", "some music", "songs", "something"):
            return self.play()

        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = httpx.get(
                f"{SPOTIFY_API_BASE}/search",
                headers=headers,
                params={"q": clean_q, "type": "track", "limit": 6},
                timeout=10.0,
            )
            if resp.status_code == 200:
                tracks = resp.json().get("tracks", {}).get("items", [])
                if not tracks:
                    return {"success": False, "error": f"No tracks found matching '{clean_q}'."}
                
                # Match track with high precision (exact title -> Spotify top-ranked result)
                q_low = clean_q.lower()
                track = next((t for t in tracks if q_low == t.get("name", "").lower()), None)
                if not track:
                    # Spotify's #1 search result is already ranked by popularity & neural relevance
                    track = tracks[0]

                track_uri = track.get("uri")
                track_name = track.get("name")
                artist = ", ".join(a.get("name", "") for a in track.get("artists", []))

                dev_id = self.ensure_active_device(play=False)
                url = f"{SPOTIFY_API_BASE}/me/player/play"
                if dev_id:
                    url += f"?device_id={dev_id}"
                payload: Dict[str, Any] = {"uris": [track_uri]}
                if position_ms is not None and position_ms >= 0:
                    payload["position_ms"] = int(position_ms)

                play_resp = httpx.put(url, json=payload, headers=headers, timeout=10.0)
                if play_resp.status_code in (200, 204):
                    res_dict = {
                        "success": True,
                        "action": "play_track",
                        "track_name": track_name,
                        "track_id": track.get("id"),
                        "track_uri": track_uri,
                        "artist": artist,
                        "status": "playing",
                    }
                    if position_ms is not None and position_ms >= 0:
                        res_dict["position_ms"] = int(position_ms)
                    return res_dict
                elif play_resp.status_code == 404 or "NOACTIVEDEVICE" in play_resp.text:
                    dev_id = self.ensure_active_device(play=True, auto_launch=True)
                    if dev_id:
                        play_resp = httpx.put(
                            f"{SPOTIFY_API_BASE}/me/player/play?device_id={dev_id}",
                            json=payload,
                            headers=headers,
                            timeout=10.0,
                        )
                        if play_resp.status_code in (200, 204):
                            res_dict = {
                                "success": True,
                                "action": "play_track",
                                "track_name": track_name,
                                "track_id": track.get("id"),
                                "track_uri": track_uri,
                                "artist": artist,
                                "status": "playing",
                            }
                            if position_ms is not None and position_ms >= 0:
                                res_dict["position_ms"] = int(position_ms)
                            return res_dict
                return {"success": False, "error": f"Playback error: {play_resp.status_code} {play_resp.text}"}
            elif resp.status_code == 401:
                return self._require_auth()
            return {"success": False, "error": f"Search failed with status {resp.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def play_playlist(self, playlist_name_or_id: str) -> Dict[str, Any]:
        """Play a specified playlist by name or Spotify URI/ID."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        # Find playlist uri
        playlists_res = self.get_playlists(limit=50)
        if not playlists_res.get("success"):
            return playlists_res

        target_uri = None
        target_name = playlist_name_or_id
        for p in playlists_res.get("playlists", []):
            if playlist_name_or_id.lower() in p["name"].lower() or playlist_name_or_id == p["id"]:
                target_uri = p["uri"]
                target_name = p["name"]
                break

        if not target_uri:
            return {
                "success": False,
                "error": f"Playlist '{playlist_name_or_id}' was not found in your Spotify account.",
            }

        headers = {"Authorization": f"Bearer {token}"}
        payload = {"context_uri": target_uri}
        dev_id = self.ensure_active_device(play=False)
        url = f"{SPOTIFY_API_BASE}/me/player/play"
        if dev_id:
            url += f"?device_id={dev_id}"
        try:
            resp = httpx.put(url, json=payload, headers=headers, timeout=10.0)
            if resp.status_code in (200, 204):
                return {
                    "success": True,
                    "action": "play_playlist",
                    "playlist": target_name,
                    "status": "playing",
                }
            if resp.status_code == 404 or "NOACTIVEDEVICE" in resp.text:
                dev_id = self.ensure_active_device(play=True, auto_launch=True)
                if dev_id:
                    resp = httpx.put(f"{SPOTIFY_API_BASE}/me/player/play?device_id={dev_id}", json=payload, headers=headers, timeout=10.0)
                    if resp.status_code in (200, 204):
                        return {
                            "success": True,
                            "action": "play_playlist",
                            "playlist": target_name,
                            "status": "playing",
                        }
            return {"success": False, "error": f"Playback error: {resp.status_code} {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def play(self, query: Optional[str] = None) -> Dict[str, Any]:
        """Resume current Spotify playback or play a specifically requested track."""
        if query and query.strip().lower() not in ("a song", "some song", "music", "some music", "songs", "something"):
            return self.search_and_play(query)

        token = self.get_access_token()
        if not token:
            return self._require_auth()

        headers = {"Authorization": f"Bearer {token}"}
        dev_id = self.ensure_active_device(play=True, auto_launch=True)

        # Check current playback state to resume exactly what the user was playing
        current_state = None
        try:
            cur_resp = httpx.get(f"{SPOTIFY_API_BASE}/me/player", headers=headers, timeout=5.0)
            if cur_resp.status_code == 200 and cur_resp.text:
                current_state = cur_resp.json()
        except Exception:
            pass

        try:
            url = f"{SPOTIFY_API_BASE}/me/player/play"
            if dev_id:
                url += f"?device_id={dev_id}"

            # Standard resume command (resumes from current track and progress)
            resp = httpx.put(url, headers=headers, timeout=10.0)

            if resp.status_code in (200, 204):
                # Trigger Windows media play key as hardware nudge for browser autoplay
                if sys.platform.startswith("win"):
                    try:
                        import ctypes
                        ctypes.windll.user32.keybd_event(0xB3, 0, 0, 0)
                        time.sleep(0.04)
                        ctypes.windll.user32.keybd_event(0xB3, 0, 2, 0)
                    except Exception:
                        pass

                return {"success": True, "action": "resume", "status": "playing"}

            # If 403 because empty queue, try resuming user's own playlists
            if resp.status_code == 403:
                # If current_state has an item, re-send that exact track URI!
                if current_state and current_state.get("item", {}).get("uri"):
                    track_uri = current_state["item"]["uri"]
                    pos = current_state.get("progress_ms", 0)
                    payload = {"uris": [track_uri], "position_ms": pos}
                    r_item = httpx.put(url, json=payload, headers=headers, timeout=10.0)
                    if r_item.status_code in (200, 204):
                        return {"success": True, "action": "resume", "status": "playing"}

                # Try user's personal playlists from their account
                p_res = self.get_playlists(limit=5)
                if p_res.get("success") and p_res.get("playlists"):
                    first_p = p_res["playlists"][0]
                    return self.play_playlist(first_p["name"])

            # Fallback: on Windows send media key
            if sys.platform.startswith("win"):
                try:
                    import ctypes
                    ctypes.windll.user32.keybd_event(0xB3, 0, 0, 0)
                    time.sleep(0.04)
                    ctypes.windll.user32.keybd_event(0xB3, 0, 2, 0)
                    return {"success": True, "action": "resume", "status": "playing"}
                except Exception:
                    pass

            return {"success": False, "error": f"Playback error: {resp.status_code} {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def pause(self) -> Dict[str, Any]:
        """Pause active playback."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        headers = {"Authorization": f"Bearer {token}"}
        dev_id = self.ensure_active_device(play=False, auto_launch=False)
        url = f"{SPOTIFY_API_BASE}/me/player/pause"
        if dev_id:
            url += f"?device_id={dev_id}"
        try:
            resp = httpx.put(url, headers=headers, timeout=10.0)
            if resp.status_code in (200, 204):
                return {"success": True, "action": "pause", "status": "paused"}
            # Windows multimedia key fallback if no active device session
            if sys.platform.startswith("win") and (resp.status_code == 404 or "NOACTIVEDEVICE" in resp.text):
                try:
                    import ctypes
                    ctypes.windll.user32.keybd_event(0xB3, 0, 0, 0)
                    time.sleep(0.05)
                    ctypes.windll.user32.keybd_event(0xB3, 0, 2, 0)
                    return {"success": True, "action": "pause", "status": "paused"}
                except Exception:
                    pass
            return {"success": False, "error": f"Playback error: {resp.status_code} {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def next_track(self) -> Dict[str, Any]:
        """Skip to next track."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        headers = {"Authorization": f"Bearer {token}"}
        dev_id = self.ensure_active_device(play=False, auto_launch=False)
        url = f"{SPOTIFY_API_BASE}/me/player/next"
        if dev_id:
            url += f"?device_id={dev_id}"
        try:
            resp = httpx.post(url, headers=headers, timeout=10.0)
            if resp.status_code in (200, 204):
                return {"success": True, "action": "next", "status": "skipped"}
            if sys.platform.startswith("win") and (resp.status_code == 404 or "NOACTIVEDEVICE" in resp.text):
                try:
                    import ctypes
                    ctypes.windll.user32.keybd_event(0xB0, 0, 0, 0)
                    time.sleep(0.05)
                    ctypes.windll.user32.keybd_event(0xB0, 0, 2, 0)
                    return {"success": True, "action": "next", "status": "skipped"}
                except Exception:
                    pass
            return {"success": False, "error": f"Playback error: {resp.status_code} {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def previous_track(self) -> Dict[str, Any]:
        """Skip to previous track."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        headers = {"Authorization": f"Bearer {token}"}
        dev_id = self.ensure_active_device(play=False, auto_launch=False)
        url = f"{SPOTIFY_API_BASE}/me/player/previous"
        if dev_id:
            url += f"?device_id={dev_id}"
        try:
            resp = httpx.post(url, headers=headers, timeout=10.0)
            if resp.status_code in (200, 204):
                return {"success": True, "action": "previous", "status": "skipped_previous"}
            if sys.platform.startswith("win") and (resp.status_code == 404 or "NOACTIVEDEVICE" in resp.text):
                try:
                    import ctypes
                    ctypes.windll.user32.keybd_event(0xB1, 0, 0, 0)
                    time.sleep(0.05)
                    ctypes.windll.user32.keybd_event(0xB1, 0, 2, 0)
                    return {"success": True, "action": "previous", "status": "skipped_previous"}
                except Exception:
                    pass
            return {"success": False, "error": f"Playback error: {resp.status_code} {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def seek(self, position_ms: int) -> Dict[str, Any]:
        """Seek to a specific position (in milliseconds) within the currently playing track."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        pos = max(0, int(position_ms))
        headers = {"Authorization": f"Bearer {token}"}
        dev_id = self.ensure_active_device(play=False, auto_launch=False)
        url = f"{SPOTIFY_API_BASE}/me/player/seek"
        params = {"position_ms": pos}
        if dev_id:
            params["device_id"] = dev_id

        try:
            resp = httpx.put(url, params=params, headers=headers, timeout=10.0)
            if resp.status_code in (200, 204):
                mins = pos // 60000
                secs = (pos % 60000) // 1000
                time_str = f"{mins}:{secs:02d}"
                return {
                    "success": True,
                    "action": "seek",
                    "position_ms": pos,
                    "time_str": time_str,
                    "status": "playing",
                }
            return {"success": False, "error": f"Seek error: {resp.status_code} {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def seek_relative(self, delta_ms: int) -> Dict[str, Any]:
        """Fast-forward or rewind relative to the currently playing position."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        headers = {"Authorization": f"Bearer {token}"}
        current_pos = 0
        try:
            cur_resp = httpx.get(f"{SPOTIFY_API_BASE}/me/player", headers=headers, timeout=5.0)
            if cur_resp.status_code == 200 and cur_resp.text:
                current_pos = cur_resp.json().get("progress_ms", 0)
        except Exception:
            pass

        target_pos = max(0, current_pos + int(delta_ms))
        return self.seek(target_pos)

    def current_playback(self) -> Dict[str, Any]:
        """Get current player playback state and track information."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = httpx.get(f"{SPOTIFY_API_BASE}/me/player", headers=headers, timeout=10.0)
            if resp.status_code == 200 and resp.text:
                data = resp.json()
                item = data.get("item") or {}
                return {
                    "success": True,
                    "is_playing": data.get("is_playing", False),
                    "track_name": item.get("name"),
                    "artist": ", ".join(a.get("name", "") for a in item.get("artists", [])),
                    "album": item.get("album", {}).get("name"),
                    "progress_ms": data.get("progress_ms"),
                }
            elif resp.status_code == 204 or not resp.text:
                return {"success": True, "is_playing": False, "message": "No active Spotify player session found"}
            return {"success": False, "error": f"Status {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_shuffle(self, state: bool = True) -> Dict[str, Any]:
        """Toggle or set Spotify playback shuffle state."""
        token = self.get_access_token()
        if not token:
            return self._require_auth()

        headers = {"Authorization": f"Bearer {token}"}
        dev_id = self.ensure_active_device(play=False, auto_launch=False)
        url = f"{SPOTIFY_API_BASE}/me/player/shuffle?state={'true' if state else 'false'}"
        if dev_id:
            url += f"&device_id={dev_id}"

        try:
            resp = httpx.put(url, headers=headers, timeout=10.0)
            if resp.status_code in (200, 204):
                return {
                    "success": True,
                    "action": "shuffle",
                    "shuffle_state": state,
                    "status": "enabled" if state else "disabled",
                }
            if resp.status_code == 404 or "NOACTIVEDEVICE" in resp.text:
                dev_id = self.ensure_active_device(play=True, auto_launch=True)
                if dev_id:
                    resp = httpx.put(f"{SPOTIFY_API_BASE}/me/player/shuffle?state={'true' if state else 'false'}&device_id={dev_id}", headers=headers, timeout=10.0)
                    if resp.status_code in (200, 204):
                        return {
                            "success": True,
                            "action": "shuffle",
                            "shuffle_state": state,
                            "status": "enabled" if state else "disabled",
                        }
            return {"success": False, "error": f"Shuffle error: {resp.status_code} {resp.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Module instance
_default_spotify_manager = SpotifyManager()

# Tool Definitions
spotify_get_playlists_tool = Tool(
    name="spotify.get_playlists",
    description="Retrieve the authenticated user's Spotify playlists.",
    parameter_schema={"type": "object", "properties": {}},
    handler=lambda args: _default_spotify_manager.get_playlists(),
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)

spotify_play_playlist_tool = Tool(
    name="spotify.play_playlist",
    description="Start playing a specific Spotify playlist by name or ID.",
    parameter_schema={
        "type": "object",
        "properties": {
            "playlist_name": {
                "type": "string",
                "description": "Name or ID of playlist to play (e.g., 'Workout', 'Chill Hits').",
            }
        },
        "required": ["playlist_name"],
    },
    handler=lambda args: _default_spotify_manager.play_playlist(args.get("playlist_name", "")),
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)

spotify_play_tool = Tool(
    name="spotify.play",
    description="Resume current Spotify playback or search and play a specific song or artist.",
    parameter_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional song name or artist to search and play (e.g., 'Starboy', 'Blinding Lights'). If omitted, resumes current playback.",
            }
        },
    },
    handler=lambda args: _default_spotify_manager.play(query=args.get("query")),
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)

spotify_pause_tool = Tool(
    name="spotify.pause",
    description="Pause current Spotify playback.",
    parameter_schema={"type": "object", "properties": {}},
    handler=lambda args: _default_spotify_manager.pause(),
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)

spotify_next_tool = Tool(
    name="spotify.next",
    description="Skip to the next Spotify track.",
    parameter_schema={"type": "object", "properties": {}},
    handler=lambda args: _default_spotify_manager.next_track(),
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)

spotify_previous_tool = Tool(
    name="spotify.previous",
    description="Skip to the previous Spotify track.",
    parameter_schema={"type": "object", "properties": {}},
    handler=lambda args: _default_spotify_manager.previous_track(),
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)

spotify_current_playback_tool = Tool(
    name="spotify.current_playback",
    description="Inspect current Spotify player state, active track, and artist.",
    parameter_schema={"type": "object", "properties": {}},
    handler=lambda args: _default_spotify_manager.current_playback(),
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)

spotify_shuffle_tool = Tool(
    name="spotify.shuffle",
    description="Enable or disable Spotify playback shuffle mode.",
    parameter_schema={
        "type": "object",
        "properties": {
            "state": {
                "type": "boolean",
                "description": "True to enable shuffle, False to disable. Defaults to True.",
            }
        },
    },
    handler=lambda args: _default_spotify_manager.set_shuffle(args.get("state", True)),
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)
