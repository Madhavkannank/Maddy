"""Media Playback Control Tool for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides structured operations for media players (Spotify, YouTube, browser, system media)
across Windows and Linux.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult
from platform_layer import PlatformInterface, get_platform


class MediaControlTool(BaseTool):
    """Controls media playback: play, pause, next, previous, stop, and status."""

    name = "media_control"
    description = (
        "Control desktop and web media playback (Spotify, system audio, browser media). "
        "Supported actions: 'play_pause', 'play', 'pause', 'next', 'previous', 'stop', 'status'."
    )
    tier = PermissionTier.SYSTEM_IMPACTING

    def __init__(self, platform: Optional[PlatformInterface] = None):
        self.platform = platform or get_platform()

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        action = str(arguments.get("action", "")).strip().lower()
        if action == "status":
            return PermissionTier.SAFE
        return PermissionTier.SYSTEM_IMPACTING

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["play_pause", "play", "pause", "next", "previous", "stop", "status"],
                    "description": "Media playback action to dispatch.",
                },
                "player": {
                    "type": "string",
                    "description": "Optional specific player target (e.g. 'spotify', 'vlc', 'browser'). Defaults to active media player.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional track name, artist, or music query to search and play.",
                },
                "playlist": {
                    "type": "string",
                    "description": "Optional playlist name or ID to play on Spotify.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def execute(self, action: str, player: Optional[str] = None, **kwargs: Any) -> ToolResult:
        act = action.strip().lower()
        os_name = self.platform.get_os_name()
        query = kwargs.get("query")
        playlist = kwargs.get("playlist")

        # 1. First-class Spotify Web API delegation if targeted or queried
        if player == "spotify" or query or playlist:
            try:
                from tools.spotify import SpotifyManager
                mgr = SpotifyManager()
                if mgr.is_authenticated():
                    if act == "play" or act == "play_pause":
                        if playlist:
                            s_res = mgr.play_playlist(playlist)
                            if s_res.get("success"):
                                return ToolResult(
                                    success=True,
                                    output=f"Spotify playing playlist '{s_res.get('playlist', playlist)}'.",
                                    metadata=s_res,
                                )
                        elif query:
                            s_res = mgr.search_and_play(query)
                            if s_res.get("success"):
                                return ToolResult(
                                    success=True,
                                    output=f"Spotify playing track '{s_res.get('track_name', query)}' by {s_res.get('artist', 'Artist')}.",
                                    metadata=s_res,
                                )
                        else:
                            s_res = mgr.play()
                            if s_res.get("success"):
                                return ToolResult(
                                    success=True,
                                    output="Spotify playback resumed.",
                                    metadata=s_res,
                                )
                    elif act == "pause":
                        s_res = mgr.pause()
                        if s_res.get("success"):
                            return ToolResult(
                                success=True,
                                output="Spotify playback paused.",
                                metadata=s_res,
                            )
                    elif act == "next":
                        s_res = mgr.next_track()
                        if s_res.get("success"):
                            return ToolResult(
                                success=True,
                                output="Spotify skipped to next track.",
                                metadata=s_res,
                            )
                    elif act == "previous":
                        s_res = mgr.previous_track()
                        if s_res.get("success"):
                            return ToolResult(
                                success=True,
                                output="Spotify returned to previous track.",
                                metadata=s_res,
                            )
                    elif act == "status":
                        s_res = mgr.current_playback()
                        if s_res.get("success"):
                            item_name = s_res.get("track_name")
                            artist_name = s_res.get("artist")
                            if item_name:
                                return ToolResult(
                                    success=True,
                                    output=f"Spotify playback: '{item_name}' by {artist_name}.",
                                    metadata=s_res,
                                )
                            return ToolResult(
                                success=True,
                                output="Spotify is open with no active track currently playing.",
                                metadata=s_res,
                            )
            except Exception:
                pass

        try:
            if act == "status":
                if os_name == "linux" and shutil.which("playerctl"):
                    try:
                        res = subprocess.run(["playerctl", "status"], capture_output=True, text=True, timeout=3)
                        if res.returncode == 0:
                            return ToolResult(success=True, output=f"Media playback status: {res.stdout.strip()}.", metadata={"status": res.stdout.strip()})
                    except Exception:
                        pass
                return ToolResult(success=False, output="", error="No active media player status provider is available.", metadata={"status": "unknown"})

            if os_name == "windows":
                return self._execute_windows(act)
            else:
                return self._execute_linux(act, player=player)

        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Media control action '{action}' failed: {str(exc)}")

    def _execute_windows(self, action: str) -> ToolResult:
        import ctypes

        user32 = ctypes.windll.user32
        KEYEVENTF_KEYUP = 0x0002

        vk_map = {
            "play_pause": 0xB3,  # VK_MEDIA_PLAY_PAUSE
            "play": 0xB3,
            "pause": 0xB3,
            "next": 0xB0,        # VK_MEDIA_NEXT_TRACK
            "previous": 0xB1,    # VK_MEDIA_PREV_TRACK
            "stop": 0xB2,        # VK_MEDIA_STOP
        }

        vk = vk_map.get(action)
        if vk is None:
            return ToolResult(success=False, output="", error=f"Unsupported Windows media action: '{action}'.")

        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

        return ToolResult(
            success=True,
            output=f"Dispatched Windows media key event '{action}'.",
            metadata={"action": action, "platform": "windows"},
        )

    def _execute_linux(self, action: str, player: Optional[str] = None) -> ToolResult:
        # 1. If playerctl is available, use it
        if shutil.which("playerctl"):
            cmd_map = {
                "play_pause": ["playerctl", "play-pause"],
                "play": ["playerctl", "play"],
                "pause": ["playerctl", "pause"],
                "next": ["playerctl", "next"],
                "previous": ["playerctl", "previous"],
                "stop": ["playerctl", "stop"],
            }
            cmd = cmd_map.get(action)
            if cmd:
                if player:
                    cmd.extend(["--player", player])
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if res.returncode == 0:
                        return ToolResult(
                            success=True,
                            output=f"Media action '{action}' executed via playerctl.",
                            metadata={"action": action, "method": "playerctl"},
                        )
                except Exception:
                    pass

        # 2. Try native DBus MPRIS2 call via dbus-send
        if shutil.which("dbus-send"):
            mpris_method_map = {
                "play_pause": "org.mpris.MediaPlayer2.Player.PlayPause",
                "play": "org.mpris.MediaPlayer2.Player.Play",
                "pause": "org.mpris.MediaPlayer2.Player.Pause",
                "next": "org.mpris.MediaPlayer2.Player.Next",
                "previous": "org.mpris.MediaPlayer2.Player.Previous",
                "stop": "org.mpris.MediaPlayer2.Player.Stop",
            }
            method = mpris_method_map.get(action)
            if method:
                dest = f"org.mpris.MediaPlayer2.{player}" if player else "org.mpris.MediaPlayer2.spotify"
                try:
                    res = subprocess.run(
                        [
                            "dbus-send",
                            "--type=method_call",
                            f"--dest={dest}",
                            "/org/mpris/MediaPlayer2",
                            method,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if res.returncode == 0:
                        return ToolResult(
                            success=True,
                            output=f"Media action '{action}' dispatched via MPRIS2 DBus to {dest}.",
                            metadata={"action": action, "method": "dbus-send", "dest": dest},
                        )
                except Exception:
                    pass

        return ToolResult(
            success=False,
            output="",
            error=f"No media control provider responded to action '{action}'.",
            metadata={"action": action, "method": "none", "executed": False},
        )
