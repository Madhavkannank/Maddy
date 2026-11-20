"""
Tests for semantic Spotify tools and authentication boundary.
Verifies unauthenticated diagnostics, instructions, and playback commands.
"""
import pytest
from tools.spotify import (
    SpotifyManager,
    spotify_get_playlists_tool,
    spotify_play_playlist_tool,
    spotify_play_tool,
    spotify_pause_tool,
    spotify_shuffle_tool,
)
from memory.user_preferences import UserPreferences

def test_spotify_unauthenticated_instructions(tmp_path):
    prefs = UserPreferences(storage_path=str(tmp_path / "prefs.json"))
    manager = SpotifyManager(user_preferences=prefs)

    # In unauthenticated state, must return instructions without fake playlists
    res = manager.get_playlists()
    assert res["success"] is False
    assert res["authenticated"] is False
    assert res["error"]["type"] == "SPOTIFY_AUTH_REQUIRED"
    assert "instructions" in res
    assert len(res["instructions"]) > 0

def test_spotify_playback_requires_auth(tmp_path):
    prefs = UserPreferences(storage_path=str(tmp_path / "prefs.json"))
    manager = SpotifyManager(user_preferences=prefs)

    play_res = manager.play()
    assert play_res["authenticated"] is False

    pause_res = manager.pause()
    assert pause_res["authenticated"] is False

    shuffle_res = manager.set_shuffle(True)
    assert shuffle_res["authenticated"] is False

def test_spotify_tools_registered_properly():
    assert spotify_get_playlists_tool.name == "spotify.get_playlists"
    assert spotify_play_playlist_tool.name == "spotify.play_playlist"
    assert spotify_play_tool.name == "spotify.play"
    assert spotify_pause_tool.name == "spotify.pause"
    assert spotify_shuffle_tool.name == "spotify.shuffle"

