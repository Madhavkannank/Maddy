"""
Voice module package initializer for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You).
"""
from voice.engine import VoiceEngine
from voice.tools import (
    voice_speak_tool,
    voice_transcribe_tool,
    voice_listen_tool,
)

__all__ = [
    "VoiceEngine",
    "voice_speak_tool",
    "voice_transcribe_tool",
    "voice_listen_tool",
]
