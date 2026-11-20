"""
Voice tool definitions for M.A.D.D.Y. (voice.speak, voice.transcribe, voice.listen).
"""
from typing import Any, Dict
from tools.base import RiskLevel, Tool
from voice.engine import VoiceEngine

_default_voice_engine = VoiceEngine()

def _handle_voice_speak(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args["text"]
    voice = args.get("voice")
    return _default_voice_engine.speak(text, voice=voice)

def _handle_voice_transcribe(args: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = args["audio_path"]
    return _default_voice_engine.transcribe(audio_path)

def _handle_voice_listen(args: Dict[str, Any]) -> Dict[str, Any]:
    duration = float(args.get("duration_seconds", 3.0))
    return _default_voice_engine.listen(duration)

voice_speak_tool = Tool(
    name="voice.speak",
    description="Synthesize spoken speech output from text input using neural voice cloning engine (Adam for deep male, Eve for sweet female).",
    parameter_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to synthesize into spoken speech.",
            },
            "voice": {
                "type": "string",
                "enum": ["adam", "eve"],
                "description": "Neural voice identity ('adam' for deep male voice, 'eve' for sweet female voice).",
            },
        },
        "required": ["text"],
    },
    handler=_handle_voice_speak,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)

voice_transcribe_tool = Tool(
    name="voice.transcribe",
    description="Transcribe spoken audio recording or sound file into text.",
    parameter_schema={
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Path to audio file (WAV, MP3, etc.).",
            }
        },
        "required": ["audio_path"],
    },
    handler=_handle_voice_transcribe,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)

voice_listen_tool = Tool(
    name="voice.listen",
    description="Capture microphone audio input stream and return transcribed string.",
    parameter_schema={
        "type": "object",
        "properties": {
            "duration_seconds": {
                "type": "number",
                "description": "Duration in seconds to listen for audio input.",
                "default": 3.0,
            }
        },
        "required": [],
    },
    handler=_handle_voice_listen,
    risk_level=RiskLevel.MEDIUM,
    timeout=15.0,
)
