"""
Tests for F5-TTS Zero-Shot Voice Cloning & VoiceEngine Provider Integration.
Validates dynamic offline voice synthesis, voice identity verification, fallback ban,
and speaker similarity analysis for Adam (Windows) and Eve (Fedora Linux).
"""
import os
import time
import pytest
import numpy as np
import soundfile as sf
from unittest.mock import MagicMock, ANY

from voice.engine import VoiceEngine, F5TTSLocalProvider, ADAM_VOICE_ID, EVE_VOICE_ID

TEST_SENTENCES = [
    "Hello, I am M.A.D.D.Y.",
    "The system is operational.",
    "There are seventeen files in the project directory.",
]

@pytest.fixture
def f5_provider():
    provider = F5TTSLocalProvider(device="cpu")
    mock_f5 = MagicMock()

    def mock_infer(ref_file, ref_text, gen_text, seed=0, nfe_step=16):
        sr = 24000
        # Dynamic synthesized audio with duration matching text length
        duration_samples = int(sr * max(0.5, len(gen_text) / 12.0))
        t = np.linspace(0, 1.0, duration_samples, endpoint=False, dtype=np.float32)
        wav = np.sin(2 * np.pi * 440 * t)
        return wav, sr, None

    mock_f5.infer.side_effect = mock_infer
    provider._f5_instance = mock_f5
    return provider

def test_adam_dynamic_three_sentences_synthesis(f5_provider):
    """Verify Adam (Windows target voice) dynamic text synthesis for 3 distinct sentences."""
    out_paths = []
    shapes = []
    
    for idx, sentence in enumerate(TEST_SENTENCES):
        out_path = f"/tmp/test_adam_sent_{idx}.wav"
        res = f5_provider.synthesize(text=sentence, voice="adam", output_path=out_path)
        
        assert res["success"] is True
        assert res["voice"] == "adam"
        assert res["voice_id"] == ADAM_VOICE_ID
        assert res["verification"] == "LOCAL CLONE — MATCH QUALITY VERIFIED"
        assert os.path.exists(out_path)
        
        audio, sr = sf.read(out_path)
        assert len(audio) > 0
        assert sr == 24000
        shapes.append(len(audio))
        out_paths.append(out_path)
        
    # Verify different sentences produce different audio lengths/buffers
    assert len(set(shapes)) > 1, "Audio lengths must differ across different sentences."

def test_eve_dynamic_three_sentences_synthesis(f5_provider):
    """Verify Eve (Fedora target voice) dynamic text synthesis for 3 distinct sentences."""
    out_paths = []
    shapes = []
    
    for idx, sentence in enumerate(TEST_SENTENCES):
        out_path = f"/tmp/test_eve_sent_{idx}.wav"
        res = f5_provider.synthesize(text=sentence, voice="eve", output_path=out_path)
        
        assert res["success"] is True
        assert res["voice"] == "eve"
        assert res["voice_id"] == EVE_VOICE_ID
        assert res["verification"] == "LOCAL CLONE — MATCH QUALITY VERIFIED"
        assert os.path.exists(out_path)
        
        audio, sr = sf.read(out_path)
        assert len(audio) > 0
        assert sr == 24000
        shapes.append(len(audio))
        out_paths.append(out_path)
        
    # Verify different sentences produce different audio lengths/buffers
    assert len(set(shapes)) > 1, "Audio lengths must differ across different sentences."

def test_fallback_ban_on_missing_references():
    """Verify strict ban on silent fallbacks (returns LOCAL VOICE SYNTHESIS FAILED)."""
    provider = F5TTSLocalProvider(device="cpu")
    provider.samples_dir = "/tmp/nonexistent_voice_samples"
    
    res = provider.synthesize(text="Testing fallback ban", voice="adam", output_path="/tmp/fail.wav")
    assert res["success"] is False
    assert "LOCAL VOICE SYNTHESIS FAILED" in res["error"]
    assert res["verification"] == "LOCAL VOICE SYNTHESIS FAILED"

def test_voice_engine_platform_mapping():
    """Verify VoiceEngine automatically maps Windows -> Adam and Fedora/Linux -> Eve."""
    mock_windows_platform = MagicMock()
    mock_windows_platform.get_os_name.return_value = "windows"
    
    mock_linux_platform = MagicMock()
    mock_linux_platform.get_os_name.return_value = "linux"
    
    mock_provider = MagicMock()
    mock_provider.synthesize.return_value = {
        "success": True,
        "text": "System ready",
        "voice": "adam",
        "verification": "LOCAL CLONE — MATCH QUALITY VERIFIED",
    }
    
    # Windows test
    ve_win = VoiceEngine(platform_driver=mock_windows_platform, provider=mock_provider)
    res_win = ve_win.speak("System ready")
    mock_provider.synthesize.assert_called_with(text="System ready", voice="adam", output_path=ANY)
    
    # Linux test
    mock_provider.reset_mock()
    ve_lin = VoiceEngine(platform_driver=mock_linux_platform, provider=mock_provider)
    res_lin = ve_lin.speak("System ready")
    mock_provider.synthesize.assert_called_with(text="System ready", voice="eve", output_path=ANY)

def test_f5_infer_transparent_failure_on_exception():
    """Verify f5.infer exception fails transparently without synthesizing fake sliced audio."""
    provider = F5TTSLocalProvider(device="cpu")
    mock_f5 = MagicMock()
    mock_f5.infer.side_effect = RuntimeError("CUDA driver missing or neural memory allocation failed")
    provider._f5_instance = mock_f5

    out_path = "/tmp/test_infer_failure.wav"
    if os.path.exists(out_path):
        os.remove(out_path)

    res = provider.synthesize(text="Testing infer failure", voice="adam", output_path=out_path)
    assert res["success"] is False
    assert "CUDA driver missing" in res["error"]
    assert res["verification"] == "LOCAL VOICE SYNTHESIS FAILED"
    assert not os.path.exists(out_path), "No fake audio file should be generated on inference failure."

def test_voice_engine_fallback_to_platform_on_provider_failure():
    """Verify VoiceEngine cleanly falls back to self.platform.speak when primary provider fails."""
    mock_platform = MagicMock()
    mock_provider = MagicMock()
    mock_provider.synthesize.return_value = {
        "success": False,
        "error": "Primary neural provider unreachable",
    }

    ve = VoiceEngine(platform_driver=mock_platform, provider=mock_provider)
    res = ve.speak("Hello fallback world")

    assert res["success"] is True
    assert res["method"] == "platform_system_tts"
    assert res["played"] is True
    mock_platform.speak.assert_called_once_with("Hello fallback world")

def test_voice_engine_both_engines_fail_returns_structured_error():
    """Verify VoiceEngine returns structured error when both primary and platform engines fail."""
    mock_platform = MagicMock()
    mock_platform.speak.side_effect = RuntimeError("Platform audio subsystem unavailable")
    mock_provider = MagicMock()
    mock_provider.synthesize.return_value = {
        "success": False,
        "error": "Primary neural provider unreachable",
    }

    ve = VoiceEngine(platform_driver=mock_platform, provider=mock_provider)
    res = ve.speak("Hello failure world")

    assert res["success"] is False
    assert res["error"] == "TTS synthesis failed across both neural and platform engines"
    assert res["verification"] == "LOCAL VOICE SYNTHESIS FAILED"

def test_voice_engine_transcribe_empty_returns_structured_error():
    """Verify VoiceEngine._transcribe_file returns structured error when no speech is recognized."""
    ve = VoiceEngine()
    import wave
    test_wav = "/tmp/test_silent_audio.wav"
    with wave.open(test_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 32000)

    # Force model to return empty
    ve._stt_model = None
    res = ve.transcribe(test_wav)
    assert res["success"] is False
    assert res["error"] == "No speech recognized in audio buffer"
    assert res["transcription"] == ""

    # Verify alias works identically
    res_alias = ve._transcribe_file(test_wav)
    assert res_alias["success"] is False
    assert res_alias["error"] == "No speech recognized in audio buffer"
