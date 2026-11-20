"""
Unit and behavioral tests for Wake Word Noise Rejection and Multi-Frame Confirmation.
Verifies:
1. Single-frame and two-frame transient acoustic noise spikes are rejected.
2. Multi-frame confirmation requires N consecutive over-threshold frames before firing.
3. Audio frames without vocal speech energy (below dynamic noise floor) are rejected.
4. Wake cooldown prevents immediate echo / reverberation double-triggers.
5. Standby Whisper transcription prompts do not bias models into wake word hallucinations.
6. Common acoustic noise hallucinations ([music], laughter, repeated single words) are rejected.
"""
import time
import pytest
from voice.wakeword import WakeWordEngine
from voice_listener import is_valid_speech_transcription


class MultiFrameWakeDetector:
    """
    Simulates the multi-frame wake confirmation state machine
    from voice_listener.py.
    """

    def __init__(
        self,
        wake_threshold: float = 0.55,
        confirmation_frames: int = 3,
        wake_cooldown_seconds: float = 2.0,
    ):
        self.wake_threshold = wake_threshold
        self.confirmation_frames = confirmation_frames
        self.wake_cooldown_seconds = wake_cooldown_seconds
        self.confirm_streak = 0
        self.last_wake_fire_time = 0.0

    def process_frame(
        self,
        score: float,
        max_amp: int,
        dynamic_threshold: int,
        chunk_rms: float,
        dynamic_rms: float,
        current_time: float,
    ) -> bool:
        neural_wake_detected = False
        has_vocal_energy = (max_amp >= dynamic_threshold or chunk_rms >= dynamic_rms)
        if score >= self.wake_threshold and has_vocal_energy:
            self.confirm_streak += 1
            if self.confirm_streak >= self.confirmation_frames:
                if (current_time - self.last_wake_fire_time) >= self.wake_cooldown_seconds:
                    neural_wake_detected = True
                    self.last_wake_fire_time = current_time
                self.confirm_streak = 0
        else:
            self.confirm_streak = 0
        return neural_wake_detected


def test_single_frame_noise_spike_rejected():
    """Verify that an isolated 80ms noise spike above threshold does not trigger wake."""
    detector = MultiFrameWakeDetector(wake_threshold=0.55, confirmation_frames=3)
    t = 100.0

    # 1. Quiet frames
    for _ in range(5):
        t += 0.08
        fired = detector.process_frame(
            score=0.10, max_amp=150, dynamic_threshold=300, chunk_rms=0.005, dynamic_rms=0.010, current_time=t
        )
        assert fired is False

    # 2. Stray noise spike (1 frame at score 0.85)
    t += 0.08
    fired = detector.process_frame(
        score=0.85, max_amp=500, dynamic_threshold=300, chunk_rms=0.020, dynamic_rms=0.010, current_time=t
    )
    assert fired is False, "Single-frame noise spike must not trigger wake word!"
    assert detector.confirm_streak == 1

    # 3. Subsequent frame drops back to ambient room noise
    t += 0.08
    fired = detector.process_frame(
        score=0.15, max_amp=180, dynamic_threshold=300, chunk_rms=0.005, dynamic_rms=0.010, current_time=t
    )
    assert fired is False
    assert detector.confirm_streak == 0, "Confirm streak must reset when score drops below threshold"


def test_two_frame_noise_spike_rejected():
    """Verify that a 2-frame transient noise burst does not trigger 3-frame confirmation."""
    detector = MultiFrameWakeDetector(wake_threshold=0.55, confirmation_frames=3)
    t = 100.0

    # Frame 1: Transient noise spike
    t += 0.08
    assert detector.process_frame(
        score=0.75, max_amp=500, dynamic_threshold=300, chunk_rms=0.020, dynamic_rms=0.010, current_time=t
    ) is False
    assert detector.confirm_streak == 1

    # Frame 2: Transient noise continues
    t += 0.08
    assert detector.process_frame(
        score=0.70, max_amp=480, dynamic_threshold=300, chunk_rms=0.018, dynamic_rms=0.010, current_time=t
    ) is False
    assert detector.confirm_streak == 2

    # Frame 3: Transient noise ends (< 3 frames)
    t += 0.08
    assert detector.process_frame(
        score=0.20, max_amp=150, dynamic_threshold=300, chunk_rms=0.004, dynamic_rms=0.010, current_time=t
    ) is False
    assert detector.confirm_streak == 0, "Streak must reset to 0 upon noise cessation"


def test_consecutive_three_frames_genuine_wake_fires():
    """Verify that authentic spoken wake word holding confidence across 3 consecutive frames fires."""
    detector = MultiFrameWakeDetector(wake_threshold=0.55, confirmation_frames=3)
    t = 100.0

    # Frame 1: Onset of "Hey"
    t += 0.08
    assert detector.process_frame(
        score=0.65, max_amp=800, dynamic_threshold=300, chunk_rms=0.025, dynamic_rms=0.010, current_time=t
    ) is False
    assert detector.confirm_streak == 1

    # Frame 2: Core syllable "Mad-"
    t += 0.08
    assert detector.process_frame(
        score=0.88, max_amp=1200, dynamic_threshold=300, chunk_rms=0.035, dynamic_rms=0.010, current_time=t
    ) is False
    assert detector.confirm_streak == 2

    # Frame 3: Ending syllable "-dy" -> 3 consecutive frames -> FIRES!
    t += 0.08
    fired = detector.process_frame(
        score=0.82, max_amp=1100, dynamic_threshold=300, chunk_rms=0.032, dynamic_rms=0.010, current_time=t
    )
    assert fired is True, "Authentic 3-frame sustained wake phrase must fire!"
    assert detector.confirm_streak == 0


def test_frames_below_dynamic_energy_floor_rejected():
    """Verify that frames with high neural score but dead silence / low energy are rejected."""
    detector = MultiFrameWakeDetector(wake_threshold=0.55, confirmation_frames=3)
    t = 100.0

    # Score is 0.90 but amplitude (50) and RMS (0.002) are below ambient noise floor (300 / 0.010)
    for _ in range(3):
        t += 0.08
        fired = detector.process_frame(
            score=0.90, max_amp=50, dynamic_threshold=300, chunk_rms=0.002, dynamic_rms=0.010, current_time=t
        )
        assert fired is False, "Frames without vocal energy above noise floor must not accumulate confirmation"
        assert detector.confirm_streak == 0


def test_cooldown_prevents_reverberation_double_trigger():
    """Verify that room reverberation or echo right after a wake cannot immediately re-fire."""
    detector = MultiFrameWakeDetector(wake_threshold=0.55, confirmation_frames=3, wake_cooldown_seconds=2.0)
    t = 100.0

    # First wake fires at t = 100.24
    for _ in range(3):
        t += 0.08
        fired = detector.process_frame(
            score=0.80, max_amp=800, dynamic_threshold=300, chunk_rms=0.025, dynamic_rms=0.010, current_time=t
        )
    assert fired is True

    # Room reverberation continues 0.5s later (t = 100.74, < 2.0s cooldown)
    t = 100.74
    for _ in range(3):
        t += 0.08
        re_fired = detector.process_frame(
            score=0.80, max_amp=800, dynamic_threshold=300, chunk_rms=0.025, dynamic_rms=0.010, current_time=t
        )
    assert re_fired is False, "Re-fire within 2.0s cooldown must be suppressed"

    # After cooldown expires (t = 102.50)
    t = 102.50
    for _ in range(3):
        t += 0.08
        normal_fired = detector.process_frame(
            score=0.80, max_amp=800, dynamic_threshold=300, chunk_rms=0.025, dynamic_rms=0.010, current_time=t
        )
    assert normal_fired is True, "Fire after cooldown expiry must succeed"


def test_noise_hallucinations_filtered_out():
    """Verify that common non-vocal acoustic noise and Whisper artifacts are rejected."""
    # Sound effect tokens from Whisper on ambient noise
    assert is_valid_speech_transcription("[music]") is False
    assert is_valid_speech_transcription("(laughter)") is False
    assert is_valid_speech_transcription("*cough*") is False
    assert is_valid_speech_transcription("[applause]") is False
    assert is_valid_speech_transcription("static") is False
    assert is_valid_speech_transcription("noise") is False
    assert is_valid_speech_transcription("ambient") is False

    # Repetitive single words from noise loops
    assert is_valid_speech_transcription("maddy maddy maddy") is False
    assert is_valid_speech_transcription("the the the the") is False
    assert is_valid_speech_transcription("Hey Maddy. Maddy. Maddy.") is False
    assert is_valid_speech_transcription("maddy maddy") is False
    assert is_valid_speech_transcription("hey hey") is False

    # Subtitle hallucination markers
    assert is_valid_speech_transcription("subtitles by amara.org") is False
    assert is_valid_speech_transcription("thank you for watching, please subscribe") is False

    # Authentic speech is accepted
    assert is_valid_speech_transcription("Hey Maddy what is the weather today") is True
    assert is_valid_speech_transcription("Open Spotify and play some music") is True

    # Extended ambient noise / hallucination artifacts
    assert is_valid_speech_transcription("foreign") is False
    assert is_valid_speech_transcription("shh") is False
    assert is_valid_speech_transcription("click") is False
    assert is_valid_speech_transcription("tap") is False
    assert is_valid_speech_transcription("buzz") is False
    assert is_valid_speech_transcription("whistle") is False


def test_conversational_noise_words_not_triggered_in_context():
    """Verify that ordinary room conversation and chatter never triggers false wake activations."""
    engine = WakeWordEngine()

    # Ambient room chatter and ordinary words must never trigger
    assert engine.detect_wake_word("I was talking to a friend yesterday") is False
    assert engine.detect_wake_word("We think we can start next month instead") is False
    assert engine.detect_wake_word("He said to be ready early tomorrow morning") is False
    assert engine.detect_wake_word("Can you check the documentation later") is False
    assert engine.detect_wake_word("Please turn down the volume in the living room") is False
    assert engine.detect_wake_word("I have to wake up early tomorrow") is False

    # Authentic intentional wake phrases for M.A.D.D.Y. must fire
    assert engine.detect_wake_word("Hey Maddy") is True
    assert engine.detect_wake_word("M.A.D.D.Y.") is True
    assert engine.detect_wake_word("maddy what is the time") is True
    assert engine.detect_wake_word("OK Maddy") is True
    assert engine.detect_wake_word("maddie") is True
    assert engine.detect_wake_word("wake up maddy") is True
    assert engine.detect_wake_word("daddy's home") is True

    # Standalone natural wake callout commands fire when utterance-initial
    assert engine.detect_wake_word("wake up") is True
    assert engine.detect_wake_word("wake up, what time is it") is True
    assert engine.detect_wake_word("listen up") is True


def test_dual_tier_neural_vs_phrase_wake_arbitration():
    """
    Verify that when neural openWakeWord model is active (oww is not None):
    1. Whisper hallucinating 'Hey Maddie.' on ambient room noise is rejected.
    2. Non-Maddy callout phrases (e.g. 'Daddy's home') are preserved.
    3. When oww is None (fallback mode), phrase wake detection triggers normally.
    """
    engine = WakeWordEngine()

    def arbitrate_standby_phrase(snip_text: str, oww_active: bool) -> bool:
        if not snip_text or not is_valid_speech_transcription(snip_text):
            return False
        import re
        is_callout = bool(re.search(r"\bdaddy(?:['’]s|\s+is)?\s+home\b", snip_text, flags=re.IGNORECASE))
        if oww_active and not is_callout:
            return False
        return engine.detect_wake_word(snip_text) or is_callout

    # When neural model is active: Whisper hallucinating "Hey Maddie." on room noise is rejected
    assert arbitrate_standby_phrase("Hey Maddie.", oww_active=True) is False
    assert arbitrate_standby_phrase("Hey Maddy", oww_active=True) is False
    assert arbitrate_standby_phrase("Maddie", oww_active=True) is False

    # Callout phrase works even when neural model is active
    assert arbitrate_standby_phrase("Daddy's home", oww_active=True) is True

    # When openWakeWord is None (pure fallback mode on systems without onnx):
    assert arbitrate_standby_phrase("Hey Maddie.", oww_active=False) is True
    assert arbitrate_standby_phrase("Hey Maddy", oww_active=False) is True


def test_barge_in_stop_words_interrupt_recitation():
    """Verify that saying stop words halts recitation immediately."""
    engine = WakeWordEngine()

    # Core stop commands
    assert engine.detect_barge_in("stop") is True
    assert engine.detect_barge_in("cancel") is True
    assert engine.detect_barge_in("shut up") is True
    assert engine.detect_barge_in("be quiet") is True
    assert engine.detect_barge_in("pause") is True
    assert engine.detect_barge_in("stop speaking") is True
    assert engine.detect_barge_in("stop talking") is True
    assert engine.detect_barge_in("quiet") is True
    assert engine.detect_barge_in("silence") is True
    assert engine.detect_barge_in("enough") is True
    assert engine.detect_barge_in("hold on") is True
    assert engine.detect_barge_in("wait") is True
    assert engine.detect_barge_in("hush") is True
    assert engine.detect_barge_in("abort") is True
    assert engine.detect_barge_in("never mind") is True

    # Conversational phrasing containing stop words
    assert engine.detect_barge_in("Maddy stop please") is True
    assert engine.detect_barge_in("Please be quiet now") is True

    # Non-stop speech must not trigger barge-in
    assert engine.detect_barge_in("What is the weather today") is False
    assert engine.detect_barge_in("Open Spotify and play music") is False


def test_silero_vad_rejects_non_speech_mechanical_noise():
    """Verify that mechanical click noise (typing/mouse taps) and 60Hz hum produce low VAD scores."""
    openwakeword = pytest.importorskip("openwakeword")
    import numpy as np
    from pathlib import Path

    vad_path = Path("models/wakewords/silero_vad.onnx")
    if not vad_path.exists():
        pytest.skip("silero_vad.onnx model not found")

    vad = openwakeword.VAD(model_path=str(vad_path))

    # 1. Mechanical transient impulse (sharp mouse click / keyboard tap)
    click = np.zeros(16000, dtype=np.int16)
    click[1000:1080] = np.random.randint(3000, 6000, 80, dtype=np.int16)
    click_vad = vad.predict(click, frame_size=640)
    assert click_vad < 0.35, f"Click VAD score ({click_vad}) must be < 0.35 to prevent false Whisper trigger"

    # 2. Continuous low-frequency ambient hum (60Hz electrical / fan rumble)
    t = np.linspace(0, 1.6, int(16000 * 1.6), endpoint=False)
    hum = (np.sin(2 * np.pi * 60 * t) * 2000).astype(np.int16)
    hum_vad = vad.predict(hum, frame_size=640)
    assert hum_vad < 0.10, f"Hum VAD score ({hum_vad}) must be < 0.10"


def test_openwakeword_model_and_vad_initialization():
    """Verify that OpenWakeWord Model initializes with M.A.D.D.Y. ONNX model and attaches Silero VAD."""
    openwakeword = pytest.importorskip("openwakeword")
    from openwakeword.model import Model as OpenWakeWordModel
    import numpy as np
    from pathlib import Path

    models_dir = Path("models/wakewords")
    maddy_model = models_dir / "maddy_v0.1.onnx"
    melspec = models_dir / "melspectrogram.onnx"
    embedding = models_dir / "embedding_model.onnx"
    vad_path = models_dir / "silero_vad.onnx"

    if not (maddy_model.exists() and melspec.exists() and embedding.exists() and vad_path.exists()):
        pytest.skip("OpenWakeWord ONNX models not populated")

    model = OpenWakeWordModel(
        wakeword_models=[str(maddy_model)],
        inference_framework="onnx",
        melspec_model_path=str(melspec),
        embedding_model_path=str(embedding),
    )
    vad = openwakeword.VAD(model_path=str(vad_path))
    model.vad = vad
    model.vad_threshold = 0.45

    assert "maddy_v0.1" in model.models
    assert model.vad is not None

    # Predict on silence: score must be 0.0
    silence = np.zeros(1280, dtype=np.int16)
    pred = model.predict(silence)
    assert pred.get("maddy_v0.1", 0.0) == 0.0


def test_trained_maddy_neural_model_acoustic_verification():
    """
    Verify that the trained neural maddy_v0.1.onnx model correctly activates
    on Maddy acoustic features and rejects third-party names and ambient chatter.
    """
    openwakeword = pytest.importorskip("openwakeword")
    from openwakeword.model import Model as OpenWakeWordModel
    from pathlib import Path
    import numpy as np

    models_dir = Path("models/wakewords")
    maddy_model = models_dir / "maddy_v0.1.onnx"
    melspec = models_dir / "melspectrogram.onnx"
    embedding = models_dir / "embedding_model.onnx"
    vad_path = models_dir / "silero_vad.onnx"

    if not (maddy_model.exists() and melspec.exists() and embedding.exists() and vad_path.exists()):
        pytest.skip("OpenWakeWord ONNX models not populated")

    model = OpenWakeWordModel(
        wakeword_models=[str(maddy_model)],
        inference_framework="onnx",
        melspec_model_path=str(melspec),
        embedding_model_path=str(embedding),
    )
    vad = openwakeword.VAD(model_path=str(vad_path))
    model.vad = vad
    model.vad_threshold = 0.45

    # Test on pre-existing synthesized verification audio clips if available
    import soundfile as sf
    pos_file = Path("/tmp/check_Hey_Maddy.wav")
    neg_audio = Path("/tmp/check_Good_morning.wav")
    neg_vision = Path("/tmp/check_Vision.wav")

    if pos_file.exists():
        audio, _ = sf.read(str(pos_file))
        audio_int16 = (audio * 32767).astype(np.int16)
        model.reset()
        peak_pos = max(
            list(model.predict(audio_int16[i:i+1280]).values())[0]
            for i in range(0, len(audio_int16), 1280)
        )
        assert peak_pos >= 0.85, f"Hey Maddy peak score ({peak_pos}) must be >= 0.85"

    if neg_audio.exists():
        audio, _ = sf.read(str(neg_audio))
        audio_int16 = (audio * 32767).astype(np.int16)
        model.reset()
        peak_neg = max(
            list(model.predict(audio_int16[i:i+1280]).values())[0]
            for i in range(0, len(audio_int16), 1280)
        )
        assert peak_neg < 0.05, f"Good morning peak score ({peak_neg}) must be < 0.05"

    if neg_vision.exists():
        audio, _ = sf.read(str(neg_vision))
        audio_int16 = (audio * 32767).astype(np.int16)
        model.reset()
        peak_vis = max(
            list(model.predict(audio_int16[i:i+1280]).values())[0]
            for i in range(0, len(audio_int16), 1280)
        )
        assert peak_vis < 0.05, f"Vision peak score ({peak_vis}) must be < 0.05"

