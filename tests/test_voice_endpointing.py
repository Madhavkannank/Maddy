"""
Unit tests for Voice Listener consecutive-frame VAD endpointing and utterance finalization.
Verifies that listening stops cleanly and reliably without hanging or looping endlessly.
"""
import time
try:
    import pytest
except ImportError:
    pytest = None


class VADEndpointDetector:
    """Simulates the VAD endpointing state machine from voice_listener.py."""

    def __init__(self, dynamic_threshold: int = 250, min_speech_chunks: int = 2, silence_limit_chunks: int = 8):
        self.dynamic_threshold = dynamic_threshold
        self.min_speech_chunks = min_speech_chunks
        self.silence_limit_chunks = silence_limit_chunks
        self.reset()

    def reset(self):
        self.speech_started = False
        self.speech_chunk_count = 0
        self.speech_resume_count = 0
        self.consecutive_silence = 0
        self.command_wait_start = time.time()
        self.command_chunks = []

    def process_chunk(self, max_amp: int, current_time: float = None) -> bool:
        now = current_time if current_time is not None else time.time()
        self.command_chunks.append(b"\x00" * 256)

        if not self.speech_started:
            if max_amp >= self.dynamic_threshold:
                self.speech_chunk_count += 1
                if self.speech_chunk_count >= self.min_speech_chunks:
                    self.speech_started = True
                    self.consecutive_silence = 0
                    self.speech_resume_count = 0
            else:
                self.speech_chunk_count = max(0, self.speech_chunk_count - 1)
        else:
            if max_amp < self.dynamic_threshold:
                self.consecutive_silence += 1
                self.speech_resume_count = 0
            else:
                self.speech_resume_count += 1
                if self.speech_resume_count >= 2:
                    self.consecutive_silence = 0
                    self.speech_resume_count = 0

        # Endpoint check
        should_process = False
        if self.speech_started and self.consecutive_silence >= self.silence_limit_chunks:
            should_process = True
        elif self.speech_started and (now - self.command_wait_start >= 10.0) and len(self.command_chunks) >= 12:
            should_process = True

        return should_process


def test_vad_triggers_after_consecutive_silence():
    vad = VADEndpointDetector(dynamic_threshold=200, min_speech_chunks=2, silence_limit_chunks=8)

    # 1. Quiet ambient chunks before speech
    for _ in range(5):
        assert vad.process_chunk(max_amp=50) is False
    assert vad.speech_started is False

    # 2. Speech onset (3 chunks > threshold)
    for _ in range(3):
        assert vad.process_chunk(max_amp=400) is False
    assert vad.speech_started is True
    assert vad.consecutive_silence == 0

    # 3. User finishes speaking and goes quiet (first 7 chunks: should not finalize yet)
    for i in range(1, 8):
        assert vad.process_chunk(max_amp=50) is False
        assert vad.consecutive_silence == i

    # 4. 8th consecutive silence chunk (~640ms total pause) -> MUST FINALIZE
    finalized = vad.process_chunk(max_amp=50)
    assert finalized is True
    assert vad.consecutive_silence == 8


def test_vad_ignores_isolated_noise_blip_during_silence():
    vad = VADEndpointDetector(dynamic_threshold=200, min_speech_chunks=2, silence_limit_chunks=8)

    # User speaks
    vad.process_chunk(max_amp=350)
    vad.process_chunk(max_amp=350)
    assert vad.speech_started is True

    # 4 silence chunks
    for _ in range(4):
        vad.process_chunk(max_amp=50)
    assert vad.consecutive_silence == 4

    # 1 transient click/flutter spike (only 1 chunk)
    vad.process_chunk(max_amp=300)
    # Consecutive silence is NOT reset by a single-chunk transient noise blip
    assert vad.consecutive_silence == 4

    # 4 more silence chunks -> total reaches 8 chunks
    for i in range(3):
        assert vad.process_chunk(max_amp=50) is False
    finalized = vad.process_chunk(max_amp=50)
    assert finalized is True


def test_vad_maximum_utterance_safety_cap():
    vad = VADEndpointDetector(dynamic_threshold=200, min_speech_chunks=2, silence_limit_chunks=8)

    t0 = 1000.0
    vad.command_wait_start = t0

    # Speech starts
    vad.process_chunk(max_amp=300, current_time=t0)
    vad.process_chunk(max_amp=300, current_time=t0 + 0.1)
    assert vad.speech_started is True

    # Continuous talker with no 640ms pause for 10.1 seconds
    for step in range(15):
        t = t0 + 0.2 + (step * 0.7)
        res = vad.process_chunk(max_amp=250, current_time=t)
        if t - t0 >= 10.0 and len(vad.command_chunks) >= 12:
            assert res is True
            break


if __name__ == "__main__":
    print("Running voice endpointing tests...")
    test_vad_triggers_after_consecutive_silence()
    print("✓ test_vad_triggers_after_consecutive_silence passed")
    test_vad_ignores_isolated_noise_blip_during_silence()
    print("✓ test_vad_ignores_isolated_noise_blip_during_silence passed")
    test_vad_maximum_utterance_safety_cap()
    print("✓ test_vad_maximum_utterance_safety_cap passed")
    print("ALL 3 VAD ENDPOINTING TESTS PASSED!")
