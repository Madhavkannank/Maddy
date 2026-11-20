"""
Unit tests for Voice Listener Standby Noise Calibration, Dual-Trigger Dispatch,
and Dynamic Threshold Scaling.
"""
import numpy as np


class StandbySpeechDispatcher:
    """Simulates the standby speech state machine and phrase dispatch from voice_listener.py."""

    def __init__(self, initial_ambient_noise: float = 3800.0):
        self.ambient_noise_ema = initial_ambient_noise
        self.standby_chunks = []
        self.standby_speech_active = False
        self.standby_silence_count = 0
        self.last_standby_phrase_check = 0.0
        self.phrase_future_done = True
        self.submitted_snippets = []

    def get_dynamic_threshold(self) -> int:
        return max(220, int(self.ambient_noise_ema * 1.45 + 100))

    def get_phrase_gate(self) -> int:
        return max(180, int(self.ambient_noise_ema * 1.30 + 60))

    def process_frame(self, max_amp: int, current_time: float) -> bool:
        # 1. Update EMA
        if max_amp < self.ambient_noise_ema:
            self.ambient_noise_ema = 0.88 * self.ambient_noise_ema + 0.12 * max_amp
        else:
            self.ambient_noise_ema = 0.995 * self.ambient_noise_ema + 0.005 * max_amp

        phrase_gate = self.get_phrase_gate()
        chunk_bytes = b"\x00" * 256

        # 2. Accumulate speech frames
        if max_amp >= phrase_gate:
            self.standby_speech_active = True
            self.standby_chunks.append(chunk_bytes)
            self.standby_silence_count = 0
        elif self.standby_speech_active:
            self.standby_chunks.append(chunk_bytes)
            self.standby_silence_count += 1

        # 3. Two-Trigger Dispatch
        should_submit = False
        if self.standby_speech_active and self.standby_silence_count >= 3 and len(self.standby_chunks) >= 5:
            should_submit = True
        elif self.standby_speech_active and len(self.standby_chunks) >= 18:
            should_submit = True

        did_dispatch = False
        if should_submit:
            if self.phrase_future_done and (current_time - self.last_standby_phrase_check >= 0.6):
                self.last_standby_phrase_check = current_time
                self.submitted_snippets.append(list(self.standby_chunks))
                self.standby_speech_active = False
                self.standby_silence_count = 0
                self.standby_chunks = []
                did_dispatch = True
            elif len(self.standby_chunks) > 25:
                self.standby_chunks = self.standby_chunks[-15:]

        return did_dispatch


def test_baseline_noise_calibration_and_gates():
    dispatcher = StandbySpeechDispatcher(initial_ambient_noise=3800.0)

    # In high hardware pre-amp environments (e.g. 3800 RMS), phrase gate must be above noise floor
    phrase_gate = dispatcher.get_phrase_gate()
    assert phrase_gate > 3800, f"Phrase gate {phrase_gate} must scale above ambient noise floor 3800"
    assert phrase_gate == int(3800.0 * 1.30 + 60)  # 5000

    # Command dynamic threshold must also scale properly without 1400 ceiling cap
    dynamic_threshold = dispatcher.get_dynamic_threshold()
    assert dynamic_threshold > 3800, f"Dynamic threshold {dynamic_threshold} must scale above ambient noise"
    assert dynamic_threshold == int(3800.0 * 1.45 + 100)  # 5610


def test_natural_speech_pause_trigger():
    dispatcher = StandbySpeechDispatcher(initial_ambient_noise=1000.0)
    # phrase_gate = 1000 * 1.30 + 60 = 1360
    t = 100.0

    # 1. User says "Hey Maddy" (8 speech frames at amplitude 2500)
    for _ in range(8):
        t += 0.08
        dispatched = dispatcher.process_frame(max_amp=2500, current_time=t)
        assert dispatched is False
    assert dispatcher.standby_speech_active is True
    assert len(dispatcher.standby_chunks) == 8

    # 2. User pauses (silence frames at amplitude 1000)
    # Frame 1 of silence
    t += 0.08
    assert dispatcher.process_frame(max_amp=1000, current_time=t) is False
    assert dispatcher.standby_silence_count == 1

    # Frame 2 of silence
    t += 0.08
    assert dispatcher.process_frame(max_amp=1000, current_time=t) is False
    assert dispatcher.standby_silence_count == 2

    # Frame 3 of silence -> 240ms natural pause reached -> MUST DISPATCH!
    t += 0.08
    dispatched = dispatcher.process_frame(max_amp=1000, current_time=t)
    assert dispatched is True
    assert len(dispatcher.submitted_snippets) == 1
    assert len(dispatcher.submitted_snippets[0]) == 11  # 8 speech + 3 pause
    assert dispatcher.standby_speech_active is False
    assert len(dispatcher.standby_chunks) == 0


def test_utterance_safety_cap_trigger():
    dispatcher = StandbySpeechDispatcher(initial_ambient_noise=1000.0)
    t = 200.0

    # User speaks continuously for 18 frames without a pause
    for i in range(17):
        t += 0.08
        dispatched = dispatcher.process_frame(max_amp=2500, current_time=t)
        assert dispatched is False, f"Should not dispatch on frame {i+1}"

    # 18th frame -> Utterance safety cap reached -> MUST DISPATCH!
    t += 0.08
    dispatched = dispatcher.process_frame(max_amp=2500, current_time=t)
    assert dispatched is True
    assert len(dispatcher.submitted_snippets) == 1
    assert len(dispatcher.submitted_snippets[0]) == 18
    assert dispatcher.standby_speech_active is False
    assert len(dispatcher.standby_chunks) == 0


def test_buffer_preservation_when_worker_busy():
    dispatcher = StandbySpeechDispatcher(initial_ambient_noise=1000.0)
    dispatcher.phrase_future_done = False  # Worker busy
    t = 300.0

    # Speech + pause reached
    for _ in range(8):
        t += 0.08
        dispatcher.process_frame(max_amp=2500, current_time=t)
    for _ in range(3):
        t += 0.08
        dispatcher.process_frame(max_amp=1000, current_time=t)

    # Dispatch could not run because worker was busy, but standby_chunks must NOT be cleared!
    assert len(dispatcher.submitted_snippets) == 0
    assert len(dispatcher.standby_chunks) == 11

    # Now worker finishes
    dispatcher.phrase_future_done = True
    t += 0.08
    dispatched = dispatcher.process_frame(max_amp=1000, current_time=t)
    assert dispatched is True
    assert len(dispatcher.submitted_snippets) == 1
    assert len(dispatcher.submitted_snippets[0]) == 12


if __name__ == "__main__":
    print("Running voice standby dispatch tests...")
    test_baseline_noise_calibration_and_gates()
    print("✓ test_baseline_noise_calibration_and_gates passed")
    test_natural_speech_pause_trigger()
    print("✓ test_natural_speech_pause_trigger passed")
    test_utterance_safety_cap_trigger()
    print("✓ test_utterance_safety_cap_trigger passed")
    test_buffer_preservation_when_worker_busy()
    print("✓ test_buffer_preservation_when_worker_busy passed")
    print("ALL STANDBY DISPATCH TESTS PASSED!")
