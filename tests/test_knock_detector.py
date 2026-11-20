"""
Unit tests for the surface knock and double-clap detector engine.
"""
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from core.knock_detector import (
    KnockDetector,
    is_maddy_running,
    default_trigger_action,
    get_running_pid,
    write_pid,
    remove_pid,
)


def make_quiet_frame(size: int = 512) -> np.ndarray:
    """Create simulated ambient quiet frame."""
    return np.random.normal(0, 0.005, size=size).astype(np.float32)


def make_knock_frame(peak: float = 0.5, size: int = 512) -> np.ndarray:
    """Create simulated sharp transient knock frame."""
    frame = np.random.normal(0, 0.01, size=size).astype(np.float32)
    # Inject steep mechanical impulse spike in center
    frame[size // 2] = peak
    frame[size // 2 + 1] = peak * 0.4
    frame[size // 2 + 2] = peak * 0.1
    return frame


def test_initialization():
    detector = KnockDetector(peak_threshold=0.25, min_interval_sec=0.15, max_interval_sec=0.8)
    assert detector.peak_threshold == 0.25
    assert detector.min_interval_sec == 0.15
    assert detector.max_interval_sec == 0.8
    assert detector.first_knock_time == 0.0


def test_single_knock_detection():
    detector = KnockDetector(peak_threshold=0.20, min_interval_sec=0.1, max_interval_sec=0.8)
    
    # Send quiet frames to settle ambient floor
    for _ in range(5):
        detector.process_frame(make_quiet_frame())
    
    t0 = 100.0
    knock_frame = make_knock_frame(peak=0.6)
    single, double, metrics = detector.process_frame(knock_frame, current_time=t0)
    
    assert single is True
    assert double is False
    assert detector.first_knock_time == t0


def test_double_knock_within_valid_window():
    mock_action = MagicMock()
    detector = KnockDetector(
        peak_threshold=0.20,
        min_interval_sec=0.15,
        max_interval_sec=0.80,
        cooldown_sec=2.0,
        on_double_knock=mock_action,
    )
    
    for _ in range(5):
        detector.process_frame(make_quiet_frame())
    
    t0 = 100.0
    single1, double1, _ = detector.process_frame(make_knock_frame(peak=0.5), current_time=t0)
    assert single1 is True
    assert double1 is False
    
    # Second knock at t0 + 0.35s (well inside [0.15s, 0.80s])
    t1 = t0 + 0.35
    single2, double2, _ = detector.process_frame(make_knock_frame(peak=0.55), current_time=t1)
    assert double2 is True
    assert detector.first_knock_time == 0.0
    assert detector.last_trigger_time == t1


def test_reject_bounce_too_fast():
    detector = KnockDetector(
        peak_threshold=0.20,
        min_interval_sec=0.15,
        max_interval_sec=0.80,
    )
    
    for _ in range(5):
        detector.process_frame(make_quiet_frame())
    
    t0 = 100.0
    single1, double1, _ = detector.process_frame(make_knock_frame(peak=0.5), current_time=t0)
    assert single1 is True
    
    # Second spike at t0 + 0.05s (50ms: too fast, mechanical rebound/drag)
    single2, double2, _ = detector.process_frame(make_knock_frame(peak=0.45), current_time=t0 + 0.05)
    assert single2 is False
    assert double2 is False
    assert detector.first_knock_time == t0  # Still waiting for a valid 2nd knock!


def test_expired_window_becomes_new_first_knock():
    detector = KnockDetector(
        peak_threshold=0.20,
        min_interval_sec=0.15,
        max_interval_sec=0.80,
    )
    
    for _ in range(5):
        detector.process_frame(make_quiet_frame())
    
    t0 = 100.0
    single1, double1, _ = detector.process_frame(make_knock_frame(peak=0.5), current_time=t0)
    assert single1 is True
    
    # Next sound arrives at t0 + 1.5s (> max_interval 0.8s)
    t_late = t0 + 1.5
    single2, double2, _ = detector.process_frame(make_knock_frame(peak=0.5), current_time=t_late)
    assert single2 is True
    assert double2 is False
    assert detector.first_knock_time == t_late  # Now waiting starting from t_late


def test_cooldown_ignores_subsequent_audio():
    detector = KnockDetector(
        peak_threshold=0.20,
        min_interval_sec=0.10,
        max_interval_sec=0.80,
        cooldown_sec=3.0,
    )
    
    t0 = 100.0
    detector.process_frame(make_knock_frame(peak=0.5), current_time=t0)
    detector.process_frame(make_knock_frame(peak=0.5), current_time=t0 + 0.25)  # Triggers double knock
    
    # Sound during cooldown (t0 + 1.0s, cooldown is 3.0s)
    single3, double3, _ = detector.process_frame(make_knock_frame(peak=0.9), current_time=t0 + 1.0)
    assert single3 is False
    assert double3 is False


def test_default_trigger_action_skips_when_running():
    with patch("core.knock_detector.is_maddy_running", return_value=12345):
        res = default_trigger_action()
        assert res is False


def test_default_trigger_action_launches_when_not_running():
    with patch("core.knock_detector.is_maddy_running", return_value=None), \
         patch("core.knock_detector.os.path.exists", return_value=True), \
         patch("core.knock_detector.subprocess.Popen") as mock_popen:
        res = default_trigger_action()
        assert res is True
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "maddy.cmd" in call_args


def test_pid_lifecycle(tmp_path):
    test_pid_file = str(tmp_path / "test_knock.pid")
    with patch("core.knock_detector.PID_FILE_PATH", test_pid_file):
        write_pid(999999)
        assert get_running_pid() is None  # 999999 is not a live python process, so should clean up
        remove_pid()
