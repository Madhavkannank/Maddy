import time
import pytest
from unittest.mock import MagicMock
from tools.timer import TimerManager, parse_duration_to_seconds, format_duration


def test_parse_duration_to_seconds():
    assert parse_duration_to_seconds("5 minutes") == 300
    assert parse_duration_to_seconds("5 mins") == 300
    assert parse_duration_to_seconds("30 seconds") == 30
    assert parse_duration_to_seconds("1 hour 15 minutes") == 4500
    assert parse_duration_to_seconds("set timer for 10") == 600
    assert parse_duration_to_seconds("set timer of 5 minutes") == 300
    assert parse_duration_to_seconds("random sentence without time") is None


def test_format_duration():
    assert format_duration(300) == "5 minutes"
    assert format_duration(65) == "1 minute and 5 seconds"
    assert format_duration(3600) == "1 hour"


def test_timer_manager_lifecycle():
    mock_voice = MagicMock()
    tm = TimerManager(voice_callback=mock_voice)

    # Set 0.5s timer
    entry = tm.set_timer(1, label="1 second")
    assert entry.id in tm.timers
    assert len(tm.get_active_timers()) == 1

    # Cancel timer
    count = tm.cancel_all_timers()
    assert count == 1
    assert len(tm.get_active_timers()) == 0


def test_maddy_timer_commands():
    from app.maddy_assistant import MaddyAssistant
    assistant = MaddyAssistant()

    # 1. Set timer
    res = assistant.execute("Set the timer for 5 minutes")
    assert res["success"] is True
    assert "Timer set for 5 minutes" in res["response"]

    # 2. Status
    res_status = assistant.execute("How much time left on the timer?")
    assert res_status["success"] is True
    assert "remaining" in res_status["response"]

    # 3. Cancel
    res_cancel = assistant.execute("Cancel the timer")
    assert res_cancel["success"] is True
    assert "Cancelled" in res_cancel["response"]

    # 4. Clock time distinction
    res_clock = assistant.execute("What is the time now?")
    assert res_clock["success"] is True
    assert "It is" in res_clock["response"]
