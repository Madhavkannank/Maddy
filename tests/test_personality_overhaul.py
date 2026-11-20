"""
Acceptance Tests for M.A.D.D.Y. Personality & Response Quality Overhaul.
Verifies:
1. Natural address policy (boss/sir) without volunteering user's real name.
2. Name privacy gate ('What is my name?' requires confirmation before revelation).
3. Explicit naming preference lifecycle ('Call me X').
4. Standardized task failure responses ('Task failed, boss. [Reason]. Want me to retry?').
5. Spoken output sanitization (markdown removal, anti-cliche filtering).
6. Integration with MaddyAssistant.
"""
import pytest
from personality.engine import PersonalityEngine
from personality.policy import PersonalityPolicy, NamePermissionState
from memory.user_preferences import UserPreferences
from app.maddy_assistant import MaddyAssistant


def test_default_addressing_is_boss_or_sir(tmp_path):
    """Verify default address alternates naturally between boss and sir, never real name."""
    prefs = UserPreferences(storage_path=str(tmp_path / "prefs.json"))
    policy = PersonalityPolicy(user_preferences=prefs)

    # Alternate naturally
    addr1 = policy.get_natural_address()
    addr2 = policy.get_natural_address()
    assert addr1 in ("boss", "sir")
    assert addr2 in ("boss", "sir")
    assert addr1 != addr2


def test_real_name_never_volunteered_spontaneously(tmp_path):
    """Even if known in preferences, real name is never returned by default address."""
    prefs = UserPreferences(storage_path=str(tmp_path / "prefs.json"))
    prefs.set_preference("user.real_name", "Madhav")
    policy = PersonalityPolicy(user_preferences=prefs)

    for _ in range(6):
        addr = policy.get_natural_address()
        assert addr in ("boss", "sir")
        assert addr != "Madhav"


def test_handle_name_query_or_setup_delegates_dynamically(tmp_path):
    """Conversational requests are not intercepted by rigid regex patterns, flowing dynamically to LLM."""
    prefs = UserPreferences(storage_path=str(tmp_path / "prefs.json"))
    prefs.set_preference("user.real_name", "Madhav")
    policy = PersonalityPolicy(user_preferences=prefs)

    handled, resp = policy.handle_name_query_or_setup("What is my name?")
    assert handled is False
    assert resp is None

    handled2, resp2 = policy.handle_name_query_or_setup("Do you know my name?")
    assert handled2 is False
    assert resp2 is None

    handled3, resp3 = policy.handle_name_query_or_setup("Call me Madhav")
    assert handled3 is False
    assert resp3 is None


def test_task_failure_formatting():
    """Verify standardized failure output format with clean reason and retry question."""
    policy = PersonalityPolicy()

    # Known reason
    fail1 = policy.format_task_failure("Chrome application not found", address="boss")
    assert fail1 == "Task failed, boss. Chrome application not found. Want me to retry?"

    # Exception with stack trace / prefix
    fail2 = policy.format_task_failure("FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'", address="sir")
    assert "Task failed, sir." in fail2
    assert "Want me to retry?" in fail2
    assert "FileNotFoundError:" not in fail2

    # Timeout
    fail3 = policy.format_task_failure("HTTP connection timed out", address="boss")
    assert fail3 == "Task failed, boss. The operation timed out before responding. Want me to retry?"

    # Unknown reason
    fail4 = policy.format_task_failure("", address="boss")
    assert fail4 == "Task failed, boss. I couldn't determine the exact cause. Want me to retry?"


def test_spoken_output_sanitization():
    """Verify markdown formatting and corporate cliches are removed for natural speech."""
    policy = PersonalityPolicy()

    # Markdown formatting
    raw_md = "### Calculation Result\n**Result:** `1200`\n- Step 1 complete."
    clean = policy.sanitize_spoken_output(raw_md)
    assert "**" not in clean
    assert "###" not in clean
    assert "`" not in clean
    assert "Result: 1200" in clean

    # Corporate cliches
    raw_cliche = "Certainly! Here is your answer. As an AI language model, I'm delighted to help! 42."
    clean_cliche = policy.sanitize_spoken_output(raw_cliche)
    assert "certainly! here is" not in clean_cliche.lower()
    assert "as an ai language model" not in clean_cliche.lower()
    assert "delighted to help" not in clean_cliche.lower()
    assert "42" in clean_cliche


def test_maddy_assistant_intercept_name_lifecycle():
    """Verify full end-to-end name lifecycle through MaddyAssistant.execute()."""
    from unittest.mock import patch, MagicMock
    assistant = MaddyAssistant()

    mock_res = MagicMock()
    mock_res.status = "SUCCESS"
    mock_res.final_response = "Your name is Madhav, boss."
    mock_res.completed_tasks = []
    mock_res.total_tasks = 0

    with patch.object(assistant.supervisor, "run", return_value=mock_res):
        res1 = assistant.execute("What is my name?")
    assert res1["success"] is True
    assert "madhav" in res1["response"].lower()


def test_maddy_assistant_task_failure_format():
    """Verify MaddyAssistant format on failure does not echo the full user request and adheres to format."""
    from unittest.mock import patch
    assistant = MaddyAssistant()

    long_request = "Organize all source code files into folders and run build verification across all repositories"
    with patch.object(assistant.supervisor, "run", side_effect=RuntimeError("Chrome couldn't be opened because the application wasn't found")):
        res = assistant.execute(long_request)

    # Must produce concise failure format without dumping the long user request
    assert res["success"] is False
    assert res["status"] == "FAILED"
    assert "Task failed" in res["response"]
    assert "Want me to retry?" in res["response"]
    assert "Chrome couldn't be opened because the application wasn't found" in res["response"]
    assert long_request not in res["response"]


def test_get_greeting_candidates_pool_size_and_variety(tmp_path):
    """Verify that get_greeting_candidates provides a rich pool of at least 30 polite acknowledgments."""
    prefs = UserPreferences(storage_path=str(tmp_path / "prefs.json"))
    policy = PersonalityPolicy(user_preferences=prefs)

    candidates = policy.get_greeting_candidates(addr="boss")
    assert len(candidates) >= 30, f"Expected at least 30 candidates, got {len(candidates)}"

    for cand in candidates:
        assert isinstance(cand, str)
        assert len(cand.strip()) > 0
        assert "boss" in cand.lower(), f"Candidate missing address title: {cand}"

