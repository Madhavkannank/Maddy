"""
Unit tests for Phase 8: Memory Architecture & Context Compression Engine.
Validates WorkingMemory, TaskMemory, EpisodicMemory, UserPreferences, ContextCompressor,
and agent memory integration across cross-platform environments.
"""
import os
import tempfile
import pytest
from unittest.mock import MagicMock

from memory.working_memory import WorkingMemory
from memory.task_memory import TaskMemory
from memory.episodic_memory import EpisodicMemory
from memory.user_preferences import UserPreferences
from memory.context_engine import ContextCompressor
from config.settings import get_memory_dir, Settings

def test_working_memory_operations():
    wm = WorkingMemory(goal="Test working memory goal")
    assert wm.goal == "Test working memory goal"

    wm.set("key1", "val1")
    wm.set("key2", 42)
    assert wm.get("key1") == "val1"
    assert wm.get("key2") == 42
    assert wm.get("nonexistent", "default") == "default"

    wm.add_observation(1, "filesystem.read", {"content": "hello"})
    obs = wm.get_observations()
    assert len(obs) == 1
    assert obs[0]["step"] == 1
    assert obs[0]["tool"] == "filesystem.read"

    wm.clear()
    assert wm.get("key1") is None
    assert len(wm.get_observations()) == 0

def test_task_memory_disk_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskMemory(task_id="task_test_123", base_dir=tmpdir)
        
        tm.save_goal("Read configuration file")
        tm.save_plan([{"id": "T1", "description": "read config"}])
        tm.record_progress(1, "RUNNING", "Reading file")
        tm.record_observation(1, "filesystem.read", {"status": "ok"})
        tm.record_error(2, "IOError", "File not found")
        artifact_path = tm.save_artifact("summary.txt", "Execution summary content")
        tm.save_result("SUCCESS", "Completed configuration read")

        assert os.path.exists(artifact_path)
        with open(artifact_path, "r", encoding="utf-8") as f:
            assert f.read() == "Execution summary content"

        # Reload and verify disk data
        loaded_goal = tm.load_goal()
        assert loaded_goal.get("goal") == "Read configuration file"

        loaded_plan = tm.load_plan()
        assert len(loaded_plan) == 1
        assert loaded_plan[0]["id"] == "T1"

        loaded_progress = tm.load_progress()
        assert len(loaded_progress) == 1

        loaded_obs = tm.load_observations()
        assert len(loaded_obs) == 1

        loaded_errs = tm.load_errors()
        assert len(loaded_errs) == 1

        loaded_res = tm.load_result()
        assert loaded_res.get("status") == "SUCCESS"
        assert loaded_res.get("result") == "Completed configuration read"

def test_episodic_memory_search_and_retrieval():
    with tempfile.TemporaryDirectory() as tmpdir:
        em = EpisodicMemory(base_dir=tmpdir)

        em.record_episode("T100", "Play Spotify music", "SUCCESS", "Started playback of jazz playlist", keywords=["spotify", "music", "audio"])
        em.record_episode("T101", "Check system logs", "SUCCESS", "Analyzed kernel logs for errors", keywords=["logs", "linux", "system"])
        em.record_episode("T102", "Search files for python code", "SUCCESS", "Found 15 python files in workspace", keywords=["filesystem", "python", "code"])

        all_episodes = em.get_all_episodes()
        assert len(all_episodes) == 3

        recent = em.get_recent_episodes(limit=2)
        assert len(recent) == 2

        # Search by keyword
        spotify_results = em.search_episodes("spotify music", limit=5)
        assert len(spotify_results) >= 1
        assert spotify_results[0]["task_id"] == "T100"

        code_results = em.search_episodes("python code search", limit=5)
        assert len(code_results) >= 1
        assert code_results[0]["task_id"] == "T102"

        fetched_ep = em.get_episode("T101")
        assert fetched_ep is not None
        assert fetched_ep["goal"] == "Check system logs"

def test_user_preferences_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        pref_file = os.path.join(tmpdir, "user_preferences.json")
        up = UserPreferences(storage_path=pref_file)

        assert up.get_preference("theme") is None
        up.set_preference("theme", "dark")
        up.set_preference("max_concurrent_workers", 4)

        assert up.get_preference("theme") == "dark"
        assert up.get_preference("max_concurrent_workers") == 4

        all_prefs = up.get_all_preferences()
        assert len(all_prefs) == 2

        # Delete preference
        deleted = up.delete_preference("theme")
        assert deleted is True
        assert up.get_preference("theme") is None
        assert len(up.get_all_preferences()) == 1

def test_context_compressor():
    compressor = ContextCompressor(max_chars=2000, max_recent_steps=2)

    obs_list = [
        {"step": 1, "tool": "filesystem.search", "data": "found file 1"},
        {"step": 2, "tool": "filesystem.read", "data": "read file 1 content"},
        {"step": 3, "tool": "filesystem.write", "data": "wrote output"},
        {"step": 4, "tool": "application.open", "data": "opened app"},
    ]

    compressed = compressor.compress_observations(obs_list)
    # Recent steps (3 and 4) kept intact + 1 summary item for older steps (1 and 2)
    assert len(compressed) == 3
    assert compressed[0]["tool"] == "context_compressor"
    assert compressed[1]["step"] == 3
    assert compressed[2]["step"] == 4

    wm = WorkingMemory(goal="Compress test")
    for o in obs_list:
        wm.add_observation(o["step"], o["tool"], o["data"])

    bounded_context = compressor.build_bounded_context(
        goal="Compress test",
        working_memory=wm,
    )
    assert "goal" in bounded_context
    assert "observations" in bounded_context
    assert len(bounded_context["observations"]) == 3

def test_get_memory_dir_env_override(monkeypatch, tmp_path):
    custom_path = str(tmp_path / "custom_memory")
    monkeypatch.setenv("MADDY_MEMORY_DIR", custom_path)

    resolved = get_memory_dir()
    assert resolved == custom_path

def test_get_memory_dir_env_override_legacy(monkeypatch, tmp_path):
    custom_path = str(tmp_path / "custom_memory_legacy")
    monkeypatch.delenv("MADDY_MEMORY_DIR", raising=False)
    monkeypatch.setenv("MADDY_MEMORY_DIR", custom_path)

    resolved = get_memory_dir()
    assert resolved == custom_path

def test_get_memory_dir_unwritable_fallback(monkeypatch):
    # Set non-writable environment override path
    non_writable_path = "/proc/invalid_read_only_path/memory"
    monkeypatch.setenv("MADDY_MEMORY_DIR", non_writable_path)

    # get_memory_dir should safely reject non-writable path and fall back to user home .maddy/memory
    resolved = get_memory_dir()
    assert resolved != non_writable_path
    assert ".maddy/memory" in resolved or "memory" in resolved

def test_is_dir_writable_probe(tmp_path):
    from config.settings import _is_dir_writable
    valid_path = str(tmp_path / "writable_dir")
    assert _is_dir_writable(valid_path) is True

    invalid_path = "/root/permission_denied_test_dir"
    assert _is_dir_writable(invalid_path) is False

def test_single_agent_memory_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskMemory(task_id="agent_mem_test", base_dir=tmpdir)
        wm = WorkingMemory(goal="SingleAgent memory goal")
        em = EpisodicMemory(base_dir=tmpdir)

        # Mock ModelProvider
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"thought": "done", "final_answer": "Task Completed Successfully"}'
        mock_model.generate.return_value = mock_response

        mock_executor = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_tool_schemas.return_value = []

        from agents.single_agent import SingleAgent
        agent = SingleAgent(
            model_provider=mock_model,
            executor=mock_executor,
            registry=mock_registry,
            task_memory=tm,
            working_memory=wm,
            episodic_memory=em,
        )

        result = agent.run("SingleAgent memory goal")
        assert result.status.name == "SUCCESS"
        assert result.final_answer == "Task Completed Successfully"

        # Verify disk persistence via TaskMemory
        res_data = tm.load_result()
        assert res_data.get("status") == "SUCCESS"
        assert res_data.get("result") == "Task Completed Successfully"

        # Verify episode recorded in EpisodicMemory
        episodes = em.get_all_episodes()
        assert len(episodes) == 1
        assert episodes[0]["task_id"] == "agent_mem_test"
        assert episodes[0]["summary"] == "Task Completed Successfully"
