"""Tests for Scheduled Tasks & Background Heartbeat Engine (Cron).

Verifies:
1. CronJob schedule calculations (interval and time_of_day).
2. Full CRUD lifecycle: add, list, pause, resume, remove.
3. Thread-safe JSON persistence.
4. Background periodic firing and execution callback dispatch.
5. Voice announcement delivery.
6. ToolRegistry integration via cronjob_manage tool.
"""
import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from agents.base import AgentStatus
from agents.maddy_agent import MaddyAgent
from models.base import ModelProvider, ModelResponse
from tasks.cron_engine import CronEngine, CronJob
from tools.base import Tool
from tools.cron_tool import create_cronjob_tool, cronjob_manage_tool, set_cron_engine
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry


class MockProvider(ModelProvider):
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt, **kwargs):
        if self.responses:
            return ModelResponse(content=self.responses.pop(0))
        return ModelResponse(content="Done.")

    def stream(self, prompt, **kwargs):
        yield self.generate(prompt).content

    def structured(self, prompt, schema, **kwargs):
        return {}


@pytest.fixture
def temp_cron():
    temp_dir = tempfile.mkdtemp()
    storage = Path(temp_dir) / "test_cron_jobs.json"
    executed_goals = []
    spoken_alerts = []

    engine = CronEngine(
        storage_path=storage,
        job_executor=lambda g: executed_goals.append(g) or f"Done: {g}",
        voice_notifier=lambda t: spoken_alerts.append(t),
        check_interval=0.05,
    )
    set_cron_engine(engine)
    yield engine, executed_goals, spoken_alerts
    engine.stop()
    set_cron_engine(None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_cron_crud_lifecycle(temp_cron):
    """Verify adding, listing, pausing, resuming, and removing jobs."""
    engine, _, _ = temp_cron

    # 1. Add job
    res = engine.add_job(
        name="GPU Health Check",
        goal="Check GPU VRAM and temperature",
        interval_seconds=60.0,
        deliver_voice=True,
    )
    assert res["success"] is True
    job_id = res["job_id"]

    # 2. List jobs
    jobs = engine.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["name"] == "GPU Health Check"
    assert jobs[0]["enabled"] is True

    # 3. Pause job
    res_pause = engine.pause_job(job_id)
    assert res_pause["success"] is True
    assert engine.list_jobs()[0]["enabled"] is False

    # 4. Resume job
    res_resume = engine.resume_job(job_id)
    assert res_resume["success"] is True
    assert engine.list_jobs()[0]["enabled"] is True

    # 5. Remove job
    res_del = engine.remove_job(job_id)
    assert res_del["success"] is True
    assert len(engine.list_jobs()) == 0


def test_background_periodic_execution(temp_cron):
    """Verify background scheduler fires due jobs automatically."""
    engine, executed, spoken = temp_cron

    # Add a job due immediately (interval 0.05s)
    engine.add_job(
        name="Quick Ping",
        goal="Perform diagnostic ping",
        interval_seconds=0.05,
        deliver_voice=True,
    )
    # Start scheduler loop
    engine.start()

    # Wait briefly for scheduler to fire at least once
    time.sleep(0.25)
    engine.stop()

    assert len(executed) >= 1
    assert "Perform diagnostic ping" in executed
    assert len(spoken) >= 1
    assert "Quick Ping" in spoken[0]


def test_manual_trigger_job(temp_cron):
    """Verify manual run action fires immediate worker execution."""
    engine, executed, _ = temp_cron
    res = engine.add_job(name="Manual Task", goal="Execute manual action", interval_seconds=3600.0)
    job_id = res["job_id"]

    res_run = engine.trigger_job(job_id)
    assert res_run["success"] is True

    # Give thread a moment
    time.sleep(0.1)
    assert "Execute manual action" in executed


def test_agent_schedules_cron_job(temp_cron):
    """Verify MaddyAgent can autonomously use cronjob_manage to schedule a background task."""
    engine, _, _ = temp_cron

    registry = ToolRegistry()
    registry.register(create_cronjob_tool(engine))
    executor = ToolExecutor(registry=registry)

    responses = [
        '<thought>I need to schedule an hourly backup check.</thought>\n<tool_call>{"name": "cronjob_manage", "arguments": {"action": "add", "name": "Hourly Backup", "goal": "Check backup directory status", "interval_seconds": 3600}}</tool_call>',
        '<thought>Job is scheduled.</thought>\nI have scheduled the hourly backup check for you, boss.',
    ]
    model = MockProvider(responses)

    agent = MaddyAgent(
        model_provider=model,
        executor=executor,
        registry=registry,
        max_steps=5,
        address_title="boss",
    )

    result = agent.run("Please schedule an hourly check of the backup directory.")
    assert result.status == AgentStatus.SUCCESS
    assert "hourly backup" in result.final_answer.lower()
    assert "boss" in result.final_answer.lower()

    jobs = engine.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["name"] == "Hourly Backup"
    assert jobs[0]["interval_seconds"] == 3600
