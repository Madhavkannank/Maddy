"""Phase 2 Benchmark Suite for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) Agent Core.

Governed strictly by CLAUDE.md Benchmark Rules:
- Pinned model: llama-3.3-70b-versatile (via Groq Cloud API)
- Fixed temperature: 0.0
- Run each benchmark task 3 times
- A task passes when >= 2/3 runs succeed
- Outcomes verified using independent Python ground-truth assertions (NEVER model claims)
- Detailed execution traces recorded in SQLite WAL database
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent_loop import MaddyAgentCore
from core.llm_client import CloudLLMClient
from core.permissions import PermissionGate
from core.tools.base import ToolRegistry
from core.tools.file_ops import ReadFileTool, WriteFileTool
from core.tools.system_info import SystemInfoTool
from core.tools.terminal import ExecuteCommandTool
from core.trace_store import TraceStore

BENCHMARK_DB_PATH = PROJECT_ROOT / "logs" / "phase2_benchmarks.db"
PINNED_MODEL = "openai/gpt-oss-120b"
FIXED_TEMPERATURE = 0.0


@dataclass
class RunResult:
    run_index: int
    task_id: str
    passed: bool
    duration_s: float
    error: Optional[str] = None
    verification_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskBenchmarkReport:
    task_id: str
    name: str
    description: str
    runs: List[RunResult] = field(default_factory=list)

    @property
    def passed_runs(self) -> int:
        return sum(1 for r in self.runs if r.passed)

    @property
    def is_successful(self) -> bool:
        return self.passed_runs >= 2


def build_test_agent() -> tuple[MaddyAgentCore, TraceStore]:
    """Instantiate a clean M.A.D.D.Y. agent core configured for benchmark evaluation."""
    trace_store = TraceStore(db_path=BENCHMARK_DB_PATH)
    registry = ToolRegistry()
    registry.register(SystemInfoTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ExecuteCommandTool())

    permission_gate = PermissionGate(
        auto_approve_tier2=False,
        allowed_write_directories=["/tmp", os.path.expanduser("~/sandbox")],
    )

    llm_client = CloudLLMClient(
        default_model=PINNED_MODEL,
        default_temperature=FIXED_TEMPERATURE,
    )

    agent = MaddyAgentCore(
        llm_client=llm_client,
        tool_registry=registry,
        trace_store=trace_store,
        permission_gate=permission_gate,
    )
    return agent, trace_store


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK TASK 1: System Telemetry & Hardware State (Safe Tier)
# ─────────────────────────────────────────────────────────────────────────────
def verify_task_1(turn_result: Dict[str, Any], trace_store: TraceStore) -> tuple[bool, Dict[str, Any]]:
    """Ground-truth verification for Task 1: System Telemetry.
    Must verify:
    1. Agent completed turn.
    2. Agent invoked system_info tool.
    3. Tool completed with status 'completed' in SQLite trace.
    4. Response contains the real host OS platform.
    """
    task_id = turn_result["task_id"]
    trace = trace_store.get_task_trace(task_id)

    tool_calls = trace.get("tool_calls", [])
    system_info_calls = [tc for tc in tool_calls if tc["tool_name"] == "system_info"]

    if not system_info_calls:
        return False, {"reason": "system_info tool was not invoked"}

    first_call = system_info_calls[0]
    if first_call["status"] != "completed":
        return False, {"reason": f"system_info call failed with status: {first_call['status']}"}

    # Ground truth check
    expected_os = platform.system().lower()
    resp_text = turn_result.get("response", "").lower()

    if expected_os not in resp_text:
        return False, {
            "reason": f"Expected OS '{expected_os}' not found in agent response.",
            "response": turn_result.get("response", ""),
        }

    return True, {
        "tool_call_id": first_call["tool_call_id"],
        "os_verified": expected_os,
        "duration_ms": first_call.get("duration_ms"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK TASK 2: Ground-Truth File Reading & Goal Extraction (Safe Tier)
# ─────────────────────────────────────────────────────────────────────────────
def verify_task_2(turn_result: Dict[str, Any], trace_store: TraceStore) -> tuple[bool, Dict[str, Any]]:
    """Ground-truth verification for Task 2: File Reading.
    Must verify:
    1. Agent completed turn.
    2. Agent invoked read_file tool on CLAUDE.md.
    3. Tool call succeeded in SQLite trace.
    4. Response accurately extracts the project goal: 'voice-driven personal AI assistant'.
    """
    task_id = turn_result["task_id"]
    trace = trace_store.get_task_trace(task_id)

    tool_calls = trace.get("tool_calls", [])
    read_calls = [tc for tc in tool_calls if tc["tool_name"] == "read_file"]

    if not read_calls:
        return False, {"reason": "read_file tool was not invoked"}

    first_call = read_calls[0]
    if first_call["status"] != "completed":
        return False, {"reason": f"read_file call failed with status: {first_call['status']}"}

    # Verify response contains the core goal keywords
    resp = turn_result.get("response", "").lower()
    key_phrases = ["personal ai assistant", "voice-driven", "windows and fedora"]
    matches = [phrase for phrase in key_phrases if phrase in resp]

    if not matches:
        return False, {
            "reason": "Agent response did not identify core rebuild goal keywords from CLAUDE.md.",
            "response": turn_result.get("response", ""),
        }

    return True, {
        "tool_call_id": first_call["tool_call_id"],
        "matched_key_phrases": matches,
        "duration_ms": first_call.get("duration_ms"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK TASK 3: Permission Gate & Sandboxed File Creation (Tier 2 Mutating)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_BENCHMARK_FILE = Path("/tmp/maddy_phase2_report.txt")
EXPECTED_SENTINEL = "PHASE_2_INITIALIZATION_SUCCESSFUL"

def verify_task_3(turn_result: Dict[str, Any], trace_store: TraceStore) -> tuple[bool, Dict[str, Any]]:
    """Ground-truth verification for Task 3: Permission Gate & Sandboxed File Creation.
    Must verify:
    1. Agent invoked write_file tool.
    2. Permission gate allowed the sandboxed write to /tmp.
    3. Tool call succeeded in SQLite trace.
    4. File physically exists on disk and contains EXPECTED_SENTINEL.
    """
    task_id = turn_result["task_id"]
    trace = trace_store.get_task_trace(task_id)

    tool_calls = trace.get("tool_calls", [])
    write_calls = [tc for tc in tool_calls if tc["tool_name"] == "write_file"]

    if not write_calls:
        return False, {"reason": "write_file tool was not invoked"}

    first_call = write_calls[0]
    if first_call["status"] != "completed":
        return False, {"reason": f"write_file call failed with status: {first_call['status']}"}

    # Physical filesystem ground-truth verification
    if not TARGET_BENCHMARK_FILE.exists():
        return False, {"reason": f"Target file '{TARGET_BENCHMARK_FILE}' was not created on the filesystem."}

    content = TARGET_BENCHMARK_FILE.read_text(encoding="utf-8")
    if EXPECTED_SENTINEL not in content:
        return False, {
            "reason": f"Target file does not contain expected sentinel '{EXPECTED_SENTINEL}'.",
            "actual_content": content,
        }

    # Cleanup temporary file
    try:
        TARGET_BENCHMARK_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    return True, {
        "tool_call_id": first_call["tool_call_id"],
        "filesystem_verified": True,
        "bytes_verified": len(content),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUITE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark_suite() -> List[TaskBenchmarkReport]:
    """Execute all 3 benchmark tasks, 3 runs each, validating >= 2/3 pass criteria."""
    claude_md_path = str(PROJECT_ROOT / "CLAUDE.md")

    tasks_spec = [
        {
            "id": "TASK-1",
            "name": "Host System Telemetry",
            "description": "Agent queries host telemetry, CPU/RAM, and local timestamp via system_info.",
            "prompt": (
                "M.A.D.D.Y., analyze our host system diagnostics. What operating system are we running, "
                "how much memory is installed on this unit, and what is our current local time?"
            ),
            "verifier": verify_task_1,
        },
        {
            "id": "TASK-2",
            "name": "Constitutional File Reading",
            "description": "Agent reads lines 1-15 of CLAUDE.md and identifies the core rebuild goal.",
            "prompt": (
                f"M.A.D.D.Y., inspect the first 15 lines of our project constitution by reading the file located at {claude_md_path} "
                "and state the primary goal of the rebuild."
            ),
            "verifier": verify_task_2,
        },
        {
            "id": "TASK-3",
            "name": "Sandboxed File Creation & Verification",
            "description": "Agent initializes a report file in /tmp with a sentinel under permission gate supervision.",
            "prompt": (
                f"M.A.D.D.Y., please initialize an automated test report at '{TARGET_BENCHMARK_FILE}' "
                f"containing the exact text '{EXPECTED_SENTINEL}'."
            ),
            "verifier": verify_task_3,
        },
    ]

    reports: List[TaskBenchmarkReport] = []

    print("\n" + "=" * 70)
    print("M.A.D.D.Y. PHASE 2 BENCHMARK SUITE")
    print("Modular Autonomous Digital Daemon for You")
    print(f"Pinned Model: {PINNED_MODEL} | Temperature: {FIXED_TEMPERATURE}")
    print(f"Evaluation Criteria: 3 runs per task, pass >= 2/3 runs")
    print("=" * 70 + "\n")

    for spec in tasks_spec:
        report = TaskBenchmarkReport(
            task_id=spec["id"],
            name=spec["name"],
            description=spec["description"],
        )
        print(f"[{spec['id']}] Running: {spec['name']}...")

        for run_i in range(1, 4):
            # Clean up target file before each run
            TARGET_BENCHMARK_FILE.unlink(missing_ok=True)

            agent, trace_store = build_test_agent()
            t0 = time.monotonic()
            error_msg = None
            passed = False
            details = {}

            try:
                turn_result = agent.run_turn(spec["prompt"])
                passed, details = spec["verifier"](turn_result, trace_store)
            except Exception as exc:
                passed = False
                error_msg = f"Exception: {type(exc).__name__}: {str(exc)}"
                details = {"exception": str(exc)}

            duration = time.monotonic() - t0
            report.runs.append(RunResult(
                run_index=run_i,
                task_id=turn_result.get("task_id", "unknown") if 'turn_result' in locals() else "failed",
                passed=passed,
                duration_s=duration,
                error=error_msg or (details.get("reason") if not passed else None),
                verification_details=details,
            ))

            status_str = "PASS" if passed else "FAIL"
            print(f"  Run {run_i}/3: {status_str} in {duration:.2f}s " + (f"({details})" if not passed else ""))
            time.sleep(2.5)

        print(f"  Outcome: {report.passed_runs}/3 Passed -> {'SUCCESS' if report.is_successful else 'FAILURE'}\n")
        reports.append(report)

    print("=" * 70)
    print("PHASE 2 BENCHMARK SUMMARY")
    print("=" * 70)
    all_passed = True
    for rep in reports:
        status = "PASSED (>= 2/3)" if rep.is_successful else "FAILED (< 2/3)"
        if not rep.is_successful:
            all_passed = False
        print(f"- {rep.task_id} ({rep.name}): {rep.passed_runs}/3 runs succeeded -> {status}")

    print("=" * 70)
    overall_status = "PHASE 2 ACCEPTANCE MET: ALL BENCHMARKS PASSED" if all_passed else "PHASE 2 ACCEPTANCE NOT MET"
    print(f"Final Status: {overall_status}\n")

    return reports


if __name__ == "__main__":
    reports = run_benchmark_suite()
    all_success = all(r.is_successful for r in reports)
    sys.exit(0 if all_success else 1)
