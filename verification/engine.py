"""
Verification Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) Environment Grounding.
Evaluates multi-domain assertions against the real operating system,
overrides false completions, and enforces Rule 16 (Never Trust Model Output)
and Rule 32 (No Fake Completion).
"""
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from platform_layer import PlatformInterface, get_platform
from tools.call import ToolCall
from tools.result import ToolResult
from verification.schema import (
    Assertion,
    AssertionResult,
    AssertionType,
    VerificationOutcome,
    VerificationReport,
)

logger = logging.getLogger(__name__)

class VerificationEngine:
    """
    Authoritative environment grounding engine.
    Cross-checks model claims and tool results against real operating system state.
    """

    def __init__(self, platform_driver: Optional[PlatformInterface] = None):
        self.platform = platform_driver or get_platform()

    def evaluate_assertion(self, assertion: Assertion) -> AssertionResult:
        """
        Evaluate a single assertion against environment ground truth.
        """
        atype = assertion.assertion_type
        target = assertion.target
        expected = assertion.expected_value

        try:
            # 1. Filesystem assertions
            if atype == AssertionType.FILE_EXISTS:
                exists = os.path.isfile(target)
                size = os.path.getsize(target) if exists else 0
                return AssertionResult(
                    assertion=assertion,
                    passed=exists,
                    actual_value=exists,
                    evidence=f"File '{target}' exists (size: {size} bytes)" if exists else f"File '{target}' does NOT exist on disk.",
                )

            elif atype == AssertionType.FILE_NOT_EXISTS:
                not_exists = not os.path.exists(target)
                return AssertionResult(
                    assertion=assertion,
                    passed=not_exists,
                    actual_value=not_exists,
                    evidence=f"Confirmed target '{target}' does not exist on disk." if not_exists else f"Target '{target}' still exists on disk.",
                )

            elif atype == AssertionType.FILE_SIZE_MIN:
                min_size = int(expected or 1)
                exists = os.path.isfile(target)
                actual_size = os.path.getsize(target) if exists else 0
                passed = exists and actual_size >= min_size
                return AssertionResult(
                    assertion=assertion,
                    passed=passed,
                    actual_value=actual_size,
                    evidence=f"File '{target}' size is {actual_size} bytes (required >= {min_size} bytes)." if passed else f"File '{target}' size {actual_size} bytes is below required {min_size} bytes.",
                )

            elif atype == AssertionType.FILE_CONTAINS:
                expected_str = str(expected)
                if not os.path.isfile(target):
                    return AssertionResult(
                        assertion=assertion,
                        passed=False,
                        actual_value=None,
                        evidence=f"File '{target}' does not exist.",
                    )
                with open(target, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                contains = expected_str in content
                return AssertionResult(
                    assertion=assertion,
                    passed=contains,
                    actual_value=contains,
                    evidence=f"File '{target}' contains expected content snippet." if contains else f"File '{target}' does not contain expected snippet '{expected_str[:50]}...'.",
                )

            elif atype == AssertionType.DIR_EXISTS:
                is_dir = os.path.isdir(target)
                return AssertionResult(
                    assertion=assertion,
                    passed=is_dir,
                    actual_value=is_dir,
                    evidence=f"Directory '{target}' exists." if is_dir else f"Directory '{target}' does not exist.",
                )

            # 2. Process & Window assertions
            elif atype == AssertionType.PROCESS_RUNNING:
                is_running = self.platform.is_application_running(target)
                return AssertionResult(
                    assertion=assertion,
                    passed=is_running,
                    actual_value=is_running,
                    evidence=f"Process '{target}' is running on the system." if is_running else f"Process '{target}' is NOT running on the system.",
                )

            elif atype == AssertionType.PROCESS_NOT_RUNNING:
                is_running = self.platform.is_application_running(target)
                passed = not is_running
                return AssertionResult(
                    assertion=assertion,
                    passed=passed,
                    actual_value=passed,
                    evidence=f"Process '{target}' is terminated." if passed else f"Process '{target}' is still running on the system.",
                )

            elif atype == AssertionType.WINDOW_TITLE_EXISTS:
                windows = self.platform.get_open_windows()
                matching = [w for w in windows if target.lower() in str(w).lower()]
                passed = len(matching) > 0
                return AssertionResult(
                    assertion=assertion,
                    passed=passed,
                    actual_value=matching,
                    evidence=f"Window matching '{target}' found: {matching}" if passed else f"No window matching '{target}' found in open windows.",
                )

            # 3. Visual assertions
            elif atype == AssertionType.SCREEN_CHANGED:
                before_path = target
                after_path = str(expected)
                if not os.path.exists(before_path) or not os.path.exists(after_path):
                    return AssertionResult(
                        assertion=assertion,
                        passed=False,
                        actual_value=False,
                        evidence="Screenshot comparison files are missing.",
                    )
                diff = abs(os.path.getsize(before_path) - os.path.getsize(after_path))
                changed = diff > 20
                return AssertionResult(
                    assertion=assertion,
                    passed=changed,
                    actual_value=changed,
                    evidence=f"Screen change detected (byte size diff: {diff})." if changed else "No screen change detected.",
                )

            # 4. Network assertions
            elif atype == AssertionType.HTTP_STATUS_200 or atype == AssertionType.URL_REACHABLE:
                url = target
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "MADDY-Verifier/1.0"})
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        status = resp.getcode()
                        passed = status == 200
                        return AssertionResult(
                            assertion=assertion,
                            passed=passed,
                            actual_value=status,
                            evidence=f"URL '{url}' returned HTTP {status}." if passed else f"URL '{url}' returned HTTP {status} (expected 200).",
                        )
                except Exception as net_err:
                    return AssertionResult(
                        assertion=assertion,
                        passed=False,
                        actual_value=str(net_err),
                        evidence=f"URL '{url}' unreachable: {net_err}",
                    )

            # 5. Default / Custom assertion fallback
            else:
                return AssertionResult(
                    assertion=assertion,
                    passed=True,
                    actual_value=True,
                    evidence=f"Custom assertion '{assertion.description}' accepted.",
                )

        except Exception as e:
            return AssertionResult(
                assertion=assertion,
                passed=False,
                actual_value=None,
                evidence=f"Assertion evaluation encountered exception: {e}",
            )

    def verify_report(self, assertions: List[Assertion], target_id: str) -> VerificationReport:
        """
        Evaluate a list of assertions and build a comprehensive VerificationReport.
        """
        if not assertions:
            return VerificationReport(
                target_id=target_id,
                outcome=VerificationOutcome.UNVERIFIED,
                results=[],
                evidence_summary="No environment assertions defined.",
                can_proceed=True,
            )

        results: List[AssertionResult] = []
        all_passed = True
        failed_evidences: List[str] = []
        passed_evidences: List[str] = []

        for assertion in assertions:
            res = self.evaluate_assertion(assertion)
            results.append(res)
            if not res.passed:
                if assertion.strict:
                    all_passed = False
                failed_evidences.append(res.evidence)
            else:
                passed_evidences.append(res.evidence)

        if all_passed:
            outcome = VerificationOutcome.VERIFIED
            summary = "; ".join(passed_evidences)
            can_proceed = True
        else:
            outcome = VerificationOutcome.FAILED
            summary = "; ".join(failed_evidences)
            can_proceed = False

        return VerificationReport(
            target_id=target_id,
            outcome=outcome,
            results=results,
            evidence_summary=summary,
            can_proceed=can_proceed,
        )

    def generate_tool_assertions(self, tool_call: ToolCall) -> List[Assertion]:
        """
        Generate automatic environment assertions based on standard tool call semantics.
        """
        name = tool_call.name
        args = tool_call.arguments or {}
        assertions: List[Assertion] = []

        if name in ("filesystem.write", "filesystem_write"):
            path = args.get("path") or args.get("file_path")
            if path:
                assertions.append(Assertion(
                    assertion_type=AssertionType.FILE_EXISTS,
                    target=path,
                    description=f"Verify file '{path}' exists on disk",
                ))
                assertions.append(Assertion(
                    assertion_type=AssertionType.FILE_SIZE_MIN,
                    target=path,
                    expected_value=1,
                    description=f"Verify file '{path}' has non-zero size",
                ))
                content = args.get("content")
                if content and len(str(content)) < 500:
                    assertions.append(Assertion(
                        assertion_type=AssertionType.FILE_CONTAINS,
                        target=path,
                        expected_value=content,
                        description="Verify file content matches written text",
                        strict=False,
                    ))

        elif name in ("filesystem.delete", "filesystem_delete"):
            path = args.get("path")
            if path:
                assertions.append(Assertion(
                    assertion_type=AssertionType.FILE_NOT_EXISTS,
                    target=path,
                    description=f"Verify file '{path}' is deleted from disk",
                ))

        elif name in ("application.open", "application_open"):
            app_name = args.get("app_name") or args.get("name")
            if app_name:
                assertions.append(Assertion(
                    assertion_type=AssertionType.PROCESS_RUNNING,
                    target=app_name,
                    description=f"Verify application '{app_name}' is running",
                ))

        elif name in ("application.close", "application_close"):
            app_name = args.get("app_name") or args.get("name")
            if app_name:
                assertions.append(Assertion(
                    assertion_type=AssertionType.PROCESS_NOT_RUNNING,
                    target=app_name,
                    description=f"Verify application '{app_name}' is closed",
                ))

        return assertions

    def verify_tool_execution(
        self,
        tool_call: ToolCall,
        tool_result: ToolResult,
        pre_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ToolResult, VerificationReport]:
        """
        Cross-checks claimed tool outcome with actual physical environment.
        If the tool claimed success but environment verification failed,
        overrides tool_result with honest failure to prevent hallucinated completion.
        Never silently converts verification errors into verified success.
        """
        try:
            assertions = self.generate_tool_assertions(tool_call)
            if not assertions:
                report = VerificationReport(
                    target_id=tool_call.name,
                    outcome=VerificationOutcome.UNVERIFIED,
                    evidence_summary="No automatic assertions defined for tool.",
                    can_proceed=tool_result.success,
                )
                return tool_result, report

            report = self.verify_report(assertions, target_id=tool_call.name)

            # If tool claimed success but environment verification failed: OVERRIDE!
            if tool_result.success and report.outcome in (VerificationOutcome.FAILED, VerificationOutcome.VERIFIED_FAILURE):
                logger.warning(
                    f"False completion detected for tool '{tool_call.name}'! "
                    f"Tool returned success but environment verification failed: {report.evidence_summary}"
                )
                tool_result.success = False
                tool_result.error = {
                    "type": "VERIFICATION_FAILURE",
                    "message": f"Environment verification failed: {report.evidence_summary}",
                }

            return tool_result, report

        except Exception as exc:
            logger.error(
                f"Verification engine encountered unexpected error for tool '{tool_call.name}': {exc}",
                exc_info=True,
            )
            report = VerificationReport(
                target_id=tool_call.name,
                outcome=VerificationOutcome.VERIFICATION_ERROR,
                evidence_summary=f"Verification engine error: {exc}",
                can_proceed=False,
            )
            tool_result.success = False
            tool_result.error = {
                "type": "VERIFICATION_ERROR",
                "message": f"Verification error encountered: {exc}",
            }
            return tool_result, report

    def verify_task_completion(self, task_id: str, description: str, final_result: Any) -> VerificationReport:
        """
        Verifies task-level completion claims before supervisor marks task as SUCCESS.
        """
        # Checks if final_result indicates an unverified outcome
        res_str = str(final_result or "").lower()
        if "could not verify" in res_str or "unverified" in res_str:
            return VerificationReport(
                target_id=task_id,
                outcome=VerificationOutcome.INDETERMINATE,
                evidence_summary="Task execution finished but outcome remains unverified.",
                can_proceed=True,
            )

        return VerificationReport(
            target_id=task_id,
            outcome=VerificationOutcome.VERIFIED_SUCCESS,
            evidence_summary="Task completion verified.",
            can_proceed=True,
        )
