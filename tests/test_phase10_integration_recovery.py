"""
Unit and integration tests for Phase 10: Full Autonomy, Recovery & System Integration.
Validates RecoveryEngine error classification, AutonomyEngine progression policies, and MaddyAssistant integration.
"""
import os
import pytest
from unittest.mock import MagicMock

from recovery.engine import RecoveryEngine, ErrorType, RecoveryStrategy
from security.autonomy import AutonomyEngine, AutonomyLevel
from tools.base import Tool, RiskLevel
from tools.call import ToolCall
from tools.result import ToolResult

def test_recovery_engine_error_classification():
    re = RecoveryEngine(max_retries=2)

    assert re.classify_error("Application 'spotify' was not found.") == ErrorType.APPLICATION_NOT_FOUND
    assert re.classify_error("File '/tmp/missing.txt' does not exist.") == ErrorType.FILE_NOT_FOUND
    assert re.classify_error("Failed to navigate to 'http://badurl': timeout") == ErrorType.NAVIGATION_ERROR
    assert re.classify_error("Action blocked by security policy") == ErrorType.PERMISSION_DENIED

def test_recovery_engine_strategy_evaluation():
    re = RecoveryEngine(max_retries=2)

    call = ToolCall(name="application.open", arguments={"app_name": "unknown_app"})
    res = ToolResult.fail(error_type="APPLICATION_NOT_FOUND", message="Application 'unknown_app' was not found.", tool_name="application.open")
    
    strat = re.evaluate_recovery(call, res, current_retry_count=0)
    assert strat.error_type == ErrorType.APPLICATION_NOT_FOUND
    assert strat.can_retry is True
    assert strat.alternative_tool == "browser.navigate"
    assert "couldn't open application" in strat.user_explanation

    # Exceeded max retries
    strat_expired = re.evaluate_recovery(call, res, current_retry_count=2)
    assert strat_expired.can_retry is False

def test_autonomy_engine_policy_levels():
    ae_trusted = AutonomyEngine(current_level=AutonomyLevel.EXECUTE_TRUSTED)
    
    low_tool = Tool(
        name="test.low",
        description="Low risk",
        parameter_schema={},
        handler=lambda x: {},
        risk_level=RiskLevel.LOW,
    )
    high_tool = Tool(
        name="test.high",
        description="High risk",
        parameter_schema={},
        handler=lambda x: {},
        risk_level=RiskLevel.HIGH,
    )

    assert ae_trusted.should_auto_execute(low_tool) is True
    assert ae_trusted.should_auto_execute(high_tool) is False

    ae_full = AutonomyEngine(current_level=AutonomyLevel.FULL_AUTONOMY)
    med_tool = Tool(
        name="test.med",
        description="Medium risk",
        parameter_schema={},
        handler=lambda x: {},
        risk_level=RiskLevel.MEDIUM,
    )
    assert ae_full.should_auto_execute(med_tool) is True
    assert ae_full.should_auto_execute(high_tool) is False

def test_maddy_assistant_initialization():
    from app.maddy_assistant import MaddyAssistant

    # Mock Ollama generation
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"thought": "done", "final_answer": "Execution Successful"}'
    mock_model.generate.return_value = mock_response

    assistant = MaddyAssistant()
    assert assistant.settings is not None
    assert assistant.memory_dir is not None
    assert len(assistant.registry.list_tools()) >= 25
    assert assistant.recovery_engine is not None
    assert assistant.autonomy_engine is not None
