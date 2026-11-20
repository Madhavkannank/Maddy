"""
Comprehensive unit and integration test suite for M.A.D.D.Y. Session Capture subsystem.
Verifies lifecycle, intervals, cryptographic integrity, storage isolation, sandboxing,
quotas, bounded delivery queue, Telegram resilience, audit ledger, and MaddyAssistant routing.
"""
import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from app.maddy_assistant import MaddyAssistant
from integrations.messaging import TelegramClient
from security.audit_ledger import AuditLedger
from security.permission_engine import PermissionEngine
from security.policy import PermissionDecision, PolicyMode
from security.quotas import ResourceQuotaConfig, ResourceQuotaManager
from security.sandbox import PathSandbox
from session_capture.artifacts import ArtifactType, CaptureArtifact, DeliveryState
from session_capture.config import (
    DEFAULT_CAPTURE_INTERVAL,
    MAX_CAPTURE_INTERVAL,
    MIN_CAPTURE_INTERVAL,
    SessionCaptureConfig,
)
from session_capture.delivery import TelegramDeliveryAdapter
from session_capture.manager import SessionCaptureManager, get_session_capture_manager
from session_capture.queue import BoundedDeliveryQueue
from session_capture.session import CaptureSession, SessionState
from session_capture.workers import CameraCaptureWorker, DisplayCaptureWorker
from tools.base import RiskLevel
from tools.call import ToolCall
from tools.camera import CameraEngine
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.session_capture_tool import (
    session_capture_start_tool,
    session_capture_status_tool,
    session_capture_stop_tool,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for test artifacts."""
    d = tempfile.mkdtemp(prefix="maddy_test_capture_")
    yield d
    try:
        shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def mock_security(temp_workspace):
    """Provide isolated sandbox, quotas, and ledger for testing."""
    sandbox = PathSandbox(allowed_workspace_roots=[temp_workspace, os.getcwd()])
    ledger_file = os.path.join(temp_workspace, "test_ledger.json")
    ledger = AuditLedger(ledger_file=ledger_file)
    quota = ResourceQuotaManager(
        config=ResourceQuotaConfig(
            max_file_write_bytes=5 * 1024 * 1024,
            max_task_write_bytes=20 * 1024 * 1024,
        )
    )
    perm_engine = PermissionEngine(
        policy_mode=PolicyMode.BALANCED,
        path_sandbox=sandbox,
        quota_manager=quota,
        audit_logger=ledger,
    )
    return {
        "sandbox": sandbox,
        "ledger": ledger,
        "quota": quota,
        "perm_engine": perm_engine,
        "workspace": temp_workspace,
    }


# =====================================================================
# 1. Configuration Validation Tests
# =====================================================================

def test_config_default_values():
    cfg = SessionCaptureConfig()
    assert cfg.display_interval == DEFAULT_CAPTURE_INTERVAL
    assert cfg.camera_interval == DEFAULT_CAPTURE_INTERVAL
    assert cfg.enable_display is True
    assert cfg.enable_camera is True
    assert cfg.enable_telegram is False
    cfg.validate()


def test_config_validation_intervals():
    # Negative interval
    with pytest.raises(ValueError, match="display_interval must be between"):
        SessionCaptureConfig(display_interval=-5.0).validate()

    # Zero interval
    with pytest.raises(ValueError, match="display_interval must be between"):
        SessionCaptureConfig(display_interval=0.0).validate()

    # Excessive interval (> 1 hour)
    with pytest.raises(ValueError, match="camera_interval must be between"):
        SessionCaptureConfig(camera_interval=5000.0).validate()


def test_config_validation_storage_and_streams():
    # Below minimum storage
    with pytest.raises(ValueError, match="max_storage_bytes must be between"):
        SessionCaptureConfig(max_storage_bytes=100).validate()

    # Both streams disabled
    with pytest.raises(ValueError, match="At least one capture stream"):
        SessionCaptureConfig(enable_display=False, enable_camera=False).validate()

    # Invalid telegram chat id
    with pytest.raises(ValueError, match="Invalid telegram_chat_id format"):
        SessionCaptureConfig(telegram_chat_id="bad id with spaces!").validate()


def test_config_hash_determinism():
    cfg1 = SessionCaptureConfig(display_interval=15.0, enable_telegram=True)
    cfg2 = SessionCaptureConfig(display_interval=15.0, enable_telegram=True)
    cfg3 = SessionCaptureConfig(display_interval=20.0, enable_telegram=True)
    assert cfg1.compute_hash() == cfg2.compute_hash()
    assert cfg1.compute_hash() != cfg3.compute_hash()


# =====================================================================
# 2. Cryptographic Integrity & Artifact Tests
# =====================================================================

def test_artifact_sha256_computation(temp_workspace):
    test_file = os.path.join(temp_workspace, "sample_artifact.bin")
    with open(test_file, "wb") as f:
        f.write(b"MADDY_SECURITY_ARTIFACT_INTEGRITY_CHECK_12345")

    computed_hash = CaptureArtifact.compute_file_sha256(test_file)
    assert len(computed_hash) == 64  # SHA-256 hex string

    artifact = CaptureArtifact(
        artifact_id="art_disp_0001",
        session_id="sess_test",
        artifact_type=ArtifactType.DISPLAY,
        file_path=test_file,
        timestamp=time.time(),
        sequence_number=1,
        file_size_bytes=len(b"MADDY_SECURITY_ARTIFACT_INTEGRITY_CHECK_12345"),
        sha256=computed_hash,
    )
    d = artifact.to_dict()
    assert d["artifact_id"] == "art_disp_0001"
    assert d["sha256"] == computed_hash
    assert d["artifact_type"] == "display"


# =====================================================================
# 3. Bounded Delivery Queue Tests
# =====================================================================

def test_bounded_delivery_queue_capacity():
    queue = BoundedDeliveryQueue(max_pending_artifacts=2, max_pending_bytes=1000)

    art1 = CaptureArtifact(
        artifact_id="a1",
        session_id="s",
        artifact_type=ArtifactType.DISPLAY,
        file_path="f1",
        timestamp=time.time(),
        sequence_number=1,
        file_size_bytes=400,
        sha256="hash1",
    )
    art2 = CaptureArtifact(
        artifact_id="a2",
        session_id="s",
        artifact_type=ArtifactType.DISPLAY,
        file_path="f2",
        timestamp=time.time(),
        sequence_number=2,
        file_size_bytes=400,
        sha256="hash2",
    )
    art3 = CaptureArtifact(
        artifact_id="a3",
        session_id="s",
        artifact_type=ArtifactType.DISPLAY,
        file_path="f3",
        timestamp=time.time(),
        sequence_number=3,
        file_size_bytes=400,
        sha256="hash3",
    )

    assert queue.enqueue(art1) is True
    assert queue.enqueue(art2) is True
    # Capacity limit reached (2 items max)
    assert queue.enqueue(art3) is False
    assert art3.delivery_state == DeliveryState.SKIPPED
    assert "limit exceeded" in art3.delivery_error

    status = queue.get_status()
    assert status["pending_count"] == 2
    assert status["total_dropped"] == 1


def test_bounded_delivery_queue_retry_and_drain():
    queue = BoundedDeliveryQueue(max_pending_artifacts=10, max_pending_bytes=10000)
    art = CaptureArtifact(
        artifact_id="a1",
        session_id="s",
        artifact_type=ArtifactType.DISPLAY,
        file_path="f1",
        timestamp=time.time(),
        sequence_number=1,
        file_size_bytes=100,
        sha256="hash1",
    )
    queue.enqueue(art)
    popped = queue.dequeue()
    assert popped.artifact_id == "a1"

    # Retry 1
    assert queue.requeue(popped, max_retries=2) is True
    assert popped.retry_count == 1

    # Retry 2
    popped2 = queue.dequeue()
    assert queue.requeue(popped2, max_retries=2) is True
    assert popped2.retry_count == 2

    # Retry 3 exceeds max_retries=2
    popped3 = queue.dequeue()
    assert queue.requeue(popped3, max_retries=2) is False
    assert popped3.delivery_state == DeliveryState.FAILED


# =====================================================================
# 4. Telegram Delivery Adapter Tests
# =====================================================================

def test_telegram_delivery_adapter_unconfigured(mock_security):
    queue = BoundedDeliveryQueue()
    mock_tg = MagicMock(spec=TelegramClient)
    mock_tg.is_configured = False

    adapter = TelegramDeliveryAdapter(
        queue=queue,
        telegram_client=mock_tg,
        audit_ledger=mock_security["ledger"],
    )

    dummy_path = os.path.join(mock_security["workspace"], "test.jpg")
    with open(dummy_path, "wb") as f:
        f.write(b"image")

    art = CaptureArtifact(
        artifact_id="a1",
        session_id="s",
        artifact_type=ArtifactType.DISPLAY,
        file_path=dummy_path,
        timestamp=time.time(),
        sequence_number=1,
        file_size_bytes=5,
        sha256="hash",
    )

    success, msg = adapter.send_artifact_sync(art)
    assert success is False
    assert "not configured" in msg


def test_telegram_delivery_adapter_success_and_audit(mock_security):
    queue = BoundedDeliveryQueue()
    mock_tg = MagicMock(spec=TelegramClient)
    mock_tg.is_configured = True
    mock_tg.send_photo.return_value = {"success": True, "message_id": 999}

    adapter = TelegramDeliveryAdapter(
        queue=queue,
        telegram_client=mock_tg,
        audit_ledger=mock_security["ledger"],
        default_chat_id="12345",
    )

    dummy_path = os.path.join(mock_security["workspace"], "test.jpg")
    with open(dummy_path, "wb") as f:
        f.write(b"dummy image data")

    art = CaptureArtifact(
        artifact_id="art_1",
        session_id="sess_123",
        artifact_type=ArtifactType.CAMERA,
        file_path=dummy_path,
        timestamp=time.time(),
        sequence_number=1,
        file_size_bytes=len(b"dummy image data"),
        sha256="abc12345",
    )

    success, msg = adapter.send_artifact_sync(art)
    assert success is True
    assert msg == "999"

    # Verify audit record exists
    records = mock_security["ledger"].get_records()
    delivery_records = [r for r in records if r.tool_name == "session.capture.delivery"]
    assert len(delivery_records) == 1
    assert delivery_records[0].allowed is True
    assert delivery_records[0].arguments["artifact_id"] == "art_1"


# =====================================================================
# 5. Session Capture Manager Lifecycle Tests
# =====================================================================

def test_manager_start_and_stop_lifecycle(mock_security):
    storage_dir = os.path.join(mock_security["workspace"], "capture_root")
    manager = SessionCaptureManager(
        permission_engine=mock_security["perm_engine"],
        path_sandbox=mock_security["sandbox"],
        quota_manager=mock_security["quota"],
        audit_ledger=mock_security["ledger"],
        base_storage_dir=storage_dir,
    )

    # Fast intervals for testing
    cfg = SessionCaptureConfig(
        display_interval=1.0,
        camera_interval=1.0,
        enable_display=True,
        enable_camera=False,
    )

    success, msg, data = manager.start_session(config=cfg, task_id="test_task_01", caller_id="owner")
    assert success is True
    assert "active" in msg.lower()
    sess_id = data["session_id"]

    status = manager.get_status()
    assert status["active"] is True
    assert status["session_id"] == sess_id
    assert status["created_by"] == "owner"
    assert status["authorized_task_id"] == "test_task_01"

    # Stop session
    stop_success, stop_msg, stop_data = manager.stop_session(task_id="test_task_01", caller_id="owner")
    assert stop_success is True
    assert "stopped" in stop_msg.lower()
    assert manager.get_status()["active"] is False

    # Check manifest.json and metadata.json were created atomically
    session_dir = stop_data["session_dir"]
    assert os.path.exists(os.path.join(session_dir, "metadata.json"))
    assert os.path.exists(os.path.join(session_dir, "manifest.json"))


def test_concurrent_session_rejection(mock_security):
    manager = SessionCaptureManager(
        permission_engine=mock_security["perm_engine"],
        path_sandbox=mock_security["sandbox"],
        quota_manager=mock_security["quota"],
        audit_ledger=mock_security["ledger"],
        base_storage_dir=os.path.join(mock_security["workspace"], "sessions"),
    )

    cfg = SessionCaptureConfig(display_interval=2.0, enable_camera=False)
    succ1, _, _ = manager.start_session(config=cfg)
    assert succ1 is True

    # Attempting to start a second session concurrently must fail
    succ2, msg2, _ = manager.start_session(config=cfg)
    assert succ2 is False
    assert "already active" in msg2

    manager.stop_session()


def test_unauthorized_session_termination_guard(mock_security):
    manager = SessionCaptureManager(
        permission_engine=mock_security["perm_engine"],
        path_sandbox=mock_security["sandbox"],
        quota_manager=mock_security["quota"],
        audit_ledger=mock_security["ledger"],
        base_storage_dir=os.path.join(mock_security["workspace"], "sessions"),
    )

    cfg = SessionCaptureConfig(display_interval=2.0, enable_camera=False)
    manager.start_session(config=cfg, task_id="authorized_task_A", caller_id="user_A")

    # Mismatched caller trying to stop should be rejected
    succ, msg, _ = manager.stop_session(task_id="foreign_task_B", caller_id="user_B")
    assert succ is False
    assert "unauthorized" in msg.lower()

    # Authorized task can stop it
    succ_auth, _, _ = manager.stop_session(task_id="authorized_task_A", caller_id="user_A")
    assert succ_auth is True


# =====================================================================
# 6. Capture Workers & Resilience Tests
# =====================================================================

def test_display_capture_worker_execution(mock_security):
    storage_dir = os.path.join(mock_security["workspace"], "disp_session")
    cfg = SessionCaptureConfig(display_interval=1.0, enable_display=True, enable_camera=False)
    session = CaptureSession("sess_disp", cfg, storage_dir)
    session.initialize_directories(mock_security["sandbox"])

    mock_controller = MagicMock()
    # Mock take_screenshot writing dummy file
    def fake_screenshot(path):
        with open(path, "wb") as f:
            f.write(b"PNG_FAKE_SCREENSHOT_DATA")
        return {"path": path, "size_bytes": len(b"PNG_FAKE_SCREENSHOT_DATA")}

    mock_controller.take_screenshot.side_effect = fake_screenshot

    worker = DisplayCaptureWorker(
        session=session,
        computer_controller=mock_controller,
        path_sandbox=mock_security["sandbox"],
        audit_ledger=mock_security["ledger"],
    )

    worker._capture_once()
    assert len(session.artifacts) == 1
    art = session.artifacts[0]
    assert art.artifact_type == ArtifactType.DISPLAY
    assert art.sequence_number == 1
    assert os.path.exists(art.file_path)
    assert len(art.sha256) == 64


def test_camera_worker_unavailable_graceful_handling(mock_security):
    storage_dir = os.path.join(mock_security["workspace"], "cam_session")
    cfg = SessionCaptureConfig(camera_interval=1.0, enable_display=False, enable_camera=True)
    session = CaptureSession("sess_cam", cfg, storage_dir)
    session.initialize_directories(mock_security["sandbox"])

    mock_cam = MagicMock()
    mock_cam.capture_photo.return_value = (False, "", "Unable to access webcam.")

    worker = CameraCaptureWorker(
        session=session,
        camera_engine=mock_cam,
        path_sandbox=mock_security["sandbox"],
        audit_ledger=mock_security["ledger"],
    )

    # Capturing when camera is unavailable should not throw exception
    worker._capture_once()
    assert worker.camera_available is False
    assert len(session.artifacts) == 0

    # Verify audit ledger recorded CAMERA_UNAVAILABLE
    records = mock_security["ledger"].get_records()
    cam_rec = [r for r in records if r.tool_name == "session.capture.camera"]
    assert len(cam_rec) == 1
    assert cam_rec[0].allowed is False
    assert cam_rec[0].decision_code == "CAMERA_UNAVAILABLE"


def test_quota_exhaustion_stops_captures(mock_security):
    storage_dir = os.path.join(mock_security["workspace"], "quota_session")
    # Set max_storage_bytes small so quota is immediately exceeded
    cfg = SessionCaptureConfig(
        display_interval=1.0,
        max_storage_bytes=10 * 1024 * 1024,
    )
    session = CaptureSession("sess_quota", cfg, storage_dir)
    session.initialize_directories(mock_security["sandbox"])

    # Simulate already used up storage
    session.artifacts.append(
        CaptureArtifact(
            artifact_id="art_heavy",
            session_id="sess_quota",
            artifact_type=ArtifactType.DISPLAY,
            file_path="f",
            timestamp=time.time(),
            sequence_number=1,
            file_size_bytes=11 * 1024 * 1024,
            sha256="fakehash",
        )
    )

    mock_controller = MagicMock()
    worker = DisplayCaptureWorker(
        session=session,
        computer_controller=mock_controller,
        path_sandbox=mock_security["sandbox"],
        audit_ledger=mock_security["ledger"],
    )

    worker._capture_once()
    # take_screenshot must NOT be called when quota exceeded
    mock_controller.take_screenshot.assert_not_called()
    assert worker.quota_exceeded is True


# =====================================================================
# 7. ACI Tool & ToolRegistry Integration Tests
# =====================================================================

def test_session_capture_tools_registration(mock_security):
    registry = ToolRegistry()
    registry.register(session_capture_start_tool)
    registry.register(session_capture_stop_tool)
    registry.register(session_capture_status_tool)

    assert registry.get("session.capture.start") is not None
    assert registry.get("session.capture.stop") is not None
    assert registry.get("session.capture.status") is not None
    assert registry.get("session.capture.start").risk_level == RiskLevel.MEDIUM


def test_session_capture_tools_execution_via_executor(mock_security):
    get_session_capture_manager(permission_engine=mock_security["perm_engine"], reset=True)

    registry = ToolRegistry()
    registry.register(session_capture_start_tool)
    registry.register(session_capture_stop_tool)
    registry.register(session_capture_status_tool)

    executor = ToolExecutor(registry=registry, permission_engine=mock_security["perm_engine"])

    # 1. Query status before starting (Low risk -> allowed)
    res_stat = executor.execute(ToolCall(name="session.capture.status", arguments={}))
    assert res_stat.success is True
    assert res_stat.data["data"]["active"] is False

    # 2. Verify PermissionEngine denial under STRICT mode when confirmation handler rejects Medium-risk tool
    mock_security["perm_engine"].policy_mode = PolicyMode.STRICT
    res_denied = executor.execute(
        ToolCall(
            name="session.capture.start",
            arguments={"display_interval": 2.0, "enable_camera": False},
        )
    )
    assert res_denied.success is False
    assert res_denied.error["type"] == "PERMISSION_DENIED"

    # 3. Authorize via confirmation handler and verify successful execution
    mock_confirm = MagicMock()
    mock_confirm.confirm.return_value = True
    mock_security["perm_engine"].confirmation_handler = mock_confirm

    res_start = executor.execute(
        ToolCall(
            name="session.capture.start",
            arguments={"display_interval": 2.0, "enable_camera": False},
        )
    )
    assert res_start.success is True
    assert "activated" in res_start.data["message"].lower() or "active" in res_start.data["message"].lower()

    # 4. Stop session via tool
    res_stop = executor.execute(ToolCall(name="session.capture.stop", arguments={}))
    assert res_stop.success is True
    assert "stopped" in res_stop.data["message"].lower()


# =====================================================================
# 8. End-to-End MaddyAssistant Command Integration Tests
# =====================================================================

def test_maddy_assistant_session_capture_commands(mock_security):
    get_session_capture_manager(reset=True)
    assistant = MaddyAssistant()

    # Fast intervals for testing
    with patch.object(
        assistant.session_capture_manager.computer_controller,
        "take_screenshot",
        return_value={"path": "C:\\mock\\test.png", "size_bytes": 100},
    ):
        # 1. Start command
        res_start = assistant.execute("Start session capture", speak_response=False)
        assert res_start["success"] is True
        assert "session capture" in res_start["response"].lower() or "active" in res_start["response"].lower()

        # 2. Status command
        res_status = assistant.execute("Session capture status", speak_response=False)
        assert res_status["success"] is True
        assert "active" in res_status["response"].lower()

        # 3. Stop command
        res_stop = assistant.execute("Stop session capture", speak_response=False)
        assert res_stop["success"] is True
        assert "stopped" in res_stop["response"].lower()

        # 4. Guest audit synonym
        res_guest_start = assistant.execute("Start guest audit", speak_response=False)
        assert res_guest_start["success"] is True
        assert "activated" in res_guest_start["response"].lower() or "active" in res_guest_start["response"].lower()

        res_guest_stop = assistant.execute("Stop guest audit", speak_response=False)
        assert res_guest_stop["success"] is True
        assert "stopped" in res_guest_stop["response"].lower()

        # 5. Dismiss command ("dismiss session")
        res_start2 = assistant.execute("Start session capture", speak_response=False)
        assert res_start2["success"] is True
        assert assistant.session_capture_manager.has_active_session() is True

        res_dismiss = assistant.execute("dismiss session", speak_response=False)
        assert res_dismiss["success"] is True
        assert "stopped" in res_dismiss["response"].lower()
        assert assistant.session_capture_manager.has_active_session() is False

        # 6. Generic "dismiss" while session is active
        res_start3 = assistant.execute("Start session capture", speak_response=False)
        assert res_start3["success"] is True
        assert assistant.session_capture_manager.has_active_session() is True

        res_generic_dismiss = assistant.execute("dismiss", speak_response=False)
        assert res_generic_dismiss["success"] is True
        assert "stopped" in res_generic_dismiss["response"].lower()
        assert assistant.session_capture_manager.has_active_session() is False

        # 7. Passkey "stop capture" while session is active
        res_start4 = assistant.execute("Start session capture", speak_response=False)
        assert res_start4["success"] is True
        assert assistant.session_capture_manager.has_active_session() is True

        res_stop_cap = assistant.execute("stop capture", speak_response=False)
        assert res_stop_cap["success"] is True
        assert "stopped" in res_stop_cap["response"].lower()
        assert assistant.session_capture_manager.has_active_session() is False

        # 8. Passkey "stop stop" while session is active
        res_start5 = assistant.execute("Start session capture", speak_response=False)
        assert res_start5["success"] is True
        assert assistant.session_capture_manager.has_active_session() is True

        res_stopstop = assistant.execute("stop stop", speak_response=False)
        assert res_stopstop["success"] is True
        assert "stopped" in res_stopstop["response"].lower()
        assert assistant.session_capture_manager.has_active_session() is False
