"""
End-to-end integration tests for Hardware Watcher, ArXiv Digest, Messaging, and Personality Rules.
"""
from unittest.mock import MagicMock, patch
import pytest

from core.router import CapabilityRouter, CapabilityType
from app.maddy_assistant import MaddyAssistant
from memory.user_preferences import UserPreferences
from personality.engine import PersonalityEngine


def test_router_classifies_superpowers():
    router = CapabilityRouter()

    # Hardware
    r_gpu = router.route("How is my GPU holding up?")
    assert CapabilityType.HARDWARE in [CapabilityType(c) for c in r_gpu["all_capabilities"]]
    assert r_gpu["fast_path_eligible"] is True

    r_vram = router.route("What is my VRAM usage?")
    assert CapabilityType.HARDWARE in [CapabilityType(c) for c in r_gpu["all_capabilities"]]

    # Research / ArXiv
    r_arxiv = router.route("Check arXiv for new research papers on concrete defect segmentation")
    assert CapabilityType.RESEARCH in [CapabilityType(c) for c in r_arxiv["all_capabilities"]]
    assert r_arxiv["fast_path_eligible"] is True

    # Messaging
    r_tele = router.route("Send a telegram alert to my phone: training epoch 10 finished")
    assert CapabilityType.MESSAGING in [CapabilityType(c) for c in r_tele["all_capabilities"]]


def test_personality_default_address_is_strictly_not_user_name(tmp_path):
    # Ensure stating "My name is Madhav" never switches the spoken address title away from boss/sir
    prefs = UserPreferences(storage_dir=str(tmp_path))
    engine = PersonalityEngine(user_preferences=prefs)

    # Initial state
    assert engine.get_address_title() in ("boss", "sir")

    # Name declaration
    handled, resp = engine.handle_conversational_intercept("My name is Madhav")
    assert handled is True
    # The address title should STILL be boss or sir, never "Madhav"
    assert engine.get_address_title() in ("boss", "sir")
    assert "Madhav" not in engine.get_address_title()

    # Formatted responses should never contain user's real name
    sample_resp = engine.format_response(f"GPU is operating normally, {engine.get_address_title()}.")
    assert "Madhav" not in sample_resp
    assert any(title in sample_resp.lower() for title in ["boss", "sir"])


def test_maddy_executes_gpu_query(tmp_path):
    with patch("config.settings.get_memory_dir", return_value=str(tmp_path)):
        assistant = MaddyAssistant()
        with patch("tools.hardware_tool._monitor.query_gpu") as mock_gpu:
            from hardware.gpu_monitor import GPUMetrics
            mock_gpu.return_value = GPUMetrics(
                name="NVIDIA GeForce RTX 3050 Laptop GPU",
                utilization_gpu=55,
                memory_total_mb=6144,
                memory_used_mb=2048,
                temperature_gpu=58,
                is_available=True,
            )
            res = assistant.execute("How is my GPU holding up?")
            assert res["success"] is True
            output = res["response"]
            assert "RTX 3050" in output
            assert ("55 percent" in output) or ("55%" in output)
            assert ("58 degrees" in output) or ("58°C" in output)
            # Must adhere to Rule 2
            assert "Madhav" not in output


def test_maddy_executes_arxiv_query(tmp_path):
    with patch("config.settings.get_memory_dir", return_value=str(tmp_path)):
        assistant = MaddyAssistant()
        with patch("tools.arxiv_digest.ArxivClient.search_papers") as mock_search:
            from tools.arxiv_digest import ArxivPaper
            mock_search.return_value = [
                ArxivPaper(
                    arxiv_id="2603.09999v1",
                    title="Real-Time Defect Detection for Civil Infrastructure",
                    authors=["Dr. Jane Doe"],
                    summary="High speed defect detection using neural networks.",
                    published="2026-03-18",
                    pdf_url="http://arxiv.org/pdf/2603.09999v1",
                )
            ]
            res = assistant.execute("Check arxiv for research papers on defect detection")
            assert res["success"] is True
            output = res["response"]
            assert "Real-Time Defect Detection" in output
            assert "Dr. Jane Doe" in output
            # Must adhere to Rule 2
            assert "Madhav" not in output


def test_process_monitoring_not_immediate_telegram_echo(tmp_path):
    with patch("config.settings.get_memory_dir", return_value=str(tmp_path)):
        assistant = MaddyAssistant()
        with patch("integrations.messaging.telegram_client.TelegramClient.send_message") as mock_tg_send, \
             patch("integrations.messaging.telegram_client.TelegramClient.send_alert") as mock_tg_alert:

            # The exact phrase spoken by the user
            query = "So there is a process running now and when it completes give me a notification on telegram"
            res = assistant.execute(query)

            assert res["success"] is True
            # Crucial: Should NEVER have sent an immediate message to Telegram
            assert not mock_tg_send.called
            assert not mock_tg_alert.called

            # Response must confirm monitoring and notification upon completion
            resp = res["response"].lower()
            assert "monitor" in resp
            assert "telegram" in resp
            assert any(title in resp for title in ["boss", "sir"])

