"""
Unit tests for Messaging Tools: Telegram and WhatsApp integrations.
"""
from unittest.mock import MagicMock, patch
import pytest

from integrations.messaging import TelegramClient
from integrations.messaging.whatsapp_client import WhatsAppClient
from tools.messaging_tool import (
    telegram_send_tool,
    whatsapp_send_tool,
    handle_telegram_send,
    handle_whatsapp_send,
)
from tools.registry import ToolRegistry


def test_telegram_client_unconfigured():
    # When neither env var nor param is provided
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
        client = TelegramClient(bot_token="", default_chat_id="")
        assert client.is_configured is False
        res = client.send_message("Test message")
        assert res["success"] is False
        assert "TELEGRAM_BOT_TOKEN" in res["error"]


def test_telegram_client_send_mock():
    client = TelegramClient(bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", default_chat_id="987654321")
    if not hasattr(client, "_client"):
        pytest.skip("Telegram integration excluded from repository checkout.")
    assert client.is_configured is True

    with patch.object(client._client, "post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 42}}
        mock_post.return_value = mock_response

        res = client.send_message("Training completed with mIoU 0.74")
        assert res["success"] is True
        assert res["message_id"] == 42
        mock_post.assert_called_once()


def test_whatsapp_client_clean_phone():
    client = WhatsAppClient()
    assert client._clean_phone("+91 98765-43210") == "+919876543210"
    assert client._clean_phone("9876543210") == "9876543210"


def test_whatsapp_client_send():
    client = WhatsAppClient()
    with patch("webbrowser.open") as mock_open:
        mock_open.return_value = True
        res = client.send_message("9876543210", "Hello from M.A.D.D.Y.")
        assert res["success"] is True
        assert "web.whatsapp.com" in res["url"]
        assert "Hello+from+M.A.D.D.Y." in res["url"] or "Hello%20from%20M.A.D.D.Y." in res["url"]


def test_messaging_tools_registration():
    from tools.messaging_tool import contact_save_tool, contact_list_tool, contact_delete_tool
    registry = ToolRegistry()
    registry.register(telegram_send_tool)
    registry.register(whatsapp_send_tool)
    registry.register(contact_save_tool)
    registry.register(contact_list_tool)
    registry.register(contact_delete_tool)

    assert registry.get("telegram.send") is not None
    assert registry.get("whatsapp.send") is not None
    assert registry.get("contact.save") is not None
    assert registry.get("contact.list") is not None
    assert registry.get("contact.delete") is not None


def test_contacts_manager_and_whatsapp_named_contact(tmp_path):
    from integrations.messaging.contacts_manager import ContactsManager
    c_file = str(tmp_path / "test_contacts.json")
    cm = ContactsManager(file_path=c_file)
    assert cm.set_contact("Rahul", "+91 99887-76655") is True
    assert cm.get_phone("Rahul") == "+919988776655"
    assert cm.get_phone("rahul") == "+919988776655"
    assert cm.get_phone("UnknownPerson") is None

    # Test WhatsAppClient with contact manager
    client = WhatsAppClient()
    client.contacts = cm
    with patch("webbrowser.open") as mock_open:
        mock_open.return_value = True
        res = client.send_message("Rahul", "Hey Rahul!", auto_send=False)
        assert res["success"] is True
        assert res["resolved_phone"] == "+919988776655"
        assert "919988776655" in res["url"]

        # Unknown contact returns helpful error
        err_res = client.send_message("UnknownGuy", "Hey!", auto_send=False)
        assert err_res["success"] is False
        assert "No phone number found" in err_res["error"]


def test_timer_manager_schedule_task():
    import time
    from tools.timer import TimerManager
    tm = TimerManager()
    ran = []
    entry = tm.schedule_task(delay_seconds=1, action=lambda: ran.append(True), label="TestTask")
    assert entry.id in tm.timers
    time.sleep(1.3)
    assert len(ran) == 1
    assert entry.id not in tm.timers

