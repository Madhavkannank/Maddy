"""
Messaging integrations for M.A.D.D.Y.
"""
from typing import Any, Dict, Optional

try:
    from integrations.messaging.telegram_client import TelegramClient
except ImportError:
    import sys
    import types

    class TelegramClient:  # type: ignore
        """Null fallback when Telegram integration files are excluded."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.bot_token = ""
            self.default_chat_id = ""
            self.is_configured = False

        def send_message(self, text: str, chat_id: Optional[str] = None) -> Dict[str, Any]:
            return {"success": False, "error": "Telegram integration not installed or configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."}

        def send_photo(self, photo_path: str, caption: Optional[str] = None, chat_id: Optional[str] = None) -> Dict[str, Any]:
            return {"success": False, "error": "Telegram integration not installed or configured."}

        def send_alert(self, title: str, details: str, chat_id: Optional[str] = None) -> Dict[str, Any]:
            return {"success": False, "error": "Telegram integration not installed or configured."}

    mock_mod = types.ModuleType("integrations.messaging.telegram_client")
    mock_mod.TelegramClient = TelegramClient
    sys.modules["integrations.messaging.telegram_client"] = mock_mod

from integrations.messaging.whatsapp_client import WhatsAppClient

__all__ = ["TelegramClient", "WhatsAppClient"]
