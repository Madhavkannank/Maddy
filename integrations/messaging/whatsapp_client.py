"""
WhatsApp Messaging Integration for M.A.D.D.Y.
Dispatches messages hands-free via WhatsApp Web with contact resolution and auto-send capability.
"""
import os
import sys
import time
import threading
import urllib.parse
import webbrowser
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from integrations.messaging.contacts_manager import contacts_manager

load_dotenv(override=True)


class WhatsAppClient:
    """
    Client for dispatching WhatsApp messages via Web URL scheme.
    Resolves contact names, opens chat, and triggers hands-free auto-send.
    """

    def __init__(self):
        self.contacts = contacts_manager

    def clean_phone(self, phone: str) -> str:
        """Sanitize phone number to digits, retaining leading plus if present."""
        return self.contacts.clean_phone(phone)

    def _clean_phone(self, phone: str) -> str:
        """Backwards-compatible alias for clean_phone."""
        return self.clean_phone(phone)

    def resolve_target(self, target: str) -> Optional[str]:
        """Resolve a contact name, self alias, or raw phone to phone digits."""
        return self.contacts.get_phone(target)

    def prepare_whatsapp_url(self, phone: str, message: str) -> str:
        """Format an official WhatsApp Web send URL."""
        clean_phone = "".join(c for c in phone if c.isdigit())
        encoded_msg = urllib.parse.quote(message)
        if clean_phone:
            return f"https://web.whatsapp.com/send?phone={clean_phone}&text={encoded_msg}"
        return f"https://web.whatsapp.com/send?text={encoded_msg}"

    def _trigger_auto_send(self, delay: float = 9.0) -> None:
        """Background worker that waits for WhatsApp Web to load and sends Enter keystroke."""
        def _worker():
            try:
                time.sleep(delay)
                if sys.platform.startswith("win"):
                    # Use Windows API to send Enter keypress
                    import ctypes
                    VK_RETURN = 0x0D
                    KEYEVENTF_KEYUP = 0x0002
                    # Press Enter
                    ctypes.windll.user32.keybd_event(VK_RETURN, 0, 0, 0)
                    time.sleep(0.05)
                    # Release Enter
                    ctypes.windll.user32.keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0)
            except Exception:
                pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def send_message(
        self,
        target: Optional[str] = None,
        message: str = "",
        auto_send: Optional[bool] = None,
        phone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Alias for open_chat_with_message with auto_send support."""
        recipient = target or phone or ""
        return self.open_chat_with_message(target=recipient, message=message, auto_send=auto_send)

    def open_chat_with_message(
        self,
        target: Optional[str] = None,
        message: str = "",
        auto_send: Optional[bool] = None,
        phone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve recipient, open WhatsApp Web chat with prefilled message,
        and trigger auto-send if enabled.
        """
        load_dotenv(override=True)
        recipient = (target or phone or "").strip()
        resolved_phone = self.resolve_target(recipient)

        # If recipient looks like a name and is not found in contacts
        if not resolved_phone and not self.contacts.is_phone_number(recipient):
            # Check if recipient is a self request without configured phone
            if recipient.strip().lower() in ("myself", "me", "self", "my phone", ""):
                # If no self number is in .env, open general send so user can choose
                url = self.prepare_whatsapp_url("", message)
                try:
                    from browser import browser_tab_manager
                    reuse_res = browser_tab_manager.open_or_reuse("whatsapp", url, force_navigate=bool(message))
                    reused = reuse_res.get("reused", False)
                    action_note = "Switched to your pre-opened WhatsApp Web" if reused else "Opened WhatsApp Web"
                    return {
                        "success": True,
                        "target": "myself",
                        "phone": None,
                        "message": message,
                        "url": url,
                        "reused_tab": reused,
                        "action": f"{action_note}. Note: Set WHATSAPP_PHONE_NUMBER in .env to message yourself directly.",
                        "needs_self_phone": True,
                    }
                except Exception as e:
                    return {"success": False, "error": f"Failed to launch WhatsApp Web: {str(e)}"}

            return {
                "success": False,
                "error": f"No phone number found for contact '{recipient}'. Please say 'save contact {recipient} as +91XXXXXXXXXX' or provide their phone number.",
                "unknown_contact": recipient,
            }

        phone_to_use = resolved_phone or recipient
        clean_digits = "".join(c for c in phone_to_use if c.isdigit())
        url = self.prepare_whatsapp_url(clean_digits, message)

        # Check if auto_send is requested or enabled in environment (default True)
        if auto_send is None:
            env_auto = os.getenv("WHATSAPP_AUTO_SEND", "true").strip().lower()
            auto_send = env_auto in ("true", "1", "yes")

        delay_str = os.getenv("WHATSAPP_SEND_DELAY", "9.0").strip()
        try:
            delay = float(delay_str)
        except ValueError:
            delay = 9.0

        try:
            from browser import browser_tab_manager
            reuse_res = browser_tab_manager.open_or_reuse("whatsapp", url, force_navigate=True)
            reused = reuse_res.get("reused", False)
            if auto_send:
                actual_delay = min(delay, 5.0) if reused else delay
                self._trigger_auto_send(delay=actual_delay)

            action_note = "Navigated pre-opened WhatsApp Web chat" if reused else "Opened WhatsApp Web chat"
            return {
                "success": True,
                "target": recipient,
                "resolved_phone": phone_to_use,
                "message": message,
                "auto_send": auto_send,
                "url": url,
                "reused_tab": reused,
                "action": f"{action_note} for {recipient} ({phone_to_use}) and queued automatic dispatch.",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to launch WhatsApp Web: {str(e)}",
            }
