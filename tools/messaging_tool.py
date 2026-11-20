"""
Messaging Tools for M.A.D.D.Y. ACI.
Provides Telegram Bot dispatching, WhatsApp message preparation/auto-send,
and Contact Book management.
"""
from typing import Any, Dict
try:
    from integrations.messaging.telegram_client import TelegramClient
except ImportError:
    from integrations.messaging import TelegramClient
from integrations.messaging.whatsapp_client import WhatsAppClient
from integrations.messaging.contacts_manager import contacts_manager
from tools.base import RiskLevel, Tool

_telegram = TelegramClient()
_whatsapp = WhatsAppClient()


def _telegram_send_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    message = params.get("message", "").strip()
    photo_path = params.get("photo_path") or params.get("image_path")
    chat_id = params.get("chat_id")
    is_alert = params.get("is_alert", False)
    level = params.get("level") or params.get("alert_level", "INFO")

    if not message and not photo_path:
        return {"status": "error", "message": "Message text or photo path is required."}

    if photo_path:
        res = _telegram.send_photo(photo_path=photo_path, caption=message or None, chat_id=chat_id)
        if res.get("success"):
            return {
                "status": "success",
                "success": True,
                "message": "Photo successfully dispatched via Telegram.",
                "recipient": res.get("recipient"),
                "spoken_response": "I have sent the photo to your Telegram, boss.",
                "spoken_summary": "I have sent the photo to your Telegram, boss.",
            }
        else:
            err_msg = res.get("error", "Failed to send photo via Telegram.")
            return {
                "status": "error",
                "success": False,
                "error": err_msg,
                "message": err_msg,
                "spoken_response": f"I couldn't send the photo to Telegram: {err_msg}",
                "spoken_summary": f"I couldn't send the photo to Telegram: {err_msg}",
            }

    if is_alert:
        title = params.get("title", "Operational Update")
        res = _telegram.send_alert(title=title, body=message, level=level, chat_id=chat_id)
    else:
        res = _telegram.send_message(text=message, chat_id=chat_id)

    if res.get("success"):
        return {
            "status": "success",
            "success": True,
            "message": "Message successfully dispatched via Telegram.",
            "recipient": res.get("recipient"),
            "spoken_response": "I have sent the message to your Telegram, boss.",
            "spoken_summary": "I have sent the message to your Telegram, boss.",
        }
    else:
        err_msg = res.get("error", "Failed to send message via Telegram.")
        return {
            "status": "error",
            "success": False,
            "error": err_msg,
            "message": err_msg,
            "spoken_response": "I couldn't send the Telegram message. Please make sure your bot token and chat ID are configured in the environment.",
            "spoken_summary": "I couldn't send the Telegram message. Please make sure your bot token and chat ID are configured in the environment.",
        }


def _whatsapp_send_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    target = (params.get("target") or params.get("phone") or params.get("contact") or "").strip()
    message = params.get("message", "").strip()
    if not message:
        return {"status": "error", "success": False, "error": "Message text is required.", "message": "Message text is required."}

    res = _whatsapp.open_chat_with_message(target=target, message=message)
    if res.get("success"):
        contact_name = res.get("target") or "recipient"
        phone_num = res.get("resolved_phone") or ""
        auto_send = res.get("auto_send", False)

        reused = res.get("reused_tab", False)
        if auto_send:
            spoken = (
                f"I've switched to your WhatsApp chat for {contact_name} and queued hands-free dispatch, boss."
                if reused else
                f"I've opened the WhatsApp chat for {contact_name} and queued hands-free dispatch, boss."
            )
        else:
            spoken = (
                f"I've switched to your open WhatsApp Web with your message for {contact_name}, boss."
                if reused else
                f"I have opened WhatsApp Web with your message for {contact_name}, boss."
            )

        return {
            "status": "success",
            "success": True,
            "action": res.get("action"),
            "target": contact_name,
            "resolved_phone": phone_num,
            "auto_send": auto_send,
            "url": res.get("url"),
            "message": f"WhatsApp message prepared for {contact_name}.",
            "spoken_response": spoken,
            "spoken_summary": spoken,
        }
    else:
        err_msg = res.get("error", "Failed to open WhatsApp Web.")
        return {
            "status": "error",
            "success": False,
            "error": err_msg,
            "message": err_msg,
            "spoken_response": err_msg,
            "spoken_summary": err_msg,
        }


def _contact_save_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name", "").strip()
    phone = params.get("phone", "").strip()
    if not name or not phone:
        return {"status": "error", "success": False, "message": "Both contact name and phone number are required."}

    success = contacts_manager.set_contact(name=name, phone=phone)
    if success:
        cleaned = contacts_manager.clean_phone(phone)
        msg = f"Saved contact '{name}' with phone number {cleaned}."
        spoken = f"I have saved {name}'s phone number, boss."
        return {"status": "success", "success": True, "message": msg, "spoken_response": spoken, "spoken_summary": spoken}
    return {"status": "error", "success": False, "message": f"Failed to save contact {name}."}


def _contact_list_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    contacts = contacts_manager.list_contacts()
    if not contacts:
        return {"status": "success", "success": True, "contacts": {}, "message": "No contacts saved yet.", "spoken_response": "You don't have any contacts saved yet, boss."}

    lines = [f"- {name.title()}: {phone}" for name, phone in sorted(contacts.items())]
    summary = "\n".join(lines)
    spoken = f"You have {len(contacts)} contacts saved: {', '.join(sorted([k.title() for k in contacts.keys()]))}, boss."
    return {"status": "success", "success": True, "contacts": contacts, "message": summary, "spoken_response": spoken}


def _contact_delete_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name", "").strip()
    if not name:
        return {"status": "error", "success": False, "message": "Contact name is required."}

    deleted = contacts_manager.delete_contact(name)
    if deleted:
        return {"status": "success", "success": True, "message": f"Deleted contact '{name}'.", "spoken_response": f"Deleted contact {name}, boss."}
    return {"status": "error", "success": False, "message": f"Contact '{name}' not found."}


handle_telegram_send = _telegram_send_handler
handle_whatsapp_send = _whatsapp_send_handler
handle_contact_save = _contact_save_handler
handle_contact_list = _contact_list_handler
handle_contact_delete = _contact_delete_handler

telegram_send_tool = Tool(
    name="telegram.send",
    description="Send a message or urgent alert to Telegram (e.g. mobile notification when training completes or OOM occurs).",
    parameter_schema={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The text content of the message or alert.",
            },
            "title": {
                "type": "string",
                "description": "Optional title for alert messages.",
            },
            "is_alert": {
                "type": "boolean",
                "description": "Whether this is an operational alert with priority icon (default: false).",
                "default": False,
            },
            "level": {
                "type": "string",
                "enum": ["INFO", "WARNING", "CRITICAL", "SUCCESS"],
                "description": "Alert severity level (default: INFO).",
                "default": "INFO",
            },
            "alert_level": {
                "type": "string",
                "enum": ["INFO", "WARNING", "CRITICAL", "SUCCESS"],
                "description": "Alert severity level alias.",
                "default": "INFO",
            },
            "chat_id": {
                "type": "string",
                "description": "Optional target Telegram chat ID (defaults to TELEGRAM_CHAT_ID in .env).",
            },
        },
        "required": ["message"],
    },
    handler=_telegram_send_handler,
    risk_level=RiskLevel.MEDIUM,
    timeout=10.0,
    supported_platforms=["windows", "linux"],
)

telegram_send_alias_tool = Tool(
    name="telegram_send",
    description="Send a message or urgent alert to Telegram (e.g. mobile notification when training completes or OOM occurs).",
    parameter_schema=telegram_send_tool.parameter_schema,
    handler=_telegram_send_handler,
    risk_level=RiskLevel.MEDIUM,
    timeout=10.0,
    supported_platforms=["windows", "linux"],
)

whatsapp_send_tool = Tool(
    name="whatsapp.send",
    description="Send a WhatsApp message to a named contact (e.g. 'Mom', 'Alex', 'myself') or phone number with automatic hands-free dispatch.",
    parameter_schema={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Contact name (e.g. 'Mom', 'Alex') or target phone number with country code (e.g. '+919876543210').",
            },
            "phone": {
                "type": "string",
                "description": "Alias for target contact name or phone number.",
            },
            "message": {
                "type": "string",
                "description": "The message body to send on WhatsApp.",
            },
        },
        "required": ["message"],
    },
    handler=_whatsapp_send_handler,
    risk_level=RiskLevel.LOW,
    timeout=10.0,
    supported_platforms=["windows", "linux"],
)

whatsapp_send_alias_tool = Tool(
    name="whatsapp_send",
    description="Send a WhatsApp message to a named contact or phone number.",
    parameter_schema=whatsapp_send_tool.parameter_schema,
    handler=_whatsapp_send_handler,
    risk_level=RiskLevel.LOW,
    timeout=10.0,
    supported_platforms=["windows", "linux"],
)

contact_save_tool = Tool(
    name="contact.save",
    description="Save a contact name and phone number into the address book.",
    parameter_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the contact (e.g. 'Mom', 'Rahul')."},
            "phone": {"type": "string", "description": "Phone number with country code (e.g. '+919876543210')."},
        },
        "required": ["name", "phone"],
    },
    handler=_contact_save_handler,
    risk_level=RiskLevel.LOW,
    timeout=5.0,
    supported_platforms=["windows", "linux"],
)

contact_list_tool = Tool(
    name="contact.list",
    description="List all saved contacts in the address book.",
    parameter_schema={"type": "object", "properties": {}},
    handler=_contact_list_handler,
    risk_level=RiskLevel.LOW,
    timeout=5.0,
    supported_platforms=["windows", "linux"],
)

contact_delete_tool = Tool(
    name="contact.delete",
    description="Delete a contact from the address book.",
    parameter_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the contact to remove."},
        },
        "required": ["name"],
    },
    handler=_contact_delete_handler,
    risk_level=RiskLevel.LOW,
    timeout=5.0,
    supported_platforms=["windows", "linux"],
)
