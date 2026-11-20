"""
Contacts Manager for M.A.D.D.Y.
Stores, resolves, and manages contacts for WhatsApp and messaging tools.
Persists contacts in config/contacts.json.
"""
import json
import os
import re
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv(override=True)

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))
CONTACTS_FILE = os.path.join(CONFIG_DIR, "contacts.json")


class ContactsManager:
    """Manages persistent contact book mapping names to phone numbers."""

    def __init__(self, file_path: str = CONTACTS_FILE):
        self.file_path = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            # Seed with default self aliases from environment if present
            default_self = os.getenv("WHATSAPP_PHONE_NUMBER", "").strip()
            initial = {}
            if default_self:
                initial["me"] = default_self
                initial["myself"] = default_self
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(initial, f, indent=2)

    def load_contacts(self) -> Dict[str, str]:
        """Load all contacts as lowercase keys."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k.strip().lower(): str(v).strip() for k, v in data.items() if str(v).strip()}
        except Exception:
            return {}

    def save_contacts(self, contacts: Dict[str, str]) -> bool:
        """Save contacts dictionary to disk."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(contacts, f, indent=2)
            return True
        except Exception:
            return False

    def clean_phone(self, phone: str) -> str:
        """Sanitize phone number to digits, retaining leading plus if present."""
        raw = phone.strip()
        has_plus = raw.startswith("+")
        digits = "".join(c for c in raw if c.isdigit())
        return f"+{digits}" if has_plus else digits

    def is_phone_number(self, target: str) -> bool:
        """Return True if target looks like a raw phone number (mostly digits)."""
        clean = "".join(c for c in target if c.isdigit())
        # If it has 7 or more digits and not many alphabetic letters
        letters = [c for c in target if c.isalpha()]
        return len(clean) >= 7 and len(letters) == 0

    def get_phone(self, name_or_phone: str) -> Optional[str]:
        """
        Resolve a contact name or raw phone number to a valid phone string.
        Returns None if name is not found in contacts.
        """
        raw = name_or_phone.strip()
        if not raw:
            return None

        # 1. Check if it's already a raw phone number
        if self.is_phone_number(raw):
            return self.clean_phone(raw)

        lower = raw.lower()

        # 2. Check self aliases against environment
        if lower in ("me", "myself", "self", "my phone", "my number"):
            load_dotenv(override=True)
            env_phone = os.getenv("WHATSAPP_PHONE_NUMBER", "").strip()
            if env_phone:
                return self.clean_phone(env_phone)

        # 3. Check saved contacts
        contacts = self.load_contacts()
        if lower in contacts:
            return self.clean_phone(contacts[lower])

        # 4. Fuzzy / partial match (e.g. "mom" in "call mom")
        for k, v in contacts.items():
            if k in lower or lower in k:
                return self.clean_phone(v)

        # 5. Fuzzy difflib match (handles slight typos or minor transcription slips)
        import difflib
        matches = difflib.get_close_matches(lower, list(contacts.keys()), n=1, cutoff=0.45)
        if matches:
            return self.clean_phone(contacts[matches[0]])

        return None

    def resolve_contact_name(self, name: str) -> Optional[str]:
        """Return the canonical saved name for a given target query, or None."""
        raw = name.strip()
        if not raw:
            return None
        lower = raw.lower()
        contacts = self.load_contacts()
        if lower in contacts:
            return lower
        for k in contacts:
            if k in lower or lower in k:
                return k
        import difflib
        matches = difflib.get_close_matches(lower, list(contacts.keys()), n=1, cutoff=0.45)
        if matches:
            return matches[0]
        return None

    def set_contact(self, name: str, phone: str) -> bool:
        """Add or update a contact."""
        clean = self.clean_phone(phone)
        if not clean:
            return False
        contacts = self.load_contacts()
        contacts[name.strip().lower()] = clean
        return self.save_contacts(contacts)

    def delete_contact(self, name: str) -> bool:
        """Remove a contact."""
        contacts = self.load_contacts()
        lower = name.strip().lower()
        if lower in contacts:
            del contacts[lower]
            return self.save_contacts(contacts)
        return False

    def list_contacts(self) -> Dict[str, str]:
        """Return all saved contacts."""
        return self.load_contacts()


# Global singleton instance
contacts_manager = ContactsManager()
