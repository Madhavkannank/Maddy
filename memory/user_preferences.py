"""
User Preferences persistent store.
Maintains persistent key-value user preferences across Windows and Linux sessions.
Stored at: <memory_dir>/user_preferences.json
"""
import json
import os
from typing import Any, Dict, List, Optional
from config.settings import get_memory_dir

class UserPreferences:
    """
    Manages persistent user preferences, rules, and default configuration overrides.
    """

    def __init__(self, storage_path: Optional[str] = None, storage_dir: Optional[str] = None):
        if storage_dir:
            self.storage_path = os.path.join(storage_dir, "user_preferences.json")
        elif storage_path:
            if os.path.isdir(storage_path):
                self.storage_path = os.path.join(storage_path, "user_preferences.json")
            else:
                self.storage_path = storage_path
        else:
            self.storage_path = os.path.join(get_memory_dir(), "user_preferences.json")
        
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored preference by key."""
        prefs = self.get_all_preferences()
        return prefs.get(key, default)

    def get_preference_list(self, key: str, default: Optional[List[str]] = None) -> List[str]:
        """Return a stored key as a normalized list of strings."""
        value = self.get_preference(key, default)
        if value is None:
            return [] if default is None else list(default)
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, tuple):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return [str(value).strip()] if str(value).strip() else []

    def add_preference_list(self, key: str, value: Any, unique: bool = True) -> List[str]:
        """Append to a stored list-like preference without duplicating equivalent entries."""
        normalized = str(value).strip()
        if not normalized:
            return self.get_preference_list(key, [])

        items = self.get_preference_list(key, [])
        if unique:
            lowered = {item.lower() for item in items}
            if normalized.lower() not in lowered:
                items.append(normalized)
        else:
            items.append(normalized)

        self.set_preference(key, items)
        return items

    def remove_preference_value(self, key: str, value: Any) -> List[str]:
        """Remove one value from a list-like preference."""
        items = self.get_preference_list(key, [])
        normalized = str(value).strip()
        filtered = [item for item in items if item.lower() != normalized.lower()]
        if filtered:
            self.set_preference(key, filtered)
        else:
            self.delete_preference(key)
        return filtered

    def set_preference(self, key: str, value: Any) -> None:
        """Store or update a user preference."""
        prefs = self.get_all_preferences()
        prefs[key] = value
        self._write_preferences(prefs)

    def delete_preference(self, key: str) -> bool:
        """Remove a preference key if it exists."""
        prefs = self.get_all_preferences()
        if key in prefs:
            del prefs[key]
            self._write_preferences(prefs)
            return True
        return False

    def get_preferred_address(self, default: str = "") -> str:
        """Return the preferred address or name for the user (e.g., 'boss', 'sir')."""
        return str(self.get_preference("user.preferred_address", default))

    def set_preferred_address(self, address: str) -> None:
        """Persist the user's preferred address."""
        self.set_preference("user.preferred_address", address.strip())

    def get_preferred_name(self, default: str = "") -> str:
        """Return the user's preferred given name."""
        return str(self.get_preference("user.preferred_name", default))

    def set_preferred_name(self, name: str) -> None:
        """Persist the user's preferred given name."""
        self.set_preference("user.preferred_name", name.strip())

    def has_configured_address(self) -> bool:
        """Check whether the user has already configured how they wish to be addressed."""
        addr = self.get_preferred_address("")
        name = self.get_preferred_name("")
        return bool(addr or name)

    def get_all_preferences(self) -> Dict[str, Any]:
        """Return dictionary of all stored preferences."""
        if not os.path.exists(self.storage_path):
            return {}
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_preferences(self, prefs: Dict[str, Any]) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
