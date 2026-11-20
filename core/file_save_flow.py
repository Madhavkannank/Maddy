"""
Conversational File Saving Manager for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Handles the multi-turn lifecycle of saving information:
1. Detecting file save commands (in .txt, .py, .json, etc.).
2. Asking the user for the target save location before writing.
3. Resolving directory names without requiring full paths (e.g. 'Desktop', 'Documents', 'tests').
4. Creating on-the-fly directories (e.g. 'create a dir called research on Desktop and save inside it').
"""
import enum
import os
import re
from typing import Any, Dict, Optional, Tuple


class FileSaveState(enum.Enum):
    IDLE = "IDLE"
    AWAITING_LOCATION = "AWAITING_LOCATION"


class FileSaveManager:
    """
    Stateful manager for interactive file saving and directory resolution.
    """

    def __init__(self, workspace_dir: Optional[str] = None, permission_gate: Optional[Any] = None):
        self.workspace_dir = workspace_dir or os.getcwd()
        self.permission_gate = permission_gate
        self.state = FileSaveState.IDLE
        self.pending_file: Optional[Dict[str, Any]] = None

    def reset(self):
        """Reset pending save state back to IDLE."""
        self.state = FileSaveState.IDLE
        self.pending_file = None

    def is_save_command(self, text: str) -> bool:
        """Check if user is requesting to save or write information to a file."""
        lower = text.lower().strip()
        patterns = [
            r"\b(?:save|write|export|store|dump)\b.*\b(?:file|txt|text\s+file|json|csv|python\s+file|script|document)\b",
            r"\b(?:save\s+this|save\s+info|save\s+the\s+info|save\s+these\s+notes|save\s+notes|save\s+data)\b",
            r"\bsave\s+to\s+(?:a\s+)?(?:file|[a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)\b",
            r"\bwrite\s+(?:this\s+)?(?:in|to)\s+(?:a\s+)?(?:txt|file|document)\b",
        ]
        return any(re.search(pat, lower) for pat in patterns)

    def extract_file_details(self, text: str, default_content: Optional[str] = None) -> Dict[str, Any]:
        """Extract filename, format extension, and content to save."""
        lower = text.lower().strip()

        # 1. Determine format / extension
        ext = ".txt"
        if re.search(r"\b(?:json|in\s+json)\b", lower):
            ext = ".json"
        elif re.search(r"\b(?:python|py|python\s+file|python\s+script)\b", lower):
            ext = ".py"
        elif re.search(r"\b(?:csv|in\s+csv)\b", lower):
            ext = ".csv"
        elif re.search(r"\b(?:markdown|md|in\s+markdown)\b", lower):
            ext = ".md"
        elif re.search(r"\b(?:html|web\s+page)\b", lower):
            ext = ".html"

        # 2. Extract filename if explicitly specified
        filename = None
        m_name = re.search(r"\b(?:called|named|as|into\s+a\s+file\s+named)\s+([a-zA-Z0-9_\-\.]+)", text, re.IGNORECASE)
        if m_name:
            cand = m_name.group(1).strip().strip("'\"")
            if "." in cand:
                filename = cand
            else:
                filename = f"{cand}{ext}"
        else:
            m_direct = re.search(r"\b([a-zA-Z0-9_\-]+\.(?:txt|py|json|csv|md|html))\b", text, re.IGNORECASE)
            if m_direct:
                filename = m_direct.group(1).strip()

        if not filename:
            filename = f"notes{ext}" if ext == ".txt" else f"data{ext}"

        # 3. Extract content
        content = None
        # Check quoted content
        m_quote = re.search(r"['\"](.+?)['\"]", text)
        if m_quote:
            content = m_quote.group(1).strip()
        else:
            # Check content after colon
            if ":" in text:
                content = text.split(":", 1)[1].strip()

        if not content:
            content = default_content or "No specific content provided."

        return {
            "filename": filename,
            "format": ext,
            "content": content,
        }

    def resolve_directory(self, location_text: str) -> Tuple[str, bool]:
        """
        Resolve user's location input into an absolute filesystem path.
        Supports:
        - Common folder names ('Desktop', 'Documents', 'Downloads')
        - Subdirectories in workspace or user home
        - Creation directives ('create a dir called X on Desktop')
        Returns: (resolved_dir_path: str, was_created: bool)
        """
        clean = location_text.strip()
        lower = clean.lower().strip(".,!?:;\"'")

        user_home = os.path.expanduser("~")
        std_folders = {
            "desktop": os.path.join(user_home, "Desktop"),
            "documents": os.path.join(user_home, "Documents"),
            "docs": os.path.join(user_home, "Documents"),
            "downloads": os.path.join(user_home, "Downloads"),
            "home": user_home,
            "workspace": self.workspace_dir,
            "current": self.workspace_dir,
            "here": self.workspace_dir,
        }

        # 1. Check for directory creation pattern
        # e.g. "create a dir called research on Desktop and save it inside"
        m_create = re.search(
            r"(?:create|make)\s+(?:a\s+)?(?:new\s+)?(?:dir|directory|folder)\s+(?:called|named)?\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?(?:\s+(?:in|on|at|inside)\s+([a-zA-Z0-9_\- ]+))?",
            lower,
        )
        if m_create:
            new_dir_name = m_create.group(1).strip()
            parent_key = (m_create.group(2) or "").strip().rstrip(".?!")
            parent_dir = self.workspace_dir
            if parent_key in std_folders:
                parent_dir = std_folders[parent_key]
            elif parent_key:
                # Check if parent is a subfolder in workspace or user home
                cand1 = os.path.join(self.workspace_dir, parent_key)
                cand2 = os.path.join(user_home, parent_key)
                if os.path.isdir(cand1):
                    parent_dir = cand1
                elif os.path.isdir(cand2):
                    parent_dir = cand2

            target_path = os.path.join(parent_dir, new_dir_name)
            os.makedirs(target_path, exist_ok=True)
            return target_path, True

        # 2. Strip filler phrases ('save in ...', 'on ...', 'folder ...')
        folder_target = re.sub(r"^(?:save\s+(?:in|to|inside|on)|in|to|inside|on)\s+", "", lower).strip()
        folder_target = re.sub(r"^(?:the\s+)?(?:folder|directory|dir)\s+", "", folder_target).strip()
        folder_target = folder_target.rstrip(".?!").strip()

        # 3. Check standard OS folders
        if folder_target in std_folders:
            target_path = std_folders[folder_target]
            os.makedirs(target_path, exist_ok=True)
            return target_path, False

        # 4. Check workspace subdirectories
        cand_ws = os.path.join(self.workspace_dir, folder_target)
        if os.path.isdir(cand_ws):
            return cand_ws, False

        # 5. Check user home subdirectories
        cand_home = os.path.join(user_home, folder_target)
        if os.path.isdir(cand_home):
            return cand_home, False

        # 6. Fallback: create directory in workspace (or Desktop if requested)
        os.makedirs(cand_ws, exist_ok=True)
        return cand_ws, True

    def handle_turn(
        self,
        text: str,
        address: str = "boss",
        recent_context: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Handle turn in the file saving conversation.
        Returns: (handled: bool, response_message: Optional[str])
        """
        lower = text.lower().strip()

        # State 1: Awaiting Location Confirmation
        if self.state == FileSaveState.AWAITING_LOCATION and self.pending_file:
            # Check for cancellation
            if any(w in lower for w in ["cancel", "never mind", "nevermind", "don't save", "dont save", "stop"]):
                self.reset()
                return True, f"File save cancelled, {address}."

            # Resolve location and save
            try:
                target_dir, created = self.resolve_directory(text)
                filename = self.pending_file["filename"]
                content = self.pending_file["content"]
                file_path = os.path.join(target_dir, filename)

                if self.permission_gate is not None:
                    if not self.permission_gate.is_path_in_allowed_sandbox(file_path):
                        self.reset()
                        return True, f"Permission denied: Target directory '{target_dir}' is outside permitted sandboxes, {address}."

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                dir_desc = f"new directory '{target_dir}'" if created else f"'{target_dir}'"
                resp = f"Saved '{filename}' into {dir_desc}, {address}."
                self.reset()
                return True, resp
            except Exception as e:
                self.reset()
                return True, f"Encountered an issue saving the file: {str(e)}"

        # State 2: Check if new file save command
        if self.is_save_command(text):
            details = self.extract_file_details(text, default_content=recent_context)
            self.pending_file = details
            self.state = FileSaveState.AWAITING_LOCATION
            filename = details["filename"]
            return True, (
                f"Where would you like me to save '{filename}', {address}? "
                f"You can specify a folder name (such as Desktop, Documents, or a project folder) "
                f"or ask me to create a new one."
            )

        return False, None
