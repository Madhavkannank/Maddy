"""Persistent Memory Store implementation for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Bounded, file-backed curated memory (MEMORY.md = agent notes, USER.md = user profile).
Entries are joined by ENTRY_DELIMITER ("\n§\n"); limits are enforced in characters.
The system prompt receives a FROZEN snapshot taken at session start (protecting prefix cache);
live writes persist to disk immediately.
"""
import errno
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tools.memory_tool")

ENTRY_DELIMITER = "\n§\n"

MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (your personal notes)",
    "user": "USER PROFILE (who the user is)",
}

DEFAULT_MEMORY_CHAR_LIMIT = 2200
DEFAULT_USER_CHAR_LIMIT = 1375

PATCH_DROP_HINT = "To drop text inside an entry, patch a longer span that contains it; 'remove' deletes the whole entry."

# Typography normalization mapping for tolerant substring matching:
# All single/double quotes, curly quotes, and backticks fold to "'";
# all hyphens, en/em dashes, and minus signs fold to "-".
_MATCH_FOLD = str.maketrans({
    **dict.fromkeys("\"`\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f", "'"),
    **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"),
})


def _normalize_for_match(text: str) -> str:
    """
    Quote/dash/whitespace-folded form for tolerant old_text matching.
    Collapses newlines and multiple whitespace runs into a single space.
    """
    return " ".join(str(text).replace("\\n", " ").translate(_MATCH_FOLD).split())


def _substring_matches(entries: List[str], old_text: str) -> List[int]:
    """Indices of entries containing old_text: byte-exact first, else typography-folded."""
    if exact := [i for i, e in enumerate(entries) if old_text in e]:
        return exact
    needle = _normalize_for_match(old_text)
    if not any(c.isalnum() for c in needle):
        return []
    return [i for i, e in enumerate(entries) if needle in _normalize_for_match(e)]


def _find_unique_match(entries: List[str], old_text: str) -> Tuple[Optional[int], bool]:
    """
    Returns (index, ambiguous) for entries matching old_text.
    A whole-entry EXACT match takes absolute priority over substring matches.
    """
    exact = [i for i, e in enumerate(entries) if e == old_text]
    matches = exact or _substring_matches(entries, old_text)
    if len({entries[i] for i in matches}) > 1:
        return None, True
    return (matches[0] if matches else None), False


def _no_match_error(entries: List[str], old_text: str, verb: str, **extra) -> Dict[str, Any]:
    """Zero-match error that suggests closest entries by word overlap."""
    def _words(text: str) -> set:
        return set(re.findall(r"\w+", text.lower()))

    words = _words(old_text)
    scored = sorted(
        ((len(words & _words(e)) / len(words), i) for i, e in enumerate(entries)) if words else [],
        key=lambda p: (-p[0], p[1]),
    )
    closest = [entries[i] for score, i in scored[:3] if score >= 0.3]
    hint = " old_text must come from ONE entry ('§' separates entries); use one op per entry." if "§" in old_text else ""
    return {
        "success": False,
        "error": f"No entry matched '{old_text}'.{hint} Retry the {verb} with old_text copied verbatim from the intended entry"
                 + (" (closest_entries below)." if closest else "."),
        "failure_class": "no_match",
        **({"closest_entries": closest} if closest else {}),
        **extra,
    }


def _span_occurrences(text: str, span: str) -> int:
    """Occurrences of span in text, overlapping ones included."""
    if not span or not text:
        return 0
    return sum(text.startswith(span, i) for i in range(len(text) - len(span) + 1))


def _locate_span_in_entry(entry: str, old_text: str) -> Optional[Tuple[int, int]]:
    """
    Finds (start, end) of old_text in entry.
    Checks exact match first; if not found, checks typography-folded match.
    """
    idx = entry.find(old_text)
    if idx >= 0:
        return idx, idx + len(old_text)

    norm_needle = _normalize_for_match(old_text)
    if not norm_needle or not any(c.isalnum() for c in norm_needle):
        return None

    # Sliding window search with slight tolerance for character mapping lengths
    target_len = len(old_text)
    min_len = max(1, target_len - 6)
    max_len = min(len(entry), target_len + 6)
    for length in range(min_len, max_len + 1):
        for i in range(len(entry) - length + 1):
            window = entry[i:i + length]
            if _normalize_for_match(window) == norm_needle:
                return i, i + length
    return None


def _patch_entry(entry: str, old_text: str, new_text: str) -> Tuple[Optional[str], str, str]:
    """
    Returns (patched_entry, "", "none"), or (None, error_message, failure_class).
    Rejects ambiguous occurrences (>1) and no-ops.
    """
    count = _span_occurrences(entry, old_text)
    if count > 1:
        return (
            None,
            f"'{old_text}' appears {count} times in the matched entry, and patch edits one span. "
            f"Include more surrounding text so old_text occurs once.",
            "ambiguous",
        )

    if count == 1:
        patched = entry.replace(old_text, new_text, 1)
    else:
        # Try tolerant span match
        span = _locate_span_in_entry(entry, old_text)
        if span is None:
            return None, f"Could not locate span '{old_text}' inside entry.", "no_match"
        start, end = span
        patched = entry[:start] + new_text + entry[end:]

    if patched == entry:
        return None, f"Patch is a no-op: '{old_text}' -> '{new_text}' leaves the entry unchanged.", "invalid_args"
    return patched, "", "none"


def atomic_write_text(path: Path, content: str, *, tmp_prefix: str = ".mem_") -> None:
    """
    Atomically writes text to path using a temp file in the same directory,
    fsyncing, and renaming with retry for Windows file lock contention.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(dir=str(path.parent), prefix=tmp_prefix, suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # On Windows, antivirus or search indexers can hold locks briefly.
        # Retry with backoff.
        replaced = False
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                replaced = True
                break
            except OSError as e:
                if attempt == 4:
                    raise
                time.sleep(0.02 * (2 ** attempt))

        if not replaced and tmp_path.exists():
            os.replace(tmp_path, path)
    except Exception:
        with suppress(OSError):
            if tmp_path.exists():
                os.unlink(tmp_path)
        raise


def get_memory_dir() -> Path:
    """
    Resolves the active memory directory.
    Priority:
    1. MADDY_MEMORY_DIR environment variable
    2. Project workspace memory/ directory
    """
    env_dir = os.environ.get("MADDY_MEMORY_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    local_p = Path(__file__).resolve().parent.parent / "memory"
    local_p.mkdir(parents=True, exist_ok=True)
    return local_p


class MemoryStore:
    """
    Bounded curated memory with file persistence.
    _system_prompt_snapshot is frozen at load time (prefix-cache stable);
    memory_entries / user_entries represent live state persisted to disk.
    """

    def __init__(
        self,
        memory_char_limit: int = DEFAULT_MEMORY_CHAR_LIMIT,
        user_char_limit: int = DEFAULT_USER_CHAR_LIMIT,
        memory_dir: Optional[Path] = None,
        memory_enabled: bool = True,
        user_profile_enabled: bool = True,
    ):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self._memory_dir = Path(memory_dir) if memory_dir is not None else None
        self.memory_enabled = memory_enabled
        self.user_profile_enabled = user_profile_enabled
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        self._consolidation_failures = 0
        self._MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir if self._memory_dir is not None else Path(get_memory_dir())

    def target_enabled(self, target: str) -> bool:
        return self.user_profile_enabled if target == "user" else self.memory_enabled

    def reset_consolidation_failures(self) -> None:
        self._consolidation_failures = 0

    def load(self, target: str) -> str:
        """Convenience method returning raw string content of target file."""
        path = self._path_for(target)
        raw, ok = self._read_raw_checked(path)
        return raw if ok else ""

    def _path_for(self, target: str) -> Path:
        return self.memory_dir / ("USER.md" if target == "user" else "MEMORY.md")

    def _entries_for(self, target: str) -> List[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: List[str]) -> None:
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        return len(ENTRY_DELIMITER.join(self._entries_for(target)))

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _usage(self, target: str) -> str:
        return f"{self._char_count(target):,}/{self._char_limit(target):,}"

    def _usage_pct(self, target: str, current: int) -> str:
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        return f"{pct}% — {current:,}/{limit:,} chars"

    def _render_block(self, target: str, entries: List[str]) -> str:
        """System prompt block: header + usage indicator + § entries."""
        if not entries:
            return ""
        content = ENTRY_DELIMITER.join(entries)
        sep = "═" * 46
        title = MEMORY_BLOCK_HEADERS["user" if target == "user" else "memory"]
        return f"{sep}\n{title} [{self._usage_pct(target, len(content))}]\n{sep}\n{content}"

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Exclusive file locking using msvcrt (Windows) or fcntl (Unix)."""
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        msvcrt = None
        fcntl = None
        try:
            import msvcrt
        except ImportError:
            try:
                import fcntl
            except ImportError:
                pass

        if fcntl is None and msvcrt is None:
            yield
            return

        flags = os.O_RDWR | os.O_CREAT
        raw_fd = os.open(str(lock_path), flags, 0o600)
        fd = os.fdopen(raw_fd, "r+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            elif msvcrt:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            with suppress(Exception):
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                elif msvcrt:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                fd.close()

    @staticmethod
    def _read_raw_checked(path: Path) -> Tuple[str, bool]:
        """Reads raw text strictly; utf-8-sig strips Notepad BOM if present."""
        if not path.exists():
            return "", True
        try:
            return path.read_text(encoding="utf-8-sig"), True
        except (OSError, UnicodeDecodeError):
            return "", False

    @staticmethod
    def _write_file(path: Path, entries: List[str]) -> None:
        """Writes entries atomically to path separated by ENTRY_DELIMITER."""
        atomic_write_text(path, ENTRY_DELIMITER.join(entries))

    @staticmethod
    def _parse_entries(raw: str) -> List[str]:
        """Stripped, non-empty entries split strictly on ENTRY_DELIMITER."""
        return [e for e in (x.strip() for x in raw.split(ENTRY_DELIMITER)) if e]

    def load_from_disk(self) -> None:
        """
        Loads MEMORY.md and USER.md from disk and captures the frozen system-prompt snapshot.
        """
        for target in ("memory", "user"):
            path = self._path_for(target)
            raw, ok = self._read_raw_checked(path)
            entries = list(dict.fromkeys(self._parse_entries(raw))) if ok else []
            self._set_entries(target, entries)
            self._system_prompt_snapshot[target] = self._render_block(target, entries)

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """Returns the frozen snapshot taken at load time."""
        return self._system_prompt_snapshot.get(target, "") or None

    def _success_response(self, target: str, message: Optional[str] = None, **extra) -> Dict[str, Any]:
        self._consolidation_failures = 0
        return {
            "success": True,
            "done": True,
            "target": target,
            "usage": self._usage_pct(target, self._char_count(target)),
            "entry_count": len(self._entries_for(target)),
            **({"message": message} if message else {}),
            **extra,
            "note": "Write saved. This update is complete — do not repeat it.",
        }

    def _mutate(self, target: str, mutate_fn) -> Dict[str, Any]:
        """
        Locks, reads live file from disk, applies mutate_fn, and atomically persists.
        """
        path = self._path_for(target)
        with self._file_lock(path):
            raw, read_ok = self._read_raw_checked(path)
            if not read_ok:
                return {
                    "success": False,
                    "error": f"Refusing to write {path.name}: file could not be read right now.",
                    "failure_class": "read_failed",
                }
            self._set_entries(target, list(dict.fromkeys(self._parse_entries(raw))))
            result = mutate_fn(self._entries_for(target), self._char_limit(target))
            if isinstance(result, dict):
                return result

            new_entries, message = result[0], result[1]
            extra_fields = result[2] if len(result) > 2 else {}
            self._set_entries(target, new_entries)
            atomic_write_text(path, ENTRY_DELIMITER.join(new_entries))
            return self._success_response(target, message, **extra_fields)

    def _consolidation_failure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Count a consolidation failure: under the per-turn cap (<=3) return response
        carrying current_entries and batch retry instructions; past it, return a
        TERMINAL result so the model stops looping memory calls and continues its reply.
        """
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {
            "success": False,
            "done": True,
            "failure_class": "retry_cap",
            "error": (
                f"Memory consolidation failed {self._consolidation_failures} times this turn. "
                "Stop retrying memory calls — leave memory unchanged for now and continue with your reply to the user. "
                "The fact can be saved in a later turn."
            ),
        }

    def add(self, target: str, content: str, auto_trim: bool = False) -> Dict[str, Any]:
        """
        Append a new entry.
        Out of budget cases:
        1. If auto_trim=True: automatically evicts the oldest entry/entries (FIFO) until the new entry fits.
        2. If auto_trim=False (default): returns an over_budget response with current_entries,
           usage statistics, exact overflow chars, and instructions to submit an atomic batch.
        3. If failures exceed 3 within the turn: terminates with retry_cap to prevent agent looping.
        """
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty.", "failure_class": "missing_content"}

        def _add(entries: List[str], limit: int):
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            working = list(entries)
            new_total = len(ENTRY_DELIMITER.join(working + [content]))

            if new_total > limit:
                if auto_trim:
                    evicted = []
                    while working and len(ENTRY_DELIMITER.join(working + [content])) > limit:
                        evicted.append(working.pop(0))

                    if len(ENTRY_DELIMITER.join(working + [content])) <= limit:
                        return (
                            working + [content],
                            f"Entry added after auto-trimming {len(evicted)} oldest entry/entries to fit budget.",
                            {"evicted_entries": evicted},
                        )

                overflow = new_total - limit
                fail_resp = {
                    "success": False,
                    "error": (
                        f"Memory at {self._char_count(target):,}/{limit:,} chars; adding this entry ({len(content)} chars) "
                        f"would exceed the limit by {overflow:,} chars. "
                        f"Retry as ONE 'operations' batch that removes or shortens (patch) stale entries from current_entries "
                        f"below to free at least {overflow:,} chars AND adds this entry — the limit is checked only on the batch result."
                    ),
                    "failure_class": "over_budget",
                    "overflow_chars": overflow,
                    "current_entries": entries,
                    "usage": self._usage(target),
                    "suggested_action": "batch",
                }
                return self._consolidation_failure(fail_resp)

            return working + [content], "Entry added."

        return self._mutate(target, _add)

    def patch(self, target: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """
        Locates the entry containing old_text and replaces that exact span with new_text.
        Preserves the remainder of the entry intact.
        """
        if not (old_text or "").strip():
            return {"success": False, "error": "old_text cannot be empty.", "failure_class": "missing_old_text"}
        if new_text is None or not str(new_text).strip():
            return {"success": False, "error": f"new_text cannot be empty. {PATCH_DROP_HINT}", "failure_class": "missing_content"}

        def _apply(entries: List[str], limit: int):
            idx, ambiguous = _find_unique_match(entries, old_text.strip())
            if ambiguous:
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "failure_class": "ambiguous",
                }
            if idx is None:
                return _no_match_error(entries, old_text.strip(), "patch", current_entries=entries)

            before = entries[idx]
            after, err_msg, fail_class = _patch_entry(before, old_text.strip(), str(new_text))
            if after is None:
                return {"success": False, "error": err_msg, "failure_class": fail_class}

            patched = entries[:idx] + [after] + entries[idx + 1:]
            new_total = len(ENTRY_DELIMITER.join(patched))
            if new_total > limit:
                return {
                    "success": False,
                    "error": f"Patch would put memory at {new_total:,}/{limit:,} chars. Shorten the text to fit.",
                    "failure_class": "over_budget",
                    "current_entries": entries,
                    "usage": self._usage(target),
                }
            return patched, "Entry patched.", {"patched_entry_before": before, "patched_entry_after": after}

        return self._mutate(target, _apply)

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Removes the entire entry containing old_text."""
        if not (old_text or "").strip():
            return {"success": False, "error": "old_text cannot be empty.", "failure_class": "missing_old_text"}

        def _apply(entries: List[str], limit: int):
            idx, ambiguous = _find_unique_match(entries, old_text.strip())
            if ambiguous:
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "failure_class": "ambiguous",
                }
            if idx is None:
                return _no_match_error(entries, old_text.strip(), "remove", current_entries=entries)

            removed_entry = entries[idx]
            remaining = entries[:idx] + entries[idx + 1:]
            return remaining, "Entry removed.", {"removed_entry": removed_entry}

        return self._mutate(target, _apply)

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Replaces the WHOLE entry containing old_text with new_content."""
        new_content = (new_content or "").strip()
        if not (old_text or "").strip():
            return {"success": False, "error": "old_text cannot be empty.", "failure_class": "missing_old_text"}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries.", "failure_class": "missing_content"}

        def _apply(entries: List[str], limit: int):
            idx, ambiguous = _find_unique_match(entries, old_text.strip())
            if ambiguous:
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "failure_class": "ambiguous",
                }
            if idx is None:
                return _no_match_error(entries, old_text.strip(), "replace", current_entries=entries)

            replaced = entries[:idx] + [new_content] + entries[idx + 1:]
            new_total = len(ENTRY_DELIMITER.join(replaced))
            if new_total > limit:
                return {
                    "success": False,
                    "error": f"Replacement would put memory at {new_total:,}/{limit:,} chars (limit {limit:,}).",
                    "failure_class": "over_budget",
                    "current_entries": entries,
                    "usage": self._usage(target),
                }
            return replaced, "Entry replaced.", {"replaced_entry": entries[idx]}

        return self._mutate(target, _apply)

    def view(self, target: str) -> Dict[str, Any]:
        """Inspects current entries and usage for the target without mutating."""
        path = self._path_for(target)
        raw, ok = self._read_raw_checked(path)
        entries = list(dict.fromkeys(self._parse_entries(raw))) if ok else []
        self._set_entries(target, entries)
        return {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": self._usage_pct(target, self._char_count(target)),
            "limit": self._char_limit(target),
            "count": len(entries),
        }

    def apply_batch(self, target: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Applies a list of add/patch/remove operations atomically against the final budget.
        All-or-nothing: any failed op rolls back the whole batch.
        """
        if not operations:
            return {"success": False, "error": "operations list is empty.", "failure_class": "invalid_args"}

        def _apply(entries: List[str], limit: int):
            working = list(entries)
            matched = []
            for i, op in enumerate(operations, 1):
                act = op.get("action")
                content = (op.get("content") or op.get("new_text") or "")
                old_text = (op.get("old_text") or "")

                if act == "add":
                    content = content.strip()
                    if not content:
                        return {"success": False, "error": f"Operation {i}: content is required.", "failure_class": "missing_content"}
                    if content not in working:
                        working.append(content)
                    matched.append(None)
                elif act == "remove":
                    old_text = old_text.strip()
                    if not old_text:
                        return {"success": False, "error": f"Operation {i}: old_text is required.", "failure_class": "missing_old_text"}
                    idx, ambiguous = _find_unique_match(working, old_text)
                    if ambiguous:
                        return {"success": False, "error": f"Operation {i}: '{old_text}' matched multiple entries.", "failure_class": "ambiguous"}
                    if idx is None:
                        miss = _no_match_error(working, old_text, "remove")
                        return {"success": False, "error": f"Operation {i}: {miss['error']}", "failure_class": "no_match"}
                    matched.append(working[idx])
                    working.pop(idx)
                elif act == "patch":
                    old_text = old_text.strip()
                    if not old_text:
                        return {"success": False, "error": f"Operation {i}: old_text is required.", "failure_class": "missing_old_text"}
                    if not content.strip():
                        return {"success": False, "error": f"Operation {i}: content is required.", "failure_class": "missing_content"}
                    idx, ambiguous = _find_unique_match(working, old_text)
                    if ambiguous:
                        return {"success": False, "error": f"Operation {i}: '{old_text}' matched multiple entries.", "failure_class": "ambiguous"}
                    if idx is None:
                        miss = _no_match_error(working, old_text, "patch")
                        return {"success": False, "error": f"Operation {i}: {miss['error']}", "failure_class": "no_match"}
                    before = working[idx]
                    after, err_msg, fail_class = _patch_entry(before, old_text, content)
                    if after is None:
                        return {"success": False, "error": f"Operation {i}: {err_msg}", "failure_class": fail_class}
                    matched.append(before)
                    working[idx] = after
                elif act == "replace":
                    old_text = old_text.strip()
                    content = content.strip()
                    if not old_text:
                        return {"success": False, "error": f"Operation {i}: old_text is required.", "failure_class": "missing_old_text"}
                    if not content:
                        return {"success": False, "error": f"Operation {i}: content is required.", "failure_class": "missing_content"}
                    idx, ambiguous = _find_unique_match(working, old_text)
                    if ambiguous:
                        return {"success": False, "error": f"Operation {i}: '{old_text}' matched multiple entries.", "failure_class": "ambiguous"}
                    if idx is None:
                        miss = _no_match_error(working, old_text, "replace")
                        return {"success": False, "error": f"Operation {i}: {miss['error']}", "failure_class": "no_match"}
                    matched.append(working[idx])
                    working[idx] = content
                else:
                    return {"success": False, "error": f"Operation {i}: unknown action '{act}'.", "failure_class": "invalid_args"}

            new_total = len(ENTRY_DELIMITER.join(working))
            if new_total > limit:
                return {
                    "success": False,
                    "error": f"Applying batch would put memory at {new_total:,}/{limit:,} chars. Over budget by {new_total - limit:,} chars.",
                    "failure_class": "over_budget",
                }
            return working, f"Applied {len(operations)} operation(s)."

        return self._mutate(target, _apply)
