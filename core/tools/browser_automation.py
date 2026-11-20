"""Browser Automation Tool with Semantic Accessibility-Tree and Tab Management.

Implements Phase 3.7:
- Semantic DOM and Accessibility Tree representation
- Tab management and intelligent deduplication
- URL navigation, in-page element clicking, and form filling
- Deterministic operation-level permissions
- Independent outcome verification
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.browser_automation import BrowserAutomationEngine
from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult


class BrowserAutomationTool(BaseTool):
    """Automates web browsing, DOM accessibility inspection, and tab interactions."""

    name = "browser_automation"
    description = (
        "Automate web browsing with semantic DOM and accessibility tree inspection, tab management, clicking, and form input. "
        "Safe read actions: 'search' (queries web search and parses results), "
        "'extract_content' (extracts clean readable markdown from active tab), "
        "'list_elements' (inspects interactive DOM elements with stable IDs like links, buttons, inputs), "
        "'list_tabs' (enumerates open browser tabs). "
        "Mutating actions: 'navigate' (loads a URL in active or new tab), "
        "'click_element' (clicks interactive DOM element by ID), "
        "'fill_form' (fills input field with value and optionally submits), "
        "'switch_tab' (switches active tab), 'close_tab' (closes browser tab)."
    )
    tier = PermissionTier.SYSTEM_IMPACTING

    SAFE_ACTIONS = {"search", "extract_content", "list_elements", "list_tabs"}

    def __init__(self, engine: Optional[BrowserAutomationEngine] = None):
        self.engine = engine or BrowserAutomationEngine()

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        action = str(arguments.get("action", "")).strip().lower()
        if action in self.SAFE_ACTIONS:
            return PermissionTier.SAFE
        return PermissionTier.SYSTEM_IMPACTING

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "search",
                        "navigate",
                        "extract_content",
                        "list_elements",
                        "click_element",
                        "fill_form",
                        "list_tabs",
                        "switch_tab",
                        "close_tab",
                    ],
                    "description": "Browser automation action to execute.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query terms for action 'search'.",
                },
                "url": {
                    "type": "string",
                    "description": "Target web URL for action 'navigate'.",
                },
                "element_id": {
                    "type": "string",
                    "description": "Semantic element ID (e.g. 'elem_1') for action 'click_element' or 'fill_form'.",
                },
                "value": {
                    "type": "string",
                    "description": "Input value to fill for action 'fill_form'.",
                },
                "submit": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to automatically submit form after filling input.",
                },
                "tab_id": {
                    "type": "string",
                    "description": "Optional specific tab ID to target.",
                },
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of search results to return.",
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()

        # ── READ-ONLY ACTIONS (SAFE) ──────────────────────────────────────────

        if action == "search":
            query = str(kwargs.get("query", "")).strip()
            if not query:
                return ToolResult(success=False, output="", error="Parameter 'query' is required for search.")

            max_res = int(kwargs.get("max_results", 5))
            res = self.engine.search(query=query, max_results=max_res)
            if not res.get("success"):
                return ToolResult(success=False, output="", error=res.get("error", "Search failed."))

            lines = [f"Search Results for '{query}' ({res.get('results_count', 0)}):"]
            for r in res.get("results", []):
                lines.append(f"  - [{r['id']}] {r['title']} -> {r['url']}")

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata=res,
            )

        elif action == "extract_content":
            tab_id = kwargs.get("tab_id")
            res = self.engine.extract_content(tab_id=tab_id)
            if not res.get("success"):
                return ToolResult(success=False, output="", error=res.get("error", "Content extraction failed."))

            return ToolResult(
                success=True,
                output=f"=== Content from '{res.get('title')}' ===\n\n{res.get('content')}",
                metadata=res,
            )

        elif action == "list_elements":
            tab_id = kwargs.get("tab_id")
            res = self.engine.list_elements(tab_id=tab_id)
            if not res.get("success"):
                return ToolResult(success=False, output="", error=res.get("error", "Failed to list elements."))

            lines = [f"Interactive Elements on '{res.get('title')}' ({res.get('total_elements')}):"]
            for elem in res.get("elements", []):
                etype = elem.get("type", "elem").upper()
                target_info = f" (target: {elem.get('target')})" if elem.get("target") else ""
                lines.append(f"  [{elem['id']}] {etype}: '{elem.get('text')}'{target_info}")

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata=res,
            )

        elif action == "list_tabs":
            tabs = self.engine.list_tabs()
            if not tabs:
                return ToolResult(success=True, output="No browser tabs are currently open.", metadata={"tabs": []})

            lines = [f"Open Browser Tabs ({len(tabs)}):"]
            for t in tabs:
                active_flag = " [ACTIVE]" if t.get("is_active") else ""
                lines.append(f"  - [{t['tab_id']}]{active_flag} '{t['title']}' ({t['url']})")

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={"tabs": tabs, "count": len(tabs)},
            )

        # ── MUTATING ACTIONS (SYSTEM_IMPACTING) WITH VERIFICATION ──────────────

        elif action == "navigate":
            url = str(kwargs.get("url", "")).strip()
            if not url:
                return ToolResult(success=False, output="", error="Parameter 'url' is required for navigate.")

            tab_id = kwargs.get("tab_id")
            res = self.engine.navigate(url=url, tab_id=tab_id)
            if not res.get("success"):
                return ToolResult(success=False, output="", error=res.get("error", "Navigation failed."))

            return ToolResult(
                success=True,
                output=res.get("message", f"Navigated to '{url}'."),
                metadata=res,
            )

        elif action == "click_element":
            elem_id = str(kwargs.get("element_id", "")).strip()
            if not elem_id:
                return ToolResult(success=False, output="", error="Parameter 'element_id' is required for click_element.")

            tab_id = kwargs.get("tab_id")
            res = self.engine.click_element(element_id=elem_id, tab_id=tab_id)
            if not res.get("success"):
                return ToolResult(success=False, output="", error=res.get("error", "Click failed."))

            return ToolResult(
                success=True,
                output=res.get("message", f"Clicked element '{elem_id}'."),
                metadata=res,
            )

        elif action == "fill_form":
            elem_id = str(kwargs.get("element_id", "")).strip()
            value = str(kwargs.get("value", ""))
            if not elem_id:
                return ToolResult(success=False, output="", error="Parameter 'element_id' is required for fill_form.")

            submit = bool(kwargs.get("submit", False))
            tab_id = kwargs.get("tab_id")
            res = self.engine.fill_form(element_id=elem_id, value=value, submit=submit, tab_id=tab_id)
            if not res.get("success"):
                return ToolResult(success=False, output="", error=res.get("error", "Form fill failed."))

            return ToolResult(
                success=True,
                output=res.get("message", f"Filled '{elem_id}' with '{value}'."),
                metadata=res,
            )

        elif action == "switch_tab":
            tab_id = str(kwargs.get("tab_id", "")).strip()
            if not tab_id:
                return ToolResult(success=False, output="", error="Parameter 'tab_id' is required for switch_tab.")

            res = self.engine.switch_tab(tab_id)
            if not res.get("success"):
                return ToolResult(success=False, output="", error=res.get("error", "Switch tab failed."))

            return ToolResult(
                success=True,
                output=res.get("message", f"Switched active tab to '{tab_id}'."),
                metadata=res,
            )

        elif action == "close_tab":
            tab_id = kwargs.get("tab_id")
            res = self.engine.close_tab(tab_id=tab_id)
            if not res.get("success"):
                return ToolResult(success=False, output="", error=res.get("error", "Close tab failed."))

            return ToolResult(
                success=True,
                output=f"Closed tab '{res.get('title')}'. Remaining open tabs: {res.get('remaining_tabs_count')}.",
                metadata=res,
            )

        return ToolResult(
            success=False,
            output="",
            error=f"Unknown browser action '{action}'.",
        )
