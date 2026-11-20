"""
Browser tool definitions for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) ACI (browser.navigate, browser.search, browser.extract_content).
"""
from typing import Any, Dict
from browser.engine import BrowserEngine
from tools.base import RiskLevel, Tool

_default_browser_engine = BrowserEngine()

def _handle_browser_navigate(args: Dict[str, Any]) -> Dict[str, Any]:
    url = args["url"]
    return _default_browser_engine.navigate(url)

def _handle_browser_search(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args["query"]
    max_results = min(max(1, int(args.get("max_results", 5))), 20)
    return _default_browser_engine.search(query=query, max_results=max_results)

def _handle_browser_extract_content(args: Dict[str, Any]) -> Dict[str, Any]:
    url = args["url"]
    nav_res = _default_browser_engine.navigate(url)
    if nav_res.get("success") is False:
        return nav_res
    
    return {
        "url": nav_res["url"],
        "title": nav_res["title"],
        "extracted_content": nav_res["content"],
    }

browser_navigate_tool = Tool(
    name="browser.navigate",
    description="Navigate to a web page URL and extract page text and title.",
    parameter_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Target web URL to navigate to (e.g., 'https://en.wikipedia.org').",
            },
        },
        "required": ["url"],
    },
    handler=_handle_browser_navigate,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)

browser_search_tool = Tool(
    name="browser.search",
    description="Perform a web search query and return result links and text snippets.",
    parameter_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Web search query terms.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum search results to return (default 5, max 20).",
            },
        },
        "required": ["query"],
    },
    handler=_handle_browser_search,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)

browser_extract_tool = Tool(
    name="browser.extract_content",
    description="Extract main readable content text from a web page URL.",
    parameter_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Web URL to extract content from.",
            },
        },
        "required": ["url"],
    },
    handler=_handle_browser_extract_content,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)

def _handle_browser_open(args: Dict[str, Any]) -> Dict[str, Any]:
    import time
    import os
    import sys
    import subprocess
    from platform_layer import get_platform
    plat = get_platform()
    browser_name = args.get("browser_name", "").strip() or "browser"
    url = args.get("url", "").strip()

    try:
        target_check = browser_name if browser_name != "browser" else getattr(plat, "get_default_browser", lambda: "chrome")()

        # Check if url belongs to a known service (whatsapp, youtube, spotify) that may already be open
        url_lower = url.lower()
        matched_service = None
        if "whatsapp" in url_lower:
            matched_service = "whatsapp"
        elif "youtube" in url_lower or "youtu.be" in url_lower:
            matched_service = "youtube"
        elif "spotify" in url_lower:
            matched_service = "spotify"

        if matched_service:
            from browser.tab_manager import browser_tab_manager
            reuse_res = browser_tab_manager.open_or_reuse(matched_service, url, force_navigate=False)
            if reuse_res.get("reused"):
                return {
                    "success": True,
                    "browser": target_check,
                    "url": url,
                    "status": "reused_existing_tab",
                    "reused_tab": True,
                    "message": f"Activated pre-opened {matched_service.capitalize()} window without opening duplicate.",
                }

        # 1. Open URL cleanly via direct browser process launch or OS handler (Zero synthetic keystrokes, Zero duplicates)
        if url:
            resolved_exe = plat.resolve_application_path(target_check) if hasattr(plat, "resolve_application_path") else ""
            if resolved_exe and os.path.isabs(resolved_exe) and os.path.exists(resolved_exe):
                subprocess.Popen([resolved_exe, url])
            elif sys.platform.startswith("win"):
                os.startfile(url)
            else:
                import webbrowser
                webbrowser.open_new_tab(url)

            time.sleep(0.3)
            try:
                plat.focus_window_or_application(target_check)
            except Exception:
                pass

            return {
                "success": True,
                "browser": target_check,
                "url": url,
                "status": "opened_new_tab" if is_already_running else "verified_running",
                "message": f"Opened {url} in {target_check}.",
            }
        else:
            plat.open_application(browser_name)

        # 3. Verify process is actually running (with retry window up to 3 seconds)
        verified = False
        for _ in range(6):
            time.sleep(0.5)
            if plat.is_application_running(target_check):
                verified = True
                break

        if verified:
            try:
                plat.focus_window_or_application(target_check)
            except Exception:
                pass
            return {
                "success": True,
                "browser": target_check,
                "url": url,
                "status": "verified_running",
                "message": f"Opened {url} in {target_check}." if url else f"Browser '{target_check}' launched and verified running.",
            }
        else:
            return {
                "success": False,
                "browser": target_check,
                "status": "launch_unverified",
                "error": f"Browser '{target_check}' launched but process could not be verified in process table.",
            }
    except Exception as e:
        return {
            "success": False,
            "browser": browser_name,
            "status": "error",
            "error": str(e),
        }

browser_open_tool = Tool(
    name="browser.open",
    description="Launch default or specified web browser (e.g. Firefox, Chrome, Edge) and verify running state.",
    parameter_schema={
        "type": "object",
        "properties": {
            "browser_name": {
                "type": "string",
                "description": "Optional name of browser (e.g. 'firefox', 'chrome', 'edge'). Defaults to system default.",
            },
            "url": {
                "type": "string",
                "description": "Optional URL to open.",
            },
        },
    },
    handler=_handle_browser_open,
    risk_level=RiskLevel.MEDIUM,
    timeout=10.0,
)
