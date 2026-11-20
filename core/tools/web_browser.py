"""Web Browser and Dynamic Search Tool for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You).
Provides dynamic web search (DuckDuckGo Lite & resilient direct site search),
URL navigation, and domain launching across Windows and Linux.
"""
from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult
from platform_layer import PlatformInterface, get_platform


class WebBrowserTool(BaseTool):
    """Navigates URLs, searches the web or specific sites, and opens web applications."""

    name = "web_browser"
    description = (
        "Search the web or open URLs and web applications in the default desktop browser. "
        "Actions: 'search' (searches web or specific site/domain, returns structured results, safe read-only), "
        "'open_search' (constructs direct search URL on site/domain and opens it in browser), "
        "'open_url' (opens an exact URL in browser), "
        "'open_domain' (opens standard or discovered web services)."
    )
    tier = PermissionTier.SYSTEM_IMPACTING

    # Minimal protocol fallback defaults for offline / mock testing
    STANDARD_SERVICES: Dict[str, str] = {
        "youtube": "https://www.youtube.com",
        "yt": "https://www.youtube.com",
        "google": "https://www.google.com",
        "github": "https://github.com",
        "reddit": "https://www.reddit.com",
        "spotify": "https://open.spotify.com",
        "twitter": "https://x.com",
        "x": "https://x.com",
        "wikipedia": "https://www.wikipedia.org",
        "amazon": "https://www.amazon.com",
        "steam": "https://store.steampowered.com",
        "gmail": "https://mail.google.com",
        "arxiv": "https://arxiv.org",
    }

    def __init__(self, platform: Optional[PlatformInterface] = None):
        self.platform = platform or get_platform()
        self._domain_cache: Dict[str, Tuple[str, str]] = {}

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        act = str(arguments.get("action", "")).strip().lower()
        if act == "search" and not arguments.get("open_in_browser"):
            return PermissionTier.SAFE
        return PermissionTier.SYSTEM_IMPACTING

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "open_search", "open_url", "open_domain"],
                    "description": "Browser action to perform.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query text for actions 'search' and 'open_search'.",
                },
                "site": {
                    "type": "string",
                    "description": "Specific website, service, or domain to search within (e.g. 'DODI Repacks', 'Steam', 'GitHub', 'Reddit', 'Wikipedia').",
                },
                "sites": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of websites, services, or domains to search across simultaneously (e.g. ['Steam', 'DODI Repacks', 'Reddit']).",
                },
                "domain": {
                    "type": "string",
                    "description": "Domain or service name for action 'open_domain' or 'open_search'.",
                },
                "url": {
                    "type": "string",
                    "description": "Full destination URL for action 'open_url'.",
                },
                "open_in_browser": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to also open the search page in the desktop browser window.",
                },
                "num_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Maximum search results to return.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def _resolve_site_domain(self, site_identifier: str) -> Tuple[str, str]:
        """Dynamically resolve any site name, phonetic moniker, or brand to its canonical domain and base URL.
        Returns: (netloc, base_url)
        """
        raw = site_identifier.strip()
        if not raw:
            return "", ""

        # 1. Full URL
        if raw.startswith("http://") or raw.startswith("https://"):
            parsed = urllib.parse.urlparse(raw)
            return parsed.netloc.lower(), f"{parsed.scheme}://{parsed.netloc}"

        # 2. Direct domain with dot (e.g. "dodi-repacks.site", "github.com")
        if "." in raw and " " not in raw:
            clean = raw.strip("/").lower()
            return clean, f"https://{clean}"

        clean_key = raw.lower()
        clean_norm = re.sub(r"[\s\-_]", "", clean_key)

        # 3. Protocol defaults (fast offline/mock fallback)
        if clean_key in self.STANDARD_SERVICES:
            base_url = self.STANDARD_SERVICES[clean_key]
            netloc = urllib.parse.urlparse(base_url).netloc.lower()
            return netloc, base_url
        if clean_norm in self.STANDARD_SERVICES:
            base_url = self.STANDARD_SERVICES[clean_norm]
            netloc = urllib.parse.urlparse(base_url).netloc.lower()
            return netloc, base_url

        # 4. Check dynamic cache
        if clean_key in self._domain_cache:
            return self._domain_cache[clean_key]

        # 5. Dynamic resolution via search
        raw_terms = [t for t in re.split(r"[\s\-_]+", clean_key) if len(t) >= 2]
        generic_domains = [
            "reddit.com", "youtube.com", "wikipedia.org", "twitter.com", "x.com",
            "facebook.com", "torrentfreak.com", "medium.com", "quora.com",
            "vimeo.com", "piratemap.org", "techradar.com", "theverge.com",
        ]

        for endpoint in ("https://lite.duckduckgo.com/lite/", "https://html.duckduckgo.com/html/"):
            try:
                q_data = urllib.parse.urlencode({"q": f"{raw} official site"}).encode("utf-8")
                req = urllib.request.Request(
                    endpoint,
                    data=q_data,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                )
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    content = resp.read().decode("utf-8", errors="replace")

                links = re.findall(r"<a[^>]+href=[\'\"]([^\'\"]+)[\'\"][^>]*class=[\'\"]result-link[\'\"]>([^<]+)</a>", content)
                if not links:
                    links = re.findall(r"<a[^>]+class=[\'\"]result-link[\'\"][^>]*href=[\'\"]([^\'\"]+)[\'\"]>([^<]+)</a>", content)

                candidates = []
                for raw_url, _ in links[:10]:
                    final_url = raw_url
                    if "/uddg=" in raw_url:
                        parsed_q = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                        if "uddg" in parsed_q:
                            final_url = parsed_q["uddg"][0]
                    p = urllib.parse.urlparse(final_url)
                    netloc = p.netloc.lower()
                    if not netloc:
                        continue
                    if netloc.startswith("www."):
                        netloc = netloc[4:]

                    is_generic = any(b in netloc for b in generic_domains)
                    if is_generic and not any(b in clean_key for b in ["reddit", "youtube", "wikipedia", "twitter", "x", "torrentfreak", "medium", "quora"]):
                        continue

                    # Score candidate by token matches against domain parts and phonetic variants
                    score = 0
                    dom_parts = re.split(r"[\.\-_]+", netloc)
                    for t in raw_terms:
                        t_stem = t.rstrip("s")
                        if any(t in part or t_stem in part for part in dom_parts):
                            score += 5
                        elif t in netloc:
                            score += 3
                        # Spoken consonant flap variation (e.g. t <-> d in American English speech)
                        t_flap_d = t.replace("t", "d")
                        t_flap_t = t.replace("d", "t")
                        if t_flap_d in netloc or t_flap_t in netloc:
                            score += 4

                    candidates.append((score, netloc))

                if candidates:
                    candidates.sort(key=lambda c: c[0], reverse=True)
                    best_netloc = candidates[0][1]
                    res = (best_netloc, f"https://{best_netloc}")
                    self._domain_cache[clean_key] = res
                    return res
            except Exception:
                continue

        # 6. Unresolved / Offline
        # Do not fabricate an arbitrary .com domain; return empty to allow honest failure reporting or search fallback
        return "", ""

    def _clean_query_for_site(self, query: str, site_identifier: str, netloc: str = "") -> str:
        """Strip site-specific names, monikers, and domain names from a query intended for that site.
        Prevents WordPress / site search failures caused by searching the site's own name on itself.
        Operates 100% dynamically on site_identifier and netloc tokens with zero hardcoded lists.
        """
        q = (query or "").strip()
        if not q:
            return ""

        terms = set()
        clean_site = site_identifier.lower().strip()
        if clean_site:
            terms.add(clean_site)
            for part in re.split(r"[\s\-_]+", clean_site):
                if len(part) >= 2:
                    terms.add(part)

        dom = (netloc or "").lower().strip()
        if dom:
            clean_dom = re.sub(r"^(?:https?://)?(?:www\.)?", "", dom).split("/")[0]
            terms.add(clean_dom)
            dom_name = clean_dom.split(".")[0]
            terms.add(dom_name)
            for part in re.split(r"[\-_\.]+", dom_name):
                if len(part) >= 2:
                    terms.add(part)

        for t in sorted(terms, key=len, reverse=True):
            q = re.sub(rf"\b{re.escape(t)}\b", "", q, flags=re.IGNORECASE)

        # Clean prepositions, conjunctions, and extra punctuation
        q = re.sub(r"\b(?:at|on|in|from|across|for|and|or)\b", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"[\?\.\!\,\;]+", " ", q)
        return re.sub(r"\s+", " ", q).strip()

    def _construct_search_url(self, base_url: str, netloc: str, query: str) -> str:
        """Construct standard search destination URL for a domain and query."""
        clean_q = (query or "").strip()
        if not clean_q:
            return base_url

        encoded_q = urllib.parse.quote_plus(clean_q)
        nl = netloc.lower()
        if "youtube.com" in nl:
            return f"https://www.youtube.com/results?search_query={encoded_q}"
        if "google.com" in nl:
            return f"https://www.google.com/search?q={encoded_q}"
        if "github.com" in nl:
            return f"https://github.com/search?q={encoded_q}"
        if "reddit.com" in nl:
            return f"https://www.reddit.com/search/?q={encoded_q}"
        if "amazon.com" in nl:
            return f"https://www.amazon.com/s?k={encoded_q}"
        if "wikipedia.org" in nl:
            return f"https://en.wikipedia.org/wiki/Special:Search?search={encoded_q}"
        if "steam" in nl:
            return f"https://store.steampowered.com/search/?term={encoded_q}"

        # Standard Web CMS / WordPress / Blog convention
        return f"{base_url}/?s={encoded_q}"

    def _launch_browser(self, target_url: str) -> bool:
        """Launch target URL in desktop browser with display environment guarantees."""
        url = target_url.strip()

        # Guard: check if target is a local file or file URI (redirect to native viewer)
        is_local_file = (
            url.startswith("file://")
            or url.startswith("/")
            or (len(url) > 2 and url[1] == ":" and url[2] in ("\\", "/"))
            or os.path.exists(url)
        )
        if is_local_file:
            if hasattr(self.platform, "open_file"):
                return self.platform.open_file(url)

        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"https://{url}"

        # 1. Try platform driver first (e.g. LinuxPlatform with GUI environment)
        if hasattr(self.platform, "open_url"):
            try:
                if self.platform.open_url(url):
                    return True
            except Exception:
                pass

        # 2. Standard Python fallback
        try:
            if webbrowser.open(url):
                return True
        except Exception:
            pass

        # 3. Direct OS launcher fallback if standard webbrowser.open failed
        if sys.platform != "win32":
            env = dict(os.environ)
            uid = os.getuid()
            rt = f"/run/user/{uid}"
            if "XDG_RUNTIME_DIR" not in env and os.path.isdir(rt):
                env["XDG_RUNTIME_DIR"] = rt
            if "DBUS_SESSION_BUS_ADDRESS" not in env:
                dbus_p = os.path.join(env.get("XDG_RUNTIME_DIR", rt), "bus")
                if os.path.exists(dbus_p):
                    env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={dbus_p}"
            if "WAYLAND_DISPLAY" not in env:
                runtime_d = env.get("XDG_RUNTIME_DIR", rt)
                for w in ["wayland-0", "wayland-1"]:
                    if os.path.exists(os.path.join(runtime_d, w)):
                        env["WAYLAND_DISPLAY"] = w
                        break
            if "DISPLAY" not in env:
                env["DISPLAY"] = ":0"

            for k, v in env.items():
                if k not in os.environ:
                    os.environ[k] = v

            if shutil.which("xdg-open"):
                try:
                    subprocess.Popen(["xdg-open", url], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except Exception:
                    pass

        elif sys.platform == "win32":
            try:
                os.startfile(url)
                return True
            except Exception:
                pass

        return False

    def execute(
        self,
        action: Optional[str] = None,
        query: Optional[str] = None,
        url: Optional[str] = None,
        domain: Optional[str] = None,
        site: Optional[str] = None,
        open_in_browser: bool = False,
        num_results: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        act = (action or "").strip().lower()
        if not act:
            if url:
                act = "open_url"
            elif domain and not query:
                act = "open_domain"
            else:
                act = "search"
        elif act in ("web_search", "search_web", "find"):
            act = "search"
        target_site = (site or domain or "").strip()

        try:
            if act == "open_url":
                target_url = (url or "").strip()
                if not target_url:
                    return ToolResult(success=False, output="", error="Parameter 'url' is required for 'open_url'.")

                # Check if target is a local file or file URI
                is_local = (
                    target_url.startswith("file://")
                    or target_url.startswith("/")
                    or (len(target_url) > 2 and target_url[1] == ":" and target_url[2] in ("\\", "/"))
                    or os.path.exists(target_url)
                )
                if is_local:
                    if hasattr(self.platform, "open_file"):
                        opened = self.platform.open_file(target_url)
                        if opened:
                            return ToolResult(
                                success=True,
                                output=f"Opened local file '{target_url}' in native desktop viewer.",
                                metadata={"url": target_url, "action": "open_file", "launched": True},
                            )
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Local file '{target_url}' cannot be opened in browser. Use app_control tool with action='open_file' and file_path='{target_url}' to open in native desktop viewer.",
                    )

                if hasattr(self.platform, "focus_window_or_application"):
                    self.platform.focus_window_or_application("browser")
                launched = self._launch_browser(target_url)
                return ToolResult(
                    success=launched,
                    output=f"Opened URL '{target_url}' in default browser.",
                    metadata={"url": target_url, "launched": launched},
                )

            elif act == "open_domain":
                d_key = target_site or (domain or "").strip()
                if not d_key:
                    return ToolResult(success=False, output="", error="Parameter 'domain' is required for 'open_domain'.")
                netloc, resolved_url = self._resolve_site_domain(d_key)
                if not resolved_url:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Could not resolve domain or service for '{d_key}'. Please specify the exact domain or URL.",
                    )
                if hasattr(self.platform, "focus_window_or_application"):
                    self.platform.focus_window_or_application(netloc)
                launched = self._launch_browser(resolved_url)
                return ToolResult(
                    success=launched,
                    output=f"Opened domain '{d_key}' ({resolved_url}) in default browser.",
                    metadata={"domain": d_key, "url": resolved_url, "launched": launched},
                )

            elif act == "open_search":
                clean_q, target_sites = self._extract_sites_and_query(query or "", site, kwargs.get("sites"), domain)
                if not clean_q:
                    return ToolResult(success=False, output="", error="Parameter 'query' is required for 'open_search'.")

                if len(target_sites) > 1:
                    opened_urls = []
                    for s in target_sites:
                        netloc, base_url = self._resolve_site_domain(s)
                        if base_url:
                            site_q = self._clean_query_for_site(clean_q, s, netloc)
                            search_url = self._construct_search_url(base_url, netloc, site_q)
                        else:
                            search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(f'{clean_q} {s}')}"
                        self._launch_browser(search_url)
                        opened_urls.append(search_url)
                    return ToolResult(
                        success=True,
                        output=f"Opened search for '{clean_q}' across {len(target_sites)} sites ({', '.join(target_sites)}) in default browser.",
                        metadata={"query": clean_q, "sites": target_sites, "urls": opened_urls, "launched": True},
                    )

                if target_sites:
                    target_s = target_sites[0]
                    netloc, base_url = self._resolve_site_domain(target_s)
                    if base_url:
                        site_q = self._clean_query_for_site(clean_q, target_s, netloc)
                        search_url = self._construct_search_url(base_url, netloc, site_q)
                        display_target = target_s
                    else:
                        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(f'{clean_q} {target_s}')}"
                        display_target = f"{target_s} (web search)"
                else:
                    search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(clean_q)}"
                    display_target = "web"

                launched = self._launch_browser(search_url)
                return ToolResult(
                    success=launched,
                    output=f"Opened search for '{clean_q}' on {display_target} ({search_url}) in default browser.",
                    metadata={"query": clean_q, "site": display_target, "url": search_url, "launched": launched},
                )

            elif act == "search":
                clean_q, target_sites = self._extract_sites_and_query(query or "", site, kwargs.get("sites"), domain)
                if not clean_q:
                    return ToolResult(success=False, output="", error="Parameter 'query' is required for web search.")

                # If multiple sites targeted, execute parallel multi-site search
                if len(target_sites) > 1:
                    return self._search_multi_sites(
                        target_sites,
                        clean_q,
                        max_results=max(1, min(10, num_results)),
                        open_browser=open_in_browser,
                    )

                # If single specific site or domain requested, execute site-specific search
                if target_sites:
                    return self._search_scoped_site(
                        target_sites[0],
                        clean_q,
                        max_results=max(1, min(10, num_results)),
                        open_browser=open_in_browser,
                    )

                # General web search
                res = self._search_duckduckgo(clean_q, max_results=max(1, min(10, num_results)))
                if open_in_browser:
                    ddg_url = f"https://duckduckgo.com/?q={urllib.parse.quote_plus(clean_q)}"
                    self._launch_browser(ddg_url)
                return res

            return ToolResult(success=False, output="", error=f"Unknown web_browser action: '{action}'.")

        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Browser action '{action}' failed: {str(exc)}")

    def _extract_sites_and_query(
        self,
        query: str,
        site: Optional[str] = None,
        sites: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """Dynamically extract clean search query and targeted site/domain list.
        Supports explicit 'sites' array, comma/'and' separated 'site' or 'domain' strings,
        and natural-language query multi-site directives (e.g. 'search for Call of Duty on Steam, DODI Repacks, and Reddit').
        """
        resolved_sites: List[str] = []

        # 1. Explicit list of sites passed in 'sites'
        if sites and isinstance(sites, (list, tuple)):
            for s in sites:
                if isinstance(s, str) and s.strip():
                    clean_s = s.strip()
                    if clean_s.lower() not in [x.lower() for x in resolved_sites]:
                        resolved_sites.append(clean_s)

        # 2. Delimited sites in 'site' or 'domain' parameter
        target_site_str = (site or domain or "").strip()
        if target_site_str:
            raw_parts = re.split(r"[,;]|\s+and\s+", target_site_str)
            for p in raw_parts:
                clean_p = p.strip()
                if clean_p and clean_p.lower() not in [x.lower() for x in resolved_sites]:
                    resolved_sites.append(clean_p)

        clean_query = (query or "").strip()

        # 3. Natural language query multi-site directives if query ends with 'on/in/across/at/from <Sites>'
        if len(resolved_sites) <= 1 and clean_query:
            clean_q_stripped = re.sub(r"[\?\.\!\,\;]+$", "", clean_query).strip()
            m = re.search(
                r"\b(?:on|in|across|at|from)\s+((?:[A-Za-z0-9\.\-\s]+,\s*)+and\s+[A-Za-z0-9\.\-\s]+|[A-Za-z0-9\.\-\s]+\s+and\s+[A-Za-z0-9\.\-\s]+|[A-Za-z0-9\.\-\s]+)$",
                clean_q_stripped,
                re.IGNORECASE,
            )
            if m:
                extracted_str = m.group(1).strip()
                extracted_parts = [p.strip() for p in re.split(r"[,;]|\s+and\s+", extracted_str) if p.strip()]
                if len(extracted_parts) >= 2 or (len(extracted_parts) == 1 and not resolved_sites):
                    q_lead = clean_q_stripped[:m.start()].strip()
                    if q_lead:
                        clean_query = q_lead
                        for ep in extracted_parts:
                            if ep.lower() not in [x.lower() for x in resolved_sites]:
                                resolved_sites.append(ep)

        # Clean conversational query prefixes like 'can you find', 'search for ', 'find ', etc.
        clean_query = re.sub(
            r"^(?:can\s+you\s+(?:please\s+)?(?:find|search|look\s+up|get|show|recommend)|could\s+you\s+(?:please\s+)?(?:find|search|look\s+up|get|show|recommend)|please\s+(?:find|search|look\s+up|get|show|recommend)|search\s+for|search|find|lookup|look\s+up|recommend|show\s+me)\s+",
            "",
            clean_query,
            flags=re.IGNORECASE,
        ).strip()
        clean_query = re.sub(r"^(?:some|a\s+few|any)\s+", "", clean_query, flags=re.IGNORECASE).strip()
        clean_query = re.sub(r"\s+(?:to\s+(?:play|watch|read|listen\s+to|download|try|buy|use))$", "", clean_query, flags=re.IGNORECASE).strip()
        clean_query = re.sub(r"[\?\.\!\,\;]+$", "", clean_query).strip()

        # If target sites are resolved, strip their monikers from clean_query dynamically
        if resolved_sites:
            site_terms = set()
            for s in resolved_sites:
                s_clean = s.lower().strip()
                if s_clean:
                    site_terms.add(s_clean)
                # Split camelCase / PascalCase into component tokens (e.g. FitGirlRepack -> Fit, Girl, Repack)
                words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", s)
                for w in words:
                    w_lower = w.lower().strip()
                    if len(w_lower) >= 2:
                        site_terms.add(w_lower)
                # Also split on whitespace and punctuation
                for part in re.split(r"[\s\.\-_]+", s_clean):
                    if len(part) >= 2:
                        site_terms.add(part)

            for st in sorted(site_terms, key=len, reverse=True):
                clean_query = re.sub(rf"\b{re.escape(st)}\b", "", clean_query, flags=re.IGNORECASE).strip()
            clean_query = re.sub(r"\b(?:at|on|in|from|across|for|and|or)\b", " ", clean_query, flags=re.IGNORECASE).strip()
            clean_query = re.sub(r"\s+", " ", clean_query).strip()

        return clean_query, resolved_sites

    def _search_multi_sites(
        self,
        sites: List[str],
        query: str,
        max_results: int = 5,
        open_browser: bool = False,
    ) -> ToolResult:
        """Search across multiple sites concurrently using ThreadPoolExecutor and aggregate results cleanly."""
        clean_q = query.strip()
        results_by_site: Dict[str, List[Dict[str, str]]] = {}
        site_search_urls: Dict[str, str] = {}
        total_found = 0

        def _do_site_search(target_site: str) -> Tuple[str, ToolResult]:
            res = self._search_scoped_site(target_site, clean_q, max_results=max_results, open_browser=open_browser)
            return target_site, res

        max_workers = min(len(sites), 5)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="maddy_multi_search") as executor:
            futures = [executor.submit(_do_site_search, s) for s in sites]
            for fut in as_completed(futures):
                try:
                    s_name, s_res = fut.result()
                    s_data = s_res.metadata.get("results", []) if s_res.metadata else []
                    results_by_site[s_name] = s_data
                    if s_res.metadata and s_res.metadata.get("search_url"):
                        site_search_urls[s_name] = s_res.metadata["search_url"]
                    total_found += len(s_data)
                except Exception:
                    pass

        lines = [f"Found {total_found} result(s) for '{clean_q}' across {len(sites)} sites ({', '.join(sites)}):"]
        all_results_flat = []

        for s in sites:
            s_hits = results_by_site.get(s, [])
            clean_dom = s
            netloc, _ = self._resolve_site_domain(s)
            if netloc:
                clean_dom = netloc
            lines.append(f"\n=== [{s}] ({clean_dom}) ===")
            if s_hits:
                for idx, hit in enumerate(s_hits, start=1):
                    h_title = hit.get("title", "Result")
                    h_url = hit.get("url", "")
                    if h_url:
                        lines.append(f"{idx}. [{h_title}]({h_url})")
                        lines.append(f"   URL: {h_url}")
                    else:
                        lines.append(f"{idx}. {h_title}")
                    all_results_flat.append(hit)
            else:
                lines.append(f"   No direct hits found on {clean_dom}.")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={
                "query": clean_q,
                "sites": sites,
                "results_by_site": results_by_site,
                "search_urls": site_search_urls,
                "results": all_results_flat,
                "count": total_found,
                "multi_site": True,
            },
        )

    def _search_scoped_site(self, site_identifier: str, query: str, max_results: int = 5, open_browser: bool = False) -> ToolResult:
        """Perform site-scoped search dynamically: attempts direct site search first,
        falling back to site-scoped search engine query if blocked or structured parsing fails.
        """
        netloc, base_url = self._resolve_site_domain(site_identifier)
        clean_domain = netloc or re.sub(r"^https?://(?:www\.)?", "", base_url).strip("/")
        eff_query = self._clean_query_for_site(query, site_identifier, clean_domain)
        search_page_url = self._construct_search_url(base_url, clean_domain, eff_query)

        # 1. Attempt direct semantic parse of site search endpoint
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        results: List[Dict[str, str]] = []
        try:
            req = urllib.request.Request(search_page_url, headers=headers)
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                content = resp.read().decode("utf-8", errors="replace")

            # Match standard WordPress/CMS entry-title pattern
            posts = re.findall(r'<h[1-3][^>]*class=[\'"][^\'"]*entry-title[^\'"]*[\'"][^>]*><a[^>]+href=[\'"]([^\'"]+)[\'"][^>]*>([^<]+)</a>', content)
            if not posts:
                # Match general article/result heading link pattern
                posts = re.findall(r'<a[^>]+class=[\'"][^\'"]*(?:title|result-link|entry-link)[^\'"]*[\'"][^>]*href=[\'"]([^\'"]+)[\'"][^>]*>([^<]+)</a>', content)

            for p_url, p_title in posts[:max_results]:
                full_p_url = p_url.strip()
                if full_p_url.startswith("/"):
                    full_p_url = f"{base_url}{full_p_url}"
                results.append({
                    "title": html.unescape(p_title).strip(),
                    "url": full_p_url,
                    "snippet": f"Release on {clean_domain}.",
                })
        except Exception:
            pass

        # 2. If direct site parse yielded results, return them
        display_q = eff_query or query
        if results:
            if open_browser:
                self._launch_browser(search_page_url)
            lines = [f"Found {len(results)} result(s) for '{display_q}' on {clean_domain}:"]
            if open_browser:
                lines.append(f"Browser opened at: {search_page_url}")
            for idx, item in enumerate(results, start=1):
                lines.append(f"{idx}. [{item['title']}]({item['url']})")
                lines.append(f"   URL: {item['url']}")

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={
                    "query": display_q,
                    "site": site_identifier,
                    "domain": clean_domain,
                    "search_url": search_page_url,
                    "results": results,
                    "count": len(results),
                },
            )

        # 3. Fallback: Query DuckDuckGo Lite with site: scope
        if open_browser:
            self._launch_browser(search_page_url)
        scoped_q = f"site:{clean_domain} {display_q}"
        ddg_res = self._search_duckduckgo(scoped_q, max_results=max_results)
        if ddg_res.success:
            prefix = f"Results for '{display_q}' on {clean_domain}"
            if open_browser:
                prefix += f" (Opened in browser: {search_page_url})"
            ddg_res.output = f"{prefix}:\n{ddg_res.output}"
            ddg_res.metadata["search_url"] = search_page_url
            ddg_res.metadata["site"] = site_identifier
            ddg_res.metadata["domain"] = clean_domain
        return ddg_res

    def _search_duckduckgo(self, query: str, max_results: int = 5) -> ToolResult:
        """Perform search query via DuckDuckGo Lite."""
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://lite.duckduckgo.com/lite/",
            data=data,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                raw_html = resp.read().decode("utf-8", errors="replace")

            links = re.findall(r"<a[^>]+href=\"([^\"]+)\"[^>]*class='result-link'>([^<]+)</a>", raw_html)
            snippets = re.findall(r"<td[^>]*class='result-snippet'>\s*([^<]+)", raw_html)

            results: List[Dict[str, str]] = []
            for (raw_url, raw_title), raw_snip in zip(links[:max_results], snippets[:max_results]):
                clean_title = html.unescape(raw_title).strip()
                clean_snip = html.unescape(raw_snip).strip()
                final_url = raw_url
                if "/uddg=" in raw_url:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                    if "uddg" in parsed:
                        final_url = parsed["uddg"][0]
                results.append({
                    "title": clean_title,
                    "url": final_url,
                    "snippet": clean_snip,
                })

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No search results returned for query: '{query}'.",
                    metadata={"query": query, "results": []},
                )

            lines = [f"Found {len(results)} search result(s) for '{query}':"]
            for idx, item in enumerate(results, start=1):
                lines.append(f"{idx}. {item['title']}")
                lines.append(f"   URL: {item['url']}")
                lines.append(f"   Snippet: {item['snippet']}")

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={"query": query, "results": results, "count": len(results)},
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Web search for '{query}' failed: {str(exc)}",
            )


class WebSearchTool(BaseTool):
    """Direct web and multi-site search tool for M.A.D.D.Y.
    Executes web searches across general search or specific sites (e.g. FitGirl Repacks, DODI Repacks, Steam, Reddit, Wikipedia).
    """

    name = "web_search"
    description = (
        "Search the web or specific sites (e.g. FitGirl Repacks, DODI Repacks, Steam, Reddit, Wikipedia) "
        "and return structured results with titles and links."
    )
    tier = PermissionTier.SAFE

    def __init__(self, platform: Optional[PlatformInterface] = None, browser_tool: Optional[WebBrowserTool] = None):
        self.platform = platform or get_platform()
        self.browser_tool = browser_tool or WebBrowserTool(platform=self.platform)

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        if arguments.get("open_in_browser"):
            return PermissionTier.SYSTEM_IMPACTING
        return PermissionTier.SAFE

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text (e.g. 'popular games', 'action RPGs').",
                },
                "site": {
                    "type": "string",
                    "description": "Specific website, service, or domain to search within (e.g. 'FitGirl Repacks', 'DODI Repacks', 'Steam').",
                },
                "sites": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of websites, services, or domains to search across simultaneously (e.g. ['FitGirl Repacks', 'DODI Repacks']).",
                },
                "num_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Maximum search results to return.",
                },
                "open_in_browser": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to also open the search in the desktop browser window.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        return self.browser_tool.operation_tier(arguments)

    def execute(self, query: str = "", **kwargs: Any) -> ToolResult:
        return self.browser_tool.execute(action="search", query=query, **kwargs)

