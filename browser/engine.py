"""
Browser Engine for fetching web content, parsing HTML into structured markdown text, and web searching.
"""
import html
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

class SimpleHTMLTextExtractor(HTMLParser):
    """Parses HTML DOM elements into clean plain text and extracts hyper-links."""

    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self.links: List[Dict[str, str]] = []
        self.title: str = ""
        self._in_title = False
        self._in_script_or_style = False
        self._current_href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style"):
            self._in_script_or_style = True
        elif tag_lower == "title":
            self._in_title = True
        elif tag_lower == "a":
            attr_dict = dict(attrs)
            if "href" in attr_dict and attr_dict["href"]:
                self._current_href = attr_dict["href"]

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style"):
            self._in_script_or_style = False
        elif tag_lower == "title":
            self._in_title = False
        elif tag_lower == "a":
            self._current_href = None
        elif tag_lower in ("p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr"):
            self.text_parts.append("\n")

    def handle_data(self, data: str):
        if self._in_script_or_style:
            return
        cleaned = data.strip()
        if not cleaned:
            return

        if self._in_title:
            self.title += cleaned + " "
        else:
            self.text_parts.append(cleaned + " ")
            if self._current_href:
                self.links.append({"text": cleaned, "url": self._current_href})

    def get_text(self) -> str:
        raw = "".join(self.text_parts)
        # Collapse multiple empty lines
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return "\n".join(lines)

class BrowserEngine:
    """
    Engine providing headless browser navigation, page parsing, and web search capabilities.
    """

    def __init__(self, user_agent: str = "MADDY-BrowserEngine/1.0"):
        self.user_agent = user_agent

    def navigate(self, url: str, timeout: float = 15.0) -> Dict[str, Any]:
        """
        Navigate to a URL and return structured page details.
        """
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        req = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"},
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                raw_bytes = resp.read(1024 * 1024)  # Limit 1 MB page body
                encoding = resp.headers.get_param("charset") or "utf-8"
                html_text = raw_bytes.decode(encoding, errors="replace")

            extractor = SimpleHTMLTextExtractor()
            extractor.feed(html_text)

            return {
                "url": url,
                "status": status,
                "title": extractor.title.strip() or "Untitled Page",
                "content": extractor.get_text()[:4000],  # Bounded content text
                "link_count": len(extractor.links),
                "top_links": extractor.links[:10],
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "type": "NAVIGATION_ERROR",
                    "message": f"Failed to navigate to '{url}': {str(e)}",
                },
            }

    def search(self, query: str, max_results: int = 5, timeout: float = 15.0) -> Dict[str, Any]:
        """
        Perform a web search query and extract search result items.
        """
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        # Use standard modern browser User-Agent so DDG does not block or return empty HTML
        ua = (
            self.user_agent
            if "Mozilla" in self.user_agent
            else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        req = urllib.request.Request(
            search_url,
            headers={
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html_text = resp.read(512 * 1024).decode("utf-8", errors="replace")

            results = []
            # Modern DuckDuckGo HTML uses result__a for title links and result__snippet for summaries
            link_pattern = re.compile(
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                re.DOTALL,
            )
            snippet_pattern = re.compile(
                r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
                re.DOTALL,
            )

            raw_titles = link_pattern.findall(html_text)
            snippets = snippet_pattern.findall(html_text)

            # Fallback patterns if classes change
            if not raw_titles:
                raw_titles = re.findall(r'<a[^>]+class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, re.DOTALL)
            if not raw_titles:
                # Generic result link fallback
                raw_titles = re.findall(r'<a[^>]+rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, re.DOTALL)

            for i in range(min(len(raw_titles), max_results)):
                raw_u, title_html = raw_titles[i]
                clean_title = re.sub(r"<[^>]+>", "", title_html).strip()
                clean_url = urllib.parse.unquote(raw_u.strip())
                snippet_text = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""

                # Handle DDG redirect links (uddg parameter)
                if "uddg=" in clean_url:
                    parsed_q = urllib.parse.parse_qs(urllib.parse.urlparse(clean_url).query)
                    clean_url = parsed_q.get("uddg", [clean_url])[0]
                elif clean_url.startswith("//duckduckgo.com/l/?"):
                    parsed_q = urllib.parse.parse_qs(clean_url.split("?", 1)[1] if "?" in clean_url else "")
                    clean_url = parsed_q.get("uddg", [clean_url])[0]

            if not results:
                # Fallback to DuckDuckGo Lite which is ultra-resilient and bypasses anti-bot JS challenges
                try:
                    lite_url = "https://lite.duckduckgo.com/lite/"
                    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
                    lite_req = urllib.request.Request(
                        lite_url,
                        data=data,
                        headers={"User-Agent": ua},
                    )
                    with urllib.request.urlopen(lite_req, timeout=timeout) as lite_resp:
                        lite_html = lite_resp.read(512 * 1024).decode("utf-8", errors="replace")
                    
                    lite_matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', lite_html)
                    for href, text in lite_matches:
                        if href.startswith("http") and "duckduckgo.com" not in href:
                            results.append({
                                "title": html.unescape(re.sub(r"<[^>]+>", "", text).strip()),
                                "url": href.strip(),
                                "snippet": "",
                            })
                            if len(results) >= max_results:
                                break
                except Exception as e_lite:
                    logger.debug(f"DDG Lite search fallback error: {e_lite}")

            if not results:
                # Fallback to Bing organic search with base64 target unwrapping
                try:
                    import base64
                    bing_url = f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}"
                    bing_req = urllib.request.Request(
                        bing_url,
                        headers={"User-Agent": ua},
                    )
                    with urllib.request.urlopen(bing_req, timeout=timeout) as bing_resp:
                        bing_html = bing_resp.read(512 * 1024).decode("utf-8", errors="replace")
                    
                    bing_matches = re.findall(
                        r'<li[^>]+class="b_algo"[^>]*>.*?<h2>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                        bing_html,
                        re.DOTALL,
                    )
                    for raw_href, raw_title in bing_matches:
                        target_url = raw_href
                        # Bing embeds the actual destination in base64 under &u=a1...
                        m_b64 = re.search(r'[?&](?:amp;)?u=a1([a-zA-Z0-9_\-\+/=]+)', raw_href)
                        if m_b64:
                            b64_str = m_b64.group(1)
                            b64_str += "=" * (-len(b64_str) % 4)
                            try:
                                decoded = base64.b64decode(b64_str.replace("-", "+").replace("_", "/")).decode("utf-8", errors="ignore")
                                if decoded.startswith("http"):
                                    target_url = decoded
                            except Exception:
                                pass
                        
                        clean_t = html.unescape(re.sub(r"<[^>]+>", "", raw_title).strip())
                        results.append({
                            "title": clean_t,
                            "url": target_url,
                            "snippet": "",
                        })
                        if len(results) >= max_results:
                            break
                except Exception as e_bing:
                    logger.debug(f"Bing search fallback error: {e_bing}")

            return {
                "query": query,
                "result_count": len(results),
                "results": results,
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "type": "SEARCH_ERROR",
                    "message": f"Failed to perform search query '{query}': {str(e)}",
                },
            }
