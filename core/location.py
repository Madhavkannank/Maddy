"""Dynamic Runtime Location Detection for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You)

Detects the operator's actual current location dynamically at runtime using
network/IP geolocation with local system timezone fallback.
Never uses hardcoded locations (e.g. Chennai).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# In-memory cache for geolocation results
_CACHED_LOCATION: Optional[Dict[str, Any]] = None
_CACHE_TIMESTAMP: float = 0.0
_CACHE_TTL_SECONDS: float = 3600.0  # 1 hour cache
_LAST_KNOWN_LOCATION: Optional[Dict[str, Any]] = None


def get_current_location(force_refresh: bool = False) -> Dict[str, Any]:
    """Dynamically determine the current runtime location of the system.

    Priorities:
    1. Real-time IP geolocation (http://ip-api.com/json/) with 3.0s timeout.
    2. Fallback secondary IP geolocation (https://ipwho.is/).
    3. Last-known-good location if available.
    4. Local system timezone inspection.

    Never defaults to static hardcoded locations.
    """
    global _CACHED_LOCATION, _CACHE_TIMESTAMP, _LAST_KNOWN_LOCATION

    now = time.time()
    if not force_refresh and _CACHED_LOCATION is not None:
        if (now - _CACHE_TIMESTAMP) < _CACHE_TTL_SECONDS:
            return dict(_CACHED_LOCATION)

    # 1. Real-time network IP geolocation
    for url, key_map in [
        ("http://ip-api.com/json/", {"city": "city", "region": "regionName", "country": "country", "tz": "timezone"}),
        ("https://ipwho.is/", {"city": "city", "region": "region", "country": "country", "tz": "timezone.id"}),
    ]:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                city = data.get(key_map["city"], "") if isinstance(data.get(key_map["city"]), str) else ""
                region = data.get(key_map["region"], "") if isinstance(data.get(key_map["region"]), str) else ""
                country = data.get(key_map["country"], "") if isinstance(data.get(key_map["country"]), str) else ""
                tz_val = data.get("timezone", "")
                if isinstance(tz_val, dict):
                    tz = tz_val.get("id", "")
                else:
                    tz = str(tz_val or "")

                if city:
                    city = city.strip()
                    region = region.strip()
                    country = country.strip()
                    parts = [p for p in [city, region, country] if p]
                    display_str = ", ".join(parts) if parts else city

                    res = {
                        "city": city,
                        "region": region,
                        "country": country,
                        "timezone": tz,
                        "display": display_str,
                        "source": "network_ip",
                    }
                    _CACHED_LOCATION = res
                    _LAST_KNOWN_LOCATION = res
                    _CACHE_TIMESTAMP = now
                    return dict(res)
        except Exception as e:
            logger.debug(f"Dynamic IP geolocation lookup failed for {url}: {e}")

    # 2. Return last known good location if network had transient failure
    if _LAST_KNOWN_LOCATION is not None:
        return dict(_LAST_KNOWN_LOCATION)

    # 2. Local system timezone inspection fallback
    tz_name = ""
    try:
        import datetime
        tz_name = datetime.datetime.now().astimezone().tzname() or ""
    except Exception:
        pass

    try:
        import time as _t
        if not tz_name and hasattr(_t, "tzname"):
            tz_name = _t.tzname[0]
    except Exception:
        pass

    res = {
        "city": "",
        "region": "",
        "country": "",
        "timezone": tz_name,
        "display": f"Local System ({tz_name})" if tz_name else "Local System",
        "source": "system_timezone",
    }
    _CACHED_LOCATION = res
    _CACHE_TIMESTAMP = now
    return dict(res)


def get_current_city(fallback: str = "Local") -> str:
    """Retrieve the current detected city name dynamically."""
    loc = get_current_location()
    return loc.get("city") or loc.get("region") or fallback


def get_location_summary() -> str:
    """Retrieve human-readable location summary string (e.g. 'Coimbatore, Tamil Nadu, India')."""
    loc = get_current_location()
    return loc.get("display") or "Local System"
