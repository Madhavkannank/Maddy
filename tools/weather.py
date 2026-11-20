"""
Live Weather and Geolocation Tools for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides real-time local and global weather, temperature, humidity, wind, and city geocoding
using Open-Meteo API and IP Geolocation (100% free, zero API keys required).
"""
import json
import urllib.request
import urllib.parse
from typing import Any, Dict, Optional, Tuple
from tools.base import RiskLevel, Tool
from tools.registry import ToolRegistry

_CACHED_LOCATION: Optional[Dict[str, Any]] = None
_LOCATION_OVERRIDE: Optional[str] = None

WMO_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def set_location_override(city_name: str) -> Optional[Dict[str, Any]]:
    """Temporarily override the current detected city using an explicit user correction."""
    global _CACHED_LOCATION, _LOCATION_OVERRIDE
    clean = (city_name or "").strip()
    if not clean:
        return None
    geocoded = geocode_city(clean)
    if not geocoded:
        return None
    city_name_norm, lat, lon, country = geocoded
    _LOCATION_OVERRIDE = clean
    _CACHED_LOCATION = {
        "city": city_name_norm,
        "region": "",
        "country": country,
        "latitude": lat,
        "longitude": lon,
    }
    return _CACHED_LOCATION


def clear_location_override() -> None:
    """Clear any explicit location override so the system falls back to network geolocation."""
    global _CACHED_LOCATION, _LOCATION_OVERRIDE
    _LOCATION_OVERRIDE = None
    _CACHED_LOCATION = None


def get_current_location() -> Optional[Dict[str, Any]]:
    """Fetch the user's live geographic location. Prefer an explicit user-provided override, then network geolocation."""
    global _CACHED_LOCATION, _LOCATION_OVERRIDE
    if _LOCATION_OVERRIDE:
        geocoded = geocode_city(_LOCATION_OVERRIDE)
        if geocoded:
            city_name_norm, lat, lon, country = geocoded
            _CACHED_LOCATION = {
                "city": city_name_norm,
                "region": "",
                "country": country,
                "latitude": lat,
                "longitude": lon,
            }
            return _CACHED_LOCATION

    if _CACHED_LOCATION:
        return _CACHED_LOCATION

    # Never trust stored location preferences as a fixed home city; live network location is authoritative.
    try:
        from memory.user_preferences import UserPreferences
        prefs = UserPreferences()
        for key in ("user.location", "user.city", "user.home_city"):
            prefs.delete_preference(key)
    except Exception:
        pass

    # Primary: ipapi.co
    try:
        req = urllib.request.Request(
            "https://ipapi.co/json/",
            headers={"User-Agent": "Maddy/1.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("city") and data.get("latitude") and data.get("longitude"):
                loc = {
                    "city": data.get("city"),
                    "region": data.get("region", ""),
                    "country": data.get("country_name", ""),
                    "latitude": float(data["latitude"]),
                    "longitude": float(data["longitude"]),
                }
                _CACHED_LOCATION = loc
                return loc
    except Exception:
        pass

    # Secondary fallback: ip-api.com
    try:
        req = urllib.request.Request(
            "http://ip-api.com/json/",
            headers={"User-Agent": "Maddy/1.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("city") and data.get("lat") and data.get("lon"):
                loc = {
                    "city": data.get("city"),
                    "region": data.get("regionName", ""),
                    "country": data.get("country", ""),
                    "latitude": float(data["lat"]),
                    "longitude": float(data["lon"]),
                }
                _CACHED_LOCATION = loc
                return loc
    except Exception:
        pass

    return None


def geocode_city(city_name: str) -> Optional[Tuple[str, float, float, str]]:
    """Geocode a city name into (normalized_name, latitude, longitude, country)."""
    clean_city = city_name.strip()
    encoded = urllib.parse.quote(clean_city)
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=en&format=json"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Maddy/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                best = results[0]
                name = best.get("name", clean_city)
                lat = float(best["latitude"])
                lon = float(best["longitude"])
                country = best.get("country", "")
                return (name, lat, lon, country)
    except Exception:
        pass

    return None


def get_weather(city: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve current weather observation from Open-Meteo API."""
    target_city = (city or "").strip()
    is_current_loc = False

    if not target_city or target_city.lower() in ("here", "current", "my location", "today", "now"):
        loc = get_current_location()
        if not loc:
            return {
                "success": False,
                "error": "LOCATION_UNAVAILABLE",
                "summary": "I couldn't determine your current location automatically. Please specify a city name.",
            }
        city_name = loc["city"]
        lat = loc["latitude"]
        lon = loc["longitude"]
        country = loc.get("country", "")
        is_current_loc = True
    else:
        geocoded = geocode_city(target_city)
        if geocoded:
            city_name, lat, lon, country = geocoded
        else:
            return {
                "success": False,
                "error": "CITY_NOT_FOUND",
                "summary": f"Could not find coordinates for '{target_city}'. Please check the city name.",
            }

    # Query Open-Meteo Current Weather
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        f"&timezone=auto"
    )

    try:
        req = urllib.request.Request(weather_url, headers={"User-Agent": "Maddy/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            w_data = json.loads(resp.read().decode("utf-8"))
            current = w_data.get("current", {})

            temp_c = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            code = current.get("weather_code", 0)
            condition = WMO_WEATHER_CODES.get(code, "clear")
            wind_speed = current.get("wind_speed_10m")

            temp_str = f"{round(temp_c)} degrees" if temp_c is not None else "normal"
            humidity_str = f"{humidity}%" if humidity is not None else "normal"

            summary = f"In {city_name}, it's currently {temp_str} and {condition} with {humidity_str} humidity."
            if wind_speed and wind_speed > 15:
                summary += f" Winds are at {round(wind_speed)} km/h."

            return {
                "success": True,
                "city": city_name,
                "country": country,
                "temperature_c": temp_c,
                "condition": condition,
                "humidity_percent": humidity,
                "wind_speed_kmh": wind_speed,
                "summary": summary,
                "is_current_location": is_current_loc,
            }
    except Exception as err:
        return {
            "success": False,
            "error": str(err),
            "summary": f"Could not retrieve weather data for {city_name}: {str(err)}",
        }


# ----------------------------------------------------------------------
# Tool handlers for ToolRegistry
# ----------------------------------------------------------------------

def _handle_get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    city = args.get("city")
    return get_weather(city)


def _handle_get_location(args: Dict[str, Any]) -> Dict[str, Any]:
    loc = get_current_location()
    if not loc:
        return {
            "success": False,
            "error": "LOCATION_UNAVAILABLE",
            "summary": "Could not determine your current location. Please check your network connection.",
        }
    return {
        "success": True,
        "city": loc["city"],
        "region": loc.get("region", ""),
        "country": loc.get("country", ""),
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "summary": f"Your current detected location is {loc['city']}, {loc.get('region', '')}, {loc.get('country', '')}.".replace(", ,", ","),
    }


def register_weather_tools(registry: ToolRegistry) -> None:
    """Register weather and location tools in the ToolRegistry."""
    weather_tool = Tool(
        name="weather.get",
        description="Retrieve real-time weather, temperature, humidity, and condition for any city or the user's current location.",
        parameter_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g. 'Paris', 'London', 'Tokyo'). Leave empty for user's current location.",
                }
            },
            "required": [],
        },
        handler=_handle_get_weather,
        risk_level=RiskLevel.LOW,
    )

    location_tool = Tool(
        name="location.get",
        description="Retrieve the user's current detected geographical location (city, region, country, coordinates).",
        parameter_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=_handle_get_location,
        risk_level=RiskLevel.LOW,
    )

    registry.register(weather_tool)
    registry.register(location_tool)
