"""Live Weather Information Tool for M.A.D.D.Y. Agent Core (Modular Autonomous Digital Daemon for You).

Queries Open-Meteo Geocoding and Weather Forecast APIs using httpx with timeout bounding.
Returns structured temperature in Celsius, WMO weather descriptions, windspeed, humidity,
and location metadata.
Deterministic Tier: SAFE (auto-allowed).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Standard WMO Weather interpretation codes (WW)
WMO_WEATHER_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_SECONDS = 10.0


class WeatherTool(BaseTool):
    """Retrieves real-time weather conditions and forecasts using Open-Meteo API."""

    name = "weather"
    description = (
        "Query real-time weather observations including temperature in Celsius, weather condition, "
        "relative humidity, wind speed, and location metadata for any city or location."
    )
    tier = PermissionTier.SAFE

    def __init__(self, client: Optional[httpx.Client] = None):
        self._client = client

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City or location name to query weather for (e.g. 'Tokyo', 'London', 'Paris').",
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        }

    def execute(
        self,
        city: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        if city == "":
            return ToolResult(
                success=False,
                output="",
                error="A city name must be specified.",
            )

        target = (city or location or kwargs.get("query") or "").strip()
        if not target or target.lower() in ("here", "current", "current location", "my location", "local", "me"):
            try:
                from core.location import get_current_city
                detected = get_current_city()
                if detected and detected.lower() != "local":
                    target = detected
            except Exception:
                pass

        if not target:
            return ToolResult(
                success=False,
                output="",
                error="A city name must be specified or current location must be detectable.",
            )

        client = self._client or httpx.Client(
            timeout=DEFAULT_TIMEOUT_SECONDS,
            headers={"User-Agent": "MADDY/2.0"},
        )

        close_client = self._client is None

        try:
            # 1. Geocode city name to lat/lon coordinates
            geo_params = {
                "name": target,
                "count": 1,
                "language": "en",
                "format": "json",
            }
            geo_resp = client.get(GEOCODING_URL, params=geo_params)
            if geo_resp.status_code != 200:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Geocoding API failed with status code {geo_resp.status_code}: {geo_resp.text}",
                    metadata={"city": target},
                )

            geo_data = geo_resp.json()
            results = geo_data.get("results", [])
            if not results:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Could not find coordinates for city: '{target}'.",
                    metadata={"city": target},
                )

            best = results[0]
            city_name = best.get("name", target)
            latitude = float(best["latitude"])
            longitude = float(best["longitude"])
            country = best.get("country", "")
            admin1 = best.get("admin1", "")
            timezone = best.get("timezone", "auto")

            # 2. Query Open-Meteo Forecast API
            forecast_params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            }
            forecast_resp = client.get(FORECAST_URL, params=forecast_params)
            if forecast_resp.status_code != 200:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Weather forecast API failed with status code {forecast_resp.status_code}: {forecast_resp.text}",
                    metadata={
                        "city": city_name,
                        "latitude": latitude,
                        "longitude": longitude,
                    },
                )

            forecast_data = forecast_resp.json()
            current = forecast_data.get("current", {})

            temperature_c = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            weather_code = current.get("weather_code", 0)
            condition = WMO_WEATHER_CODES.get(weather_code, f"Code {weather_code}")
            wind_speed = current.get("wind_speed_10m")

            # Build human-readable output summary
            location_str = f"{city_name}, {country}" if country else city_name
            summary = (
                f"Weather in {location_str}: {condition}. "
                f"Temperature: {temperature_c}°C. "
                f"Humidity: {humidity}%. "
                f"Wind Speed: {wind_speed} km/h."
            )

            metadata = {
                "city": city_name,
                "country": country,
                "region": admin1,
                "latitude": latitude,
                "longitude": longitude,
                "temperature_c": temperature_c,
                "condition": condition,
                "weather_code": weather_code,
                "humidity_percent": humidity,
                "wind_speed_kmh": wind_speed,
                "timezone": forecast_data.get("timezone", timezone),
            }

            return ToolResult(
                success=True,
                output=summary,
                metadata=metadata,
            )

        except httpx.TimeoutException as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Weather API request timed out after {DEFAULT_TIMEOUT_SECONDS}s: {str(exc)}",
                metadata={"city": target},
            )
        except httpx.RequestError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Weather API network error: {type(exc).__name__}: {str(exc)}",
                metadata={"city": target},
            )
        except Exception as exc:
            logger.exception("Unexpected error querying weather for %s", target)
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to query weather for '{target}': {str(exc)}",
                metadata={"city": target},
            )
        finally:
            if close_client:
                client.close()
