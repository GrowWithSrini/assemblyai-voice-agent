"""Tools the agent can call.

Each `@tool` function's signature + docstring is what LangChain serializes to a
JSON-Schema function definition and sends to the model. Keep names and docstrings
precise — that text is the model's only guide to when and how to call them.

`get_weather` returns deterministic mock data (no external API / key); swap the body
for a real request when you have a provider.
"""

from __future__ import annotations

import datetime as dt
import zoneinfo

from langchain_core.tools import tool

_CONDITIONS = ["clear skies", "partly cloudy", "overcast", "light rain", "gusty winds"]


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in an IANA timezone.

    Args:
        timezone: IANA timezone name, e.g. "Asia/Tokyo", "America/New_York", "Europe/London".
                  Defaults to "UTC".
    """
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except Exception:
        return f"Unknown timezone {timezone!r}. Use an IANA name like 'Asia/Tokyo'."
    return dt.datetime.now(tz).strftime("%A, %d %B %Y at %H:%M") + f" ({timezone})"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Demo tool: returns representative mock data, not a live forecast.

    Args:
        city: City name, e.g. "Paris" or "San Francisco".
    """
    seed = sum(ord(c) for c in city.strip().lower())
    condition = _CONDITIONS[seed % len(_CONDITIONS)]
    temp_c = 6 + seed % 26
    return f"{city.strip().title()}: {condition}, {temp_c}°C (demo data)"


TOOLS = [get_current_time, get_weather]
