"""Module for fetching and analyzing tide predictions from RWS API."""

import logging
from datetime import datetime as dt
from datetime import timedelta, timezone
from threading import Lock
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from scipy.signal import find_peaks

_LOGGER = logging.getLogger(__name__)

RWS_API_URL = "https://waterwebservices.rijkswaterstaat.nl/ONLINEWAARNEMINGENSERVICES_DBO/OphalenWaarnemingen"
# API timezone is always +01:00 (CET without DST)
API_TIMEZONE = timezone(timedelta(hours=1))
# Tide detection parameters
PEAK_DISTANCE = 30  # Minimum points between tides (5 hours for 10-min intervals)
PEAK_PROMINENCE = 5  # Minimum height difference in cm
CACHE_DURATION = 300

_tide_cache: dict[str, tuple[dt, pd.DataFrame, pd.DataFrame]] = {}
_cache_locks: dict[str, Lock] = {}
_locks_lock = Lock()  # Lock to protect the _cache_locks dictionary


def _get_lock_for_location(location_code: str) -> Lock:
    """Get or create a lock for a specific location."""
    with _locks_lock:
        if location_code not in _cache_locks:
            _cache_locks[location_code] = Lock()
        return _cache_locks[location_code]


def _get_cached_tides(location_code: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Get cached tide extremes if available and not expired.

    Args:
        location_code: Station code

    Returns:
        Tuple of (high_tides, low_tides) if cache valid, None otherwise
    """
    if location_code not in _tide_cache:
        return None

    cached_time, high_tides, low_tides = _tide_cache[location_code]

    now = dt.now(timezone.utc)
    if (now - cached_time).total_seconds() < CACHE_DURATION:
        return high_tides, low_tides

    del _tide_cache[location_code]
    return None


def _fetch_and_cache_tides(location: pd.Series, location_code: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Fetch tide predictions from API and cache the extremes.

    Args:
        location: Location data from ddlpy.locations()
        location_code: Station code

    Returns:
        Tuple of (high_tides, low_tides) or None if unavailable
    """
    start_date = dt.now(API_TIMEZONE) - timedelta(hours=2)
    end_date = start_date + timedelta(hours=24)

    body = _build_api_request_body(location_code, location["X"], location["Y"], start_date, end_date)

    try:
        response = requests.post(RWS_API_URL, json=body)
        response.raise_for_status()
        data = response.json()

        df = _parse_api_response(data)
        if df is None or df.empty:
            _LOGGER.warning("No data received from API for %s", location_code)
            return None

        high_tides, low_tides = find_tide_extremes(df)

        # Cache the results with current timestamp
        _tide_cache[location_code] = (dt.now(timezone.utc), high_tides, low_tides)

        return high_tides, low_tides

    except (requests.RequestException, KeyError, ValueError) as err:
        _LOGGER.error("Error fetching tide data for %s: %s", location_code, err)
        return None


def _build_api_request_body(location_code: str, x: float, y: float, start_date: dt, end_date: dt) -> dict[str, Any]:
    """Build the request body for the RWS API."""
    return {
        "Locatie": {
            "Code": location_code,
            "X": x,
            "Y": y,
        },
        "AquoPlusWaarnemingMetadata": {
            "AquoMetadata": {
                "Compartiment": {"Code": "OW"},
                "Grootheid": {"Code": "WATHTEVERWACHT"},
            }
        },
        "Periode": {
            "Begindatumtijd": start_date.strftime("%Y-%m-%dT%H:%M:%S.000+01:00"),
            "Einddatumtijd": end_date.strftime("%Y-%m-%dT%H:%M:%S.000+01:00"),
        },
    }


def _parse_api_response(data: dict[str, Any]) -> pd.DataFrame | None:
    """Parse RWS API response into a DataFrame."""
    if not data.get("WaarnemingenLijst"):
        return None

    records = []
    for waarneming in data["WaarnemingenLijst"]:
        for metingen in waarneming.get("MetingenLijst", []):
            timestamp = pd.to_datetime(metingen["Tijdstip"]).tz_convert("UTC")
            waarde = metingen.get("Meetwaarde", {}).get("Waarde_Numeriek")

            if waarde is not None:
                records.append({"Tijdstip": timestamp, "Waarde": waarde})

    if not records:
        return None

    df = pd.DataFrame(records)
    df.set_index("Tijdstip", inplace=True)
    return df


def has_wathteverwacht(location: pd.Series, location_code: str) -> bool:
    """
    Check if WATHTEVERWACHT data is available for a location.

    This parameter is not in the catalog but can be requested directly.
    Based on https://rijkswaterstaatdata.nl/waterdata/

    Args:
        location: Location data from ddlpy.locations()
        location_code: Station code

    Returns:
        True if weather-based predictions are available
    """
    start_date = dt.now(API_TIMEZONE) - timedelta(hours=2)
    end_date = start_date + timedelta(hours=24)

    body = _build_api_request_body(location_code, location["X"], location["Y"], start_date, end_date)

    try:
        response = requests.post(RWS_API_URL, json=body)
        response.raise_for_status()
        data = response.json()

        if "WaarnemingenLijst" in data and data["WaarnemingenLijst"]:
            return len(data["WaarnemingenLijst"][0].get("MetingenLijst", [])) > 0

    except (requests.RequestException, KeyError, ValueError):
        return False

    return False


def get_wathteverwacht(location: pd.Series, location_code: str, low_tide: bool) -> pd.DataFrame | None:
    """
    Get next tide extreme from WATHTEVERWACHT (weather-based predictions).

    This parameter is not in the catalog but can be requested directly.
    Based on https://rijkswaterstaatdata.nl/waterdata/

    Results are cached for 5 minutes to avoid duplicate API calls when requesting
    both high and low tide in quick succession.

    Args:
        location: Location data from ddlpy.locations()
        location_code: Location code
        low_tide: If True, return next low tide; if False, return next high tide

    Returns:
        DataFrame with single row containing the next tide extreme, or None if unavailable
    """
    # Get location-specific lock to prevent concurrent API calls
    lock = _get_lock_for_location(location_code)

    with lock:
        # Check cache again inside the lock (double-check pattern)
        cached = _get_cached_tides(location_code)

        if cached is not None:
            high_tides, low_tides = cached
        else:
            # Cache miss or expired, fetch from API
            result = _fetch_and_cache_tides(location, location_code)
            if result is None:
                return None
            high_tides, low_tides = result

    # Return the requested tide type
    tides = low_tides if low_tide else high_tides
    return tides.head(1) if not tides.empty else None


def find_tide_extremes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find tide extremes using scipy.signal.find_peaks.

    Args:
        df: DataFrame with 'Waarde' column containing water level predictions and datetime index

    Returns:
        Tuple of (high_tides, low_tides) DataFrames
    """
    if df.empty or "Waarde" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    values: npt.NDArray[np.float64] = df["Waarde"].to_numpy()

    # Find peaks (high tides)
    high_indices, _ = find_peaks(values, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)

    # Find troughs (low tides) by inverting the signal
    low_indices, _ = find_peaks(-values, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)

    high_tides = df.iloc[high_indices]
    low_tides = df.iloc[low_indices]

    return high_tides, low_tides
