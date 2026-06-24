"""
EPA AQS (Air Quality System) API client.

Pulls hourly sample data for PM2.5 (88101), NO2 (42602), and ozone (44201).
Paginates by year to stay within the API's per-request row limits.
Handles rate limits (HTTP 429) and transient errors with exponential backoff.

API docs: https://aqs.epa.gov/data/api
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Iterator

import pandas as pd
import requests
from dotenv import load_dotenv
import os

load_dotenv()

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

AQS_BASE = "https://aqs.epa.gov/data/api"

# Parameter codes → human-readable names
PARAMETERS: dict[str, str] = {
    "88101": "PM2.5",
    "42602": "NO2",
    "44201": "Ozone",
}

# How long to wait on a 429 before retrying (seconds); doubled each attempt
_BACKOFF_BASE = 10
_MAX_RETRIES = 6

# AQS returns at most ~1 year of hourly data per call per site, so we
# paginate by calendar year to stay within their undocumented row cap.
_PAGE_YEARS = 1


# ── Low-level HTTP ────────────────────────────────────────────────────────────


def _get(session: requests.Session, endpoint: str, params: dict) -> dict:
    """GET an AQS endpoint, retrying on 429 / 5xx with exponential backoff."""
    url = f"{AQS_BASE}/{endpoint}"
    wait = _BACKOFF_BASE
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=120)
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES:
                raise
            log.warning("Network error (attempt %d/%d): %s — retrying in %ds",
                        attempt, _MAX_RETRIES, exc, wait)
            time.sleep(wait)
            wait *= 2
            continue

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", wait))
            log.warning("Rate-limited (attempt %d/%d) — sleeping %ds",
                        attempt, _MAX_RETRIES, retry_after)
            time.sleep(retry_after)
            wait = max(wait * 2, retry_after)
            continue

        if resp.status_code >= 500:
            if attempt == _MAX_RETRIES:
                resp.raise_for_status()
            log.warning("Server error %d (attempt %d/%d) — retrying in %ds",
                        resp.status_code, attempt, _MAX_RETRIES, wait)
            time.sleep(wait)
            wait *= 2
            continue

        resp.raise_for_status()
        payload = resp.json()

        # AQS wraps errors/empty results in the JSON body
        header = payload.get("Header", [{}])[0]
        status = header.get("status", "Success")

        # "No data matched" is not an error — the county just has no monitors
        if status != "Success":
            msg = header.get("error", header.get("message", status))
            no_data_phrases = ("no data", "no matching data", "no results")
            if any(p in msg.lower() for p in no_data_phrases) or any(
                p in status.lower() for p in no_data_phrases
            ):
                log.debug("No data for this query: %s", msg)
                payload.setdefault("Data", [])
                return payload
            log.error("AQS header: %s", header)
            raise RuntimeError(f"AQS API error: {msg}")

        return payload

    raise RuntimeError("Exceeded max retries for AQS request")


# ── Year-range helpers ────────────────────────────────────────────────────────


def _year_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield (window_start, window_end) pairs split at calendar-year boundaries."""
    cursor = start
    while cursor <= end:
        year_end = date(cursor.year, 12, 31)
        window_end = min(year_end, end)
        yield cursor, window_end
        cursor = date(window_end.year + 1, 1, 1)


def _fmt(d: date) -> str:
    return d.strftime("%Y%m%d")


# ── Public client ─────────────────────────────────────────────────────────────


class AQSClient:
    """
    Client for the EPA AQS hourly data API.

    Parameters
    ----------
    email : str
        Registered AQS account email (from env var EPA_AQS_EMAIL).
    api_key : str
        AQS API key (from env var EPA_AQS_KEY).
    """

    def __init__(self, email: str | None = None, api_key: str | None = None) -> None:
        self.email = email or os.environ["EPA_AQS_EMAIL"]
        self.api_key = api_key or os.environ["EPA_AQS_KEY"]
        if not self.email or not self.api_key:
            raise ValueError(
                "AQS credentials missing. Set EPA_AQS_EMAIL and EPA_AQS_KEY in .env"
            )
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "air-quality-platform/0.1"})

    # -- internals ------------------------------------------------------------

    def _base_params(self) -> dict:
        return {"email": self.email, "key": self.api_key}

    def _hourly_by_county(
        self,
        param: str,
        state: str,
        county: str,
        bdate: date,
        edate: date,
    ) -> list[dict]:
        """Fetch hourly data for one param / state / county / year window."""
        params = {
            **self._base_params(),
            "param": param,
            "bdate": _fmt(bdate),
            "edate": _fmt(edate),
            "state": state,
            "county": county,
        }
        log.debug(
            "Fetching %s  state=%s county=%s  %s→%s",
            PARAMETERS[param], state, county, bdate, edate,
        )
        payload = _get(self._session, "sampleData/byCounty", params)
        return payload.get("Data", [])

    # -- public ---------------------------------------------------------------

    def pull_hourly(
        self,
        param_codes: list[str],
        state_counties: list[tuple[str, str]],
        start: date,
        end: date,
        inter_request_delay: float = 1.0,
    ) -> pd.DataFrame:
        """
        Pull hourly sample data for the given parameters, states/counties,
        and date range.  Paginates by year automatically.

        Parameters
        ----------
        param_codes :
            List of AQS parameter codes, e.g. ["88101", "42602", "44201"].
        state_counties :
            List of (state_fips, county_fips) 2-tuples, e.g. [("06", "037")].
        start / end :
            Inclusive date range.
        inter_request_delay :
            Seconds to sleep between API calls (be a polite citizen).

        Returns
        -------
        pd.DataFrame with one row per hourly observation.
        """
        all_rows: list[dict] = []
        combos = [
            (param, state, county, bdate, edate)
            for param in param_codes
            for state, county in state_counties
            for bdate, edate in _year_windows(start, end)
        ]
        total = len(combos)

        skipped: list[str] = []

        for idx, (param, state, county, bdate, edate) in enumerate(combos, 1):
            log.info(
                "[%d/%d] %s  state=%s county=%s  %s→%s",
                idx, total, PARAMETERS[param], state, county, bdate, edate,
            )
            try:
                rows = self._hourly_by_county(param, state, county, bdate, edate)
            except RuntimeError as exc:
                label = f"{PARAMETERS[param]} state={state} county={county} {bdate}→{edate}"
                log.warning("Skipping %s — %s", label, exc)
                skipped.append(label)
                rows = []

            all_rows.extend(rows)
            log.info("  → %d rows (running total: %d)", len(rows), len(all_rows))

            if idx < total:
                time.sleep(inter_request_delay)

        if skipped:
            log.warning("%d combos skipped due to API errors:\n  %s",
                        len(skipped), "\n  ".join(skipped))

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df = self._normalize(df)
        return df

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Cast types and add a unified UTC timestamp column."""
        # AQS returns date_local + time_local (HH:MM) + gmt_offset
        if "date_local" in df.columns and "time_local" in df.columns:
            df["datetime_local"] = pd.to_datetime(
                df["date_local"] + " " + df["time_local"],
                format="%Y-%m-%d %H:%M",
                errors="coerce",
            )

        numeric_cols = [
            "sample_measurement", "mdl", "uncertainty",
            "latitude", "longitude",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def list_counties(self, state: str) -> pd.DataFrame:
        """Return all counties for a state — useful for building state_counties lists."""
        payload = _get(
            self._session,
            "list/countiesByState",
            {**self._base_params(), "state": state},
        )
        return pd.DataFrame(payload.get("Data", []))

    def list_states(self) -> pd.DataFrame:
        """Return all state FIPS codes and names."""
        payload = _get(self._session, "list/states", self._base_params())
        return pd.DataFrame(payload.get("Data", []))
