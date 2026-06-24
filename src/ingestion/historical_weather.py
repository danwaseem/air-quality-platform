"""
Historical weather fetcher using the Open-Meteo Archive API.

Fetches hourly 2023 weather for every AQS monitoring station in
data/processed/clean_hourly.parquet and joins it to the cleaned
air-quality data.

API:  https://archive-api.open-meteo.com/v1/archive
      Free, no API key, ~10 req/s polite limit.

Weather variables fetched
  temperature_2m         °C
  wind_speed_10m         m/s   (wind_speed_unit=ms)
  relative_humidity_2m   %

Outputs
  data/raw/weather_2023.parquet              raw long-format weather (all stations)
  data/processed/clean_hourly_weather.parquet  AQ + weather joined on (station_id, timestamp_utc)

Caching
  Each station's weather is saved to data/raw/.weather_chunks_2023/<station_id>.parquet
  before moving on.  Re-running the script skips already-fetched stations so an
  interrupted run resumes exactly where it left off.

Run:
  python src/ingestion/historical_weather.py [--fresh]
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_API_URL      = "https://archive-api.open-meteo.com/v1/archive"
_START_DATE   = "2023-01-01"
_END_DATE     = "2023-12-31"
_VARIABLES    = "temperature_2m,wind_speed_10m,relative_humidity_2m"
_EXPECTED_ROWS = 8760          # 365 days × 24 h (2023 is not a leap year)
_REQUEST_DELAY = 0.2           # seconds between calls — stays well under rate limit
_MAX_RETRIES   = 4
_RETRY_BASE    = 2.0           # seconds; doubles on each retry

# Paths
_ROOT         = Path(__file__).resolve().parent.parent.parent
_CLEAN_PQ     = _ROOT / "data" / "processed" / "clean_hourly.parquet"
_CHUNKS_DIR   = _ROOT / "data" / "raw" / ".weather_chunks_2023"
_WEATHER_PQ   = _ROOT / "data" / "raw" / "weather_2023.parquet"
_JOINED_PQ    = _ROOT / "data" / "processed" / "clean_hourly_weather.parquet"


# ── Per-station fetch ─────────────────────────────────────────────────────────

def _fetch_station(
    session:    requests.Session,
    station_id: str,
    lat:        float,
    lon:        float,
) -> pd.DataFrame | None:
    """
    Call the Open-Meteo archive API for one station and return a DataFrame.

    Columns: station_id, timestamp_utc, temperature_2m, wind_speed_10m,
             relative_humidity_2m.
    Returns None on unrecoverable failure (logged, station skipped).
    """
    params = {
        "latitude":       lat,
        "longitude":      lon,
        "start_date":     _START_DATE,
        "end_date":       _END_DATE,
        "hourly":         _VARIABLES,
        "timezone":       "UTC",
        "wind_speed_unit": "ms",
    }

    for attempt in range(_MAX_RETRIES):
        try:
            resp = session.get(_API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                wait = _RETRY_BASE * (2 ** attempt)
                log.warning("%s  rate-limited; sleeping %.0fs", station_id, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES - 1:
                log.error("%s  request failed after %d attempts: %s",
                          station_id, _MAX_RETRIES, exc)
                return None
            wait = _RETRY_BASE * (2 ** attempt)
            log.warning("%s  attempt %d failed (%s); retry in %.0fs",
                        station_id, attempt + 1, exc, wait)
            time.sleep(wait)
    else:
        log.error("%s  exhausted retries", station_id)
        return None

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        log.warning("%s  API returned empty hourly data", station_id)
        return None

    # Parse UTC timestamps; cast to match clean_hourly.parquet dtype (datetime64[us, UTC])
    ts = (
        pd.to_datetime(times, format="%Y-%m-%dT%H:%M", utc=True)
        .astype("datetime64[us, UTC]")
    )

    df = pd.DataFrame({
        "station_id":            station_id,
        "timestamp_utc":         ts,
        "temperature_2m":        pd.array(hourly.get("temperature_2m",        [None] * len(times)), dtype="Float64"),
        "wind_speed_10m":        pd.array(hourly.get("wind_speed_10m",        [None] * len(times)), dtype="Float64"),
        "relative_humidity_2m":  pd.array(hourly.get("relative_humidity_2m",  [None] * len(times)), dtype="Float64"),
    })

    if len(df) != _EXPECTED_ROWS:
        log.warning("%s  expected %d rows, got %d",
                    station_id, _EXPECTED_ROWS, len(df))

    return df


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _chunk_path(station_id: str) -> Path:
    return _CHUNKS_DIR / f"{station_id}.parquet"


def _cached_stations() -> set[str]:
    """Return station IDs whose chunk file exists and has the expected row count."""
    if not _CHUNKS_DIR.exists():
        return set()
    good: set[str] = set()
    for f in _CHUNKS_DIR.glob("*.parquet"):
        try:
            n = pq.read_metadata(f).num_rows
            if n >= _EXPECTED_ROWS:
                good.add(f.stem)
        except Exception:
            pass
    return good


# ── Main fetch orchestrator ───────────────────────────────────────────────────

def fetch_all_weather(stations: pd.DataFrame, fresh: bool = False) -> pd.DataFrame:
    """
    Fetch 2023 hourly weather for every station in `stations`
    (must have columns station_id, latitude, longitude).

    Saves per-station chunks to the cache directory and merges them
    into a single DataFrame on completion.

    Returns the merged weather DataFrame.
    """
    _CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if fresh and _CHUNKS_DIR.exists():
        shutil.rmtree(_CHUNKS_DIR)
        _CHUNKS_DIR.mkdir(parents=True)
        log.info("--fresh: cleared weather cache")

    already_done = _cached_stations()
    todo = stations[~stations["station_id"].isin(already_done)]

    log.info(
        "Weather fetch: %d stations total, %d cached, %d to fetch",
        len(stations), len(already_done), len(todo),
    )
    if already_done:
        print(f"  {len(already_done)} stations already cached — skipping")

    failed: list[str] = []
    session = requests.Session()

    with tqdm(total=len(todo), unit="station", desc="Fetching weather") as pbar:
        for _, row in todo.iterrows():
            sid = row["station_id"]
            df  = _fetch_station(session, sid, row["latitude"], row["longitude"])
            if df is not None:
                df.to_parquet(_chunk_path(sid), index=False, compression="snappy")
            else:
                failed.append(sid)

            pbar.set_postfix({"failed": len(failed)}, refresh=False)
            pbar.update(1)
            time.sleep(_REQUEST_DELAY)

    if failed:
        log.warning("Failed stations (%d): %s", len(failed), failed)

    # ── Merge chunks → single DataFrame ──────────────────────────────────────
    chunk_files = sorted(_CHUNKS_DIR.glob("*.parquet"))
    if not chunk_files:
        raise RuntimeError("No weather chunks found after fetch. Check logs.")

    print(f"\nMerging {len(chunk_files)} station chunks …", flush=True)
    df_weather = pd.concat(
        [pd.read_parquet(f) for f in chunk_files],
        ignore_index=True,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_stations_ok = df_weather["station_id"].nunique()
    null_frac = {
        col: round(df_weather[col].isna().mean() * 100, 1)
        for col in ("temperature_2m", "wind_speed_10m", "relative_humidity_2m")
    }
    print(
        f"  Stations with weather: {n_stations_ok} / {len(stations)}\n"
        f"  Total weather rows:    {len(df_weather):,}\n"
        f"  Expected (215×8760):   {len(stations) * _EXPECTED_ROWS:,}\n"
        f"  Null rates:  temp={null_frac['temperature_2m']}%  "
        f"wind={null_frac['wind_speed_10m']}%  "
        f"hum={null_frac['relative_humidity_2m']}%"
    )
    if failed:
        print(f"  Failed stations ({len(failed)}): {failed}")

    return df_weather


# ── Join ──────────────────────────────────────────────────────────────────────

def join_weather(df_aq: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join weather onto the cleaned AQ DataFrame on (station_id, timestamp_utc).

    AQ rows without matching weather (failed/missing stations) retain NaN
    in the weather columns — the model can use them with HistGBR's NaN handling.
    """
    # Ensure identical timestamp dtypes before merge (both datetime64[us, UTC])
    if df_weather["timestamp_utc"].dtype != df_aq["timestamp_utc"].dtype:
        df_weather = df_weather.copy()
        df_weather["timestamp_utc"] = df_weather["timestamp_utc"].astype(
            df_aq["timestamp_utc"].dtype
        )

    df_joined = df_aq.merge(
        df_weather[["station_id", "timestamp_utc",
                    "temperature_2m", "wind_speed_10m", "relative_humidity_2m"]],
        on=["station_id", "timestamp_utc"],
        how="left",
    )

    # Join coverage diagnostics
    n_total     = len(df_joined)
    n_with_temp = df_joined["temperature_2m"].notna().sum()
    coverage_pct = round(n_with_temp / n_total * 100, 2)

    stations_with_weather = (
        df_joined.groupby("station_id")["temperature_2m"]
        .apply(lambda s: s.notna().any())
        .sum()
    )
    stations_without = (
        df_joined["station_id"].nunique() - stations_with_weather
    )

    log.info("Join coverage: %.2f%% of rows have weather data", coverage_pct)
    log.info(
        "Stations with weather: %d  |  without: %d",
        stations_with_weather, stations_without,
    )

    return df_joined, {
        "total_rows":             n_total,
        "rows_with_weather":      int(n_with_temp),
        "coverage_pct":           coverage_pct,
        "stations_with_weather":  int(stations_with_weather),
        "stations_without_weather": int(stations_without),
    }


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(fresh: bool = False) -> None:
    """
    Full pipeline: fetch → save raw → join → save processed.
    """
    if not _CLEAN_PQ.exists():
        sys.exit(f"ERROR: {_CLEAN_PQ} not found — run scripts/clean_data.py first.")

    # ── Load station list ─────────────────────────────────────────────────────
    print(f"Loading station list from {_CLEAN_PQ.name} …", flush=True)
    df_aq = pd.read_parquet(_CLEAN_PQ)
    stations = (
        df_aq.groupby("station_id")[["latitude", "longitude", "state"]]
        .first()
        .reset_index()
    )
    print(
        f"  {len(stations)} unique stations  ·  "
        f"AQ rows: {len(df_aq):,}  ·  "
        f"Weather rows to fetch: {len(stations) * _EXPECTED_ROWS:,}\n"
    )

    # ── Fetch weather ─────────────────────────────────────────────────────────
    df_weather = fetch_all_weather(stations, fresh=fresh)

    # ── Save raw weather parquet ──────────────────────────────────────────────
    _WEATHER_PQ.parent.mkdir(parents=True, exist_ok=True)
    df_weather.to_parquet(_WEATHER_PQ, index=False, compression="snappy")
    size_mb = _WEATHER_PQ.stat().st_size / 1e6
    print(f"\nSaved raw weather  → {_WEATHER_PQ}  ({size_mb:.1f} MB)")

    # ── Join ──────────────────────────────────────────────────────────────────
    print("\nJoining weather onto AQ data …", flush=True)
    df_joined, stats = join_weather(df_aq, df_weather)

    # ── Save joined parquet ───────────────────────────────────────────────────
    _JOINED_PQ.parent.mkdir(parents=True, exist_ok=True)
    df_joined.to_parquet(_JOINED_PQ, index=False, compression="snappy")
    joined_mb = _JOINED_PQ.stat().st_size / 1e6
    print(f"Saved joined data  → {_JOINED_PQ}  ({joined_mb:.1f} MB)")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Historical Weather — Done")
    print("=" * 60)
    print(f"  Stations fetched:     {df_weather['station_id'].nunique()} / {len(stations)}")
    print(f"  Raw weather rows:     {len(df_weather):,}")
    print(f"  AQ rows:              {len(df_aq):,}")
    print(f"  Joined rows:          {len(df_joined):,}")
    print(f"  Join coverage:        {stats['coverage_pct']:.1f}%")
    print(f"  Stations with wx:     {stats['stations_with_weather']}")
    print(f"  Stations without wx:  {stats['stations_without_weather']}")
    print(f"  New columns:          temperature_2m (°C)  wind_speed_10m (m/s)")
    print(f"                        relative_humidity_2m (%)")
    print(f"  Output:               {_JOINED_PQ.name}")
    print("=" * 60)
    print(
        "\nNext step: re-run feature engineering and training against\n"
        "  data/processed/clean_hourly_weather.parquet\n"
        "to include temperature_2m, wind_speed_10m, relative_humidity_2m\n"
        "as additional model features."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fetch 2023 hourly weather from Open-Meteo and join to AQ data."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete cached weather chunks and re-fetch everything from scratch.",
    )
    args = parser.parse_args()

    run(fresh=args.fresh)
