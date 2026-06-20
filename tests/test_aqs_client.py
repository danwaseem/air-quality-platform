"""Unit tests for AQSClient (no live API calls)."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.aqs_client import AQSClient, _year_windows


def test_year_windows_single_year():
    windows = list(_year_windows(date(2023, 3, 1), date(2023, 11, 30)))
    assert windows == [(date(2023, 3, 1), date(2023, 11, 30))]


def test_year_windows_multi_year():
    windows = list(_year_windows(date(2022, 6, 1), date(2024, 2, 28)))
    assert windows[0] == (date(2022, 6, 1), date(2022, 12, 31))
    assert windows[1] == (date(2023, 1, 1), date(2023, 12, 31))
    assert windows[2] == (date(2024, 1, 1), date(2024, 2, 28))
    assert len(windows) == 3


def test_client_raises_without_credentials(monkeypatch):
    monkeypatch.delenv("EPA_AQS_EMAIL", raising=False)
    monkeypatch.delenv("EPA_AQS_KEY", raising=False)
    with pytest.raises((ValueError, KeyError)):
        AQSClient()


@patch("src.ingestion.aqs_client._get")
def test_pull_hourly_returns_dataframe(mock_get):
    fake_row = {
        "date_local": "2023-01-01",
        "time_local": "13:00",
        "sample_measurement": "12.3",
        "parameter": "PM2.5 - Local Conditions",
        "state_name": "California",
    }
    mock_get.return_value = {"Header": [{"status": "Success"}], "Data": [fake_row] * 5}

    client = AQSClient(email="test@test.com", api_key="testkey")
    df = client.pull_hourly(
        param_codes=["88101"],
        state_counties=[("06", "037")],
        start=date(2023, 1, 1),
        end=date(2023, 1, 31),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "datetime_local" in df.columns
