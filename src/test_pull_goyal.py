"""Unit tests for pull_goyal.py: parsing Goyal's workbook without the network.

Each test targets one parsing rule from Appendix C.2 (thousands separators,
mixed NaN/blank missing values, period-end dates) or one failure mode that must
be loud rather than silent (a changed link, an unexpected cell, a bad header).
"""

import numpy as np
import pandas as pd
import pytest

import pull_goyal


def test_clean_numeric_column_handles_separators_and_missing_markers():
    """Text numbers with thousands separators parse; "NaN", blanks and None are NaN."""
    raw = pd.Series(["6,845.50", "NaN", "", None, 4.44, " 1,000 "], dtype=object)
    cleaned = pull_goyal.clean_numeric_column(raw)
    expected = pd.Series([6845.5, np.nan, np.nan, np.nan, 4.44, 1000.0])
    pd.testing.assert_series_equal(cleaned, expected)


def test_clean_numeric_column_rejects_unexpected_text():
    """A cell that is neither a number nor a missing marker raises instead of becoming NaN."""
    with pytest.raises(ValueError):
        pull_goyal.clean_numeric_column(pd.Series(["0.01", "see note"], dtype=object))


@pytest.mark.parametrize(
    "freq,codes,expected",
    [
        ("monthly", [187101, 202512], ["1871-01-31", "2025-12-31"]),
        ("quarterly", [18711, 20253], ["1871-03-31", "2025-09-30"]),
        ("annual", [1871, 2025], ["1871-12-31", "2025-12-31"]),
    ],
)
def test_parse_period_end_dates(freq, codes, expected):
    """Goyal's date codes become period-end timestamps."""
    dates = pull_goyal.parse_period_end_dates(codes, freq)
    assert dates.equals(pd.DatetimeIndex(expected))
    assert dates.name == "date"


@pytest.mark.parametrize(
    "freq,codes",
    [
        ("quarterly", [20255]),  # quarter digit outside 1-4
        ("monthly", [18711]),  # a quarterly code in the monthly sheet
        ("monthly", [202512.7]),  # fractional code would be truncated
        ("annual", [202512]),  # a monthly code in the annual sheet
    ],
)
def test_parse_period_end_dates_rejects_malformed_codes(freq, codes):
    """Malformed date codes raise instead of silently becoming a wrong date."""
    with pytest.raises(ValueError):
        pull_goyal.parse_period_end_dates(codes, freq)


def test_parse_goyal_sheet_rejects_values_without_a_date():
    """A row with data but a blank date code raises instead of being dropped."""
    raw = pd.DataFrame({"yyyymm": [202511, np.nan], "Index": [6849.09, 6845.5]})
    with pytest.raises(ValueError):
        pull_goyal.parse_goyal_sheet(raw, "monthly")


def test_save_goyal_records_the_workbook_vintage(tmp_path):
    """The saved vintage record holds the SHA-256 of exactly the bytes written."""
    import hashlib
    import json

    content = b"PK fake workbook bytes"
    sheet = pd.DataFrame({"Index": [1.0]}, index=pd.DatetimeIndex(["2025-12-31"]))
    pull_goyal.save_goyal(content, {"monthly": sheet}, data_dir=tmp_path)
    vintage = json.loads((tmp_path / pull_goyal.VINTAGE_FILE_NAME).read_text())
    assert vintage["sha256"] == hashlib.sha256(content).hexdigest()
    assert (tmp_path / pull_goyal.RAW_WORKBOOK_NAME).read_bytes() == content


def test_parse_goyal_sheet_builds_clean_float_panel():
    """A raw sheet with text numbers and a trailing blank row becomes a float panel."""
    raw = pd.DataFrame(
        {
            "yyyymm": [202511, 202512, np.nan],
            "Index": ["6,849.09", "6,845.50", np.nan],
            "D12": [78.77, "NaN", np.nan],
        }
    )
    sheet = pull_goyal.parse_goyal_sheet(raw, "monthly")
    assert sheet.index.equals(pd.DatetimeIndex(["2025-11-30", "2025-12-31"]))
    assert sheet.index.name == "date"
    assert sheet.dtypes.eq(float).all()
    assert sheet.loc["2025-12-31", "Index"] == 6845.5
    assert np.isnan(sheet.loc["2025-12-31", "D12"])


def test_parse_goyal_sheet_rejects_unexpected_layout():
    """If Goyal renames or moves the date column, parsing stops."""
    raw = pd.DataFrame({"date": [202512], "Index": [6845.5]})
    with pytest.raises(ValueError):
        pull_goyal.parse_goyal_sheet(raw, "monthly")


def test_download_rejects_html_instead_of_workbook(monkeypatch):
    """A moved or restricted Google link returns HTML; that must raise, not be saved."""

    class FakeResponse:
        content = b"<!DOCTYPE html><html>Sign in</html>"
        headers = {"Content-Type": "text/html"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(pull_goyal.requests, "get", lambda *a, **k: FakeResponse())
    with pytest.raises(ValueError):
        pull_goyal.download_goyal_workbook("https://example.invalid/workbook")
