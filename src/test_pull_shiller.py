"""Unit tests for pull_shiller.py, offline.

The risky parts of Shiller's file are its decimal month codes (October is
2026.1), a header row that is found by content rather than position, and a
download link that changes with every update.
"""

import numpy as np
import pandas as pd
import pytest

import pull_shiller


@pytest.mark.parametrize(
    "code,expected",
    [(1871.01, "1871-01-31"), (2026.1, "2026-10-31"), (2026.12, "2026-12-31")],
)
def test_shiller_month_codes(code, expected):
    """2026.1 is October, not January; 2026.12 is December."""
    assert pull_shiller.parse_shiller_dates([code])[0] == pd.Timestamp(expected)


def test_shiller_month_code_out_of_range_raises():
    """A code like 2026.13 is not a month."""
    with pytest.raises(ValueError):
        pull_shiller.parse_shiller_dates([2026.13])


def test_parse_shiller_sheet_finds_header_and_drops_notes():
    """Title rows above the header and notes below the data are ignored."""
    raw = pd.DataFrame(
        [
            ["Stock Market Data", None, None, None, None, None],
            [None, None, None, None, None, None],
            ["Date", "P", "D", "E", "CPI", "Fraction"],
            [2025.09, 6600.5, 78.1, 240.2, 324.8, 2025.708],
            [2025.1, 6700.25, "NaN", None, 325.0, 2025.792],
            ["Note: data are monthly averages", None, None, None, None, None],
        ]
    )
    out = pull_shiller.parse_shiller_sheet(raw)
    assert list(out.columns) == ["P", "D", "E", "CPI"]
    assert out.index.equals(pd.DatetimeIndex(["2025-09-30", "2025-10-31"]))
    assert out.loc["2025-10-31", "P"] == 6700.25
    assert np.isnan(out.loc["2025-10-31", "D"])


def test_parse_shiller_sheet_without_header_raises():
    """If the 'Date' header row is gone, parsing stops instead of guessing."""
    raw = pd.DataFrame([[2025.09, 6600.5, 78.1, 240.2, 324.8]])
    with pytest.raises(ValueError):
        pull_shiller.parse_shiller_sheet(raw)


def test_find_workbook_link_handles_protocol_relative_links():
    """shillerdata.com links start with // and carry a changing version tag."""
    html = '<a href="//img1.wsimg.com/blobby/go/x/downloads/y/ie_data.xls?ver=123">Data</a>'
    assert (
        pull_shiller.find_workbook_link(html)
        == "https://img1.wsimg.com/blobby/go/x/downloads/y/ie_data.xls?ver=123"
    )


def test_find_workbook_link_raises_when_missing():
    """No link means the page changed; fail rather than fall back to stale data."""
    with pytest.raises(ValueError):
        pull_shiller.find_workbook_link("<html>No data here</html>")
