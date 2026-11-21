"""Unit tests for pull_crsp.py without a WRDS connection.

The NYSE aggregation SQL, the one non-trivial query, is run against a small
in-memory SQLite table with hand-checkable values (the course's approach to
testing WRDS queries locally). The rows exercise each filter: other exchanges,
closed-end funds, new listings without a previous cap, and dates outside the
window.
"""

import sqlite3

import pandas as pd
import pytest

import pull_crsp

MSF_ROWS = [
    # permno, yyyymm, mthcaldt, mthcap, mthprevcap, mthretx, primaryexch, securitytype
    (1, 200001, "2000-01-31", 100.0, 90.0, 0.10, "N", "EQTY"),
    (2, 200001, "2000-01-31", 50.0, 50.0, 0.00, "N", "EQTY"),
    (1, 200002, "2000-02-29", 110.0, 100.0, 0.05, "N", "EQTY"),
    (2, 200002, "2000-02-29", 60.0, 50.0, 0.02, "N", "EQTY"),  # issued shares
    (3, 200002, "2000-02-29", 30.0, None, None, "N", "EQTY"),  # new listing
    (4, 200002, "2000-02-29", 1000.0, 990.0, 0.01, "N", "FUND"),  # closed-end fund
    (5, 200002, "2000-02-29", 500.0, 480.0, 0.04, "Q", "EQTY"),  # Nasdaq
    (1, 199001, "1990-01-31", 80.0, 79.0, 0.01, "N", "EQTY"),  # outside window
]


class SQLiteWRDS:
    """Minimal stand-in for wrds.Connection backed by an in-memory SQLite database."""

    def __init__(self, rows):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("ATTACH DATABASE ':memory:' AS crsp_m_stock")
        self.conn.execute(
            "CREATE TABLE crsp_m_stock.msf_v2 (permno INT, yyyymm INT, mthcaldt TEXT, "
            "mthcap REAL, mthprevcap REAL, mthretx REAL, primaryexch TEXT, "
            "securitytype TEXT)"
        )
        self.conn.executemany(
            "INSERT INTO crsp_m_stock.msf_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )

    def raw_sql(self, sql, date_cols=None):
        return pd.read_sql_query(sql, self.conn, parse_dates=date_cols)


@pytest.fixture
def nyse():
    """NYSE aggregates from the SQLite table for 1999-01 to 2000-12."""
    return pull_crsp.pull_nyse_monthly(SQLiteWRDS(MSF_ROWS), "1999-01-01", "2000-12-31")


def test_nyse_aggregation_keeps_only_nyse_equities_in_the_window(nyse):
    """Funds, Nasdaq stocks and out-of-window months are excluded."""
    assert nyse.index.equals(pd.DatetimeIndex(["2000-01-31", "2000-02-29"]))
    assert nyse.loc["2000-02-29", "mcap"] == pytest.approx(110 + 60 + 30)
    assert nyse.loc["2000-02-29", "n_stocks"] == 3


def test_nyse_value_weighted_ex_dividend_return(nyse):
    """vwretx weights each return by last month's cap; new listings carry no weight."""
    february = nyse.loc["2000-02-29"]
    vwretx = february["lagcap_x_retx"] / february["lagcap_with_retx"]
    assert vwretx == pytest.approx((100 * 0.05 + 50 * 0.02) / 150)


@pytest.mark.parametrize(
    "dates,expected",
    [
        (["2024-11-29"], "2024-11-30"),  # last trading day -> month end
        ([202402], "2024-02-29"),  # yyyymm code -> month end
    ],
)
def test_month_end_index(dates, expected):
    """CRSP dates are last trading days or yyyymm codes; panels use month ends."""
    index = pull_crsp.month_end_index(dates)
    assert index[0] == pd.Timestamp(expected)
    assert index.name == "date"


def test_two_rows_in_one_month_raise():
    """Two rows mapping to the same month would double-count; that must fail."""
    frame = pd.DataFrame({"caldt": ["2024-11-28", "2024-11-29"], "x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        pull_crsp._by_month(frame, "caldt")


def test_pull_window_includes_lookback_and_whole_last_month():
    """Two years before START_DATE, through the end of END_DATE's month."""
    first, last = pull_crsp.pull_window(
        pd.Timestamp("1947-01-01"), pd.Timestamp("2025-11-21")
    )
    assert (first, last) == ("1945-01-01", "2025-11-30")
