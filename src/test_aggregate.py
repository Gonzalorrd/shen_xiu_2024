"""Unit tests for aggregate.py on small hand-checkable inputs.

Each aggregation rule (end-of-period, compounding, summing, Goyal's Rfree) gets
one test with a number you can verify by hand, plus the edge cases that would
silently corrupt a panel: partial periods, missing months, and a last-non-null
"end-of-period" value. The check against Goyal's real sheets lives in
test_data_validation.py.
"""

import numpy as np
import pandas as pd
import pytest

from aggregate import aggregate_to_k_months, goyal_rfree, period_end_dates


def _monthly(values, start="2000-01-31"):
    """Monthly frame on month-end dates starting at ``start``."""
    n = len(next(iter(values.values())))
    index = pd.date_range(start, periods=n, freq="ME", name="date")
    return pd.DataFrame(values, index=index)


def test_period_end_dates_are_calendar_aligned():
    """Quarters end in Mar/Jun/Sep/Dec, half-years in Jun/Dec, years in Dec."""
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    assert period_end_dates(dates, 3)[0] == pd.Timestamp("2000-03-31")
    assert period_end_dates(dates, 3)[3] == pd.Timestamp("2000-06-30")
    assert period_end_dates(dates, 6)[6] == pd.Timestamp("2000-12-31")
    assert set(period_end_dates(dates, 12)) == {pd.Timestamp("2000-12-31")}


def test_end_of_period_takes_the_last_month_not_the_mean():
    """Prices and yields take the value of the period's last month."""
    monthly = _monthly({"Index": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]})
    quarterly = aggregate_to_k_months(monthly, 3)
    assert quarterly["Index"].tolist() == [120.0, 150.0]


def test_end_of_period_does_not_fill_a_missing_last_month():
    """A missing period-end value stays missing; it is not replaced by an earlier month."""
    monthly = _monthly({"tbl": [0.01, 0.02, np.nan]})
    assert np.isnan(aggregate_to_k_months(monthly, 3)["tbl"].iloc[0])


def test_compounding_returns():
    """Three monthly returns of 1%, 2%, 3% compound to 1.01*1.02*1.03 - 1."""
    monthly = _monthly({"CRSP_SPvw": [0.01, 0.02, 0.03]})
    quarterly = aggregate_to_k_months(monthly, 3)
    assert quarterly["CRSP_SPvw"].iloc[0] == pytest.approx(1.01 * 1.02 * 1.03 - 1)


def test_summing_svar():
    """svar is a sum of squared daily returns, so the quarter is the sum of its months."""
    monthly = _monthly({"svar": [0.001, 0.002, 0.004]})
    assert aggregate_to_k_months(monthly, 3)["svar"].iloc[0] == pytest.approx(0.007)


def test_missing_month_makes_compounded_period_missing():
    """A quarter with a missing monthly return is NaN, not a two-month return."""
    monthly = _monthly({"CRSP_SPvw": [0.01, np.nan, 0.03]})
    assert np.isnan(aggregate_to_k_months(monthly, 3)["CRSP_SPvw"].iloc[0])


def test_partial_periods_at_the_edges_are_dropped():
    """Data starting in February cannot produce a first-quarter value."""
    monthly = _monthly({"CRSP_SPvw": [0.01] * 5}, start="2000-02-29")
    quarterly = aggregate_to_k_months(monthly, 3)
    assert quarterly.index.equals(pd.DatetimeIndex(["2000-06-30"]))


def test_goyal_rfree_uses_start_of_quarter_tbill_and_compounds_quarters():
    """Quarter q earns tbl at the end of quarter q-1 times 3/12; a year compounds four."""
    tbl = pd.Series(
        [0.04] * 3 + [0.08] * 3 + [0.12] * 3 + [0.16] * 3 + [0.20] * 3,
        index=pd.date_range("1999-10-31", periods=15, freq="ME"),
    )
    quarterly = goyal_rfree(tbl, 3)
    # 2000Q1 earns the Dec-1999 yield (0.04); 2000Q2 the Mar-2000 yield (0.08).
    assert quarterly.loc["2000-03-31"] == pytest.approx(0.04 / 4)
    assert quarterly.loc["2000-06-30"] == pytest.approx(0.08 / 4)
    annual = goyal_rfree(tbl, 12)
    expected = (1 + 0.04 / 4) * (1 + 0.08 / 4) * (1 + 0.12 / 4) * (1 + 0.16 / 4) - 1
    assert annual.loc["2000-12-31"] == pytest.approx(expected)


def test_both_rfree_conventions_are_returned():
    """Rfree follows Goyal's convention; Rfree_comp compounds the monthly Rfree."""
    monthly = _monthly(
        {"Rfree": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006], "tbl": [0.12] * 6},
        start="2000-01-31",
    )
    quarterly = aggregate_to_k_months(monthly, 3)
    assert quarterly.loc["2000-06-30", "Rfree"] == pytest.approx(0.12 / 4)
    assert quarterly.loc["2000-06-30", "Rfree_comp"] == pytest.approx(
        1.004 * 1.005 * 1.006 - 1
    )


def test_monthly_k1_returns_inputs_unchanged():
    """k = 1 is the identity, so the monthly panel goes through the same function exactly."""
    monthly = _monthly({"CRSP_SPvw": [0.01, -0.02], "Rfree": [0.001, 0.002]})
    out = aggregate_to_k_months(monthly, 1)
    assert out["CRSP_SPvw"].tolist() == [0.01, -0.02]
    assert out["Rfree"].tolist() == out["Rfree_comp"].tolist() == [0.001, 0.002]


@pytest.mark.parametrize("k", [2, 4, 24])
def test_unsupported_k_raises(k):
    """Only calendar-aligned 1, 3, 6 and 12-month periods are defined."""
    with pytest.raises(ValueError):
        aggregate_to_k_months(_monthly({"Index": [1.0] * 24}), k)


def test_non_month_end_dates_raise():
    """Mid-month dates mean the input is not a clean monthly panel."""
    frame = pd.DataFrame(
        {"Index": [1.0, 2.0]}, index=pd.to_datetime(["2000-01-15", "2000-02-15"])
    )
    with pytest.raises(ValueError):
        aggregate_to_k_months(frame, 3)


def test_month_end_dates_with_time_of_day_raise():
    """Month-ends at 12:00 would silently turn end-of-period columns into NaN."""
    index = pd.date_range("2000-01-31", periods=3, freq="ME") + pd.Timedelta(hours=12)
    frame = pd.DataFrame({"Index": [1.0, 2.0, 3.0]}, index=index)
    with pytest.raises(ValueError):
        aggregate_to_k_months(frame, 3)
