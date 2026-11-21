"""Unit tests for pull_fred_predictors.py, offline (FRED is monkeypatched).

They pin down the two things that can silently misalign Panel R with Panel G:
FRED's period-start dates must become period ends, and the pull must start
early enough for the predictors that need history before START_DATE.
"""

import pandas as pd

import pull_fred_predictors as pfp


def test_fred_dates_move_to_period_ends():
    """FRED's 1947-01-01 means January, Q1 or the year 1947, depending on frequency."""
    frame = pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex(["1947-01-01"]))
    assert pfp.to_period_end(frame, "monthly").index[0] == pd.Timestamp("1947-01-31")
    assert pfp.to_period_end(frame, "quarterly").index[0] == pd.Timestamp("1947-03-31")
    assert pfp.to_period_end(frame, "annual").index[0] == pd.Timestamp("1947-12-31")
    assert pfp.to_period_end(frame, "monthly").index.name == "date"


def test_pull_starts_before_the_sample_and_keeps_fred_units(monkeypatch):
    """The request starts two years before START_DATE; values are not rescaled."""
    calls = {}

    def fake_datareader(series, source, start, end):
        calls["start"] = pd.Timestamp(start)
        index = pd.date_range("1945-01-01", periods=3, freq="MS")
        return pd.DataFrame({s: [4.5, 4.6, 4.7] for s in series}, index=index)

    monkeypatch.setattr(pfp.web, "DataReader", fake_datareader)
    out = pfp.pull_fred_series(
        "monthly", pd.Timestamp("1947-01-01"), pd.Timestamp("2025-12-31")
    )
    assert calls["start"] == pd.Timestamp("1945-01-01")
    assert list(out.columns) == pfp.FRED_SERIES["monthly"]
    assert out["TB3MS"].tolist() == [4.5, 4.6, 4.7]  # still percent
    assert out.index[0] == pd.Timestamp("1945-01-31")
