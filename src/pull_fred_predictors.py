"""Pull the FRED series behind the rebuilt Welch-Goyal predictors (Panel R).

No API key is needed. The series are saved in FRED's own units (percent
yields, index levels, billions or millions of dollars), only re-dated to
period ends. ``rebuild_inputs.py`` turns them into Goyal-coded inputs.

=================  =========  ===============================================
FRED series        Frequency  Used for
=================  =========  ===============================================
TB3MS              monthly    tbl (3-month T-bill, secondary market)
AAA, BAA           monthly    dfy = BAA - AAA (Moody's seasoned yields)
LTGOVTBD           monthly    lty proxy, 1925-2000 (long-term government bonds)
GS20               monthly    lty proxy after 2000-06 (20-year constant maturity)
CPIAUCNS           monthly    infl (CPI-U, not seasonally adjusted)
PNFI               quarterly  ik numerator (private nonresidential fixed investment)
K1NTOTL1ES000      annual     ik denominator (net stock of private nonresidential
                              fixed assets, current cost)
=================  =========  ===============================================

Outputs in DATA_DIR (never committed): ``fred_monthly.parquet``,
``fred_quarterly.parquet``, ``fred_annual.parquet``. Each has a period-end
DatetimeIndex named ``date``: FRED dates periods by their first day
(1947-01-01 is January, Q1 or the year 1947), and they are moved to the last day.

The pull starts PULL_LOOKBACK_YEARS before START_DATE, because some
predictors need history before the sample begins: d/y needs the previous
month's price, ntis needs twelve months of issuance, and ik needs the previous
year's capital stock.
"""

from pathlib import Path

import pandas as pd
import pandas_datareader.data as web

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = pd.Timestamp(config("START_DATE"))
END_DATE = pd.Timestamp(config("END_DATE"))
PULL_LOOKBACK_YEARS = 2

FRED_SERIES = {
    "monthly": ["TB3MS", "AAA", "BAA", "LTGOVTBD", "GS20", "CPIAUCNS"],
    "quarterly": ["PNFI"],
    "annual": ["K1NTOTL1ES000"],
}
PERIOD_END = {
    "monthly": pd.offsets.MonthEnd(0),
    "quarterly": pd.offsets.QuarterEnd(0),
    "annual": pd.offsets.YearEnd(0),
}


def pull_start(start_date=START_DATE):
    """First date to pull: START_DATE minus the lookback the predictors need."""
    return pd.Timestamp(start_date) - pd.DateOffset(years=PULL_LOOKBACK_YEARS)


def to_period_end(frame, freq):
    """Move FRED's period-start dates to period ends and name the index ``date``."""
    out = frame.copy()
    out.index = pd.DatetimeIndex(out.index + PERIOD_END[freq], name="date")
    return out


def pull_fred_series(freq, start_date=START_DATE, end_date=END_DATE):
    """Download the FRED series for one frequency, in FRED's units, period-end dated."""
    raw = web.DataReader(FRED_SERIES[freq], "fred", pull_start(start_date), end_date)
    return to_period_end(raw, freq).astype(float)


def pull_fred_predictors(start_date=START_DATE, end_date=END_DATE):
    """Download every FRED series; returns {"monthly": df, "quarterly": df, "annual": df}."""
    return {freq: pull_fred_series(freq, start_date, end_date) for freq in FRED_SERIES}


def load_fred_predictors(freq="monthly", data_dir=DATA_DIR):
    """Load one saved frequency. Run this module first to pull the data."""
    if freq not in FRED_SERIES:
        raise ValueError(f"Unknown frequency {freq!r}; use one of {list(FRED_SERIES)}.")
    return pd.read_parquet(Path(data_dir) / f"fred_{freq}.parquet")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in pull_fred_predictors().items():
        df.to_parquet(DATA_DIR / f"fred_{name}.parquet")
        print(f"fred_{name}: {df.shape[0]} rows, {df.index.min().date()} "
              f"to {df.index.max().date()}")  # fmt: skip
