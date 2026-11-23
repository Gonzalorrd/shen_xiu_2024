"""Rebuild Goyal-coded monthly inputs from primary sources (Panel R): cleaning only.

The pulls (pull_crsp.py, pull_fred_predictors.py, pull_shiller.py) save
sources in their own units. This module turns them into the same input
columns Goyal posts, in decimals and on a month-end index, so Panel R goes
through exactly the same aggregation and predictor code as Panel G
(build_predictors.py).

======  =====================================================  ===============
Input   Construction                                           Status
======  =====================================================  ===============
Index   CRSP S&P 500 level (spindx) at month end               rebuilt
D12     Shiller D (already a trailing 12-month value)           rebuilt
E12     Shiller E (already a trailing 12-month value)           rebuilt
tbl     FRED TB3MS / 100                                       rebuilt
AAA     FRED AAA / 100                                         rebuilt
BAA     FRED BAA / 100                                         rebuilt
lty     FRED LTGOVTBD to 2000-06, then GS20; / 100             proxy
ntis    12-month sum of NYSE net issues / NYSE market cap       rebuilt
Rfree   CRSP 30-day T-bill return; after it ends (2024-12),     rebuilt
        the 1-month bill yield at the previous month end,
        exp(y * days / 365) - 1
infl    FRED CPIAUCNS, monthly change                          rebuilt
ltr     CRSP 20-year fixed-term Treasury index return           proxy
svar    sum of squared daily CRSP S&P 500 returns (vwretd)     rebuilt
CRSP_*  CRSP S&P 500 returns with (vwretd) / without (vwretx)  rebuilt
b/m     Goyal (Value Line book values are commercial)          from Goyal
corpr   Goyal (Ibbotson; dfr is then replaced by Goyal's dfr)  from Goyal
ik      FRED PNFI / 4 over the previous year-end BEA net       proxy
        stock of private nonresidential fixed assets; annual
        value = Q4 (Goyal's convention)
======  =====================================================  ===============

Choices fixed before comparing with Goyal:

- svar uses CRSP's daily value-weighted S&P 500 return (vwretd), because the
  daily price return (sprtrn) is missing before July 1962.
- Net issues follow Welch and Goyal (2008): Mcap_t - Mcap_{t-1} (1 + vwretx_t).
- October 2025 CPI was never published (federal shutdown). A single missing
  CPI month is filled geometrically between its neighbours, which splits the
  two-month change evenly; longer gaps stay missing. Goyal's file does the same.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pull_crsp
import pull_fred_predictors
import pull_goyal
import pull_shiller
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

LTY_SWITCH_DATE = pd.Timestamp("2000-06-30")  # last month of FRED's LTGOVTBD
MONTHS_PER_YEAR = 12
DAYS_PER_YEAR = 365
MILLIONS_PER_BILLION = 1_000
REBUILT_INPUTS = [
    "Index", "D12", "E12", "b/m", "tbl", "AAA", "BAA", "lty", "ntis",
    "Rfree", "infl", "ltr", "corpr", "svar", "CRSP_SPvw", "CRSP_SPvwx",
]  # fmt: skip


def splice_long_term_yield(ltgovtbd, gs20, switch_date=LTY_SWITCH_DATE):
    """lty proxy in decimals: LTGOVTBD through switch_date, GS20 afterwards."""
    spliced = ltgovtbd.where(ltgovtbd.index <= switch_date, gs20)
    return (spliced / 100).rename("lty")


def fill_single_missing_months(level):
    """Fill isolated one-month gaps in a level series with the geometric mean of its neighbours."""
    previous, following = level.shift(1), level.shift(-1)
    isolated = level.isna() & previous.notna() & following.notna()
    filled = level.copy()
    filled[isolated] = np.sqrt(previous[isolated] * following[isolated])
    return filled


def inflation_from_cpi(cpi):
    """Monthly CPI inflation, after filling isolated missing months."""
    return fill_single_missing_months(cpi).pct_change(fill_method=None).rename("infl")


def _require_full_monthly_index(series_or_frame, name):
    """Raise unless the index is every month-end between its first and last dates."""
    index = series_or_frame.index
    full = pd.date_range(index.min(), index.max(), freq="ME")
    if not index.equals(full):
        raise ValueError(f"{name} must be on a gap-free month-end index.")


def net_equity_expansion(nyse):
    """ntis: 12-month sum of NYSE net issues over current NYSE market cap.

    nyse has columns mcap, lagcap_x_retx and lagcap_with_retx (pull_crsp.py).
    """
    _require_full_monthly_index(nyse, "NYSE aggregates")
    vwretx = nyse["lagcap_x_retx"] / nyse["lagcap_with_retx"]
    net_issues = nyse["mcap"] - nyse["mcap"].shift(1) * (1 + vwretx)
    rolling_issues = net_issues.rolling(
        MONTHS_PER_YEAR, min_periods=MONTHS_PER_YEAR
    ).sum()
    return (rolling_issues / nyse["mcap"]).rename("ntis")


def stock_variance(daily_returns):
    """svar: sum of squared daily returns within each month, on a month-end index."""
    month_ends = pd.DatetimeIndex(
        daily_returns.index + pd.offsets.MonthEnd(0), name="date"
    )
    return (daily_returns**2).groupby(month_ends).sum(min_count=1).rename("svar")


def risk_free_return(treasury):
    """Monthly Rfree: CRSP t30ret, extended past its last month with the 1-month bill yield.

    Only months AFTER the last t30ret observation are filled; a gap inside the
    t30ret history stays missing.
    """
    t30ret = treasury["t30ret"]
    last_t30 = t30ret.last_valid_index()
    previous_yield = treasury["rf1m_ytm_pct"].shift(1) / 100
    days_held = treasury["rf1m_caldt"].diff().dt.days
    from_yield = np.exp(previous_yield * days_held / DAYS_PER_YEAR) - 1
    after_t30 = treasury.index > last_t30
    return t30ret.where(~after_t30, from_yield).rename("Rfree")


def investment_capital_ratio(pnfi_quarterly, capital_annual):
    """ik by quarter: PNFI (SAAR, billions) / 4 over last year-end's net stock (millions).

    Returns the quarterly series; Goyal's annual ik is its Q4 value.
    """
    previous_year_capital = capital_annual.copy()
    previous_year_capital.index = previous_year_capital.index.year + 1
    capital = pnfi_quarterly.index.year.map(previous_year_capital)
    capital_billions = (
        pd.Series(capital, index=pnfi_quarterly.index) / MILLIONS_PER_BILLION
    )
    return (pnfi_quarterly / 4 / capital_billions).rename("ik")


def annual_ik(pnfi_quarterly, capital_annual):
    """Annual ik on a year-end index: the Q4 quarterly value, as in Goyal's sheet."""
    quarterly = investment_capital_ratio(pnfi_quarterly, capital_annual)
    return quarterly[quarterly.index.month == 12]


def build_rebuilt_monthly_inputs(data_dir=DATA_DIR):
    """Goyal-coded monthly inputs for Panel R, from the saved pulls in data_dir."""
    sp500 = pull_crsp.load_crsp("crsp_sp500_monthly", data_dir)
    daily = pull_crsp.load_crsp("crsp_sp500_daily", data_dir)
    treasury = pull_crsp.load_crsp("crsp_treasury_monthly", data_dir)
    nyse = pull_crsp.load_crsp("crsp_nyse_monthly", data_dir)
    fred = pull_fred_predictors.load_fred_predictors("monthly", data_dir)
    shiller = pull_shiller.load_shiller(data_dir)
    goyal = pull_goyal.load_goyal("monthly", data_dir)

    index = sp500.index
    _require_full_monthly_index(sp500, "CRSP S&P 500 monthly file")
    inputs = pd.DataFrame(
        {
            "Index": sp500["spindx"],
            "D12": shiller["D"],
            "E12": shiller["E"],
            "b/m": goyal["b/m"],
            "tbl": fred["TB3MS"] / 100,
            "AAA": fred["AAA"] / 100,
            "BAA": fred["BAA"] / 100,
            "lty": splice_long_term_yield(fred["LTGOVTBD"], fred["GS20"]),
            "ntis": net_equity_expansion(nyse),
            "Rfree": risk_free_return(treasury),
            "infl": inflation_from_cpi(fred["CPIAUCNS"]),
            "ltr": treasury["ft20_ret_pct"] / 100,
            "corpr": goyal["corpr"],
            "svar": stock_variance(daily["vwretd"]),
            "CRSP_SPvw": sp500["vwretd"],
            "CRSP_SPvwx": sp500["vwretx"],
        },
        index=index,
    )
    inputs.index.name = "date"
    return inputs[REBUILT_INPUTS].astype(float)


def build_rebuilt_annual_ik(data_dir=DATA_DIR):
    """Annual ik for the Finance 1 panel R, from the saved FRED pulls."""
    pnfi = pull_fred_predictors.load_fred_predictors("quarterly", data_dir)["PNFI"]
    capital = pull_fred_predictors.load_fred_predictors("annual", data_dir)[
        "K1NTOTL1ES000"
    ]
    return annual_ik(pnfi, capital)
