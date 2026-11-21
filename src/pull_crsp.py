"""Pull the CRSP series behind the rebuilt predictors (Panel R) from WRDS.

Needs WRDS_USERNAME in .env (the password is read from the pgpass file the
wrds package sets up on first login). Uses the CRSP CIZ (v2) tables, which run
past 2024. Legacy tables are used only where the CIZ files have no equivalent.

============================  ===============================================
CRSP table                    Used for
============================  ===============================================
crsp_m_indexes.msp500_v2      Index (spindx at month end); S&P 500 returns with
                              (vwretd) and without (vwretx) dividends
crsp_m_indexes.dsp500_v2      svar: sum of squared daily S&P 500 returns (sprtrn)
crsp_m_indexes.mcti           Rfree: 30-day T-bill return (t30ret). Legacy table,
                              ends 2024-12
crsp_m_treasuries.tfz_mth_rf  Rfree after mcti ends: 1-month bill yield (tmytm)
crsp_m_treasuries.tfz_mth_ft  ltr proxy: 20-year fixed-term index return (tmretadj),
                              identical to mcti's b20ret where both exist
crsp_m_stock.msf_v2           ntis: NYSE equity market cap and value-weighted
                              ex-dividend return, aggregated in SQL by month
============================  ===============================================

Values are saved in CRSP's units (tmytm and tmretadj in percent; market caps
in thousands of dollars), re-dated to month ends, and turned into Goyal-coded
inputs in ``rebuild_inputs.py``. Outputs in DATA_DIR (never committed):
``crsp_sp500_monthly.parquet``, ``crsp_sp500_daily.parquet``,
``crsp_treasury_monthly.parquet``, ``crsp_nyse_monthly.parquet``.

The NYSE aggregation follows Welch and Goyal (2008): net issues are
Mcap_t - Mcap_{t-1} (1 + vwretx_t), where Mcap is the total market cap of NYSE
stocks and vwretx is their value-weighted return without dividends. This
counts listings and delistings as issues and repurchases, as Welch and Goyal do.
"NYSE stocks" are rows with primaryexch = 'N' and securitytype = 'EQTY'
(common shares, ADRs, units), which excludes closed-end funds and ETFs.
"""

from pathlib import Path

import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = pd.Timestamp(config("START_DATE"))
END_DATE = pd.Timestamp(config("END_DATE"))
WRDS_USERNAME = config("WRDS_USERNAME", default="", cast=str)
PULL_LOOKBACK_YEARS = 2

RISK_FREE_1M_ID = 2000001  # CRSP Risk Free Rates - 1-Month (Nominal)
FIXED_TERM_20Y_ID = 2000008  # CRSP Fixed Term Index - 20-Year (Nominal)


def pull_window(start_date=START_DATE, end_date=END_DATE):
    """(first, last) dates to pull as ISO strings, with the lookback before START_DATE."""
    first = pd.Timestamp(start_date) - pd.DateOffset(years=PULL_LOOKBACK_YEARS)
    last = pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)
    return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")


def sp500_monthly_query(first, last):
    """SQL for the monthly S&P 500 index file."""
    return f"""
        SELECT caldt, spindx, sprtrn, vwretd, vwretx
        FROM crsp_m_indexes.msp500_v2
        WHERE caldt BETWEEN '{first}' AND '{last}'
        ORDER BY caldt
    """


def sp500_daily_query(first, last):
    """SQL for the daily S&P 500 index file."""
    return f"""
        SELECT caldt, sprtrn, vwretd
        FROM crsp_m_indexes.dsp500_v2
        WHERE caldt BETWEEN '{first}' AND '{last}'
        ORDER BY caldt
    """


def treasury_queries(first, last):
    """SQL for the 30-day T-bill return, the 1-month bill yield and the 20-year index."""
    return {
        "t30ret": f"""
            SELECT caldt, t30ret
            FROM crsp_m_indexes.mcti
            WHERE caldt BETWEEN '{first}' AND '{last}'
            ORDER BY caldt
        """,
        "rf1m": f"""
            SELECT mcaldt AS caldt, tmytm AS rf1m_ytm_pct
            FROM crsp_m_treasuries.tfz_mth_rf
            WHERE kytreasnox = {RISK_FREE_1M_ID}
              AND mcaldt BETWEEN '{first}' AND '{last}'
            ORDER BY mcaldt
        """,
        "ft20": f"""
            SELECT mcaldt AS caldt, tmretadj AS ft20_ret_pct
            FROM crsp_m_treasuries.tfz_mth_ft
            WHERE kytreasnox = {FIXED_TERM_20Y_ID}
              AND mcaldt BETWEEN '{first}' AND '{last}'
            ORDER BY mcaldt
        """,
    }


def nyse_aggregates_query(first, last):
    """SQL that aggregates NYSE equities to one row per month.

    lagcap_x_retx / lagcap_with_retx is the value-weighted ex-dividend return,
    using only stocks that have both a previous market cap and a return.
    """
    return f"""
        SELECT yyyymm,
               SUM(mthcap) AS mcap,
               SUM(CASE WHEN mthretx IS NOT NULL AND mthprevcap IS NOT NULL
                        THEN mthprevcap * mthretx END) AS lagcap_x_retx,
               SUM(CASE WHEN mthretx IS NOT NULL AND mthprevcap IS NOT NULL
                        THEN mthprevcap END) AS lagcap_with_retx,
               COUNT(mthcap) AS n_stocks
        FROM crsp_m_stock.msf_v2
        WHERE primaryexch = 'N'
          AND securitytype = 'EQTY'
          AND mthcaldt BETWEEN '{first}' AND '{last}'
        GROUP BY yyyymm
        ORDER BY yyyymm
    """


def month_end_index(dates):
    """Month-end DatetimeIndex named ``date`` from trading dates or yyyymm codes."""
    dates = pd.Series(dates)
    if pd.api.types.is_numeric_dtype(dates):
        dates = pd.to_datetime(dates.astype(int).astype(str), format="%Y%m")
    else:
        dates = pd.to_datetime(dates)
    return pd.DatetimeIndex(dates + pd.offsets.MonthEnd(0), name="date")


def _by_month(frame, date_column):
    """Index a monthly query result by month end, checking one row per month."""
    out = frame.copy()
    out.index = month_end_index(out.pop(date_column))
    if not out.index.is_unique:
        raise ValueError(
            f"More than one row per month in a CRSP result ({date_column})."
        )
    return out.astype(float)


def pull_sp500_monthly(db, first, last):
    """Monthly S&P 500 level and returns, indexed by month end."""
    return _by_month(db.raw_sql(sp500_monthly_query(first, last)), "caldt")


def pull_sp500_daily(db, first, last):
    """Daily S&P 500 returns, indexed by trading date."""
    daily = db.raw_sql(sp500_daily_query(first, last), date_cols=["caldt"])
    daily = daily.set_index(pd.DatetimeIndex(daily.pop("caldt"), name="date"))
    return daily.astype(float)


def pull_treasury_monthly(db, first, last):
    """t30ret, the 1-month bill yield (with its date) and the 20-year index return, by month."""
    queries = treasury_queries(first, last)
    t30 = _by_month(db.raw_sql(queries["t30ret"]), "caldt")
    rf1m = db.raw_sql(queries["rf1m"], date_cols=["caldt"])
    rf1m_dates = pd.to_datetime(rf1m["caldt"])
    rf1m = _by_month(rf1m, "caldt")
    rf1m["rf1m_caldt"] = rf1m_dates.values  # actual quote date, for day counts
    ft20 = _by_month(db.raw_sql(queries["ft20"]), "caldt")
    return pd.concat([t30, rf1m, ft20], axis=1)


def pull_nyse_monthly(db, first, last):
    """Monthly NYSE equity aggregates for ntis, indexed by month end."""
    return _by_month(db.raw_sql(nyse_aggregates_query(first, last)), "yyyymm")


def pull_crsp(start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME):
    """Run every CRSP query on one WRDS connection; returns {file stem: DataFrame}."""
    import wrds  # imported here so the SQL can be tested without WRDS installed

    first, last = pull_window(start_date, end_date)
    db = wrds.Connection(wrds_username=wrds_username)
    try:
        return {
            "crsp_sp500_monthly": pull_sp500_monthly(db, first, last),
            "crsp_sp500_daily": pull_sp500_daily(db, first, last),
            "crsp_treasury_monthly": pull_treasury_monthly(db, first, last),
            "crsp_nyse_monthly": pull_nyse_monthly(db, first, last),
        }
    finally:
        db.close()


def load_crsp(name, data_dir=DATA_DIR):
    """Load one saved CRSP file, e.g. load_crsp("crsp_sp500_monthly")."""
    return pd.read_parquet(Path(data_dir) / f"{name}.parquet")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for stem, df in pull_crsp().items():
        df.to_parquet(DATA_DIR / f"{stem}.parquet")
        print(f"{stem}: {df.shape[0]} rows, {df.index.min().date()} "
              f"to {df.index.max().date()}")  # fmt: skip
