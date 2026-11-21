"""Pre-registered data-validation tests for the predictor panels (Appendix C.3).

The tolerances at the top of this file were fixed BEFORE any comparison between
our series and Goyal's was run. Do not loosen them after looking at results: a
failure is a finding to report (Table C1), not a number to tune. If a tolerance
must change, do it in its own commit and say why in the commit message.

The checks follow Appendix C.3:

1. Reconciliation of each rebuilt series (Panel R) against Goyal (Panel G),
   with correlation thresholds 0.99 / 0.97 / 0.95.
2. Accounting identities, exact up to floating point.
3. Independent-source cross-check: CRSP total return vs. Shiller's implied return.
4. Coverage and units: no gaps from START_DATE to END_DATE, decimals, period-end dates.
5. Aggregation: our monthly -> quarterly / annual rules reproduce Goyal's own
   Quarterly and Annual sheets.

Tests that need data on disk skip (with a message) until `doit` has built it, so
the suite stays green in CI before the WRDS pulls exist.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = pd.Timestamp(config("START_DATE"))
END_DATE = pd.Timestamp(config("END_DATE"))

###############################################################################
## Pre-registered tolerances (frozen; see module docstring)
###############################################################################

## 1. Reconciliation: minimum correlation of our series with Goyal's, by source.
CORR_MIN_PRIMARY = 0.99  # Shiller, FRED and CRSP-index series
CORR_MIN_CONSTRUCTED = 0.97  # built from stock-level or daily data
CORR_MIN_PROXY = (
    0.95  # close proxies for Ibbotson/BEA series, and series built from them
)

RECONCILIATION_CORR_MIN = {
    # Shiller D, E with CRSP month-end price
    "d/p": CORR_MIN_PRIMARY,
    "d/y": CORR_MIN_PRIMARY,
    "e/p": CORR_MIN_PRIMARY,
    "d/e": CORR_MIN_PRIMARY,
    # FRED
    "tbl": CORR_MIN_PRIMARY,
    "dfy": CORR_MIN_PRIMARY,
    "infl": CORR_MIN_PRIMARY,
    # CRSP index files and CRSP 30-day T-bill
    "ret": CORR_MIN_PRIMARY,
    "rf": CORR_MIN_PRIMARY,
    # Built from daily index returns / NYSE stock-level issuance
    "svar": CORR_MIN_CONSTRUCTED,
    "ntis": CORR_MIN_CONSTRUCTED,
    # Proxies, and tms because it uses the proxy lty
    "lty": CORR_MIN_PROXY,
    "ltr": CORR_MIN_PROXY,
    "tms": CORR_MIN_PROXY,
}
# ik exists only at quarterly/annual frequency; checked on the Finance 1 panel.
RECONCILIATION_CORR_MIN_ANNUAL_ONLY = {"ik": CORR_MIN_PROXY}

# Taken from Goyal in both panels, so they must be identical, not merely close.
GOYAL_SOURCED_MONTHLY = ["b/m", "dfr"]
GOYAL_SOURCED_ANNUAL = ["b/m", "eqis", "dfr"]

## 2. Identities hold to floating-point precision.
IDENTITY_ATOL = 1e-10
# The log price change must track CRSP's ex-dividend index return.
PRICE_CHANGE_VS_RETX_CORR_MIN = CORR_MIN_PRIMARY

## 3. CRSP vs. Shiller annual total return (Shiller P is a monthly average, so
##    the two are not identical; they must move together with no level bias).
SHILLER_CROSSCHECK_CORR_MIN = 0.95
SHILLER_CROSSCHECK_MAX_ABS_MEAN_GAP = 0.01  # 1 percentage point per year

## 4. Units: yields in decimals (a percent-coded yield would exceed 1).
MAX_DECIMAL_YIELD = 0.25
# Returns in decimals: a percent-coded return series has median |r| of several units.
MAX_MEDIAN_ABS_RETURN = 0.2

## 5. Aggregation reproduces Goyal's own Quarterly and Annual sheets.
AGGREGATION_RTOL = 1e-6
AGGREGATION_ATOL = 1e-6

###############################################################################
## Panel contract (see build_predictors.py)
###############################################################################

FREQUENCIES = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
MONTHLY_PREDICTORS = [
    "d/p", "d/y", "e/p", "d/e", "svar", "b/m", "ntis",
    "tbl", "lty", "ltr", "tms", "dfy", "dfr", "infl",
]  # fmt: skip
FINANCE1_PREDICTORS = MONTHLY_PREDICTORS + ["eqis", "ik"]
TARGET_COLUMNS = ["ret", "retx", "rf", "rf_comp", "eq_prem", "eq_prem_log",
                  "eq_prem_rfcomp", "eq_prem_log_rfcomp"]  # fmt: skip
YIELD_COLUMNS = ["tbl", "lty"]
RETURN_COLUMNS = ["ret", "retx", "rf", "ltr", "infl", "eq_prem"]
# Goyal input codes that aggregate.py aggregates (one rule per code).
AGGREGATED_INPUTS = [
    "Index", "D12", "E12", "b/m", "tbl", "AAA", "BAA", "lty", "ntis",
    "Rfree", "infl", "ltr", "corpr", "svar", "CRSP_SPvw", "CRSP_SPvwx",
]  # fmt: skip


def _read_parquet_or_skip(name):
    """Load a parquet file from DATA_DIR, or skip the test if it is not built yet."""
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"{path.name} not found in DATA_DIR; run `doit` first.")
    return pd.read_parquet(path)


def _panel_files():
    """All (panel, frequency) file names in the contract."""
    return [
        f"panel_{panel}_{freq}.parquet" for panel in ["G", "R"] for freq in FREQUENCIES
    ]


def _expected_period_ends(k, start=START_DATE, end=END_DATE):
    """Calendar-aligned period-end dates for k-month periods between start and end."""
    month_ends = pd.date_range(start, end, freq="ME")
    return month_ends[month_ends.month % k == 0]


###############################################################################
## 1. Reconciliation against Goyal
###############################################################################


@pytest.mark.parametrize("series,corr_min", RECONCILIATION_CORR_MIN.items())
def test_rebuilt_series_reconcile_with_goyal_monthly(series, corr_min):
    """Each rebuilt monthly series correlates with Goyal's above its threshold."""
    panel_g = _read_parquet_or_skip("panel_G_monthly.parquet")
    panel_r = _read_parquet_or_skip("panel_R_monthly.parquet")
    both = pd.concat([panel_r[series], panel_g[series]], axis=1, keys=["R", "G"])
    both = both.dropna()
    assert len(both) > 0, f"No overlapping observations for {series}"
    corr = both["R"].corr(both["G"])
    assert corr >= corr_min, f"{series}: corr {corr:.4f} < {corr_min}"


@pytest.mark.parametrize("series,corr_min", RECONCILIATION_CORR_MIN_ANNUAL_ONLY.items())
def test_rebuilt_series_reconcile_with_goyal_annual(series, corr_min):
    """Series that exist only at annual frequency reconcile on the Finance 1 panel."""
    panel_g = _read_parquet_or_skip("finance1_panel_G.parquet")
    panel_r = _read_parquet_or_skip("finance1_panel_R.parquet")
    both = pd.concat([panel_r[series], panel_g[series]], axis=1, keys=["R", "G"])
    both = both.dropna()
    assert len(both) > 0, f"No overlapping observations for {series}"
    corr = both["R"].corr(both["G"])
    assert corr >= corr_min, f"{series}: corr {corr:.4f} < {corr_min}"


def test_goyal_sourced_series_identical_in_both_panels():
    """b/m, eqis and dfr come from Goyal in Panel R too, so they match exactly."""
    monthly_g = _read_parquet_or_skip("panel_G_monthly.parquet")
    monthly_r = _read_parquet_or_skip("panel_R_monthly.parquet")
    pd.testing.assert_frame_equal(
        monthly_r[GOYAL_SOURCED_MONTHLY], monthly_g[GOYAL_SOURCED_MONTHLY]
    )
    annual_g = _read_parquet_or_skip("finance1_panel_G.parquet")
    annual_r = _read_parquet_or_skip("finance1_panel_R.parquet")
    pd.testing.assert_frame_equal(
        annual_r[GOYAL_SOURCED_ANNUAL], annual_g[GOYAL_SOURCED_ANNUAL]
    )


###############################################################################
## 2. Accounting identities
###############################################################################


@pytest.mark.parametrize("file_name", _panel_files())
def test_identity_payout_equals_dp_minus_ep(file_name):
    """d/e = d/p - e/p, because all three are differences of the same logs."""
    panel = _read_parquet_or_skip(file_name)
    np.testing.assert_allclose(
        panel["d/e"], panel["d/p"] - panel["e/p"], rtol=0, atol=IDENTITY_ATOL
    )


@pytest.mark.parametrize("file_name", _panel_files())
def test_identity_term_spread_equals_lty_minus_tbl(file_name):
    """tms = lty - tbl."""
    panel = _read_parquet_or_skip(file_name)
    np.testing.assert_allclose(
        panel["tms"], panel["lty"] - panel["tbl"], rtol=0, atol=IDENTITY_ATOL
    )


@pytest.mark.parametrize("file_name", _panel_files())
def test_identity_dy_minus_dp_tracks_crsp_ex_dividend_return(file_name):
    """d/y - d/p is the log price change, which must track CRSP's ex-dividend return."""
    panel = _read_parquet_or_skip(file_name)
    log_price_change = panel["d/y"] - panel["d/p"]
    log_retx = np.log1p(panel["retx"])
    both = pd.concat([log_price_change, log_retx], axis=1).dropna()
    corr = both.iloc[:, 0].corr(both.iloc[:, 1])
    assert corr >= PRICE_CHANGE_VS_RETX_CORR_MIN, f"corr {corr:.4f}"


@pytest.mark.parametrize("panel", ["G", "R"])
def test_identity_compounded_monthly_returns_equal_annual_return(panel):
    """Twelve compounded monthly returns equal the annual return in the annual panel."""
    monthly = _read_parquet_or_skip(f"panel_{panel}_monthly.parquet")
    annual = _read_parquet_or_skip(f"panel_{panel}_annual.parquet")
    by_year = (1 + monthly["ret"]).groupby(monthly.index.year)
    compounded = (by_year.prod() - 1)[by_year.count() == 12]
    annual_ret = annual["ret"].set_axis(annual.index.year)
    years = annual_ret.index.intersection(compounded.index)
    assert len(years) > 0
    np.testing.assert_allclose(
        annual_ret.loc[years], compounded.loc[years], rtol=0, atol=IDENTITY_ATOL
    )


###############################################################################
## 3. Independent-source cross-check
###############################################################################


def test_crsp_total_return_matches_shiller_implied_return():
    """Annual CRSP S&P 500 total return vs. Shiller's (P_t + D_t/12)/P_{t-1}, compounded."""
    shiller = _read_parquet_or_skip("shiller.parquet")
    panel_r = _read_parquet_or_skip("panel_R_annual.parquet")
    monthly_implied = (shiller["P"] + shiller["D"] / 12) / shiller["P"].shift(1) - 1
    annual_implied = (1 + monthly_implied).groupby(shiller.index.year).prod() - 1
    crsp = panel_r["ret"].groupby(panel_r.index.year).last()
    both = pd.concat([crsp, annual_implied], axis=1, keys=["crsp", "shiller"]).dropna()
    assert both["crsp"].corr(both["shiller"]) >= SHILLER_CROSSCHECK_CORR_MIN
    mean_gap = (both["crsp"] - both["shiller"]).mean()
    assert abs(mean_gap) <= SHILLER_CROSSCHECK_MAX_ABS_MEAN_GAP


###############################################################################
## 4. Coverage and units
###############################################################################


@pytest.mark.parametrize("file_name", _panel_files())
def test_panel_covers_sample_with_no_gaps(file_name):
    """Period-end dates from START_DATE to END_DATE, one row each, no missing values."""
    panel = _read_parquet_or_skip(file_name)
    freq = file_name.removesuffix(".parquet").split("_")[-1]
    expected_dates = _expected_period_ends(FREQUENCIES[freq])
    assert panel.index.equals(pd.DatetimeIndex(expected_dates, name="date"))
    columns = MONTHLY_PREDICTORS + TARGET_COLUMNS
    missing = panel[columns].isna().sum()
    assert missing.sum() == 0, f"Missing values:\n{missing[missing > 0]}"


@pytest.mark.parametrize("panel", ["G", "R"])
def test_finance1_panel_covers_sample_with_no_gaps(panel):
    """The annual Finance 1 panel has all 16 predictors for every year."""
    df = _read_parquet_or_skip(f"finance1_panel_{panel}.parquet")
    expected_dates = _expected_period_ends(12)
    assert df.index.equals(pd.DatetimeIndex(expected_dates, name="date"))
    assert df[FINANCE1_PREDICTORS + TARGET_COLUMNS].isna().sum().sum() == 0


@pytest.mark.parametrize("file_name", _panel_files())
def test_panel_units_are_decimals(file_name):
    """Yields and returns are stored as decimals, not percent."""
    panel = _read_parquet_or_skip(file_name)
    for col in YIELD_COLUMNS:
        assert panel[col].abs().max() < MAX_DECIMAL_YIELD, col
    for col in RETURN_COLUMNS:
        assert panel[col].abs().median() < MAX_MEDIAN_ABS_RETURN, col


###############################################################################
## 5. Aggregation reproduces Goyal's Quarterly and Annual sheets
###############################################################################


@pytest.mark.parametrize("sheet,k", [("quarterly", 3), ("annual", 12)])
@pytest.mark.parametrize("series", AGGREGATED_INPUTS)
def test_aggregation_reproduces_goyal_sheet(sheet, k, series):
    """aggregate.py applied to Goyal's Monthly sheet reproduces his coarser sheet."""
    aggregate = pytest.importorskip("aggregate")
    monthly = _read_parquet_or_skip("goyal_monthly.parquet")
    published = _read_parquet_or_skip(f"goyal_{sheet}.parquet")
    ours = aggregate.aggregate_to_k_months(monthly, k)
    window = slice(START_DATE, END_DATE)
    both = pd.concat(
        [ours.loc[window, series], published.loc[window, series]],
        axis=1,
        keys=["ours", "goyal"],
    ).dropna()
    assert len(both) > 0, f"No overlapping observations for {series}"
    np.testing.assert_allclose(
        both["ours"],
        both["goyal"],
        rtol=AGGREGATION_RTOL,
        atol=AGGREGATION_ATOL,
        err_msg=f"{series} ({sheet})",
    )
