"""Unit tests for rebuild_inputs.py: one hand-checkable case per construction rule.

Each test pins down a choice that would silently change Panel R if broken:
the lty splice date, the CPI gap rule, the Welch-Goyal net-issue formula,
the T-bill extension past 2024, svar's sum of squares, and ik's timing.
"""

import numpy as np
import pandas as pd
import pytest

import rebuild_inputs as ri


def _months(start, n):
    return pd.date_range(start, periods=n, freq="ME", name="date")


def test_lty_splice_switches_after_june_2000_and_converts_to_decimals():
    """LTGOVTBD through 2000-06, GS20 from 2000-07, both divided by 100."""
    index = _months("2000-05-31", 3)
    ltgovtbd = pd.Series([6.0, 6.1, np.nan], index=index)
    gs20 = pd.Series([5.0, 5.1, 5.2], index=index)
    lty = ri.splice_long_term_yield(ltgovtbd, gs20)
    assert lty.tolist() == pytest.approx([0.060, 0.061, 0.052])


def test_single_missing_cpi_month_is_filled_geometrically_and_longer_gaps_are_not():
    """One missing month gets the geometric mean of its neighbours; two in a row stay missing."""
    cpi = pd.Series(
        [100.0, np.nan, 121.0, np.nan, np.nan, 130.0], index=_months("2025-01-31", 6)
    )
    filled = ri.fill_single_missing_months(cpi)
    assert filled.iloc[1] == pytest.approx(110.0)
    assert filled.iloc[3:5].isna().all()


def test_inflation_splits_a_two_month_change_evenly_across_a_missing_month():
    """With October missing, September->November inflation is split into two equal months."""
    cpi = pd.Series([324.8, np.nan, 324.122], index=_months("2025-09-30", 3))
    infl = ri.inflation_from_cpi(cpi)
    expected = np.sqrt(324.122 / 324.8) - 1
    assert infl.iloc[1] == pytest.approx(expected)
    assert infl.iloc[2] == pytest.approx(expected)


def test_net_equity_expansion_counts_issues_beyond_price_changes():
    """Cap growing only with returns is zero issuance; a 10 issue shows for 12 months."""
    n = 14
    vwretx = np.full(n, 0.01)
    mcap = 1000 * np.cumprod(1 + vwretx)
    mcap[13] += 10  # a 10 issue in the last month, on top of price growth
    nyse = pd.DataFrame(
        {"mcap": mcap, "lagcap_x_retx": vwretx, "lagcap_with_retx": np.ones(n)},
        index=_months("2000-01-31", n),
    )
    ntis = ri.net_equity_expansion(nyse)
    # Month 0 has no previous cap, so the first full 12-month sum is at month 12.
    assert ntis.iloc[:12].isna().all()
    assert ntis.iloc[12] == pytest.approx(0.0, abs=1e-12)
    assert ntis.iloc[13] == pytest.approx(10 / mcap[13])


def test_net_equity_expansion_rejects_a_gap_in_months():
    """A missing month would make the 12-month sum span 13 months."""
    index = pd.DatetimeIndex(["2000-01-31", "2000-03-31"])
    nyse = pd.DataFrame(
        {
            "mcap": [1.0, 1.0],
            "lagcap_x_retx": [0.0, 0.0],
            "lagcap_with_retx": [1.0, 1.0],
        },
        index=index,
    )
    with pytest.raises(ValueError):
        ri.net_equity_expansion(nyse)


def test_stock_variance_sums_squared_daily_returns_by_month():
    """svar for January = 0.01^2 + 0.02^2; dates move to the month end."""
    daily = pd.Series(
        [0.01, -0.02, 0.03],
        index=pd.to_datetime(["2000-01-03", "2000-01-31", "2000-02-01"]),
    )
    svar = ri.stock_variance(daily)
    assert svar.loc["2000-01-31"] == pytest.approx(0.0005)
    assert svar.loc["2000-02-29"] == pytest.approx(0.0009)


def test_risk_free_uses_t30ret_then_previous_month_yield_after_it_ends():
    """Months after the last t30ret use exp(previous yield * days / 365) - 1; gaps inside stay NaN."""
    index = _months("2024-10-31", 4)
    treasury = pd.DataFrame(
        {
            "t30ret": [0.0040, np.nan, 0.0037, np.nan],
            "rf1m_ytm_pct": [4.5, 4.2, 4.0, 3.9],
            "rf1m_caldt": pd.to_datetime(
                ["2024-10-31", "2024-11-29", "2024-12-31", "2025-01-31"]
            ),
        },
        index=index,
    )
    rfree = ri.risk_free_return(treasury)
    assert rfree.iloc[0] == 0.0040
    assert np.isnan(rfree.iloc[1])  # interior gap is not filled
    assert rfree.iloc[3] == pytest.approx(np.exp(0.040 * 31 / 365) - 1)


def test_ik_divides_quarterly_investment_by_previous_year_end_capital():
    """ik_q = (PNFI_q / 4) / K_{previous year}, with K converted from millions to billions."""
    pnfi = pd.Series(
        [4000.0, 4400.0], index=pd.DatetimeIndex(["2001-03-31", "2001-12-31"])
    )
    capital = pd.Series(
        [40_000_000.0, 50_000_000.0],
        index=pd.DatetimeIndex(["2000-12-31", "2001-12-31"]),
    )
    ik = ri.investment_capital_ratio(pnfi, capital)
    assert ik.loc["2001-03-31"] == pytest.approx(1000 / 40_000)
    annual = ri.annual_ik(pnfi, capital)
    assert annual.index.equals(pd.DatetimeIndex(["2001-12-31"]))
    assert annual.iloc[0] == pytest.approx(1100 / 40_000)  # Q4 value, 2000 capital
