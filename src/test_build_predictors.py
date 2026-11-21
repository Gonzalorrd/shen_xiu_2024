"""Unit tests for build_predictors.py: predictor formulas, timing and the contract.

The formulas are checked on a two-period input whose answers are known by
hand. The timing test guards the most consequential choice in the panel:
d/y uses the PREVIOUS period's price, and nothing else is lagged.
"""

import numpy as np
import pandas as pd
import pytest

import build_predictors as bp


def _inputs():
    """Two annual periods of Goyal-coded inputs with round numbers."""
    index = pd.DatetimeIndex(["2000-12-31", "2001-12-31"], name="date")
    return pd.DataFrame(
        {
            "Index": [100.0, 125.0],
            "D12": [2.0, 2.5],
            "E12": [5.0, 4.0],
            "b/m": [0.3, 0.35],
            "tbl": [0.05, 0.02],
            "AAA": [0.07, 0.06],
            "BAA": [0.08, 0.075],
            "lty": [0.06, 0.05],
            "ntis": [0.01, -0.01],
            "infl": [0.03, 0.02],
            "ltr": [0.04, 0.10],
            "corpr": [0.05, 0.09],
            "svar": [0.02, 0.03],
            "CRSP_SPvw": [0.10, -0.05],
            "CRSP_SPvwx": [0.08, -0.07],
            "Rfree": [0.05, 0.02],
            "Rfree_comp": [0.049, 0.021],
        },
        index=index,
    )


def test_predictor_formulas_match_welch_goyal_definitions():
    """Log ratios and spreads follow the definitions in the contract docstring."""
    out = bp.compute_predictors(_inputs()).loc["2001-12-31"]
    assert out["d/p"] == pytest.approx(np.log(2.5 / 125.0))
    assert out["e/p"] == pytest.approx(np.log(4.0 / 125.0))
    assert out["d/e"] == pytest.approx(np.log(2.5 / 4.0))
    assert out["tms"] == pytest.approx(0.05 - 0.02)
    assert out["dfy"] == pytest.approx(0.075 - 0.06)
    assert out["dfr"] == pytest.approx(0.09 - 0.10)


def test_dividend_yield_uses_previous_period_price_and_nothing_else_is_lagged():
    """d/y = log D12_t - log P_{t-1}; the first period has no d/y; other columns are current."""
    out = bp.compute_predictors(_inputs())
    assert np.isnan(out["d/y"].iloc[0])
    assert out["d/y"].iloc[1] == pytest.approx(np.log(2.5 / 100.0))
    assert out["tbl"].tolist() == [0.05, 0.02]


def test_targets_simple_and_log_premium_with_both_rf_conventions():
    """eq_prem = ret - rf; the log and rf_comp variants use their own definitions."""
    inputs = _inputs()
    targets = bp.compute_targets(inputs, inputs["Rfree"], inputs["Rfree_comp"])
    row = targets.loc["2000-12-31"]
    assert row["eq_prem"] == pytest.approx(0.10 - 0.05)
    assert row["eq_prem_log"] == pytest.approx(np.log(1.10) - np.log(1.05))
    assert row["eq_prem_rfcomp"] == pytest.approx(0.10 - 0.049)
    assert row["eq_prem_log_rfcomp"] == pytest.approx(np.log(1.10) - np.log(1.049))
    assert list(targets.columns) == bp.TARGET_COLUMNS


def _synthetic_monthly(start="1999-01-31", end="2001-12-31"):
    """Flat synthetic Goyal-coded monthly inputs, with infl = 0.001 * month number."""
    index = pd.date_range(start, end, freq="ME", name="date")
    n = len(index)
    return pd.DataFrame(
        {
            "Index": np.linspace(100, 135, n),
            "D12": np.full(n, 2.0),
            "E12": np.full(n, 5.0),
            "b/m": np.full(n, 0.3),
            "tbl": np.full(n, 0.04),
            "AAA": np.full(n, 0.06),
            "BAA": np.full(n, 0.07),
            "lty": np.full(n, 0.05),
            "ntis": np.full(n, 0.01),
            "Rfree": np.full(n, 0.003),
            "infl": 0.001 * np.arange(1, n + 1),
            "ltr": np.full(n, 0.004),
            "corpr": np.full(n, 0.005),
            "svar": np.full(n, 0.002),
            "CRSP_SPvw": np.full(n, 0.01),
            "CRSP_SPvwx": np.full(n, 0.008),
        },
        index=index,
    )


def test_frequency_panel_has_contract_columns_and_window():
    """A built panel has the 14 predictors, the targets and infl_pit, cut to [start, end]."""
    panel = bp.build_frequency_panel(
        _synthetic_monthly(),
        "quarterly",
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("2001-12-31"),
    )
    expected_columns = (
        bp.MONTHLY_PREDICTORS + bp.TARGET_COLUMNS + bp.POINT_IN_TIME_COLUMNS
    )
    assert list(panel.columns) == expected_columns
    assert panel.index[0] == pd.Timestamp("2000-03-31")
    assert panel.index[-1] == pd.Timestamp("2001-12-31")
    assert panel.notna().all().all()


def test_infl_pit_uses_only_inflation_published_by_period_end():
    """infl_pit for 2000Q1 compounds Dec-1999 to Feb-2000 inflation, not Jan-Mar 2000."""
    monthly = _synthetic_monthly()  # infl in Dec 1999 = 0.012, Jan 2000 = 0.013, ...
    quarterly = bp.aggregate_with_point_in_time_inflation(monthly, 3)
    expected = 1.012 * 1.013 * 1.014 - 1
    assert quarterly.loc["2000-03-31", "infl_pit"] == pytest.approx(expected)
    monthly_out = bp.aggregate_with_point_in_time_inflation(monthly, 1)
    assert monthly_out.loc["2000-01-31", "infl_pit"] == pytest.approx(0.012)


def test_build_refuses_to_write_a_panel_with_missing_values():
    """Starting in the first month of data leaves d/y and infl_pit missing, which raises."""
    with pytest.raises(ValueError, match="Missing values"):
        bp.build_frequency_panel(
            _synthetic_monthly(),
            "monthly",
            pd.Timestamp("1999-01-01"),
            pd.Timestamp("2001-12-31"),
        )


def test_build_refuses_an_end_date_beyond_the_data():
    """An END_DATE after the last observation raises instead of silently ending early."""
    with pytest.raises(ValueError, match="END_DATE"):
        bp.build_frequency_panel(
            _synthetic_monthly(),
            "monthly",
            pd.Timestamp("2000-01-01"),
            pd.Timestamp("2002-06-30"),
        )


def test_compute_predictors_rejects_an_index_with_gaps():
    """With a missing year, d/y would silently use a two-period-old price."""
    inputs = _inputs()
    gappy = pd.concat([inputs, inputs.iloc[[1]].set_axis(["2003-12-31"])])
    gappy.index = pd.DatetimeIndex(gappy.index, name="date")
    with pytest.raises(ValueError):
        bp.compute_predictors(gappy)


def test_panel_r_takes_goyal_sourced_columns_from_panel_g():
    """b/m and dfr in Panel R are Panel G's values; other columns are untouched."""
    index = pd.DatetimeIndex(["2000-12-31", "2001-12-31"], name="date")
    panel_r = pd.DataFrame(
        {"b/m": [9.0, 9.0], "dfr": [9.0, 9.0], "tbl": [0.05, 0.02]}, index
    )
    panel_g = pd.DataFrame(
        {"b/m": [0.3, 0.4], "dfr": [0.01, -0.01], "tbl": [0.0, 0.0]}, index
    )
    out = bp.replace_with_goyal_sourced(panel_r, panel_g, ["b/m", "dfr"])
    assert out["b/m"].tolist() == [0.3, 0.4]
    assert out["dfr"].tolist() == [0.01, -0.01]
    assert out["tbl"].tolist() == [0.05, 0.02]


def test_panel_r_and_g_must_cover_the_same_dates():
    """Copying across panels with different dates would misalign the Goyal columns."""
    panel_r = pd.DataFrame({"b/m": [1.0]}, index=pd.DatetimeIndex(["2000-12-31"]))
    panel_g = pd.DataFrame({"b/m": [1.0]}, index=pd.DatetimeIndex(["2001-12-31"]))
    with pytest.raises(ValueError):
        bp.replace_with_goyal_sourced(panel_r, panel_g, ["b/m"])


def test_contract_lists_are_consistent():
    """14 monthly predictors, 16 for Finance 1, and every Goyal-sourced series is a predictor."""
    assert len(bp.MONTHLY_PREDICTORS) == 14
    assert len(bp.FINANCE1_PREDICTORS) == 16
    assert set(bp.GOYAL_SOURCED) <= set(bp.FINANCE1_PREDICTORS)
    assert bp.TARGET in bp.TARGET_COLUMNS
