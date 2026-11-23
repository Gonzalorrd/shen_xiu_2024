"""Build the tidy predictor panels: cleaning only, no analysis.

PANEL CONTRACT
==============
This is the interface the replication (Appendix B) and the extension (Appendix C,
Part III) code against. Change it only by agreement with the team.

Files (in DATA_DIR, built by ``doit``, never committed)
-------------------------------------------------------
- ``panel_{P}_{freq}.parquet`` for P in {G, R} and freq in
  {monthly, quarterly, semiannual, annual}: the 14 monthly predictors and the
  target columns at that frequency. Coarser frequencies are aggregated from the
  monthly inputs by ``aggregate.py``, so every rung of the frequency ladder uses
  the same rules.
- ``finance1_panel_{P}.parquet``: annual, all 16 Finance 1 predictors (the 14
  plus eqis and ik) and the target columns. For Panel G it is built from Goyal's
  own Annual sheet, the authors' input for Table 2.

Panel G is built from Goyal's posted workbook. Panel R has the same file
layout and goes through the same aggregation and predictor code, starting
from inputs rebuilt from CRSP, FRED and Shiller (``rebuild_inputs.py``); its
b/m, eqis and dfr are copied from Panel G. Load with
``load_panel("G", "monthly")`` or ``load_finance1_panel("R")``.

Index
-----
A DatetimeIndex named ``date`` holding period-end timestamps: month-ends;
quarters end in Mar/Jun/Sep/Dec, half-years in Jun/Dec, years on Dec 31. There
is one row per period from START_DATE to END_DATE (settings.py), with no gaps
and no missing values.

Timing
------
Row t holds the predictors observed at the END of period t and the return
earned DURING period t (from the end of t-1 to the end of t). Predictors are
NOT lagged. To forecast y_t with x_{t-1}, use
``X = panel[PREDICTORS].shift(1)`` and ``y = panel[TARGET]``.

Exception, infl: CPI for month t is published around mid-month t+1, so ``infl``
in row t is NOT yet known at the end of period t, and the recipe above would
use about two weeks of future information. Every panel therefore also carries
``infl_pit``: monthly inflation lagged one month for the release, then
aggregated, so it IS known at the end of period t. Use ``infl_pit`` in place of
``infl`` for point-in-time forecasts (the extension). ``infl`` stays as Goyal
posts it, which is what the paper's Finance 1 exercise used.

Units
-----
Decimals throughout. Yields are annualized (tbl = 0.0359 means 3.59% a year).
Returns and inflation are simple returns per period: monthly at the monthly
frequency, annual at the annual one. d/p, d/y, e/p and d/e are natural-log
ratios. svar is the sum of squared daily returns within the period.

Predictor columns (Goyal codes)
-------------------------------
``MONTHLY_PREDICTORS`` (14): d/p, d/y, e/p, d/e, svar, b/m, ntis, tbl, lty,
ltr, tms, dfy, dfr, infl. ``FINANCE1_PREDICTORS`` (16) adds eqis and ik, which
Goyal posts only at annual (eqis) or quarterly and annual (ik) frequency.

- d/p = log D12 - log P_t;  d/y = log D12 - log P_{t-1};  e/p = log E12 - log P_t;
  d/e = log D12 - log E12;  tms = lty - tbl;  dfy = BAA - AAA;  dfr = corpr - ltr.
- ``GOYAL_SOURCED`` = b/m, eqis, dfr: their primary inputs are commercial or
  discontinued (Appendix C.2), so they are taken from Goyal in BOTH panels and
  are identical in G and R.
- infl is inflation during period t as Goyal posts it; ``infl_pit`` is the
  release-lagged version (see Timing). infl_pit is an extra column, not one of
  the 14 / 16 predictors.

The build refuses to write a panel that breaks this contract (a gap, a
missing value, or dates outside the data), so START_DATE and END_DATE must
lie inside the sample where every predictor exists (1947-01 to 2025-12 for
Goyal's current file).

Target columns
--------------
- ``ret``: S&P 500 total return (CRSP value-weighted, with dividends);
- ``retx``: S&P 500 return without dividends;
- ``rf``: risk-free return over the period, in Goyal's convention (see aggregate.py);
- ``rf_comp``: compounded monthly risk-free return (equal to rf at monthly frequency);
- ``eq_prem`` = ret - rf (simple premium);
- ``eq_prem_log`` = log(1 + ret) - log(1 + rf) (log premium);
- ``eq_prem_rfcomp``, ``eq_prem_log_rfcomp``: the same two with rf_comp.

``TARGET`` is ``eq_prem_log_rfcomp``: the log S&P 500 return minus the log
compounded 1-month T-bill return (Appendix B.2, C.2). Appendix B.8 pre-registered
the simple premium but left the OLS anchor (B.6(1)) to decide; on Panel G the
simple premium gives R2oos -94.86 and the log premium -82.67, against the paper's
-81.08 +/- 2 (test_build_predictors.py). Because it is a log return on compounded
rf, the k-month target is the sum of the k monthly targets.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pull_goyal
from aggregate import (
    AGGREGATION_RULES,
    COMPOUND,
    aggregate_to_k_months,
    to_full_monthly_range,
)
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = pd.Timestamp(config("START_DATE"))
END_DATE = pd.Timestamp(config("END_DATE"))

FREQUENCIES = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
PANELS = ("G", "R")

MONTHLY_PREDICTORS = [
    "d/p", "d/y", "e/p", "d/e", "svar", "b/m", "ntis",
    "tbl", "lty", "ltr", "tms", "dfy", "dfr", "infl",
]  # fmt: skip
FINANCE1_PREDICTORS = MONTHLY_PREDICTORS + ["eqis", "ik"]
GOYAL_SOURCED = ["b/m", "eqis", "dfr"]
TARGET = "eq_prem_log_rfcomp"  # chosen by the OLS anchor; see docstring
TARGET_COLUMNS = [
    "ret", "retx", "rf", "rf_comp",
    "eq_prem", "eq_prem_log", "eq_prem_rfcomp", "eq_prem_log_rfcomp",
]  # fmt: skip
POINT_IN_TIME_COLUMNS = ["infl_pit"]
CPI_RELEASE_LAG_MONTHS = 1


def panel_file_name(panel, freq):
    """File name of one frequency panel, e.g. panel_G_monthly.parquet."""
    if panel not in PANELS or freq not in FREQUENCIES:
        raise ValueError(f"Unknown panel {panel!r} or frequency {freq!r}.")
    return f"panel_{panel}_{freq}.parquet"


def finance1_file_name(panel):
    """File name of the annual Finance 1 panel, e.g. finance1_panel_G.parquet."""
    if panel not in PANELS:
        raise ValueError(f"Unknown panel {panel!r}.")
    return f"finance1_panel_{panel}.parquet"


def compute_predictors(inputs):
    """Compute the Welch-Goyal predictors from Goyal-coded inputs at one frequency.

    ``inputs`` must be on a gap-free period-end index, because d/y uses the
    previous period's price. eqis and ik are passed through when present.
    """
    month_numbers = inputs.index.year * 12 + inputs.index.month
    if len(set(np.diff(month_numbers))) > 1:
        raise ValueError(
            "inputs must be on a gap-free index: d/y needs the previous period."
        )
    price = inputs["Index"]
    log_d12 = np.log(inputs["D12"])
    log_e12 = np.log(inputs["E12"])
    predictors = pd.DataFrame(
        {
            "d/p": log_d12 - np.log(price),
            "d/y": log_d12 - np.log(price.shift(1)),
            "e/p": log_e12 - np.log(price),
            "d/e": log_d12 - log_e12,
            "svar": inputs["svar"],
            "b/m": inputs["b/m"],
            "ntis": inputs["ntis"],
            "tbl": inputs["tbl"],
            "lty": inputs["lty"],
            "ltr": inputs["ltr"],
            "tms": inputs["lty"] - inputs["tbl"],
            "dfy": inputs["BAA"] - inputs["AAA"],
            "dfr": inputs["corpr"] - inputs["ltr"],
            "infl": inputs["infl"],
        },
        index=inputs.index,
    )
    for column in ["eqis", "ik"]:
        if column in inputs.columns:
            predictors[column] = inputs[column]
    return predictors


def compute_targets(returns, rf, rf_comp):
    """Build the target columns from the S&P 500 returns and both risk-free rates.

    ``returns`` has columns CRSP_SPvw (with dividends) and CRSP_SPvwx (without).
    """
    ret = returns["CRSP_SPvw"]
    return pd.DataFrame(
        {
            "ret": ret,
            "retx": returns["CRSP_SPvwx"],
            "rf": rf,
            "rf_comp": rf_comp,
            "eq_prem": ret - rf,
            "eq_prem_log": np.log1p(ret) - np.log1p(rf),
            "eq_prem_rfcomp": ret - rf_comp,
            "eq_prem_log_rfcomp": np.log1p(ret) - np.log1p(rf_comp),
        }
    )


def aggregate_with_point_in_time_inflation(monthly_inputs, k):
    """Aggregate monthly inputs to k months, adding release-lagged inflation ``infl_pit``.

    infl_pit compounds the k monthly inflation rates that are published by the
    end of the period, i.e. the k months ending one month before the period end.
    """
    monthly = to_full_monthly_range(monthly_inputs)
    monthly["infl_pit"] = monthly["infl"].shift(CPI_RELEASE_LAG_MONTHS)
    rules = {**AGGREGATION_RULES, "infl_pit": COMPOUND}
    return aggregate_to_k_months(monthly, k, rules=rules)


def check_panel_contract(panel, k, columns, start, end):
    """Raise if a panel breaks the contract: wrong dates, gaps or missing values."""
    month_ends = pd.date_range(start, end, freq="ME")
    expected = month_ends[month_ends.month % k == 0]
    if not panel.index.equals(expected):
        raise ValueError(
            f"Panel dates run {panel.index.min().date()} to {panel.index.max().date()} "
            f"({len(panel)} rows); the contract needs every {k}-month period end from "
            f"{start.date()} to {end.date()}. Do START_DATE and END_DATE lie "
            "inside the data?"
        )
    missing = panel[columns].isna().sum()
    if missing.any():
        first_gap = panel.index[panel[columns].isna().any(axis=1)][0].date()
        raise ValueError(
            f"Missing values (first on {first_gap}) in: "
            f"{missing[missing > 0].to_dict()}. Set START_DATE (settings.py or .env) "
            "inside the sample where every predictor exists, e.g. 1947-01-01."
        )


def build_frequency_panel(monthly_inputs, freq, start=START_DATE, end=END_DATE):
    """Aggregate monthly inputs to ``freq`` and return the contract panel for [start, end]."""
    k = FREQUENCIES[freq]
    inputs = aggregate_with_point_in_time_inflation(monthly_inputs, k)
    predictors = compute_predictors(inputs)[MONTHLY_PREDICTORS]
    targets = compute_targets(inputs, inputs["Rfree"], inputs["Rfree_comp"])
    panel = pd.concat([predictors, targets, inputs[POINT_IN_TIME_COLUMNS]], axis=1)
    panel = panel.loc[start:end]
    panel.index.name = "date"
    columns = MONTHLY_PREDICTORS + TARGET_COLUMNS + POINT_IN_TIME_COLUMNS
    check_panel_contract(panel, k, columns, start, end)
    return panel


def build_finance1_panel_from_goyal(
    annual_sheet, monthly_sheet, start=START_DATE, end=END_DATE
):
    """Annual Finance 1 panel (16 predictors) from Goyal's own Annual sheet.

    rf is Goyal's annual Rfree; rf_comp compounds his monthly Rfree, the
    alternative convention the replication also runs. infl_pit comes from the
    monthly sheet (December-to-November inflation).
    """
    from_monthly = aggregate_with_point_in_time_inflation(monthly_sheet, 12)
    from_monthly = from_monthly.reindex(annual_sheet.index)
    predictors = compute_predictors(annual_sheet)[FINANCE1_PREDICTORS]
    targets = compute_targets(
        annual_sheet,
        rf=annual_sheet["Rfree"],
        rf_comp=from_monthly["Rfree_comp"],
    )
    panel = pd.concat(
        [predictors, targets, from_monthly[POINT_IN_TIME_COLUMNS]], axis=1
    ).loc[start:end]
    panel.index.name = "date"
    columns = FINANCE1_PREDICTORS + TARGET_COLUMNS + POINT_IN_TIME_COLUMNS
    check_panel_contract(panel, 12, columns, start, end)
    return panel


def build_panel_G(data_dir=DATA_DIR, start=START_DATE, end=END_DATE):
    """Build every Panel G file from the parsed Goyal sheets.

    Returns a dict {file name: DataFrame}.
    """
    monthly = pull_goyal.load_goyal("monthly", data_dir=data_dir)
    annual = pull_goyal.load_goyal("annual", data_dir=data_dir)
    panels = {
        panel_file_name("G", freq): build_frequency_panel(monthly, freq, start, end)
        for freq in FREQUENCIES
    }
    panels[finance1_file_name("G")] = build_finance1_panel_from_goyal(
        annual, monthly, start, end
    )
    return panels


def replace_with_goyal_sourced(panel_r, panel_g, columns):
    """Copy the Goyal-sourced columns from Panel G into Panel R (same dates required).

    b/m, eqis and dfr cannot be rebuilt (Appendix C.2), so Panel R carries
    Goyal's values and the two panels are identical in these columns.
    """
    if not panel_r.index.equals(panel_g.index):
        raise ValueError("Panels R and G must cover the same dates.")
    out = panel_r.copy()
    out[columns] = panel_g[columns]
    return out


def build_finance1_panel_rebuilt(
    monthly_inputs, ik, finance1_g, start=START_DATE, end=END_DATE
):
    """Annual Finance 1 panel R: aggregated rebuilt inputs + rebuilt ik + Goyal's b/m, eqis, dfr.

    rf follows Goyal's convention from our tbl; rf_comp compounds CRSP's T-bill returns.
    """
    inputs = aggregate_with_point_in_time_inflation(monthly_inputs, 12)
    predictors = compute_predictors(inputs)[MONTHLY_PREDICTORS]
    predictors["eqis"] = np.nan  # from Goyal below
    predictors["ik"] = ik.reindex(inputs.index)
    targets = compute_targets(inputs, inputs["Rfree"], inputs["Rfree_comp"])
    panel = pd.concat([predictors, targets, inputs[POINT_IN_TIME_COLUMNS]], axis=1)
    panel = panel.loc[start:end]
    panel.index.name = "date"
    panel = replace_with_goyal_sourced(panel, finance1_g, GOYAL_SOURCED)
    columns = FINANCE1_PREDICTORS + TARGET_COLUMNS + POINT_IN_TIME_COLUMNS
    check_panel_contract(panel, 12, columns, start, end)
    return panel[columns]


def build_panel_R(data_dir=DATA_DIR, start=START_DATE, end=END_DATE):
    """Build every Panel R file from the rebuilt inputs, with Goyal's b/m, eqis and dfr.

    Needs the Panel G files (for the Goyal-sourced columns) and the saved pulls.
    Returns a dict {file name: DataFrame}.
    """
    import rebuild_inputs  # Panel G does not need the CRSP/FRED/Shiller pulls

    monthly = rebuild_inputs.build_rebuilt_monthly_inputs(data_dir)
    monthly_sourced = [c for c in GOYAL_SOURCED if c in MONTHLY_PREDICTORS]
    panels = {}
    for freq in FREQUENCIES:
        panel_r = build_frequency_panel(monthly, freq, start, end)
        panel_g = load_panel("G", freq, data_dir)
        panels[panel_file_name("R", freq)] = replace_with_goyal_sourced(
            panel_r, panel_g, monthly_sourced
        )
    panels[finance1_file_name("R")] = build_finance1_panel_rebuilt(
        monthly,
        rebuild_inputs.build_rebuilt_annual_ik(data_dir),
        load_finance1_panel("G", data_dir),
        start,
        end,
    )
    return panels


def save_panels(panels, data_dir=DATA_DIR):
    """Write each panel to data_dir as parquet."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_name, panel in panels.items():
        panel.to_parquet(data_dir / file_name)


def load_panel(panel="G", freq="monthly", data_dir=DATA_DIR):
    """Load one frequency panel (14 predictors + targets). Build it with doit first."""
    return pd.read_parquet(Path(data_dir) / panel_file_name(panel, freq))


def load_finance1_panel(panel="G", data_dir=DATA_DIR):
    """Load the annual Finance 1 panel (16 predictors + targets)."""
    return pd.read_parquet(Path(data_dir) / finance1_file_name(panel))


if __name__ == "__main__":
    # `python build_predictors.py` builds Panel G; `python build_predictors.py R` builds Panel R.
    which = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in PANELS else "G"
    built = build_panel_G() if which == "G" else build_panel_R()
    save_panels(built)
    for name, df in built.items():
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} columns, "
              f"{df.index.min().date()} to {df.index.max().date()}")  # fmt: skip
