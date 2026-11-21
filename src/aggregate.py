"""Aggregate monthly Welch-Goyal inputs to k-month periods (k = 1, 3, 6, 12).

One function, ``aggregate_to_k_months``, turns a monthly frame of Goyal-coded
inputs into calendar-aligned k-month periods: quarters end in Mar/Jun/Sep/Dec,
half-years in Jun/Dec, years in Dec. The same function builds the quarterly,
semi-annual and annual panels for both Panel G (Goyal's monthly sheet) and
Panel R (our rebuilt monthly data), so all frequencies share one set of rules.

Aggregation rules (Appendix C; team plan section 2.1, item 6):

- end-of-period value: prices, dividend and earnings sums, yields, ntis, b/m
  (``Index, D12, E12, b/m, tbl, AAA, BAA, lty, ntis``);
- compounding: returns and inflation
  (``CRSP_SPvw, CRSP_SPvwx, ltr, corpr, infl``);
- summing: ``svar`` (a sum of squared daily returns stays a sum).

The risk-free rate is built two ways:

- ``Rfree_comp``: the monthly ``Rfree`` returns, compounded;
- ``Rfree``: Goyal's own convention. Each quarter earns the T-bill yield at the
  start of the quarter times 3/12, and longer periods compound those quarterly
  returns. This reproduces Goyal's Quarterly and Annual sheets exactly, which
  is why his annual Rfree (3.22% in 1960) differs from the compounded monthly
  rate (2.67%). At k = 1 it is Goyal's monthly ``Rfree`` unchanged. The rule was
  identified by comparing candidate formulas with his sheets (2026-11-21).

Only complete periods are returned: a partial first or last period is dropped,
and a period with any missing month in a compounded or summed series is NaN.
``test_data_validation.py`` checks that these rules reproduce Goyal's own
Quarterly and Annual sheets.
"""

import numpy as np
import pandas as pd

END_OF_PERIOD = "end_of_period"
COMPOUND = "compound"
SUM = "sum"

AGGREGATION_RULES = {
    "Index": END_OF_PERIOD,
    "D12": END_OF_PERIOD,
    "E12": END_OF_PERIOD,
    "b/m": END_OF_PERIOD,
    "tbl": END_OF_PERIOD,
    "AAA": END_OF_PERIOD,
    "BAA": END_OF_PERIOD,
    "lty": END_OF_PERIOD,
    "ntis": END_OF_PERIOD,
    "infl": COMPOUND,
    "ltr": COMPOUND,
    "corpr": COMPOUND,
    "CRSP_SPvw": COMPOUND,
    "CRSP_SPvwx": COMPOUND,
    "svar": SUM,
}
ALLOWED_K = (1, 3, 6, 12)


def to_full_monthly_range(monthly):
    """Check the index and reindex to every month-end between the first and last date."""
    if not isinstance(monthly.index, pd.DatetimeIndex):
        raise TypeError("monthly must have a DatetimeIndex of month-end dates.")
    if not monthly.index.is_unique:
        raise ValueError("monthly has duplicated dates.")
    if not monthly.index.is_month_end.all():
        raise ValueError("monthly dates must be month-ends.")
    if not monthly.index.equals(monthly.index.normalize()):
        # 2000-01-31 12:00 is a "month end" but would not align with period ends.
        raise ValueError(
            "monthly dates must be month-ends at midnight (no time of day)."
        )
    full_range = pd.date_range(monthly.index.min(), monthly.index.max(), freq="ME")
    return monthly.reindex(full_range)


def period_end_dates(dates, k):
    """Map each month-end date to the end of its calendar-aligned k-month period."""
    end_month = ((dates.month - 1) // k + 1) * k
    period_ends = pd.to_datetime(
        pd.DataFrame({"year": dates.year, "month": end_month, "day": 1})
    ) + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex(period_ends, name="date")


def goyal_rfree(tbl, k):
    """Goyal's k-month risk-free return from the monthly T-bill yield (k = 3, 6, 12).

    Quarter q earns tbl at the end of quarter q-1 times 3/12; a k-month period
    compounds the k/3 quarters it contains. ``tbl`` must be on a full monthly range.
    """
    if k % 3 != 0:
        raise ValueError("Goyal's Rfree convention is defined for k = 3, 6, 12.")
    quarter_ends = tbl.index[tbl.index.month % 3 == 0]
    quarterly_rf = (tbl.shift(3) / 4).loc[quarter_ends]
    keys = period_end_dates(quarter_ends, k)
    compounded = (1 + quarterly_rf).groupby(keys).prod(min_count=k // 3) - 1
    return compounded


def aggregate_to_k_months(monthly, k, rules=AGGREGATION_RULES):
    """Aggregate monthly Goyal-coded inputs to calendar-aligned k-month periods.

    Parameters
    ----------
    monthly : DataFrame
        Month-end DatetimeIndex; columns named with Goyal's codes. Columns that
        have no rule are ignored. ``Rfree`` and ``tbl`` are needed for the two
        risk-free columns.
    k : int
        Months per period: 1, 3, 6 or 12.
    rules : dict
        Column -> END_OF_PERIOD, COMPOUND or SUM.

    Returns
    -------
    DataFrame indexed by period-end ``date``, one row per complete period,
    with the ruled columns plus ``Rfree`` (Goyal's convention) and ``Rfree_comp``
    (compounded monthly Rfree).
    """
    if k not in ALLOWED_K:
        raise ValueError(f"k must be one of {ALLOWED_K}, got {k}.")
    full = to_full_monthly_range(monthly)
    keys = period_end_dates(full.index, k)
    months_in_period = pd.Series(1, index=full.index).groupby(keys).sum()
    complete_periods = months_in_period.index[months_in_period == k]

    out = {}
    for column, rule in rules.items():
        if column not in full.columns:
            continue
        series = full[column]
        if k == 1:  # monthly: every rule is the identity; avoid (1 + r) - 1 rounding
            values = series.rename_axis("date")
        elif rule == END_OF_PERIOD:
            values = series[full.index.month % k == 0]
            values.index = pd.DatetimeIndex(values.index, name="date")
        elif rule == COMPOUND:
            values = (1 + series).groupby(keys).prod(min_count=k) - 1
        elif rule == SUM:
            values = series.groupby(keys).sum(min_count=k)
        else:
            raise ValueError(f"Unknown rule {rule!r} for column {column!r}.")
        out[column] = values

    if "Rfree" in full.columns:
        if k == 1:
            out["Rfree"] = full["Rfree"].rename_axis("date")
            out["Rfree_comp"] = full["Rfree"].rename_axis("date")
        else:
            out["Rfree_comp"] = (1 + full["Rfree"]).groupby(keys).prod(min_count=k) - 1
            if "tbl" in full.columns:
                out["Rfree"] = goyal_rfree(full["tbl"], k)

    aggregated = pd.DataFrame(out).reindex(complete_periods)
    aggregated.index.name = "date"
    return aggregated.astype(np.float64)
