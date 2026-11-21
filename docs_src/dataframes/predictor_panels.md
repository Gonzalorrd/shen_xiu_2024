## Description

These panels hold the Welch-Goyal predictors of the US equity premium, used by
the Shen and Xiu (2024) Finance 1 replication and by our frequency extension.
Panel G is built from Amit Goyal's posted workbook. Panel R, with the same
layout, is rebuilt from primary sources (CRSP, FRED, Shiller) and is added later.

- `panel_G_{monthly, quarterly, semiannual, annual}`: 14 predictors plus targets.
  The coarser frequencies are aggregated from the monthly data with one set of
  rules (end-of-period values for prices, yields and ntis; compounding for
  returns and inflation; summing for svar). Applied to Goyal's Monthly sheet,
  these rules reproduce his Quarterly and Annual sheets.
- `finance1_panel_G`: annual, all 16 predictors, from Goyal's own Annual sheet.
  This is the input to the Table 2 replication.

Each row is a period-end date. Predictors are measured at the end of the period
and are not lagged. The target is the return earned during the period. To
forecast period t, use the predictors from row t-1. b/m, eqis and dfr are taken
from Goyal in both panels, because their primary inputs are commercial or
discontinued.

## Data Dictionary

- **date**: `datetime64[ns]` Period-end date (index).
- **d/p**: `float64` Log dividend-price ratio, log D12 - log P.
- **d/y**: `float64` Log dividend yield, log D12 - log P of the previous period.
- **e/p**: `float64` Log earnings-price ratio, log E12 - log P.
- **d/e**: `float64` Log payout ratio, log D12 - log E12.
- **svar**: `float64` Stock variance: sum of squared daily S&P 500 returns in the period.
- **b/m**: `float64` Dow Jones book-to-market ratio (from Goyal).
- **ntis**: `float64` Net equity expansion of NYSE stocks.
- **tbl**: `float64` 3-month T-bill yield, annualized, decimal.
- **lty**: `float64` Long-term government bond yield, annualized, decimal.
- **ltr**: `float64` Long-term government bond return over the period.
- **tms**: `float64` Term spread, lty - tbl.
- **dfy**: `float64` Default yield spread, BAA - AAA.
- **dfr**: `float64` Default return spread, corporate minus government bond return (from Goyal).
- **infl**: `float64` CPI inflation over the period, as Goyal posts it. It is not yet published at the period end.
- **infl_pit**: `float64` Point-in-time inflation: monthly CPI inflation lagged one month for its release, then compounded over the period. It is known at the period end, so use it for point-in-time forecasts.
- **eqis**: `float64` Percent equity issuing (Finance 1 panel only; from Goyal).
- **ik**: `float64` Investment-to-capital ratio (Finance 1 panel only).
- **ret**: `float64` S&P 500 total return over the period (CRSP value-weighted, with dividends).
- **retx**: `float64` S&P 500 return without dividends.
- **rf**: `float64` Risk-free return over the period, Goyal's convention (start-of-quarter T-bill yield times 3/12, compounded across quarters).
- **rf_comp**: `float64` Compounded monthly risk-free return.
- **eq_prem**: `float64` Equity premium, ret - rf: the pre-registered target.
- **eq_prem_log**: `float64` Log equity premium, log(1 + ret) - log(1 + rf).
- **eq_prem_rfcomp**: `float64` ret - rf_comp.
- **eq_prem_log_rfcomp**: `float64` log(1 + ret) - log(1 + rf_comp).
