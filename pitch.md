# Shen & Xiu (2024), "Can Machines Learn Weak Signals?" — Pitch

Fernando Perez-Millan, Emilio Diaz, Alexander Cruz, Gonzalo Romero · repo `shen_xiu_2024`

**1. What does the paper claim?**
When each predictor carries only a weak signal, Ridge still beats a forecast of zero while Lasso cannot for any choice of penalty, so weak signals, not a lack of sparsity, are what hold Lasso back in economic prediction.

**2. What does the paper not do that someone would want?**
For the equity premium, the paper never varies signal strength. It uses one horizon (annual) and 57 forecasts, with no measure of uncertainty (Ridge 0.80% vs. Lasso −12.19%).

Horizon varies signal strength on the same data: monthly returns are nearly unpredictable, while longer-horizon returns accumulate the slow-moving predictable part. Someone choosing between Ridge and Lasso would want to know whether Ridge's edge shrinks as signals strengthen, as the paper predicts, and whether the gaps exceed noise.

**3. What will we build?** A `doit` pipeline, published as a ChartBook site.
- **Data.** Rebuild the 16 Welch–Goyal predictors where primary sources allow:
  - Shiller dividends and earnings, with the CRSP S&P 500 index: d/p, d/y, e/p, d/e.
  - FRED: tbl, lty, tms, dfy, infl.
  - CRSP: the equity premium, svar, ntis.

  The other five (b/m, eqis, ltr, dfr, ik) come from Goyal's file, which ends in December 2025. Rebuilt series run through the latest month available. Only 14 predictors are monthly (eqis is annual, ik quarterly).
- **Replication.** Figure 5 with fixed seeds. Table 2, Finance 1 row: seven methods, expanding window, forecasts 1965–2021.
- **Update.** The Table 2 row through 2025 on our rebuilt data, stating whether the result held.
- **Extension (to agree).** Ridge vs. Lasso vs. the historical mean at 1, 3, 6 and 12 months, using the 14 monthly predictors.
  - *Hypothesis, from our proposal:* Ridge's advantage is largest at 1 month and narrows with horizon.
  - *Signal strength:* measured at each horizon by Ridge's out-of-sample R², which the paper shows tracks it (Sec. 2.7).
  - *Overlap:* at 12 months, a 60-year test period gives ~720 forecasts but only ~60 non-overlapping ones, so naive standard errors are far too small. We handle it in three places:
    1. Training uses only returns realized by the forecast date.
    2. The penalty is tuned by blocked cross-validation with an h−1 month gap, so no leakage into validation.
    3. Ridge vs. Lasso is tested with Diebold–Mariano, using Newey–West standard errors with h−1 lags and the Harvey–Leybourne–Newbold correction.

    As a check, we rerun on non-overlapping forecasts.

**4. Who would clone it, and what would they do with it?**
Researchers and students who use Goyal's predictor file. Today they wait for his yearly update and can't rebuild or extend the series themselves. With our repo, they run `doit` and get the rebuilt predictors through last month, reconciled to his. They can then add a predictor or method and compare it with Ridge and Lasso at 1–12 month horizons.

**5. How will we know it worked?** Tolerances go into the tests before we run anything.
- **Figure 5:** Table 1, the paper's numerical summary of Figure 5, matches within Monte Carlo error. Same seeds give identical output.
- **Table 2, run on Goyal's file:** every method has the paper's sign, Ridge beats Lasso, and Ridge and Lasso are within ±2 pp.
- **Extension:** the conclusions survive the non-overlapping rerun.

**Data validation** (a reconciliation table plus overlay charts)
- **Reconciliation with Goyal's file:** correlation and largest absolute difference per series.
  - *tbl, dfy, infl* share his sources and should match to rounding.
  - *lty* should match from 2022, when he switched to FRED.
  - CRSP-built series should be close, not exact.
- **Identities:** tms = lty − tbl; dfy = BAA − AAA; log d/e = log d/p − log e/p.
- **Filter counts:** observations removed by each CRSP filter for ntis, vs. what we expect.
- **Coverage:** no unexplained gaps through the end date.
- **Source check:** Table 2 on our data vs. Goyal's.

**Questions for you**
1. Goyal's file now ends in 2025, the same year as our annual update. Only our monthly data go further. Is that enough for the update?
2. b/m, ltr and dfr have no free primary source and end in December 2025. Should we drop them after that, or end the monthly sample there?
