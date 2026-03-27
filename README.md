## Results & Analysis

### Performance Summary (2023-2024 Test Period)

**Winning Assets:**
- **WTI Crude**: 10.86% annual return, 0.49 Sharpe, +30.51% vs buy-and-hold
- **Heating Oil**: 8.20% annual return, 0.41 Sharpe, +42.49% vs buy-and-hold  
- **Natural Gas**: 2.55% annual return, 0.34 Sharpe, +69.86% vs buy-and-hold

**Underperforming Assets:**
- **Brent Crude**: -8.51% to -13.41% annual return
- **Gasoline**: -13.20% to -17.95% annual return

### Strategy Strengths

1. **Trending Market Performance**: Successfully captures momentum in directional markets (WTI, Heating Oil)
2. **Downside Protection**: Consistent outperformance vs buy-and-hold, even when absolute returns are negative
3. **Data Quality Discipline**: Identified and resolved futures contract rollover issues, demonstrating production-ready validation practices
4. **Regime Adaptability**: GMM clustering successfully identifies distinct market states

### Known Limitations

1. **Mean-Reverting Markets**: Strategy underperforms in choppy, range-bound markets (Gasoline, Brent)
2. **High Volatility Exposure**: Natural Gas shows 59.7% annualized volatility with -61.88% max drawdown
3. **Modest Sharpe Ratios**: 0.34-0.49 range indicates room for improved risk management
4. **Market-Specific Performance**: Not all energy commodities exhibit momentum characteristics

### Risk Metrics

- **Maximum Drawdown**: -61.88% (Natural Gas) to -20.64% (WTI Crude)
- **Win Rate**: ~50% across assets
- **Volatility**: 29-60% annualized, varies significantly by commodity
- **Transaction Costs**: 0.1% per trade included in all results

### Technical Insights

**What We Learned:**
- GARCH volatility forecasting effectively captures clustering in energy markets
- Regime detection identifies market inflection points but requires asset-specific calibration
- Momentum strategies require trending markets to be profitable
- Data quality (contract rollovers, missing data) can completely invalidate results if not properly handled

**Future Improvements:**
- Implement risk parity position sizing to reduce volatility drag
- Add mean-reversion signals for Gasoline and Brent
- Explore multi-timeframe approaches (combine daily momentum with weekly trends)
- Incorporate fundamental factors (inventory data, OPEC announcements)