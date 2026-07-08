# CTIF estimate series

1. `compute_daily_weights.py` - daily portfolio weights (options excluded, BOXX+Cash folded into "cash").
2. `build_estimate_series.py` - weighted aggregate FY1/FY2 estimate series vs CTIF price, dual-axis chart.

## Run
    pip install pandas matplotlib yfinance
    python compute_daily_weights.py
    python build_estimate_series.py
