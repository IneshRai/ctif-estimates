"""
compute_daily_weights.py
------------------------
Builds daily portfolio weights for each stock in the CTIF strategy from the
holdings history file.

Rules (per project spec):
  1. Option positions are EXCLUDED from the denominator entirely.
  2. BOXX and "Cash&Other" are treated as a single cash-management bucket and
     summed together into one line called "cash".
  3. Weight = position market value / (sum of all non-option market values that day).

Outputs:
  - daily_weights_long.csv : tidy format (date, ticker, market_value, weight)
  - daily_weights_wide.csv : matrix (date index, one column per ticker)

Diagnostics print to console so you can eyeball correctness before using the
series downstream.
"""

import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config: edit INPUT_FILE to point at your local copy, e.g.
#   INPUT_FILE = Path(r"C:\Users\inesh\OneDrive\TI Estimates\etf_holdings_history.csv")
# ---------------------------------------------------------------------------
INPUT_FILE = Path("etf_holdings_history.csv")
OUTPUT_DIR = Path(".")

# Anything in this set is folded into the single "cash" bucket.
CASH_TICKERS = {"BOXX", "Cash", "Cash&Other"}

# OCC option symbol: optional leading digit, 1-6 char root, 6-digit date,
# C or P, 8-digit strike. Spaces are used to pad the root, so allow them.
OCC_PATTERN = re.compile(r"^\s*\d?[A-Z]{1,6}\s*\d{6}[CP]\d{8}\s*$")


def classify(ticker: str) -> str:
    """Return 'option', 'cash', or 'stock' for a STOCK_TICKER value."""
    if ticker in CASH_TICKERS:
        return "cash"
    if OCC_PATTERN.match(ticker):
        return "option"
    return "stock"


def main() -> None:
    if not INPUT_FILE.exists():
        sys.exit(f"Input file not found: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE)

    # Parse dates. Source format is m/d/yy (e.g. 6/25/25).
    df["DATE"] = pd.to_datetime(df["EFFECTIVE_DATE"], format="%m/%d/%y")

    # Classify every row.
    df["kind"] = df["STOCK_TICKER"].apply(classify)

    # --- Diagnostics: what did we find? ---------------------------------
    n_days = df["DATE"].nunique()
    print("Loaded %d rows" % len(df))
    print("Date range: %s to %s (%d unique days)"
          % (df["DATE"].min().date(), df["DATE"].max().date(), n_days))
    print("Row breakdown:")
    print(df["kind"].value_counts().to_string())

    # Sanity: options should be short calls (mostly negative MV). Flag if not.
    pos_opt = df[(df.kind == "option") & (df.MARKET_VALUE > 0)]
    if len(pos_opt):
        print("\nNote: %d option rows have positive market value "
              "(expected for short calls to be negative). Review if unexpected."
              % len(pos_opt))

    # --- Build denominator base (drop options) --------------------------
    base = df[df.kind != "option"].copy()

    # Collapse BOXX + Cash&Other into one "cash" bucket; stocks keep ticker.
    base["bucket"] = base.apply(
        lambda r: "cash" if r.kind == "cash" else r.STOCK_TICKER, axis=1
    )

    # Sum market value by day and bucket (handles the two cash rows merging).
    agg = (base.groupby(["DATE", "bucket"], as_index=False)["MARKET_VALUE"]
                .sum())

    # Daily denominator = total non-option market value.
    denom = agg.groupby("DATE")["MARKET_VALUE"].sum().rename("denom")
    agg = agg.merge(denom, on="DATE")
    agg["weight"] = agg["MARKET_VALUE"] / agg["denom"]

    # --- Validation -----------------------------------------------------
    wsum = agg.groupby("DATE")["weight"].sum()
    max_err = (wsum - 1.0).abs().max()
    print("\nValidation: max |sum(weights) - 1| across all days = %.2e" % max_err)
    if max_err > 1e-9:
        print("WARNING: weights do not sum to 1 on some days. Investigate.")
    else:
        print("OK: weights sum to 1.0 on every day.")

    # --- Write outputs --------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    long_out = agg.rename(columns={"bucket": "ticker",
                                   "MARKET_VALUE": "market_value"})
    long_out = long_out[["DATE", "ticker", "market_value", "weight"]]
    long_out = long_out.sort_values(["DATE", "ticker"]).reset_index(drop=True)
    long_path = OUTPUT_DIR / "daily_weights_long.csv"
    long_out.to_csv(long_path, index=False)

    wide = (agg.pivot(index="DATE", columns="bucket", values="weight")
               .sort_index())
    wide.index.name = "DATE"
    wide_path = OUTPUT_DIR / "daily_weights_wide.csv"
    wide.to_csv(wide_path)

    print("\nWrote:")
    print("  %s  (%d rows)" % (long_path, len(long_out)))
    print("  %s  (%d days x %d tickers)"
          % (wide_path, wide.shape[0], wide.shape[1]))

    # Peek at the most recent day for a quick sanity check.
    last = long_out[long_out.DATE == long_out.DATE.max()]
    print("\nMost recent day (%s), top holdings:" % last.DATE.max().date())
    print(last.sort_values("weight", ascending=False)
              .head(10)[["ticker", "weight"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
