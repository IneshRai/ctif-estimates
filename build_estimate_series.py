"""
build_estimate_series.py
------------------------
Builds a weighted aggregate earnings-estimate series for the CTIF strategy and
charts it against CTIF's own price history on dual axes.

Inputs:
  - daily_weights_long.csv   (from compute_daily_weights.py)
  - estimates_ti/*.csv        (one file per ticker: Date, Ticker, BEST_EPS, BEST_EPS_NXT_YR)
  - CTIF price via yfinance

Method:
  1. Estimates are event-driven (a row only when consensus changes), so each
     name is forward-filled onto the holdings dates via a backward as-of join.
     Backward only, so there is never any lookahead.
  2. Cash (BOXX + Cash&Other) has no EPS. By default we RENORMALIZE the stock
     weights to sum to 1 across names that have an estimate that day, so the
     aggregate is a clean held-stock estimate and is not dragged by cash level.
     Set RENORMALIZE = False to keep cash in the denominator at zero estimate.
  3. Aggregate FY1 = sum(weight_i * BEST_EPS_i); aggregate FY2 likewise with
     BEST_EPS_NXT_YR. FY1 and FY2 are kept as separate series (same axis).

Outputs:
  - aggregate_estimate_series.csv : date, agg_eps_fy1, agg_eps_fy2, ctif_price,
                                    est_weight_coverage, cash_weight
  - per_name_estimates.csv        : date, ticker, weight, eps_fy1, eps_fy2
                                    (as-of filled; foundation for name-level work)
  - estimate_vs_price.png         : dual-axis chart

CAVEATS (read before drawing conclusions):
  - BEST_EPS is FY1 = the CURRENT fiscal year, which ROLLS forward each year.
    Names have different fiscal year-ends, so aggregate FY1 blends horizons and
    will show step discontinuities as individual names roll. Aggregate FY2 is a
    bit cleaner but shifts too. For "warning signs" you will likely want a
    revision/rebased series rather than raw levels. This script gives levels
    (what was asked for); ask and I will add a rebased-index variant.
"""

import glob
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; safe on any machine
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS_FILE = Path("daily_weights_long.csv")
ESTIMATES_DIR = Path("estimates_ti")
OUTPUT_DIR = Path(".")

CTIF_TICKER = "CTIF"
AUTO_ADJUST = True   # True = dividend/total-return adjusted close (recommended
                     # for a fund that pays quarterly income). False = raw price.

RENORMALIZE = True   # See method note 2 above.

# ---------------------------------------------------------------------------
# 1. Load and forward-fill estimates
# ---------------------------------------------------------------------------
def load_estimates() -> pd.DataFrame:
    """Return long df: ticker, Date, eps_fy1, eps_fy2 (per-file ffilled)."""
    frames = []
    for f in sorted(glob.glob(str(ESTIMATES_DIR / "*.csv"))):
        ticker = os.path.splitext(os.path.basename(f))[0]
        d = pd.read_csv(f, parse_dates=["Date"])
        d = d.rename(columns={"BEST_EPS": "eps_fy1",
                              "BEST_EPS_NXT_YR": "eps_fy2"})
        d = d[["Date", "eps_fy1", "eps_fy2"]].sort_values("Date")
        # Fill gaps forward within the name (estimate persists until revised).
        d[["eps_fy1", "eps_fy2"]] = d[["eps_fy1", "eps_fy2"]].ffill()
        d["ticker"] = ticker
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. As-of join estimates onto holdings dates
# ---------------------------------------------------------------------------
def asof_join(weights: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    """For each (date, ticker) weight row, attach the most recent estimate
    on or before that date. Backward as-of, so no lookahead."""
    out = []
    for ticker, wgrp in weights.groupby("ticker"):
        egrp = est[est.ticker == ticker]
        if egrp.empty:
            # No estimate file for this name (e.g. cash). Keep with NaN EPS.
            tmp = wgrp.copy()
            tmp["eps_fy1"] = pd.NA
            tmp["eps_fy2"] = pd.NA
            out.append(tmp)
            continue
        merged = pd.merge_asof(
            wgrp.sort_values("DATE"),
            egrp.sort_values("Date")[["Date", "eps_fy1", "eps_fy2"]],
            left_on="DATE", right_on="Date", direction="backward",
        )
        merged = merged.drop(columns=["Date"])
        out.append(merged)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Aggregate
# ---------------------------------------------------------------------------
def aggregate(joined: pd.DataFrame) -> pd.DataFrame:
    """Weighted FY1/FY2 per day, renormalized across estimate-bearing names."""
    joined = joined.copy()
    joined["is_cash"] = joined["ticker"] == "cash"
    # A name contributes only if it has an FY1 estimate that day.
    has_est = joined["eps_fy1"].notna() & ~joined["is_cash"]

    records = []
    for date, g in joined.groupby("DATE"):
        est_rows = g[has_est.loc[g.index]]
        base_w = est_rows["weight"].sum()  # non-cash, estimate-bearing weight
        cash_w = g.loc[g.is_cash, "weight"].sum()

        if base_w == 0:
            records.append((date, pd.NA, pd.NA, 0.0, cash_w))
            continue

        if RENORMALIZE:
            w = est_rows["weight"] / base_w
        else:
            w = est_rows["weight"]  # cash sits in denom at zero contribution

        fy1 = (w * est_rows["eps_fy1"]).sum()
        fy2 = (w * est_rows["eps_fy2"]).sum() if est_rows["eps_fy2"].notna().all() \
              else (w * est_rows["eps_fy2"].astype(float)).sum(min_count=1)
        # coverage = share of non-cash weight that had an estimate
        noncash_w = g.loc[~g.is_cash, "weight"].sum()
        coverage = base_w / noncash_w if noncash_w else pd.NA
        records.append((date, fy1, fy2, coverage, cash_w))

    agg = pd.DataFrame(records, columns=[
        "date", "agg_eps_fy1", "agg_eps_fy2", "est_weight_coverage", "cash_weight"])
    return agg.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. CTIF price (yfinance)  <-- the only step this sandbox could not run live
# ---------------------------------------------------------------------------
def fetch_ctif_price(start, end) -> pd.Series:
    import yfinance as yf
    df = yf.download(CTIF_TICKER, start=start, end=end + pd.Timedelta(days=1),
                     auto_adjust=AUTO_ADJUST, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned no data for CTIF. Check ticker/network.")
    # yfinance may return single- or multi-index columns depending on version.
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    close.name = "ctif_price"
    return close


# ---------------------------------------------------------------------------
# 5. Chart
# ---------------------------------------------------------------------------
def make_chart(agg: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.family"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(agg["date"], agg["agg_eps_fy1"], color="black",
             linewidth=1.6, label="Aggregate EPS, current year (FY1)")
    ax1.plot(agg["date"], agg["agg_eps_fy2"], color="black", linestyle="--",
             linewidth=1.6, label="Aggregate EPS, next year (FY2)")
    ax1.set_ylabel("Weighted aggregate EPS estimate")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(agg["date"], agg["ctif_price"], color="0.55",
             linewidth=1.6, label="CTIF price")
    ax2.set_ylabel("CTIF price ($)")

    # Combined legend
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", frameon=False, fontsize=9)

    ax1.set_title("CTIF: weighted aggregate earnings estimates vs price",
                  fontsize=12)
    ax1.grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    weights = pd.read_csv(WEIGHTS_FILE, parse_dates=["DATE"])
    est = load_estimates()

    print("Estimates: %d rows across %d tickers, %s to %s"
          % (len(est), est.ticker.nunique(),
             est.Date.min().date(), est.Date.max().date()))

    joined = asof_join(weights, est)

    # Coverage diagnostics: did any held name lack an estimate?
    stock = joined[joined.ticker != "cash"]
    missing = stock[stock.eps_fy1.isna()]
    if len(missing):
        print("\nHeld name-days with no available FY1 estimate: %d" % len(missing))
        print(missing.groupby("ticker").size().to_string())
    else:
        print("\nEvery held stock has an FY1 estimate on every holdings date.")

    agg = aggregate(joined)
    print("\nEstimate weight coverage (share of non-cash weight with an "
          "estimate): min %.4f max %.4f"
          % (agg.est_weight_coverage.min(), agg.est_weight_coverage.max()))

    # Price
    price = fetch_ctif_price(agg.date.min(), agg.date.max())
    agg = agg.merge(price.rename("ctif_price"),
                    left_on="date", right_index=True, how="left")
    n_missing_px = agg.ctif_price.isna().sum()
    if n_missing_px:
        print("Note: %d holdings dates had no CTIF price (non-trading day "
              "misalignment); forward-filling price for the chart." % n_missing_px)
        agg["ctif_price"] = agg["ctif_price"].ffill()

    # Outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg_path = OUTPUT_DIR / "aggregate_estimate_series.csv"
    agg.to_csv(agg_path, index=False)

    per_name = joined.rename(columns={"DATE": "date"})[
        ["date", "ticker", "weight", "eps_fy1", "eps_fy2"]]
    per_name = per_name.sort_values(["date", "ticker"])
    per_name_path = OUTPUT_DIR / "per_name_estimates.csv"
    per_name.to_csv(per_name_path, index=False)

    chart_path = OUTPUT_DIR / "estimate_vs_price.png"
    make_chart(agg, chart_path)

    print("\nWrote:")
    print("  %s (%d days)" % (agg_path, len(agg)))
    print("  %s (%d rows)" % (per_name_path, len(per_name)))
    print("  %s" % chart_path)


if __name__ == "__main__":
    main()
