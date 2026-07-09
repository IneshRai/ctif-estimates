"""
build_estimate_series.py
------------------------
Builds aggregate earnings-estimate series for the CTIF strategy, charts them
against CTIF's price history (yfinance), and produces a PDF chartbook with
the three aggregate charts up front followed by one page per stock (holding
period only) in the same format.

Aggregate constructions (FY1 = current year, FY2 = next year):

  A. REVISION INDEX, computed two ways:
       - ACTUAL WEIGHTS: prior-day portfolio weights (what the fund held)
       - EQUAL WEIGHT:  1/N across held names each day (daily rebalance)
     Chain-linked index of same-name, same-fiscal-year estimate changes.
     Excluded from the chain:
       - fiscal-year ROLL days (detected as |change| > ROLL_THRESHOLD on the
         first business day of a month; matches every roll in this dataset
         and no genuine revision)
       - names entering or exiting the portfolio (no same-name prior value)
     Result: lines move only on actual analyst revisions. Start = 100.

  B. LEVEL SERIES (reference). Weighted average EPS level; contains
     mechanical steps from fiscal-year rolls and turnover.

Per-stock chartbook pages: BEST_EPS (FY1) and BEST_EPS_NXT_YR (FY2) levels
vs the stock's own price, restricted to the dates the fund held the name.
Note: per-name FY1/FY2 levels include the fiscal-year roll step; on a
single-name chart that step is visible and interpretable, unlike in the
aggregate.

Colors: FY1 dark blue, FY2 light blue, price black (right axis).

Inputs:
  - daily_weights_long.csv   (from compute_daily_weights.py)
  - estimates_ti/*.csv       (Date, Ticker, BEST_EPS, BEST_EPS_NXT_YR)
  - prices via yfinance (CTIF + each stock)

Outputs:
  - aggregate_estimate_series.csv
  - per_name_estimates.csv
  - estimate_revisions_actual_weight.png
  - estimate_revisions_equal_weight.png
  - estimate_revisions_weighting_comparison.png
  - estimate_levels_vs_price.png
  - ctif_estimate_chartbook.pdf   (3 aggregate pages + 1 page per stock)

No lookahead: estimates joined with a backward as-of merge; revision chain
uses prior-day weights.
"""

import glob
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS_FILE = Path("daily_weights_long.csv")
ESTIMATES_DIR = Path("estimates_ti")
OUTPUT_DIR = Path(".")

CTIF_TICKER = "CTIF"
AUTO_ADJUST = True      # dividend-adjusted close (total return basis)
RENORMALIZE = True
ROLL_THRESHOLD = 0.045

# Colors
COL_FY1 = "#1F4E79"     # dark blue
COL_FY2 = "#5B9BD5"     # light blue
COL_PRICE = "black"

plt.rcParams["font.family"] = ["Arial", "Liberation Sans", "DejaVu Sans"]


# ---------------------------------------------------------------------------
# Load estimates (event-driven files -> per-name ffill)
# ---------------------------------------------------------------------------
def load_estimates() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(ESTIMATES_DIR / "*.csv"))):
        ticker = os.path.splitext(os.path.basename(f))[0]
        d = pd.read_csv(f, parse_dates=["Date"])
        d = d.rename(columns={"BEST_EPS": "eps_fy1",
                              "BEST_EPS_NXT_YR": "eps_fy2"})
        d = d[["Date", "eps_fy1", "eps_fy2"]].sort_values("Date")
        d[["eps_fy1", "eps_fy2"]] = d[["eps_fy1", "eps_fy2"]].ffill()
        d["ticker"] = ticker
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Backward as-of join onto holdings dates (no lookahead)
# ---------------------------------------------------------------------------
def asof_join(weights: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ticker, wgrp in weights.groupby("ticker"):
        egrp = est[est.ticker == ticker]
        if egrp.empty:
            tmp = wgrp.copy()
            tmp["eps_fy1"] = pd.NA
            tmp["eps_fy2"] = pd.NA
            out.append(tmp)
            continue
        merged = pd.merge_asof(
            wgrp.sort_values("DATE"),
            egrp.sort_values("Date")[["Date", "eps_fy1", "eps_fy2"]],
            left_on="DATE", right_on="Date", direction="backward",
        ).drop(columns=["Date"])
        out.append(merged)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Roll detection
# ---------------------------------------------------------------------------
def flag_rolls(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined[joined.ticker != "cash"].sort_values(["ticker", "DATE"]).copy()
    df["eps_fy1_prev"] = df.groupby("ticker")["eps_fy1"].shift()
    df["eps_fy2_prev"] = df.groupby("ticker")["eps_fy2"].shift()
    df["w_prev"] = df.groupby("ticker")["weight"].shift()
    df["date_prev"] = df.groupby("ticker")["DATE"].shift()

    df["chg_fy1"] = df["eps_fy1"] / df["eps_fy1_prev"] - 1
    df["chg_fy2"] = df["eps_fy2"] / df["eps_fy2_prev"] - 1

    days = pd.Series(sorted(joined["DATE"].unique()))
    first_bday = set(days.groupby([days.dt.year, days.dt.month]).min())
    df["is_first_bday"] = df["DATE"].isin(first_bday)
    df["is_roll"] = df["is_first_bday"] & (df["chg_fy1"].abs() > ROLL_THRESHOLD)
    return df


# ---------------------------------------------------------------------------
# Chain-linked revision indices (actual or equal weighting)
# ---------------------------------------------------------------------------
def revision_indices(named: pd.DataFrame, all_dates: list,
                     weighting: str = "actual") -> pd.DataFrame:
    date_pos = {d: i for i, d in enumerate(all_dates)}
    records = []
    idx1, idx2 = 100.0, 100.0
    for i, d in enumerate(all_dates):
        if i == 0:
            records.append((d, idx1, idx2, pd.NA))
            continue
        g = named[named.DATE == d]
        ok = (g.date_prev.notna()
              & g.date_prev.map(lambda x: date_pos.get(x, -2)).eq(i - 1)
              & ~g.is_roll
              & g.w_prev.notna())
        g1 = g[ok & g.chg_fy1.notna()]
        g2 = g[ok & g.chg_fy2.notna()]

        def wavg(sub, col):
            if len(sub) == 0:
                return 0.0
            if weighting == "equal":
                return sub[col].mean()
            wsum = sub.w_prev.sum()
            return (sub.w_prev * sub[col]).sum() / wsum if wsum > 0 else 0.0

        idx1 *= (1 + wavg(g1, "chg_fy1"))
        idx2 *= (1 + wavg(g2, "chg_fy2"))
        cov = g1.w_prev.sum() / g.w_prev.sum() if g.w_prev.sum() > 0 else pd.NA
        records.append((d, idx1, idx2, cov))
    suffix = "aw" if weighting == "actual" else "ew"
    return pd.DataFrame(records, columns=[
        "date", f"rev_index_fy1_{suffix}", f"rev_index_fy2_{suffix}",
        f"chain_coverage_{suffix}"])


# ---------------------------------------------------------------------------
# Weighted level series (reference)
# ---------------------------------------------------------------------------
def level_series(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.copy()
    df["is_cash"] = df["ticker"] == "cash"
    records = []
    for date, g in df.groupby("DATE"):
        est_rows = g[g.eps_fy1.notna() & ~g.is_cash]
        base_w = est_rows.weight.sum()
        cash_w = g.loc[g.is_cash, "weight"].sum()
        if base_w == 0:
            records.append((date, pd.NA, pd.NA, 0.0, cash_w))
            continue
        w = est_rows.weight / base_w if RENORMALIZE else est_rows.weight
        fy1 = (w * est_rows.eps_fy1).sum()
        fy2 = (w * est_rows.eps_fy2.astype(float)).sum(min_count=1)
        noncash_w = g.loc[~g.is_cash, "weight"].sum()
        cov = base_w / noncash_w if noncash_w else pd.NA
        records.append((date, fy1, fy2, cov, cash_w))
    return pd.DataFrame(records, columns=[
        "date", "agg_eps_fy1", "agg_eps_fy2",
        "est_weight_coverage", "cash_weight"]).sort_values("date")


# ---------------------------------------------------------------------------
# Prices (yfinance): CTIF + all stocks in one batch
# ---------------------------------------------------------------------------
def fetch_prices(tickers: list, start, end) -> pd.DataFrame:
    """Return DataFrame indexed by date, one column per ticker (Close)."""
    import yfinance as yf
    df = yf.download(tickers, start=start, end=end + pd.Timedelta(days=1),
                     auto_adjust=AUTO_ADJUST, progress=False)
    close = df["Close"]
    if isinstance(close, pd.Series):        # single ticker
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index)
    missing = [t for t in tickers if t not in close.columns
               or close[t].isna().all()]
    if missing:
        print("WARNING: no price data for: %s" % ", ".join(missing))
    return close


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def _pct_aligned_limits(p_start, p_lo, p_hi, a_start, e_lo, e_hi,
                        pad_frac=0.05):
    """Give both axes the same percentage scale anchored at the starting
    values: a +x% move in price spans the same vertical distance as a +x%
    estimate change, and both series start at the same height (Bloomberg
    style). Returns (left_lims, right_lims)."""
    # ranges expressed as percent moves from each series' start
    p_lo_pct, p_hi_pct = p_lo / p_start - 1, p_hi / p_start - 1
    e_lo_pct, e_hi_pct = e_lo / a_start - 1, e_hi / a_start - 1
    lo_pct = min(p_lo_pct, e_lo_pct)
    hi_pct = max(p_hi_pct, e_hi_pct)
    pad = (hi_pct - lo_pct) * pad_frac or 0.01
    lo_pct, hi_pct = lo_pct - pad, hi_pct + pad
    left = (p_start * (1 + lo_pct), p_start * (1 + hi_pct))
    right = (a_start * (1 + lo_pct), a_start * (1 + hi_pct))
    return left, right


def _dual_axis_fig(df, ycols, ylabels, ylab_right, title,
                   price_col="ctif_price", price_label="CTIF price (left)",
                   figsize=(11, 6)):
    fig, ax1 = plt.subplots(figsize=figsize)

    # LEFT axis: price (black)
    ax1.plot(df["date"], df[price_col], color=COL_PRICE,
             linewidth=1.2, label=price_label)
    ax1.set_ylabel("Price ($)")
    ax1.set_xlabel("Date")

    # RIGHT axis: estimates (blues)
    ax2 = ax1.twinx()
    styles = [(COL_FY1, "-"), (COL_FY2, "-")]
    for (col, lab), (color, ls) in zip(zip(ycols, ylabels), styles):
        ax2.plot(df["date"], df[col], color=color, linestyle=ls,
                 linewidth=1.7, label=lab)
    ax2.set_ylabel(ylab_right)

    # Same percentage scale on both axes, anchored at starting values,
    # so the FY1 line and the price line start at the same height and
    # equal percent moves cover equal vertical distance.
    price_vals = pd.to_numeric(df[price_col], errors="coerce").dropna()
    est_vals = df[ycols].astype(float)
    anchor = est_vals[ycols[0]].dropna()
    if len(price_vals) and len(anchor) and price_vals.iloc[0] > 0 \
            and anchor.iloc[0] > 0:
        left, right = _pct_aligned_limits(
            price_vals.iloc[0], price_vals.min(), price_vals.max(),
            anchor.iloc[0], est_vals.min().min(), est_vals.max().max())
        ax1.set_ylim(*left)
        ax2.set_ylim(*right)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", frameon=False,
               fontsize=9)
    ax1.set_title(title, fontsize=12)
    ax1.grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    return fig


def save_chart(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chartbook
# ---------------------------------------------------------------------------
def make_chartbook(agg, joined, prices, path):
    """3 aggregate pages up front, then one page per stock (holding period)."""
    stock_days = (joined[joined.ticker != "cash"]
                  .groupby("ticker")["DATE"].agg(["min", "max", "count"]))
    with PdfPages(path) as pdf:
        # Page 1: revisions, actual weights
        fig = _dual_axis_fig(
            agg, ["rev_index_fy1_aw", "rev_index_fy2_aw"],
            ["EPS revision index, current year (FY1)",
             "EPS revision index, next year (FY2)"],
            "Estimate revision index (start = 100)",
            "CTIF: estimate revisions, actual portfolio weights, vs price")
        pdf.savefig(fig); plt.close(fig)

        # Page 2: revisions, equal weight
        fig = _dual_axis_fig(
            agg, ["rev_index_fy1_ew", "rev_index_fy2_ew"],
            ["EPS revision index, current year (FY1)",
             "EPS revision index, next year (FY2)"],
            "Estimate revision index (start = 100)",
            "CTIF: estimate revisions, equal weight (daily rebalance), "
            "vs price")
        pdf.savefig(fig); plt.close(fig)

        # Page 3: levels (reference)
        fig = _dual_axis_fig(
            agg, ["agg_eps_fy1", "agg_eps_fy2"],
            ["Aggregate EPS level, current year (FY1)",
             "Aggregate EPS level, next year (FY2)"],
            "Weighted aggregate EPS estimate ($)",
            "CTIF: aggregate estimate levels vs price (reference; contains "
            "fiscal-year roll and turnover steps)")
        pdf.savefig(fig); plt.close(fig)

        # One page per stock, holding period only
        for ticker in sorted(stock_days.index):
            sub = joined[(joined.ticker == ticker)].sort_values("DATE")
            sub = sub.rename(columns={"DATE": "date"})
            if ticker in prices.columns:
                px = prices[ticker].reindex(sub["date"]).ffill().values
            else:
                px = [float("nan")] * len(sub)
            sub = sub.assign(stock_price=px)
            start = stock_days.loc[ticker, "min"].date()
            end = stock_days.loc[ticker, "max"].date()
            fig = _dual_axis_fig(
                sub, ["eps_fy1", "eps_fy2"],
                ["EPS estimate, current year (FY1)",
                 "EPS estimate, next year (FY2)"],
                "Consensus EPS estimate ($)",
                "%s: EPS estimates vs price, holding period %s to %s"
                % (ticker, start, end),
                price_col="stock_price",
                price_label="%s price (left)" % ticker)
            pdf.savefig(fig); plt.close(fig)
    print("Chartbook written: %s (%d pages)" % (path, 3 + len(stock_days)))


# ---------------------------------------------------------------------------
def main() -> None:
    weights = pd.read_csv(WEIGHTS_FILE, parse_dates=["DATE"])
    est = load_estimates()
    print("Estimates: %d rows, %d tickers, %s to %s"
          % (len(est), est.ticker.nunique(),
             est.Date.min().date(), est.Date.max().date()))

    joined = asof_join(weights, est)
    stock = joined[joined.ticker != "cash"]
    print("Held name-days without FY1 estimate: %d"
          % stock.eps_fy1.isna().sum())

    named = flag_rolls(joined)
    rolls = named[named.is_roll]
    print("\nFiscal-year rolls excluded from revision chain (%d):" % len(rolls))
    print(rolls[["DATE", "ticker", "chg_fy1"]]
          .assign(chg_fy1=lambda d: (100 * d.chg_fy1).round(1))
          .rename(columns={"chg_fy1": "fy1_chg_pct"}).to_string(index=False))
    kept = named[~named.is_roll & (named.chg_fy1.abs() > ROLL_THRESHOLD)]
    print("\nLarge genuine revisions KEPT in the chain (%d):" % len(kept))
    print(kept[["DATE", "ticker", "chg_fy1"]]
          .assign(chg_fy1=lambda d: (100 * d.chg_fy1).round(1))
          .rename(columns={"chg_fy1": "fy1_chg_pct"}).to_string(index=False))

    all_dates = sorted(joined.DATE.unique())
    rev_aw = revision_indices(named, all_dates, weighting="actual")
    rev_ew = revision_indices(named, all_dates, weighting="equal")
    lev = level_series(joined)
    agg = rev_aw.merge(rev_ew, on="date").merge(lev, on="date")

    print("\nRevision index endpoints (start = 100):")
    print("  actual-weight: FY1 %.2f  FY2 %.2f"
          % (agg.rev_index_fy1_aw.iloc[-1], agg.rev_index_fy2_aw.iloc[-1]))
    print("  equal-weight:  FY1 %.2f  FY2 %.2f"
          % (agg.rev_index_fy1_ew.iloc[-1], agg.rev_index_fy2_ew.iloc[-1]))

    # Prices: CTIF + every stock, one batch
    stock_tickers = sorted(stock.ticker.unique())
    prices = fetch_prices([CTIF_TICKER] + stock_tickers,
                          agg.date.min(), agg.date.max())
    agg = agg.merge(prices[CTIF_TICKER].rename("ctif_price"),
                    left_on="date", right_index=True, how="left")
    if agg.ctif_price.isna().any():
        agg["ctif_price"] = agg["ctif_price"].ffill()

    # Outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUTPUT_DIR / "aggregate_estimate_series.csv", index=False)
    per_name = named.rename(columns={"DATE": "date"})[
        ["date", "ticker", "weight", "eps_fy1", "eps_fy2",
         "chg_fy1", "chg_fy2", "is_roll"]].sort_values(["date", "ticker"])
    per_name.to_csv(OUTPUT_DIR / "per_name_estimates.csv", index=False)

    save_chart(_dual_axis_fig(
        agg, ["rev_index_fy1_aw", "rev_index_fy2_aw"],
        ["EPS revision index, current year (FY1)",
         "EPS revision index, next year (FY2)"],
        "Estimate revision index (start = 100)",
        "CTIF: estimate revisions, actual portfolio weights, vs price"),
        OUTPUT_DIR / "estimate_revisions_actual_weight.png")

    save_chart(_dual_axis_fig(
        agg, ["rev_index_fy1_ew", "rev_index_fy2_ew"],
        ["EPS revision index, current year (FY1)",
         "EPS revision index, next year (FY2)"],
        "Estimate revision index (start = 100)",
        "CTIF: estimate revisions, equal weight (daily rebalance), vs price"),
        OUTPUT_DIR / "estimate_revisions_equal_weight.png")

    save_chart(_dual_axis_fig(
        agg, ["rev_index_fy1_aw", "rev_index_fy1_ew"],
        ["FY1 revisions, actual weights", "FY1 revisions, equal weight"],
        "Estimate revision index (start = 100)",
        "CTIF: FY1 revision index, actual vs equal weighting, vs price"),
        OUTPUT_DIR / "estimate_revisions_weighting_comparison.png")

    save_chart(_dual_axis_fig(
        agg, ["agg_eps_fy1", "agg_eps_fy2"],
        ["Aggregate EPS level, current year (FY1)",
         "Aggregate EPS level, next year (FY2)"],
        "Weighted aggregate EPS estimate ($)",
        "CTIF: aggregate estimate levels vs price (reference; contains "
        "fiscal-year roll and turnover steps)"),
        OUTPUT_DIR / "estimate_levels_vs_price.png")

    make_chartbook(agg, joined, prices,
                   OUTPUT_DIR / "ctif_estimate_chartbook.pdf")

    print("\nAll outputs written to %s" % OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()