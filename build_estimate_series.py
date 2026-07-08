"""
build_estimate_series.py
------------------------
Builds aggregate earnings-estimate series for the CTIF strategy and charts
them against CTIF's price history (yfinance) on dual axes.

Two aggregate constructions, FY1 (current year) and FY2 (next year) each:

  A. REVISION INDEX (main output / main charts), computed two ways:
       - ACTUAL WEIGHTS: prior-day portfolio weights (what the fund held)
       - EQUAL WEIGHT:  1/N across held names each day (daily rebalance),
                        so every name's revisions count the same
     Chain-linked index of same-name, same-fiscal-year estimate changes.
     Each day: weighted average of each held name's percent estimate change
     vs the prior holdings date, using prior-day weights renormalized across
     included names. Excluded from the chain:
       - fiscal-year ROLL days (FY1/FY2 relabel to a new year; detected as
         |change| > ROLL_THRESHOLD on the first business day of a month,
         which matches every roll in this dataset and no genuine revision)
       - names entering or exiting the portfolio (no same-name prior value)
     Result: moves reflect actual analyst revisions only. Index starts at 100.

  B. LEVEL SERIES (secondary chart, appendix use).
     Weighted average EPS level. Contains mechanical steps from fiscal-year
     rolls and composition turnover; kept for reference, not for inference.

Cash (BOXX + Cash&Other) carries no estimate; stock weights are renormalized
to sum to 1 across estimate-bearing names each day (RENORMALIZE = False keeps
cash in the denominator at zero contribution instead).

Inputs:
  - daily_weights_long.csv   (from compute_daily_weights.py)
  - estimates_ti/*.csv       (Date, Ticker, BEST_EPS, BEST_EPS_NXT_YR)
  - CTIF price via yfinance

Outputs:
  - aggregate_estimate_series.csv  (date, revision indices, levels, price,
                                    coverage, cash weight)
  - per_name_estimates.csv         (per name-day: weight, eps, change, roll flag)
  - estimate_revisions_vs_price.png  (MAIN chart: revision indices vs price)
  - estimate_levels_vs_price.png     (secondary: levels vs price)

No lookahead anywhere: estimates joined with a backward as-of merge; revision
chain uses prior-day weights.
"""

import glob
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS_FILE = Path("daily_weights_long.csv")
ESTIMATES_DIR = Path("estimates_ti")
OUTPUT_DIR = Path(".")

CTIF_TICKER = "CTIF"
AUTO_ADJUST = True      # dividend-adjusted close (total return basis)
RENORMALIZE = True
ROLL_THRESHOLD = 0.045  # |day-over-day estimate change| above this, on the
                        # first business day of a month, is a fiscal-year roll

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
    """Add per-name day-over-day changes and a fiscal-year roll flag."""
    df = joined[joined.ticker != "cash"].sort_values(["ticker", "DATE"]).copy()
    df["eps_fy1_prev"] = df.groupby("ticker")["eps_fy1"].shift()
    df["eps_fy2_prev"] = df.groupby("ticker")["eps_fy2"].shift()
    df["w_prev"] = df.groupby("ticker")["weight"].shift()
    # prior holdings date per name; a gap (name exited and re-entered) breaks
    # the same-name chain, so require consecutive holdings dates
    df["date_prev"] = df.groupby("ticker")["DATE"].shift()

    df["chg_fy1"] = df["eps_fy1"] / df["eps_fy1_prev"] - 1
    df["chg_fy2"] = df["eps_fy2"] / df["eps_fy2_prev"] - 1

    # First business day of each month on the holdings calendar
    days = pd.Series(sorted(joined["DATE"].unique()))
    first_bday = set(days.groupby([days.dt.year, days.dt.month]).min())
    df["is_first_bday"] = df["DATE"].isin(first_bday)

    df["is_roll"] = df["is_first_bday"] & (df["chg_fy1"].abs() > ROLL_THRESHOLD)
    return df


# ---------------------------------------------------------------------------
# A. Chain-linked revision indices
# ---------------------------------------------------------------------------
def revision_indices(named: pd.DataFrame, all_dates: list,
                     weighting: str = "actual") -> pd.DataFrame:
    """Chain same-name estimate changes into FY1/FY2 revision indices.

    weighting = "actual": prior-day portfolio weights (what the fund held).
    weighting = "equal":  1/N across included names each day (daily rebalance),
                          so every stock's revisions count the same regardless
                          of position size.
    """
    date_pos = {d: i for i, d in enumerate(all_dates)}
    records = []
    idx1, idx2 = 100.0, 100.0
    for i, d in enumerate(all_dates):
        if i == 0:
            records.append((d, idx1, idx2, pd.NA))
            continue
        g = named[named.DATE == d]
        # include: has prior value on the immediately preceding holdings date,
        # not a roll day for that name, valid weights
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

        r1 = wavg(g1, "chg_fy1")
        r2 = wavg(g2, "chg_fy2")
        idx1 *= (1 + r1)
        idx2 *= (1 + r2)
        cov = g1.w_prev.sum() / g.w_prev.sum() if g.w_prev.sum() > 0 else pd.NA
        records.append((d, idx1, idx2, cov))
    suffix = "aw" if weighting == "actual" else "ew"
    return pd.DataFrame(records, columns=[
        "date", f"rev_index_fy1_{suffix}", f"rev_index_fy2_{suffix}",
        f"chain_coverage_{suffix}"])


# ---------------------------------------------------------------------------
# B. Weighted level series (reference)
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
# CTIF price (yfinance)
# ---------------------------------------------------------------------------
def fetch_ctif_price(start, end) -> pd.Series:
    import yfinance as yf
    df = yf.download(CTIF_TICKER, start=start, end=end + pd.Timedelta(days=1),
                     auto_adjust=AUTO_ADJUST, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned no data for CTIF.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    close.name = "ctif_price"
    return close


# ---------------------------------------------------------------------------
# Charts (dual axes: FY1 + FY2 on left, CTIF price on right)
# ---------------------------------------------------------------------------
def dual_axis_chart(df, ycols, ylabels, ylab_left, title, path):
    fig, ax1 = plt.subplots(figsize=(11, 6))
    styles = [("black", "-"), ("black", "--")]
    for (col, lab), (color, ls) in zip(zip(ycols, ylabels), styles):
        ax1.plot(df["date"], df[col], color=color, linestyle=ls,
                 linewidth=1.6, label=lab)
    ax1.set_ylabel(ylab_left)
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["ctif_price"], color="0.55",
             linewidth=1.6, label="CTIF price (right)")
    ax2.set_ylabel("CTIF price ($)")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", frameon=False,
               fontsize=9)
    ax1.set_title(title, fontsize=12)
    ax1.grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    weights = pd.read_csv(WEIGHTS_FILE, parse_dates=["DATE"])
    est = load_estimates()
    print("Estimates: %d rows, %d tickers, %s to %s"
          % (len(est), est.ticker.nunique(),
             est.Date.min().date(), est.Date.max().date()))

    joined = asof_join(weights, est)
    stock = joined[joined.ticker != "cash"]
    n_missing = stock.eps_fy1.isna().sum()
    print("Held name-days without FY1 estimate: %d" % n_missing)

    named = flag_rolls(joined)
    rolls = named[named.is_roll]
    print("\nFiscal-year rolls excluded from revision chain (%d):" % len(rolls))
    print(rolls[["DATE", "ticker", "chg_fy1"]]
          .assign(chg_fy1=lambda d: (100 * d.chg_fy1).round(1))
          .rename(columns={"chg_fy1": "fy1_chg_pct"})
          .to_string(index=False))
    kept_big = named[~named.is_roll & (named.chg_fy1.abs() > ROLL_THRESHOLD)]
    print("\nLarge genuine revisions KEPT in the chain (%d):" % len(kept_big))
    print(kept_big[["DATE", "ticker", "chg_fy1"]]
          .assign(chg_fy1=lambda d: (100 * d.chg_fy1).round(1))
          .rename(columns={"chg_fy1": "fy1_chg_pct"})
          .to_string(index=False))

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
    print("Chain coverage (weight share chained): min %.3f"
          % agg.chain_coverage_aw.dropna().min())

    price = fetch_ctif_price(agg.date.min(), agg.date.max())
    agg = agg.merge(price.rename("ctif_price"),
                    left_on="date", right_index=True, how="left")
    if agg.ctif_price.isna().any():
        agg["ctif_price"] = agg["ctif_price"].ffill()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUTPUT_DIR / "aggregate_estimate_series.csv", index=False)

    per_name = named.rename(columns={"DATE": "date"})[
        ["date", "ticker", "weight", "eps_fy1", "eps_fy2",
         "chg_fy1", "chg_fy2", "is_roll"]].sort_values(["date", "ticker"])
    per_name.to_csv(OUTPUT_DIR / "per_name_estimates.csv", index=False)

    dual_axis_chart(
        agg, ["rev_index_fy1_aw", "rev_index_fy2_aw"],
        ["EPS revision index, current year (FY1)",
         "EPS revision index, next year (FY2)"],
        "Estimate revision index (start = 100)",
        "CTIF: estimate revisions, actual portfolio weights, vs price",
        OUTPUT_DIR / "estimate_revisions_actual_weight.png")

    dual_axis_chart(
        agg, ["rev_index_fy1_ew", "rev_index_fy2_ew"],
        ["EPS revision index, current year (FY1)",
         "EPS revision index, next year (FY2)"],
        "Estimate revision index (start = 100)",
        "CTIF: estimate revisions, equal weight (daily rebalance), vs price",
        OUTPUT_DIR / "estimate_revisions_equal_weight.png")

    dual_axis_chart(
        agg, ["rev_index_fy1_aw", "rev_index_fy1_ew"],
        ["FY1 revisions, actual weights",
         "FY1 revisions, equal weight"],
        "Estimate revision index (start = 100)",
        "CTIF: FY1 revision index, actual vs equal weighting, vs price",
        OUTPUT_DIR / "estimate_revisions_weighting_comparison.png")

    dual_axis_chart(
        agg, ["agg_eps_fy1", "agg_eps_fy2"],
        ["Aggregate EPS level, current year (FY1)",
         "Aggregate EPS level, next year (FY2)"],
        "Weighted aggregate EPS estimate ($)",
        "CTIF: aggregate estimate levels vs price (reference; contains "
        "fiscal-year roll and turnover steps)",
        OUTPUT_DIR / "estimate_levels_vs_price.png")

    print("\nWrote: aggregate_estimate_series.csv, per_name_estimates.csv,")
    print("       estimate_revisions_actual_weight.png,")
    print("       estimate_revisions_equal_weight.png,")
    print("       estimate_revisions_weighting_comparison.png,")
    print("       estimate_levels_vs_price.png (reference)")


if __name__ == "__main__":
    main()