"""Walk-forward backtest, technicals + market context only (no look-ahead).

At each sampled date we show the model ONLY data up to that date, then compare
its signal to the real forward return.

Usage:
    python backtest.py --tickers AAPL MSFT NVDA --step 10 --horizon 5 --max-calls 60
Each LLM call costs money, so --step and --max-calls keep it cheap.
"""
import argparse

import pandas as pd

import config
from data import get_prices, load_market_frames, market_context
from indicators import summarize
from reasoner import decide
from store import score

WARMUP_BARS = 60


def run(tickers, step, horizon, bars, max_calls, model=None):
    spy, vix = load_market_frames("3y")
    rows, calls = [], 0
    for t in tickers:
        df = get_prices(t, "3y")
        last_start = len(df) - horizon - 1
        first = max(WARMUP_BARS, last_start - bars)
        for i in range(first, last_start + 1, step):
            if calls >= max_calls:
                break
            hist = df.iloc[: i + 1]  # <-- only data up to and including day i
            asof = hist.index[-1]
            tech = summarize(hist)
            ctx = market_context(spy, vix, asof)
            try:
                d = decide(t, horizon, tech, ctx, news=None, model=model)
            except Exception as e:  # keep going if one call fails
                print(f"skip {t} {asof.date()}: {e}")
                continue
            calls += 1
            fwd = (df["close"].iloc[i + horizon] / df["close"].iloc[i] - 1) * 100
            rows.append({"ticker": t, "asof": str(asof.date()), "signal": d.signal,
                         "confidence": d.confidence, "fwd_return_pct": round(float(fwd), 3),
                         "guardrail": d.guardrail_note})
            print(f"{t} {asof.date()} -> {d.signal:8s} conf={d.confidence:.2f} fwd={fwd:+.2f}%")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "JPM", "XOM"])
    ap.add_argument("--step", type=int, default=10, help="bars between samples")
    ap.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON_DAYS)
    ap.add_argument("--bars", type=int, default=250, help="how far back to sample")
    ap.add_argument("--max-calls", type=int, default=60)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="backtest_results.csv")
    a = ap.parse_args()

    res = run(a.tickers, a.step, a.horizon, a.bars, a.max_calls, a.model)
    res.to_csv(a.out, index=False)
    print("\nSaved", a.out)
    print(score(res))
