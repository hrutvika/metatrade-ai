"""Run with: pytest -q   (no network or API key needed)"""
import numpy as np
import pandas as pd

from indicators import add_indicators, candlestick_patterns, rsi, summarize


def make_df(n=120, drift=0.001, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.01, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(1_000_000, 2_000_000, n)}, index=idx)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up).iloc[-1] == 100
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi(down).iloc[-1] < 5
    r = rsi(make_df()["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_engulfing_and_doji():
    idx = pd.bdate_range("2025-01-01", periods=3)
    df = pd.DataFrame({
        "open": [10.0, 10.0, 9.0],
        "high": [10.5, 10.1, 11.5],
        "low": [9.5, 9.0, 8.9],
        "close": [10.0, 9.2, 11.2],  # bar1 bearish, bar2 big bullish engulfing
        "volume": [1, 1, 1]}, index=idx)
    p = candlestick_patterns(df)
    assert bool(p["bullish_engulfing"].iloc[2])
    assert bool(p["doji"].iloc[0])


def test_no_lookahead_in_indicators():
    """Indicators on a truncated frame must equal the same rows of the full frame."""
    df = make_df()
    full = add_indicators(df)
    part = add_indicators(df.iloc[:80])
    cols = ["sma20", "sma50", "rsi14", "macd", "macd_signal", "atr14"]
    pd.testing.assert_frame_equal(full.iloc[:80][cols], part[cols])


def test_summarize_shape():
    s = summarize(make_df())
    for key in ["as_of", "close", "trend", "rsi14", "macd_state", "volume_note",
                "recent_candlestick_patterns_last_5_bars"]:
        assert key in s
