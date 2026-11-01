"""Deterministic technical analysis. Pure pandas, no LLM, easy to test.

The LLM never computes indicators. It only *interprets* what this module reports.
"""
import numpy as np
import pandas as pd

PATTERN_COLS = ["doji", "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing"]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(loss == 0) & (gain > 0)] = 100.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    prev_o, prev_c = o.shift(1), c.shift(1)

    out = pd.DataFrame(index=df.index)
    out["doji"] = (rng > 0) & (body <= 0.1 * rng)
    out["hammer"] = (body > 0) & (lower >= 2 * body) & (upper <= body)
    out["shooting_star"] = (body > 0) & (upper >= 2 * body) & (lower <= body)
    out["bullish_engulfing"] = (prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)
    out["bearish_engulfing"] = (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)
    return out.fillna(False)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["rsi14"] = rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["atr14"] = atr(df)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    pats = candlestick_patterns(df)
    for col in PATTERN_COLS:
        df[col] = pats[col]
    return df


def _pct(a: float, b: float) -> float:
    return float((a / b - 1) * 100) if b else float("nan")


def summarize(df: pd.DataFrame) -> dict:
    """Compact, LLM-friendly description of the latest bar. Needs >= 60 bars."""
    if len(df) < 60:
        raise ValueError("need at least 60 bars of price history")
    d = add_indicators(df)
    last, prev = d.iloc[-1], d.iloc[-2]

    if last["close"] > last["sma20"] > last["sma50"]:
        trend = "uptrend (price > SMA20 > SMA50)"
    elif last["close"] < last["sma20"] < last["sma50"]:
        trend = "downtrend (price < SMA20 < SMA50)"
    else:
        trend = "mixed / sideways (moving averages not aligned)"

    r = float(last["rsi14"])
    rsi_zone = "overbought (>70)" if r > 70 else "oversold (<30)" if r < 30 else "neutral (30-70)"

    if prev["macd"] <= prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        macd_state = "bullish crossover just occurred"
    elif prev["macd"] >= prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        macd_state = "bearish crossover just occurred"
    else:
        macd_state = "MACD above signal (bullish momentum)" if last["macd"] > last["macd_signal"] \
            else "MACD below signal (bearish momentum)"

    recent = d.tail(5)
    patterns = []
    for idx, row in recent.iterrows():
        for col in PATTERN_COLS:
            if bool(row[col]):
                patterns.append({"date": str(idx.date()), "pattern": col})

    vr = float(last["vol_ratio"])
    return {
        "as_of": str(d.index[-1].date()),
        "close": round(float(last["close"]), 2),
        "return_5d_pct": round(_pct(last["close"], d["close"].iloc[-6]), 2),
        "return_20d_pct": round(_pct(last["close"], d["close"].iloc[-21]), 2),
        "trend": trend,
        "sma20": round(float(last["sma20"]), 2),
        "sma50": round(float(last["sma50"]), 2),
        "rsi14": round(r, 1),
        "rsi_zone": rsi_zone,
        "macd_state": macd_state,
        "macd_hist": round(float(last["macd_hist"]), 4),
        "atr14_pct_of_price": round(float(last["atr14"] / last["close"] * 100), 2),
        "volume_vs_20d_avg": round(vr, 2),
        "volume_note": "high volume" if vr > 1.5 else "low volume" if vr < 0.7 else "normal volume",
        "recent_candlestick_patterns_last_5_bars": patterns,
    }
