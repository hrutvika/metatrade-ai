"""Market data via yfinance (free, no key).

IMPORTANT for honest backtests: yfinance gives historical PRICES but only
CURRENT news. So backtests are technicals + market-context only; news is used
in live mode. Never feed today's news into a past prediction (look-ahead bias).
"""
from typing import Optional

import pandas as pd
import yfinance as yf


def get_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No price data found for '{ticker}'. Check the symbol.")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def get_news(ticker: str, limit: int = 10) -> list:
    """Normalize yfinance news (its format has changed over versions)."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    items = []
    for r in raw[:limit]:
        c = r.get("content", r) if isinstance(r, dict) else {}
        title = c.get("title") or ""
        if not title:
            continue
        provider = c.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else c.get("publisher", "")
        items.append({
            "title": title,
            "summary": c.get("summary") or c.get("description") or "",
            "published": c.get("pubDate") or c.get("providerPublishTime") or "",
            "publisher": publisher or "",
        })
    return items


def load_market_frames(period: str = "2y"):
    """SPY (market trend) and ^VIX (fear gauge)."""
    return get_prices("SPY", period), get_prices("^VIX", period)


def market_context(spy: pd.DataFrame, vix: pd.DataFrame, asof: Optional[pd.Timestamp] = None) -> dict:
    """Broad-market backdrop as of a date (uses only data up to that date)."""
    if asof is not None:
        spy, vix = spy.loc[:asof], vix.loc[:asof]
    if len(spy) < 51 or len(vix) < 21:
        return {"note": "not enough market data"}
    spy_close = spy["close"]
    sma50 = spy_close.rolling(50).mean().iloc[-1]
    vix_now = float(vix["close"].iloc[-1])
    return {
        "spy_trend": "above its 50-day average (bullish backdrop)" if spy_close.iloc[-1] > sma50
        else "below its 50-day average (weak backdrop)",
        "spy_return_20d_pct": round(float((spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100), 2),
        "vix_level": round(vix_now, 1),
        "vix_regime": "high fear (>25)" if vix_now > 25 else "elevated (18-25)" if vix_now > 18 else "calm (<18)",
        "vix_change_20d": round(vix_now - float(vix["close"].iloc[-21]), 1),
    }
