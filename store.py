"""Paper-trading log (SQLite). Every prediction is stored, then scored later
against what the price actually did. This is the 'honest scorecard'."""
import json
import sqlite3
from datetime import datetime

import pandas as pd

import config
from data import get_prices

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT, created_at TEXT, asof TEXT,
    signal TEXT, confidence REAL, horizon_days INTEGER,
    price_at_signal REAL, payload TEXT,
    resolved INTEGER DEFAULT 0, price_at_horizon REAL, fwd_return_pct REAL
);
"""


def _conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(SCHEMA)
    return conn


def log_prediction(ticker, decision, tech, horizon_days, payload_extra=None):
    payload = {"decision": decision.model_dump(), "technicals": tech, **(payload_extra or {})}
    with _conn() as c:
        c.execute(
            "INSERT INTO predictions (ticker, created_at, asof, signal, confidence, horizon_days,"
            " price_at_signal, payload) VALUES (?,?,?,?,?,?,?,?)",
            (ticker, datetime.now().isoformat(timespec="seconds"), tech["as_of"], decision.signal,
             decision.confidence, horizon_days, tech["close"], json.dumps(payload, default=str)))


def resolve_due() -> int:
    """Fill in outcomes for predictions whose horizon has passed. Returns count resolved."""
    n = 0
    with _conn() as c:
        rows = c.execute("SELECT id, ticker, asof, horizon_days, price_at_signal FROM predictions "
                         "WHERE resolved = 0").fetchall()
        for pid, ticker, asof, horizon, p0 in rows:
            try:
                df = get_prices(ticker, "1y")
            except ValueError:
                continue
            pos = df.index.searchsorted(pd.Timestamp(asof), side="right") - 1
            target = pos + horizon
            if pos < 0 or target >= len(df):
                continue  # horizon not reached yet
            p1 = float(df["close"].iloc[target])
            c.execute("UPDATE predictions SET resolved=1, price_at_horizon=?, fwd_return_pct=? WHERE id=?",
                      (p1, (p1 / p0 - 1) * 100, pid))
            n += 1
    return n


def load_predictions() -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", c)


def score(df: pd.DataFrame) -> dict:
    """Scorecard metrics from any frame with columns: signal, fwd_return_pct.

    - hit rate is computed only over BUY/SELL calls (BUY right if price rose, SELL if it fell)
    - strategy return: BUY earns +ret, SELL earns -ret, HOLD/NO TRADE earn 0
    - compared against 'buy and hold' = average forward return over all sampled points
    """
    d = df.dropna(subset=["fwd_return_pct"])
    if d.empty:
        return {"n": 0}
    trades = d[d["signal"].isin(["BUY", "SELL"])]
    direction = trades["signal"].map({"BUY": 1, "SELL": -1})
    strat = (direction * trades["fwd_return_pct"])
    no_trade = d[d["signal"] == "NO TRADE"]
    return {
        "n": int(len(d)),
        "trades": int(len(trades)),
        "no_trade_rate_pct": round(len(no_trade) / len(d) * 100, 1),
        "hit_rate_pct": round(float((strat > 0).mean() * 100), 1) if len(trades) else None,
        "avg_trade_return_pct": round(float(strat.mean()), 2) if len(trades) else None,
        "buy_and_hold_avg_pct": round(float(d["fwd_return_pct"].mean()), 2),
        "avg_abs_move_when_no_trade_pct": round(float(no_trade["fwd_return_pct"].abs().mean()), 2)
        if len(no_trade) else None,
    }
