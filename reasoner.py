"""The decision engine: reasons ACROSS technicals, news and market context.

Two layers of safety:
1. The prompt tells the model to prefer NO TRADE when evidence conflicts.
2. apply_guardrails() enforces it in code, so a confident-but-wrong model
   can't force a trade through.
"""
import json
from typing import Optional

import config
from llm import chat_json
from schemas import Decision, NewsSummary

SYSTEM = """You are the reasoning core of MetaTrade AI, a transparent decision-support tool for
retail traders. You receive pre-computed technical indicators, structured news analysis (may be
absent), and broad-market context. You output BUY, HOLD, SELL or NO TRADE for the given time horizon.

How to reason:
1. Evaluate each evidence family separately: trend, momentum (RSI/MACD), volume, candlestick
   patterns, news, market backdrop.
2. Explicitly note where families REINFORCE each other and where they CONFLICT.
3. Choose NO TRADE when: major signals conflict, evidence is thin, news is absent/low quality
   AND technicals are ambiguous, or volatility/market regime makes outcomes unpredictable.
   NO TRADE is a good answer, not a failure. HOLD means "if already in the stock, stay in";
   NO TRADE means "do not open a position".
4. Confidence is a calibrated probability-like number in [0,1]. Most situations deserve 0.4-0.65.
   Only go above 0.75 when several independent families agree strongly.
5. Never claim certainty. Never invent data that was not provided. Quote the numbers you were given.
6. List concrete future events/levels that would change your view (what_would_change_my_mind)."""


def apply_guardrails(d: Decision, news_quality: Optional[str]) -> Decision:
    """Code-level overrides. Returns a (possibly modified) copy."""
    d = d.model_copy()
    if d.signal in ("BUY", "SELL") and d.confidence < config.MIN_CONFIDENCE:
        d.guardrail_note = (f"Overridden from {d.signal} to NO TRADE: confidence {d.confidence:.2f} "
                            f"is below the {config.MIN_CONFIDENCE:.2f} minimum.")
        d.signal = "NO TRADE"
    elif d.signal in ("BUY", "SELL") and len(d.conflicts) >= 3:
        d.guardrail_note = (f"Overridden from {d.signal} to NO TRADE: {len(d.conflicts)} "
                            "conflicting signals were reported.")
        d.signal = "NO TRADE"
    elif d.signal in ("BUY", "SELL") and news_quality == "none" and d.confidence < 0.7:
        d.guardrail_note = (f"Overridden from {d.signal} to NO TRADE: no usable news and "
                            "only moderate confidence from technicals alone.")
        d.signal = "NO TRADE"
    return d


def decide(ticker: str, horizon_days: int, tech: dict, market: dict,
           news: Optional[NewsSummary] = None, model: str = None) -> Decision:
    news_block = (news.model_dump() if news is not None
                  else "NOT PROVIDED (technical-only mode). Do not assume any news.")
    user = (f"Ticker: {ticker}\nTime horizon: {horizon_days} trading days\n\n"
            f"TECHNICALS:\n{json.dumps(tech, indent=2)}\n\n"
            f"MARKET CONTEXT:\n{json.dumps(market, indent=2)}\n\n"
            f"NEWS ANALYSIS:\n{json.dumps(news_block, indent=2, default=str)}")
    raw = chat_json(SYSTEM, user, Decision, model or config.MODEL_REASON, temperature=0.2)
    return apply_guardrails(raw, news.news_quality if news is not None else None)
