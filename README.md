# Meta Global AI Developer Hackathon

## MetaTrade AI

A transparent trading **decision-support** engine that combines technical analysis with
LLM-interpreted news and market context, then outputs **BUY / HOLD / SELL / NO TRADE** with the evidence, conflicts, risks, and "what would change my mind", and scores itself against real outcomes through paper trading and walk-forward backtesting.

> Educational tool. Not financial advice. Does not place real trades.

Built for the Meta hackathon with **Muse Spark 1.3** via the Meta Model API.

## Why it's different
- **NO TRADE is a first-class answer.** The model is told to abstain when signals conflict, and guardrails in *code* (`reasoner.apply_guardrails`) enforce it.
- **Code does the math, the LLM does the reasoning.** RSI, MACD, candlestick patterns, and volume are computed deterministically (`indicators.py`, unit-tested). The model only interprets them.
- **Honest evaluation.** Walk-forward backtest with no look-ahead, plus a live paper-trading log.

## Architecture
```
yfinance prices ──► indicators.py ──┐
yfinance news ───► news.py (LLM) ───┼──► reasoner.py (LLM + guardrails) ──► Decision JSON
SPY + VIX ───────► data.market_context ┘                                    │
                                                        store.py (SQLite) ◄──┤
                                                        backtest.py ◄────────┘
                                                        app.py (Streamlit UI)
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your MODEL_API_KEY
pytest -q                   # indicator tests, no API key needed
streamlit run app.py
python backtest.py --tickers AAPL MSFT NVDA --step 10 --horizon 5 --max-calls 60
```

## Known limitations
- **Backtests are technicals + market context only.** Free data has no point-in-time news, so feeding today's news into past dates would leak the future. News is used in live mode.
- Small backtest samples are noisy. Treat hit rates as indicative, not proof of edge.
- No transaction costs, slippage, or position sizing.
- LLMs can't reliably predict prices; the value here is structured reasoning and abstention.
