"""MetaTrade AI - Streamlit UI.   Run:  streamlit run app.py"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from data import get_news, get_prices, load_market_frames, market_context
from indicators import add_indicators, summarize
from news import analyze_news
from reasoner import decide
from store import load_predictions, log_prediction, resolve_due, score

st.set_page_config(page_title="MetaTrade AI", page_icon="📈", layout="wide")
st.title("📈 MetaTrade AI")
st.caption(config.DISCLAIMER)

COLORS = {"BUY": "#1a9850", "SELL": "#d73027", "HOLD": "#f0a30a", "NO TRADE": "#6b7280"}

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    horizon = st.slider("Horizon (trading days)", 1, 20, config.DEFAULT_HORIZON_DAYS)
    use_news = st.checkbox("Include news analysis", True)
    run = st.button("Analyze", type="primary")
    st.caption(f"Model: {config.MODEL_REASON}")


def candle_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = add_indicators(df).tail(120)
    fig = go.Figure(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"],
                                   close=d["close"], name=ticker))
    fig.add_trace(go.Scatter(x=d.index, y=d["sma20"], name="SMA20", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=d.index, y=d["sma50"], name="SMA50", line=dict(width=1.2)))
    marks = d[d[["hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing"]].any(axis=1)]
    fig.add_trace(go.Scatter(x=marks.index, y=marks["high"] * 1.01, mode="markers", name="Pattern",
                             marker=dict(symbol="diamond", size=8)))
    fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def show_list(title, items):
    st.markdown(f"**{title}**")
    for i in items or ["(none)"]:
        st.markdown(f"- {i}")


tab_analysis, tab_news, tab_score, tab_backtest = st.tabs(
    ["Analysis", "News", "Paper-trading scorecard", "Backtest"])

if run:
    try:
        with st.spinner("Fetching data and reasoning..."):
            df = get_prices(ticker, "1y")
            tech = summarize(df)
            spy, vix = load_market_frames("1y")
            ctx = market_context(spy, vix)
            news_summary, raw_news = None, []
            if use_news:
                raw_news = get_news(ticker)
                news_summary = analyze_news(ticker, raw_news)
            decision = decide(ticker, horizon, tech, ctx, news_summary)
        st.session_state["result"] = dict(ticker=ticker, df=df, tech=tech, ctx=ctx, news=news_summary,
                                          raw_news=raw_news, decision=decision, horizon=horizon)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

res = st.session_state.get("result")

with tab_analysis:
    if not res:
        st.info("Pick a ticker in the sidebar and click **Analyze**.")
    else:
        d = res["decision"]
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(candle_chart(res["df"], res["ticker"]), use_container_width=True)
        with right:
            st.markdown(
                f"<div style='padding:16px;border-radius:12px;background:{COLORS[d.signal]};color:white'>"
                f"<div style='font-size:14px'>{res['ticker']} · {res['horizon']}-day horizon</div>"
                f"<div style='font-size:34px;font-weight:700'>{d.signal}</div>"
                f"<div>Confidence {d.confidence:.0%}</div></div>", unsafe_allow_html=True)
            if d.guardrail_note:
                st.warning(d.guardrail_note)
            st.write(d.summary)
            if st.button("Log as paper trade"):
                log_prediction(res["ticker"], d, res["tech"], res["horizon"])
                st.success("Logged. Resolve it later in the scorecard tab.")
        c1, c2 = st.columns(2)
        with c1:
            show_list("Supporting evidence", d.supporting_evidence)
            show_list("Conflicting signals", d.conflicts)
        with c2:
            show_list("Risks", d.risks)
            show_list("What would change my mind", d.what_would_change_my_mind)
        with st.expander("Raw technical + market inputs the model saw"):
            st.json({"technicals": res["tech"], "market": res["ctx"]})

with tab_news:
    if not res or res["news"] is None:
        st.info("Run an analysis with news enabled.")
    else:
        n = res["news"]
        st.write(f"Overall sentiment: **{n.overall_sentiment}** · news quality: **{n.news_quality}**")
        if n.items:
            st.dataframe(pd.DataFrame([i.model_dump() for i in n.items]), use_container_width=True)
        show_list("Key events", n.key_events)

with tab_score:
    if st.button("Resolve due predictions"):
        st.success(f"Resolved {resolve_due()} prediction(s).")
    preds = load_predictions()
    if preds.empty:
        st.info("No paper trades yet. Log one from the Analysis tab.")
    else:
        st.json(score(preds[preds["resolved"] == 1]))
        st.dataframe(preds.drop(columns=["payload"]), use_container_width=True)

with tab_backtest:
    st.write("Generate results with `python backtest.py`, then reload this page.")
    if os.path.exists("backtest_results.csv"):
        bt = pd.read_csv("backtest_results.csv")
        st.json(score(bt))
        st.dataframe(bt, use_container_width=True)
    else:
        st.info("No backtest_results.csv found yet.")
