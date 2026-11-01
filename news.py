"""Turn raw headlines into structured, scored signals (one LLM call for all items)."""
import json

import config
from llm import chat_json
from schemas import NewsSummary

SYSTEM = """You are a financial news analyst. For each headline decide the sentiment for the
STOCK (not the world), the event type, and how material it is to the stock price (1-5).
Rules:
- Be skeptical: most headlines are noise (materiality 1-2). Reserve 4-5 for earnings, guidance,
  major legal/regulatory events, M&A, or large product news.
- Do not invent facts beyond the headline and summary you are given.
- If there is nothing relevant, return an empty items list and news_quality 'none'."""


def analyze_news(ticker: str, items: list, model: str = None) -> NewsSummary:
    if not items:
        return NewsSummary(items=[], overall_sentiment="neutral", key_events=[], news_quality="none")
    payload = [{"headline": i["title"], "summary": i["summary"][:400], "published": str(i["published"])}
               for i in items]
    user = f"Ticker: {ticker}\nHeadlines (JSON):\n{json.dumps(payload, indent=2)}"
    return chat_json(SYSTEM, user, NewsSummary, model or config.MODEL_NEWS)
