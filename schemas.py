"""Pydantic schemas = the contract between the LLM and the rest of the code."""
from typing import List, Literal

from pydantic import BaseModel, Field

Signal = Literal["BUY", "HOLD", "SELL", "NO TRADE"]


class NewsItemAnalysis(BaseModel):
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    event_type: str = Field(description="e.g. earnings, guidance, lawsuit, product, macro, analyst, other")
    materiality: int = Field(ge=1, le=5, description="1 = noise, 5 = likely to move the stock")
    confidence: float = Field(ge=0, le=1)


class NewsSummary(BaseModel):
    items: List[NewsItemAnalysis]
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    key_events: List[str]
    news_quality: Literal["rich", "thin", "none"] = Field(
        description="How much useful news there was. 'none' if there was nothing relevant."
    )


class Decision(BaseModel):
    signal: Signal
    confidence: float = Field(ge=0, le=1)
    summary: str
    supporting_evidence: List[str]
    conflicts: List[str] = Field(description="Signals that disagree with each other")
    risks: List[str]
    what_would_change_my_mind: List[str]
    guardrail_note: str = ""  # filled by code if a guardrail overrides the model
