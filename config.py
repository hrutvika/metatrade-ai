"""Central settings. Everything can be overridden from .env"""
import os

from dotenv import load_dotenv

load_dotenv()

# Meta Model API is OpenAI-SDK compatible. The OpenAI client does NOT read
# MODEL_API_KEY automatically, so we pass it explicitly (see llm.py).
BASE_URL = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
API_KEY = os.getenv("MODEL_API_KEY", "")

# Use the non-"contributor" model for anything private: the -contributor
# variants are cheaper but your data is used to improve Meta's products.
MODEL_NEWS = os.getenv("MODEL_NEWS", "muse-spark-1.3")
MODEL_REASON = os.getenv("MODEL_REASON", "muse-spark-1.3")

# Guardrail enforced in CODE, not just in the prompt.
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))

DEFAULT_HORIZON_DAYS = 5
DB_PATH = os.getenv("DB_PATH", "predictions.db")

DISCLAIMER = (
    "MetaTrade AI is an educational decision-support and paper-trading tool. "
    "It is not financial advice and does not place real trades."
)
