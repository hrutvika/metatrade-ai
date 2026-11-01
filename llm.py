"""Thin wrapper around the Meta Model API (OpenAI-compatible).

chat_json() asks for JSON, validates it with Pydantic, and retries with the
validation error fed back to the model if the output is malformed.
"""
import json
import re
from typing import Optional, Type, TypeVar

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError

import config

T = TypeVar("T", bound=BaseModel)

_client: Optional[OpenAI] = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not config.API_KEY:
            raise RuntimeError("MODEL_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
    return _client


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def _complete(model: str, messages: list, temperature: Optional[float]) -> str:
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        resp = client().chat.completions.create(**kwargs)
    except BadRequestError:
        # Some models reject sampling params; retry without them.
        kwargs.pop("temperature", None)
        resp = client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def chat_json(system: str, user: str, schema: Type[T], model: str, retries: int = 2,
              temperature: Optional[float] = 0.2) -> T:
    schema_text = json.dumps(schema.model_json_schema())
    messages = [
        {"role": "system", "content": f"{system}\n\nReturn ONLY one JSON object (no prose, no markdown) "
                                      f"that matches this JSON schema:\n{schema_text}"},
        {"role": "user", "content": user},
    ]
    last_error = None
    for _ in range(retries + 1):
        raw = _complete(model, messages, temperature)
        try:
            return schema.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as e:  # json errors are ValueError subclasses
            last_error = e
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"That was invalid: {e}. "
                                                        "Return ONLY the corrected JSON object."})
    raise RuntimeError(f"Model failed to return valid JSON after {retries + 1} tries: {last_error}")
