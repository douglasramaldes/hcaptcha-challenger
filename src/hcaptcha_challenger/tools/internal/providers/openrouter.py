# -*- coding: utf-8 -*-
"""
OpenRouterProvider - OpenRouter API implementation via OpenAI-compatible interface.

Supports any vision model on https://openrouter.ai (free and paid).
Images are sent as base64 data URIs since OpenRouter has no Files API.

Recommended models for hCaptcha:
  - qwen/qwen3.6-plus:free          (free, good vision)
  - google/gemma-4-26b-a4b-it       ($0.13/M tokens, strong spatial)
  - google/gemini-flash-1.5          (via OR, same model as direct Gemini)
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import List, Type, TypeVar

from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

ResponseT = TypeVar("ResponseT", bound=BaseModel)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _extract_first_json_block(text: str) -> dict | None:
    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


class OpenRouterProvider:
    """
    OpenRouter-based chat provider.

    Uses the OpenAI-compatible API at openrouter.ai.
    Images are base64-encoded inline (no upload step needed).
    """

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model
        self._client = None
        self._last_content: str | None = None

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=_OPENROUTER_BASE_URL,
                api_key=self._api_key,
            )
        return self._client

    def _encode_image(self, path: Path) -> str:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode()
        suffix = path.suffix.lower().lstrip(".")
        mime = "image/png" if suffix == "png" else f"image/{suffix or 'jpeg'}"
        return f"data:{mime};base64,{b64}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"OpenRouter retry ({retry_state.attempt_number}/3) - "
            f"{retry_state.outcome.exception()}"
        ),
    )
    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        content: list[dict] = []

        valid_images = [Path(p) for p in images if p and Path(p).exists()]
        for img_path in valid_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(img_path)},
                }
            )

        if user_prompt:
            content.append({"type": "text", "text": user_prompt})

        messages = []
        if description:
            messages.append({"role": "system", "content": description})
        messages.append({"role": "user", "content": content})

        schema = response_schema.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": False, "schema": schema},
        }

        try:
            resp = await self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format=response_format,
            )
        except Exception:
            # Fallback: some models don't support json_schema — use json_object
            resp = await self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
            )

        if not resp.choices:
            raise ValueError(
                f"OpenRouter returned no choices (model rate-limited or unavailable): "
                f"model={self._model}, id={getattr(resp, 'id', '?')}"
            )

        first = resp.choices[0]
        msg = getattr(first, "message", None)
        self._last_content = (getattr(msg, "content", None) or "") if msg else ""

        try:
            return response_schema.model_validate_json(self._last_content)
        except Exception:
            json_data = _extract_first_json_block(self._last_content)
            if isinstance(json_data, list) and len(json_data) == 1:
                json_data = json_data[0]
            if isinstance(json_data, dict):
                return response_schema(**json_data)
            raise ValueError(f"OpenRouter: failed to parse response: {self._last_content[:300]}")

    def cache_response(self, path: Path) -> None:
        if not self._last_content:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"raw": self._last_content}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"OpenRouter: failed to cache response: {e}")
