"""
LLM client wrapper.

Wraps OpenAI calls with:
  - Retry logic (exponential backoff)
  - Structured JSON output requests
  - Token tracking
  - Timeout enforcement
  - Prompt safety (anti-hallucination system prompt injected always)

The ANTI_FABRICATION_SYSTEM_PROMPT is prepended to every extraction call.
It instructs the LLM to return MISSING rather than guess.
"""
from __future__ import annotations
import json
import time
import asyncio
from typing import Any, Optional

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

from app.configs.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# This prompt is the backbone of the no-fabrication guarantee.
# It is injected into every extraction call.
ANTI_FABRICATION_SYSTEM_PROMPT = """You are a clinical documentation assistant helping extract information from patient source notes.

CRITICAL SAFETY RULES — you MUST follow these without exception:
1. NEVER invent, infer, or guess any clinical fact.
2. NEVER fabricate diagnoses, medications, dates, lab values, or patient details.
3. If information is not explicitly stated in the provided text, return null or the string "MISSING".
4. If information appears to be pending (e.g., "awaiting results", "pending"), return "PENDING".
5. If two sources contradict each other, return "CONFLICT: <brief description>" — do NOT pick one.
6. Extract ONLY what is directly stated. Do not paraphrase in a way that changes clinical meaning.
7. Return ONLY valid JSON matching the schema requested. No prose, no markdown fences.
8. For every fact you extract, you MUST include the exact supporting text from the source.

You are producing a DRAFT for clinician review. Accuracy and explicit uncertainty are more
important than completeness. A missing field that is flagged is far safer than a fabricated one."""


class LLMClient:
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_model
        self.total_tokens_used = 0

    async def extract_structured(
        self,
        user_prompt: str,
        system_extra: str = "",
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> Optional[dict[str, Any]]:
        """
        Call the LLM with anti-fabrication system prompt and return parsed JSON.
        Returns None if all retries fail.
        """
        system = ANTI_FABRICATION_SYSTEM_PROMPT
        if system_extra:
            system += f"\n\nAdditional context:\n{system_extra}"

        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.settings.openai_temperature,
                        max_tokens=self.settings.openai_max_tokens,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                    ),
                    timeout=timeout,
                )

                usage = response.usage
                if usage:
                    self.total_tokens_used += usage.total_tokens

                raw = response.choices[0].message.content
                if not raw:
                    logger.warning("llm_empty_response", attempt=attempt)
                    continue

                return json.loads(raw)

            except (RateLimitError, APITimeoutError) as e:
                wait = 2 ** attempt
                logger.warning("llm_rate_limit_or_timeout", attempt=attempt, wait=wait, error=str(e))
                await asyncio.sleep(wait)

            except APIError as e:
                logger.error("llm_api_error", attempt=attempt, error=str(e))
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(1)

            except json.JSONDecodeError as e:
                logger.error("llm_json_parse_error", attempt=attempt, error=str(e))
                if attempt == max_retries - 1:
                    return None

            except asyncio.TimeoutError:
                logger.warning("llm_timeout", attempt=attempt)
                if attempt == max_retries - 1:
                    return None

        return None

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> Optional[str]:
        """General-purpose chat for planning and reasoning steps."""
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        temperature=0.0,
                        max_tokens=1024,
                        messages=messages,
                    ),
                    timeout=timeout,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning("llm_chat_error", attempt=attempt, error=str(e))
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return None
