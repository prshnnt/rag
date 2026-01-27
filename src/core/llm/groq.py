from __future__ import annotations

import os
from openai import OpenAI
from core.llm.base import BaseLLM


class GroqLLM(BaseLLM):
    def __init__(self, api_key_env: str, model: str, max_tokens: int):
        self.client = OpenAI(
            api_key=os.getenv(api_key_env),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
