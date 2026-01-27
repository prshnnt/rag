import google.generativeai as genai
from core.llm.base import BaseLLM
import os

class GeminiLLM(BaseLLM):
    def __init__(self, api_key_env: str, model: str, max_tokens: int):
        genai.configure(api_key=os.getenv(api_key_env))
        self.model = genai.GenerativeModel(model)
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str) -> str:
        response = self.model.generate_content(
            f"{system_prompt}\n\n{prompt}",
            generation_config={"temperature": 0.0},
        )
        return response.text
