import yaml
# from core.llm.anthropic import AnthropicLLM
# from core.llm.openai import OpenAILLM
from core.llm.gemini import GeminiLLM
from core.llm.local import LocalLLM
from core.llm.groq import GroqLLM

def load_llm(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    provider = cfg["provider"]

    if provider == "groq":
        return GroqLLM(**cfg["groq"])

    # if provider == "anthropic":
    #     return AnthropicLLM(**cfg["anthropic"])

    # if provider == "openai":
    #     return OpenAILLM(**cfg["openai"])

    if provider == "gemini":
        return GeminiLLM(**cfg["gemini"])

    if provider == "local":
        return LocalLLM(**cfg["local"])

    raise ValueError(f"Unsupported LLM provider: {provider}")
