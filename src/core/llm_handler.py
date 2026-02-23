# src/core/llm_handler.py
from __future__ import annotations

from typing import List, Dict
from anthropic import Anthropic

from config.prompts import LEGAL_SYSTEM_PROMPT

class LegalLLMHandler:
    """Handles LLM interaction with strict legal guardrails."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.api_key = api_key
        
        # Initialize appropriate client
        if api_key.startswith("gsk_"):
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                self.provider = "groq"
                # If model name seems wrong for Groq, maybe default to a safe one
                # but for now we trust the user or fallback later
                if "openai" in model: # Handle the weird "openai/gpt-oss..." if intended for Groq
                     # It's better to log a warning, but we must use a valid model name if possible
                     # For now, we'll try to use the model as is, but if it fails, maybe we should catch it.
                     pass
            except ImportError:
                raise ImportError("Groq client not installed. Run `pip install groq`")
        else:
            self.client = Anthropic(api_key=api_key)
            self.provider = "anthropic"
    
    def build_context(self, chunks: List[Dict], max_tokens: int = 8000) -> str:
        """Build citation-first legal context."""
        context_parts = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_text = self._format_chunk(chunk)
            chunk_tokens = len(chunk_text.split()) * 1.3  # Rough token estimate
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_chunk(self, chunk: Dict) -> str:
        """Format chunk with full citation."""
        parts = [
            f"**{chunk['law_name']}**",
            f"{chunk['identifier_type']} {chunk['identifier_number']}" + 
            (f" - {chunk['title']}" if chunk.get('title') else ""),
            f"\n{chunk['text']}",
        ]
        
        if chunk.get('proviso'):
            parts.append(f"\nProviso: {chunk['proviso']}")
        
        if chunk.get('explanation'):
            parts.append(f"\nExplanation: {chunk['explanation']}")
        
        parts.append(f"\nSource: {chunk['source_url']}")
        
        return "\n".join(parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer with strict legal guardrails."""
        
        messages = [
            {
                "role": "system",
                "content": LEGAL_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""Query: {query}

Legal Context:
{context}

Provide a factual legal position based ONLY on the context above."""
            }
        ]
        
        if self.provider == "anthropic":
            # Anthropic uses system param separately
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.0,
                system=LEGAL_SYSTEM_PROMPT,
                messages=[messages[1]] # Only user message
            )
            return message.content[0].text
            
        elif self.provider == "groq":
            # Groq (OpenAI compatible)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        
        return "Error: Unsupported provider"

