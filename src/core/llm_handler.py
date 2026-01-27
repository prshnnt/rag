# src/core/llm_handler.py
from __future__ import annotations

from typing import List, Dict
from anthropic import Anthropic

from config.prompts import LEGAL_SYSTEM_PROMPT

class LegalLLMHandler:
    """Handles LLM interaction with strict legal guardrails."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
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
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.0,
            system=LEGAL_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"""Query: {query}

Legal Context:
{context}

Provide a factual legal position based ONLY on the context above."""
                }
            ]
        )
        
        return message.content[0].text

