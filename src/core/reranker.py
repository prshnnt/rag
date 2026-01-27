
# src/core/reranker.py
from __future__ import annotations

from typing import List, Dict

from sentence_transformers import CrossEncoder

class LegalReranker:
    """Cross-encoder reranker for legal relevance."""
    
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank candidates using cross-encoder."""
        
        # Prepare pairs
        pairs = []
        for candidate in candidates:
            text = f"{candidate.get('title', '')} {candidate['text']}"
            pairs.append([query, text])
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Attach scores
        for candidate, score in zip(candidates, scores):
            candidate['rerank_score'] = float(score)
        
        # Sort and return top-k
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        return candidates[:top_k]