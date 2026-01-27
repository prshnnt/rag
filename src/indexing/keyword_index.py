
# src/indexing/keyword_index.py
from __future__ import annotations
from rank_bm25 import BM25Okapi
from typing import List, Dict, TYPE_CHECKING
import pickle
import numpy as np


if TYPE_CHECKING:
    from core.chunker import LegalChunk

class KeywordIndex:
    """BM25-based keyword index for exact legal term matching."""
    
    def __init__(self):
        self.bm25 = None
        self.metadata = []
        self.corpus_tokens = []
    
    def add_chunks(self, chunks: List[LegalChunk]):
        """Add chunks to keyword index."""
        corpus = []
        for chunk in chunks:
            # Create searchable text with legal keywords
            text = f"{chunk.identifier_type} {chunk.identifier_number} {chunk.title or ''} {chunk.text}"
            tokens = text.lower().split()
            corpus.append(tokens)
            self.corpus_tokens.append(tokens)
            self.metadata.append(chunk.dict())
        
        self.bm25 = BM25Okapi(corpus)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Keyword-based search."""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(scores[idx])
                results.append(result)
        
        return results
    
    def save(self, path: str):
        with open(f"{path}/bm25.pkl", 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'metadata': self.metadata,
                'corpus_tokens': self.corpus_tokens
            }, f)
    
    def load(self, path: str):
        with open(f"{path}/bm25.pkl", 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.metadata = data['metadata']
            self.corpus_tokens = data['corpus_tokens']

