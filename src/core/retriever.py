
# src/core/retriever.py

from __future__ import annotations
from typing import List, Dict
from indexing.vector_store import VectorStore
from indexing.keyword_index import KeywordIndex


class HybridRetriever:
    """Hybrid retrieval combining vector and keyword search."""
    
    def __init__(self, vector_store: VectorStore, keyword_index: KeywordIndex, 
                 vector_weight: float = 0.6, keyword_weight: float = 0.4):
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Hybrid retrieval with weighted merging."""
        
        # Get vector results
        vector_results = self.vector_store.search(query, top_k=top_k)
        
        # Get keyword results
        keyword_results = self.keyword_index.search(query, top_k=top_k)
        
        # Merge and reweight
        merged = self._merge_results(vector_results, keyword_results)
        
        # Sort by final score
        merged.sort(key=lambda x: x['final_score'], reverse=True)
        
        return merged[:top_k]
    
    def _merge_results(self, vector_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """Merge and reweight results from both indices."""
        results_map = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result['chunk_id']
            results_map[chunk_id] = result.copy()
            results_map[chunk_id]['vector_score'] = result['score']
            results_map[chunk_id]['keyword_score'] = 0.0
        
        # Add/update keyword results
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id in results_map:
                results_map[chunk_id]['keyword_score'] = result['score']
            else:
                results_map[chunk_id] = result.copy()
                results_map[chunk_id]['vector_score'] = 0.0
                results_map[chunk_id]['keyword_score'] = result['score']
        
        # Calculate final scores
        for chunk_id in results_map:
            v_score = results_map[chunk_id].get('vector_score', 0)
            k_score = results_map[chunk_id].get('keyword_score', 0)
            results_map[chunk_id]['final_score'] = (
                self.vector_weight * v_score + self.keyword_weight * k_score
            )
        
        return list(results_map.values())
