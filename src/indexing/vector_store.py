# src/indexing/vector_store.py

from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from core.chunker import LegalChunk

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class VectorStore:
    """FAISS-based vector store for legal chunks."""
    
    def __init__(self, embedding_model: str):
        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
    
    def add_chunks(self, chunks: List[LegalChunk]):
        """Add legal chunks to vector index."""
        texts = []
        for chunk in chunks:
            # Create searchable text
            search_text = f"{chunk.title or ''} {chunk.text} {chunk.proviso or ''}"
            texts.append(search_text)
            self.metadata.append(chunk.model_dump())
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Add to FAISS
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic search."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(1 / (1 + dist))  # Convert distance to similarity
                results.append(result)
        
        return results
    
    def save(self, path: str):
        """Save index and metadata."""
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self, path: str):
        """Load index and metadata."""
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
