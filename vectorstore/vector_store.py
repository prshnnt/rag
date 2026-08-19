import chromadb
import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from typing import List
from dotenv import load_dotenv
load_dotenv()

def _chroma(path: str | Path | None = None):
        db_path = path or os.getenv("CHROMA_PATH", "./chroma")
        return chromadb.PersistentClient(path=str(db_path))

def _embeddings() -> OllamaEmbeddings:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(model=model,base_url=base)

class VectorStore:
    def __init__(self,path: str | Path | None = None):
            self.client = _chroma(path)
            self.embedding_model = _embeddings()

    def get_collection(self,collection_name)-> chromadb.Collection:
        return self.client.get_or_create_collection(collection_name,metadata={"hnsw:space": "cosine"})
          
    def search(self,collection_name,query,k=5,where=None):
        vec = self.embedding_model.embed_query(query)
        collection = self.get_collection(collection_name)
        result = collection.query(
              query_embeddings=[vec],
              n_results=k,
              where=where
        )
        return result
    
    def ingest(
            self,
            collection_name,
            ids:List[str],
            content:List[str],
            metadatas:List[dict]
            ):
        collection = self.get_collection(collection_name)
        embeddings = self.embedding_model.embed_documents(content)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
            )