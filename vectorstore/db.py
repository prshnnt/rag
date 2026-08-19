import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB = "rag_db"
DEFAULT_COLLECTION = "document_chunks"


def mongo_uri() -> str:
    return os.getenv("MONGO_URI", "mongodb://localhost:27017/")


_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Module-level singleton MongoClient. Reuses connection across calls."""
    global _client
    if _client is None:
        _client = MongoClient(
            mongo_uri(),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
    return _client


def get_chunks_collection():
    return get_client()[DEFAULT_DB][DEFAULT_COLLECTION]
