import json
import uuid
from core.chunker import LegalChunk
from indexing.vector_store import VectorStore

def process_legal_json(json_file_path):
    # 1. Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = []

    # 2. Convert JSON items to LegalChunk objects
    for item in data:
        chunk = LegalChunk(
            law_name=item['act'],
            chapter_title=item['chapter'],
            identifier_number=str(item['section']),
            text=item['content'],
            chunk_id=str(uuid.uuid4())
        )
        if chunk.validate_completeness():
            chunks.append(chunk)

    # 3. Store in ChromaDB
    print(f"Loading {len(chunks)} chunks into ChromaDB...")
    vdb = VectorStore(persist_directory="./legal_chroma_db")
    vdb.add_chunks(chunks)
    print("Database updated successfully.")

if __name__ == "__main__":
    # Ensure you have your 'adhiniyam_data.json' ready
    process_legal_json("adhiniyam_data.json")