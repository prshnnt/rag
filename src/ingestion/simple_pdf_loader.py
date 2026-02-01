import json
from langchain.docstore.document import Document
from indexing.vector_store import VectorStore

def process_legal_json(json_file_path):
    # 1. Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []

    # 2. Iterate through each point in the Adhiniyam
    for item in data:
        # Create a self-contained string for embedding
        # This ensures the vector 'knows' the context even without metadata
        page_content = f"Act: {item['act']}\nChapter: {item['chapter']}\nSection: {item['section']}\nContent: {item['content']}"

        # Store structured metadata for precise filtering during RAG
        metadata = {
            "chapter": item['chapter'],
            "section": item['section'],
            "act": item['act']
        }

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    # 3. Store in ChromaDB using your updated vectorstore.py
    print(f"Loading {len(documents)} sections into ChromaDB...")
    vector_db = VectorStore()
    vector_db.add_chunks(documents=documents)
    print("Vector database created and persisted successfully.")

if __name__ == "__main__":
    process_legal_json("adhiniyam_data.json")