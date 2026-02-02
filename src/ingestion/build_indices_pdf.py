from pathlib import Path
import sys
from typing import List, Optional
import uuid
from tqdm import tqdm
from loguru import logger

# Add src to pythonpath
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from ingestion.simple_pdf_loader import SimplePDFLoader
from indexing.vector_store import VectorStore
from indexing.keyword_index import KeywordIndex
from core.chunker import LegalChunk

class IndexBuilder:
    """Builds FAISS indices from PDF Documents."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vector_store = VectorStore(embedding_model=settings.embedding_model)
        self.keyword_index = KeywordIndex()
        self.pdf_dir = Path("data/pdfs") # Default, could be configurable
        self.index_dir = Path(settings.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build_all(self, laws: Optional[List[str]] = None):
        """
        Build indices for specified laws or all available PDFs.
        
        Args:
            laws: List of law codes (filenames without extension) to process.
                  If None, process all found PDFs.
        """
        logger.info("Initializing PDF Loader...")
        # Create a temporary output dir for the loader if needed, or use processed data dir
        loader_output = Path(self.settings.processed_data_dir)
        loader = SimplePDFLoader(self.pdf_dir, loader_output)
        
        logger.info("Loading PDFs...")
        documents = loader.load_all_pdfs()
        
        if not documents:
            logger.warning("No PDF documents found to index.")
            return

        all_chunks = []
        
        logger.info(f"Processing {len(documents)} documents...")
        for law_code, doc_data in documents.items():
            if laws and law_code not in laws:
                continue
                
            logger.info(f"Chunking {law_code}...")
            chunks = self._create_chunks(doc_data)
            all_chunks.extend(chunks)
            logger.info(f"Generated {len(chunks)} chunks for {law_code}")

        if not all_chunks:
            logger.warning("No chunks generated.")
            return

        logger.info(f"Building vector index with {len(all_chunks)} total chunks...")
        self.vector_store.add_chunks(all_chunks)
        self.keyword_index.add_chunks(all_chunks)
        
        logger.info(f"Saving index to {self.index_dir}...")
        self.vector_store.save(str(self.index_dir))
        self.keyword_index.save(str(self.index_dir))
        logger.success("Index build complete!")

    def _create_chunks(self, doc_data: dict) -> List[LegalChunk]:
        """Convert document pages into LegalChunk objects."""
        chunks = []
        law_code = doc_data.get("law_code", "UNKNOWN")
        filename = doc_data.get("filename", "unknown.pdf")
        
        # Simple chunking by page for now
        # In a real system, we'd want more sophisticated text splitting
        for page in doc_data.get("pages", []):
            page_text = page.get("text", "").strip()
            if not page_text:
                continue
                
            # Create a chunk for the page
            chunk = LegalChunk(
                law_code=law_code,
                law_name=filename, # Use filename as law name for now
                identifier_type="Page",
                identifier_number=str(page.get("page_number")),
                text=page_text,
                chunk_id=str(uuid.uuid4()),
                page_number=page.get("page_number"),
                metadata={
                    "filename": filename,
                    "total_pages": doc_data.get("total_pages")
                }
            )
            chunks.append(chunk)
            
        return chunks

if __name__ == "__main__":
    # Support running this file directly
    settings = Settings()
    builder = IndexBuilder(settings)
    builder.build_all()
