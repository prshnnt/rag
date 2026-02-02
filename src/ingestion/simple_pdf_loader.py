from pathlib import Path
from typing import Dict, List, Optional
import json
import pypdf
from loguru import logger

class SimplePDFLoader:
    """
    Simple PDF loader that extracts text from PDFs without complex chunking.
    Used for creating the base database.
    """
    
    def __init__(self, pdf_dir: Path, output_dir: Path):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_all_pdfs(self) -> Dict[str, dict]:
        """Load all PDFs from the directory."""
        documents = {}
        
        # files like "BNS.pdf", "Constitution.pdf", etc.
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}")
        
        for pdf_file in pdf_files:
            try:
                doc_data = self._process_pdf(pdf_file)
                if doc_data:
                    # Use filename stem (e.g., "BNS") as law code
                    law_code = pdf_file.stem
                    # Or verify against supported laws if needed
                    doc_data["law_code"] = law_code
                    documents[law_code] = doc_data
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                
        return documents
    
    def _process_pdf(self, pdf_path: Path) -> Optional[dict]:
        """Process a single PDF file."""
        logger.info(f"Processing {pdf_path.name}...")
        
        try:
            reader = pypdf.PdfReader(pdf_path)
            total_pages = len(reader.pages)
            full_text = ""
            pages_data = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                full_text += text + "\n\n"
                
                pages_data.append({
                    "page_number": i + 1,
                    "text": text
                })
            
            return {
                "filename": pdf_path.name,
                "total_pages": total_pages,
                "full_text": full_text.strip(),
                "metadata": {
                    "file_size_bytes": pdf_path.stat().st_size
                },
                "pages": pages_data
            }
            
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return None

    def save_to_json(self, documents: Dict[str, dict]):
        """Save all documents to a single JSON file."""
        output_file = self.output_dir / "all_documents.json"
        
        # Convert to list for JSON serialization if needed, or keep as dict
        data_to_save = {
            "meta": {
                "count": len(documents),
                "timestamp": "now" # TODO: add timestamp
            },
            "documents": documents
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
        logger.success(f"Saved all documents to {output_file}")
        
    def save_individual_files(self, documents: Dict[str, dict]):
        """Save each document to its own JSON file."""
        docs_dir = self.output_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        for law_code, doc in documents.items():
            output_file = docs_dir / f"{law_code}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)