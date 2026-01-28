"""
Simple PDF loader that extracts text and stores in database.
NO chunking logic - just raw text extraction for prototyping.
"""

import sys
from pathlib import Path
import json
from typing import Dict, List
from datetime import datetime
import pypdf
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/pdf_loader_{time}.log", rotation="10 MB")


class SimplePDFLoader:
    """
    Simple PDF loader - extracts full text from PDFs.
    No chunking, no complex parsing - just load and store.
    """
    
    # PDF filename mapping
    PDF_FILES = {
        "constitution": "constitution_of_india.pdf",
        "bns": "bharatiya_nyaya_sanhita_2023.pdf",
        # "bnss": "bharatiya_nagarik_suraksha_sanhita_2023.pdf",
        # "ipc": "indian_penal_code_1860.pdf",
        # "crpc": "code_criminal_procedure_1973.pdf",
        # "cpc": "code_civil_procedure_1908.pdf",
        # "it_act": "information_technology_act_2000.pdf",
    }
    
    def __init__(self, pdf_dir: Path, output_dir: Path):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_pdfs(self) -> Dict[str, dict]:
        """
        Load all available PDFs and return as dictionary.
        
        Returns:
            Dictionary with law_code as key and document data as value
        """
        documents = {}
        
        logger.info(f"Loading PDFs from: {self.pdf_dir}")
        
        for law_code, filename in self.PDF_FILES.items():
            pdf_path = self.pdf_dir / filename
            
            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                continue
            
            try:
                logger.info(f"Loading {law_code}: {filename}")
                doc_data = self.load_single_pdf(pdf_path, law_code)
                documents[law_code] = doc_data
                logger.success(f"✓ Loaded {law_code}: {doc_data['total_pages']} pages")
                
            except Exception as e:
                logger.error(f"✗ Failed to load {law_code}: {e}")
        
        logger.info(f"Successfully loaded {len(documents)}/{len(self.PDF_FILES)} PDFs")
        return documents
    
    def load_single_pdf(self, pdf_path: Path, law_code: str) -> dict:
        """
        Load a single PDF and extract text.
        
        Returns:
            Dictionary with document metadata and text
        """
        reader = pypdf.PdfReader(str(pdf_path))
        
        # Extract text from all pages
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                pages_text.append({
                    "page_number": i + 1,
                    "text": text
                })
            except Exception as e:
                logger.warning(f"Failed to extract page {i+1}: {e}")
                pages_text.append({
                    "page_number": i + 1,
                    "text": ""
                })
        
        # Combine all text
        full_text = "\n\n".join([p["text"] for p in pages_text])
        
        # Create document dictionary
        doc_data = {
            "law_code": law_code,
            "filename": pdf_path.name,
            "total_pages": len(pages_text),
            "full_text": full_text,
            "pages": pages_text,
            "metadata": {
                "file_size_bytes": pdf_path.stat().st_size,
                "loaded_at": datetime.now().isoformat(),
            }
        }
        
        return doc_data
    
    def save_to_json(self, documents: Dict[str, dict], filename: str = "legal_documents.json"):
        """
        Save all documents to a single JSON file (simple database).
        
        Args:
            documents: Dictionary of document data
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving to JSON database: {output_path}")
        
        # Prepare data for JSON
        db_data = {
            "total_documents": len(documents),
            "created_at": datetime.now().isoformat(),
            "documents": documents
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, indent=2, ensure_ascii=False)
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.success(f"✓ Saved database: {output_path} ({file_size_mb:.2f} MB)")
        
        return output_path
    
    def save_individual_files(self, documents: Dict[str, dict]):
        """
        Save each document as a separate JSON file.
        
        Args:
            documents: Dictionary of document data
        """
        logger.info(f"Saving individual document files...")
        
        for law_code, doc_data in documents.items():
            output_path = self.output_dir / f"{law_code}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Saved: {output_path}")
        
        logger.success(f"✓ Saved {len(documents)} individual files")

