
# ----------------------------------------------------------------------------
# FILE 3: src/ingestion/build_indices_pdf.py
# ----------------------------------------------------------------------------
"""
Index builder for PDF-based legal documents.
Replaces web scraping with PDF parsing.
"""

import sys
from pathlib import Path
import json
from typing import List, Dict, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import Settings
from config.legal_guardrails import SUPPORTED_LAWS
from core.chunker import LegalChunk
from indexing.vector_store import VectorStore
from indexing.keyword_index import KeywordIndex
from ingestion.pdf_parser import LegalPDFParser
from ingestion.pdf_chunker import PDFLegalChunker
from ingestion.validators import ContentValidator


# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/build_indices_pdf_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="DEBUG"
)


class PDFIndexBuilder:
    """Index builder for PDF-based legal documents."""
    
    # PDF filename mapping
    PDF_FILENAMES = {
        "constitution": "constitution_of_india.pdf",
        "bns": "bharatiya_nyaya_sanhita_2023.pdf",
        # "bnss": "bharatiya_nagarik_suraksha_sanhita_2023.pdf",
        # "ipc": "indian_penal_code_1860.pdf",
        # "crpc": "code_criminal_procedure_1973.pdf",
        # "cpc": "code_civil_procedure_1908.pdf",
        # "it_act": "information_technology_act_2000.pdf",
    }
    
    def __init__(self, settings: Settings, pdf_dir: Path):
        self.settings = settings
        self.pdf_dir = pdf_dir
        self.parser = LegalPDFParser()
        self.chunker = PDFLegalChunker()
        self.validator = ContentValidator()
        
        # Create directories
        Path(settings.raw_data_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.processed_data_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.index_dir).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def build_all(self, laws: Optional[List[str]] = None):
        """Build indices for all or specified laws."""
        
        if laws is None:
            laws = [code for code, info in SUPPORTED_LAWS.items() if info.get("active")]
        
        logger.info(f"Building indices from PDFs for {len(laws)} laws")
        logger.info(f"PDF directory: {self.pdf_dir}")
        
        all_chunks = []
        stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "failed_documents": 0,
            "laws": {}
        }
        
        # Process each law
        for law_code in laws:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing: {SUPPORTED_LAWS[law_code]['name']}")
                logger.info(f"{'='*60}")
                
                chunks = self.process_pdf(law_code)
                
                if chunks:
                    all_chunks.extend(chunks)
                    stats["laws"][law_code] = {
                        "chunks": len(chunks),
                        "status": "success"
                    }
                    stats["total_chunks"] += len(chunks)
                    stats["total_documents"] += 1
                    
                    logger.success(f"✓ {law_code}: {len(chunks)} chunks created")
                else:
                    stats["laws"][law_code] = {
                        "chunks": 0,
                        "status": "failed"
                    }
                    stats["failed_documents"] += 1
                    logger.error(f"✗ {law_code}: No chunks created")
                
            except Exception as e:
                logger.error(f"✗ {law_code} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                stats["laws"][law_code] = {
                    "chunks": 0,
                    "status": "error",
                    "error": str(e)
                }
                stats["failed_documents"] += 1
        
        # Build indices if we have chunks
        if all_chunks:
            logger.info(f"\n{'='*60}")
            logger.info("Building vector and keyword indices...")
            logger.info(f"{'='*60}")
            
            self.build_indices(all_chunks)
            
            # Save stats
            self.save_stats(stats)
            
            logger.success(f"\n{'='*60}")
            logger.success("INDEX BUILD COMPLETE")
            logger.success(f"{'='*60}")
            logger.success(f"Total documents: {stats['total_documents']}")
            logger.success(f"Total chunks: {stats['total_chunks']}")
            logger.success(f"Failed documents: {stats['failed_documents']}")
        else:
            logger.error("No chunks created. Index build failed.")
    
    def process_pdf(self, law_code: str) -> List[LegalChunk]:
        """Process a single PDF: parse, chunk, validate."""
        
        # Get PDF filename
        pdf_filename = self.PDF_FILENAMES.get(law_code)
        if not pdf_filename:
            logger.error(f"No PDF filename mapping for {law_code}")
            return []
        
        pdf_path = self.pdf_dir / pdf_filename
        
        # Check if PDF exists
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            logger.info(f"Please place the PDF file at: {pdf_path}")
            return []
        
        # Step 1: Parse PDF
        logger.info(f"Step 1/4: Parsing PDF {pdf_filename}...")
        sections = self.parser.parse_pdf(pdf_path, law_code)
        
        if not sections:
            logger.error(f"Failed to parse PDF: {pdf_path}")
            return []
        
        logger.info(f"✓ Parsed {len(sections)} sections from PDF")
        
        # Save raw parsed data
        raw_path = Path(self.settings.raw_data_dir) / f"{law_code}_parsed.json"
        raw_data = [
            {
                "page": s.page_number,
                "type": s.identifier_type,
                "number": s.identifier_number,
                "title": s.title,
                "text": s.text
            }
            for s in sections
        ]
        raw_path.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False), encoding='utf-8')
        logger.info(f"✓ Saved raw parsed data: {raw_path}")
        
        # Step 2: Chunk
        logger.info(f"Step 2/4: Creating legal chunks...")
        chunks = self.chunker.chunk_from_pdf_sections(sections, law_code)
        logger.info(f"✓ Created {len(chunks)} chunks")
        
        # Step 3: Validate
        logger.info(f"Step 3/4: Validating chunks...")
        valid_chunks = []
        invalid_count = 0
        
        for chunk in chunks:
            if self.validator.validate_chunk(chunk):
                valid_chunks.append(chunk)
            else:
                invalid_count += 1
                logger.warning(f"Invalid chunk: {chunk.chunk_id}")
        
        logger.info(f"✓ Valid chunks: {len(valid_chunks)}, Invalid: {invalid_count}")
        
        # Step 4: Save processed chunks
        logger.info(f"Step 4/4: Saving processed chunks...")
        processed_path = Path(self.settings.processed_data_dir) / f"{law_code}.json"
        
        chunks_dict = [chunk.dict() for chunk in valid_chunks]
        processed_path.write_text(
            json.dumps(chunks_dict, indent=2, ensure_ascii=False), 
            encoding='utf-8'
        )
        logger.info(f"✓ Saved processed chunks: {processed_path}")
        
        return valid_chunks
    
    def build_indices(self, chunks: List[LegalChunk]):
        """Build vector and keyword indices."""
        
        # Initialize stores
        logger.info("Initializing vector store...")
        vector_store = VectorStore(self.settings.embedding_model)
        
        logger.info("Initializing keyword index...")
        keyword_index = KeywordIndex()
        
        # Add chunks
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        vector_store.add_chunks(chunks)
        
        logger.info(f"Adding {len(chunks)} chunks to keyword index...")
        keyword_index.add_chunks(chunks)
        
        # Save indices
        logger.info("Saving indices...")
        index_dir = Path(self.settings.index_dir)
        
        vector_store.save(str(index_dir))
        logger.success(f"✓ Vector index saved: {index_dir}/faiss.index")
        
        keyword_index.save(str(index_dir))
        logger.success(f"✓ Keyword index saved: {index_dir}/bm25.pkl")
    
    def save_stats(self, stats: Dict):
        """Save build statistics."""
        stats["build_time"] = datetime.now().isoformat()
        stats["source_type"] = "pdf"
        stats_path = Path(self.settings.index_dir) / "build_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        logger.info(f"✓ Stats saved: {stats_path}")


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build indices from PDF legal documents"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
        help="Directory containing PDF files (default: data/pdfs)"
    )
    parser.add_argument(
        "--laws",
        nargs="+",
        choices=list(SUPPORTED_LAWS.keys()),
        help="Specific laws to process (default: all active laws)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild indices even if they exist"
    )
    parser.add_argument(
        "--list-pdfs",
        action="store_true",
        help="List expected PDF filenames and exit"
    )
    
    args = parser.parse_args()
    
    # List PDFs mode
    if args.list_pdfs:
        print("\nExpected PDF files in directory:", args.pdf_dir)
        print("-" * 60)
        for law_code, filename in PDFIndexBuilder.PDF_FILENAMES.items():
            law_name = SUPPORTED_LAWS[law_code]["name"]
            print(f"{law_code:12} -> {filename}")
            print(f"              ({law_name})")
        print("-" * 60)
        return
    
    # Load settings
    settings = Settings()
    
    # Check PDF directory
    if not args.pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {args.pdf_dir}")
        logger.info(f"Creating directory: {args.pdf_dir}")
        args.pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Please place PDF files in: {args.pdf_dir}")
        logger.info("Run with --list-pdfs to see expected filenames")
        return
    
    # Check if indices already exist
    index_dir = Path(settings.index_dir)
    if (index_dir / "faiss.index").exists() and not args.rebuild:
        logger.warning("Indices already exist. Use --rebuild to force rebuild.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            return
    
    # Build indices
    builder = PDFIndexBuilder(settings, args.pdf_dir)
    
    try:
        builder.build_all(laws=args.laws)
        logger.success("\n✓ Index build completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("\n\nBuild interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n\n✗ Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
