
# ----------------------------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------------------------

from pathlib import Path
from loguru import logger

from ingestion.load_to_database import LegalDocumentDB
from ingestion.simple_pdf_loader import SimplePDFLoader


def main():
    """Main entry point for loading PDFs to database."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load PDFs to database (no chunking)"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/database"),
        help="Output directory for database"
    )
    parser.add_argument(
        "--format",
        choices=["json", "sqlite", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List documents in database and exit"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # List mode
    if args.list:
        db_path = args.output_dir / "legal_docs.db"
        if not db_path.exists():
            print("Database not found. Run without --list to create it.")
            return
        
        db = LegalDocumentDB(db_path)
        docs = db.list_documents()
        
        print("\n" + "="*60)
        print("Documents in Database:")
        print("="*60)
        for doc in docs:
            print(f"{doc['law_code']:12} | {doc['total_pages']:3} pages | {doc['filename']}")
        print("="*60)
        
        db.close()
        return
    
    # Load PDFs
    loader = SimplePDFLoader(args.pdf_dir, args.output_dir)
    documents = loader.load_all_pdfs()
    
    if not documents:
        logger.error("No documents loaded. Check PDF directory.")
        return
    
    # Save to JSON
    if args.format in ["json", "both"]:
        logger.info("\n" + "="*60)
        logger.info("Saving to JSON format...")
        logger.info("="*60)
        
        loader.save_to_json(documents)
        loader.save_individual_files(documents)
    
    # Save to SQLite
    if args.format in ["sqlite", "both"]:
        logger.info("\n" + "="*60)
        logger.info("Saving to SQLite database...")
        logger.info("="*60)
        
        db_path = args.output_dir / "legal_docs.db"
        db = LegalDocumentDB(db_path)
        db.insert_all(documents)
        
        # Show summary
        docs = db.list_documents()
        logger.success(f"\n✓ Database created with {len(docs)} documents")
        
        db.close()
    
    logger.success("\n" + "="*60)
    logger.success("LOADING COMPLETE!")
    logger.success("="*60)
    logger.success(f"Output directory: {args.output_dir}")
    logger.success(f"Total documents: {len(documents)}")


if __name__ == "__main__":
    main()

