
# ----------------------------------------------------------------------------
# FILE: src/ingestion/load_to_database.py
# ----------------------------------------------------------------------------
"""
Load PDFs to actual database (SQLite for demo).
"""

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List
from loguru import logger

# Add src to pythonpath if running directly
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.simple_pdf_loader import SimplePDFLoader



class LegalDocumentDB:
    """
    Simple SQLite database for storing legal documents.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database and create tables."""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                law_code TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                total_pages INTEGER,
                full_text TEXT,
                file_size_bytes INTEGER,
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create pages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                page_number INTEGER,
                text TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    def insert_document(self, doc_data: dict):
        """
        Insert a document into the database.
        
        Args:
            doc_data: Document dictionary from SimplePDFLoader
        """
        cursor = self.conn.cursor()
        
        # Insert main document
        cursor.execute("""
            INSERT OR REPLACE INTO documents 
            (law_code, filename, total_pages, full_text, file_size_bytes)
            VALUES (?, ?, ?, ?, ?)
        """, (
            doc_data["law_code"],
            doc_data["filename"],
            doc_data["total_pages"],
            doc_data["full_text"],
            doc_data["metadata"]["file_size_bytes"]
        ))
        
        document_id = cursor.lastrowid
        
        # Insert pages
        for page in doc_data["pages"]:
            cursor.execute("""
                INSERT INTO pages (document_id, page_number, text)
                VALUES (?, ?, ?)
            """, (
                document_id,
                page["page_number"],
                page["text"]
            ))
        
        self.conn.commit()
        logger.info(f"✓ Inserted {doc_data['law_code']} into database")
    
    def insert_all(self, documents: Dict[str, dict]):
        """Insert all documents."""
        for law_code, doc_data in documents.items():
            try:
                self.insert_document(doc_data)
            except Exception as e:
                logger.error(f"Failed to insert {law_code}: {e}")
    
    def get_document(self, law_code: str) -> dict:
        """Retrieve a document by law code."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM documents WHERE law_code = ?
        """, (law_code,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            "id": row[0],
            "law_code": row[1],
            "filename": row[2],
            "total_pages": row[3],
            "full_text": row[4],
            "file_size_bytes": row[5],
            "loaded_at": row[6]
        }
    
    def list_documents(self) -> List[dict]:
        """List all documents in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT law_code, filename, total_pages FROM documents")
        
        return [
            {"law_code": row[0], "filename": row[1], "total_pages": row[2]}
            for row in cursor.fetchall()
        ]
    
    def search_text(self, query: str, law_code: str = None) -> List[dict]:
        """
        Simple text search in documents.
        
        Args:
            query: Search query
            law_code: Optional law code to search within
        """
        cursor = self.conn.cursor()
        
        if law_code:
            cursor.execute("""
                SELECT law_code, page_number, text 
                FROM pages p
                JOIN documents d ON p.document_id = d.id
                WHERE d.law_code = ? AND p.text LIKE ?
            """, (law_code, f"%{query}%"))
        else:
            cursor.execute("""
                SELECT law_code, page_number, text 
                FROM pages p
                JOIN documents d ON p.document_id = d.id
                WHERE p.text LIKE ?
            """, (f"%{query}%",))
        
        return [
            {"law_code": row[0], "page": row[1], "text": row[2][:200]}
            for row in cursor.fetchall()
        ]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# ----------------------------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------------------------

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

