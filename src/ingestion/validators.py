
# src/ingestion/validators.py
# src/ingestion/validators.py
"""
Validators for legal content and chunks.
Ensures all legal data meets strict quality and completeness standards.
"""

import re
from typing import Dict, List, Optional
from loguru import logger

# Import required types and registry
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.chunker import LegalChunk

"""
Validators for legal content and chunks.
"""

class ContentValidator:
    """Validates legal chunks and content."""
    
    @staticmethod
    def validate_chunk(chunk: LegalChunk) -> bool:
        """Validate that chunk meets all requirements."""
        
        # Check mandatory fields
        if not chunk.validate_completeness():
            logger.warning(f"Chunk {chunk.chunk_id} missing mandatory fields")
            return False
        
        # Check text length
        if len(chunk.text.strip()) < 10:
            logger.warning(f"Chunk {chunk.chunk_id} has insufficient text")
            return False
        
        # Check identifier format
        if not ContentValidator._validate_identifier(chunk.identifier_number):
            logger.warning(f"Chunk {chunk.chunk_id} has invalid identifier format")
            return False
        
        # Check source URL
        if not SourceRegistry.validate_source_domain(chunk.source_url):
            logger.warning(f"Chunk {chunk.chunk_id} has invalid source URL")
            return False
        
        return True
    
    @staticmethod
    def _validate_identifier(identifier: str) -> bool:
        """Validate Article/Section number format."""
        import re
        # Should match patterns like: 123, 123A, 123-A, etc.
        pattern = r'^\d+[A-Za-z]?(-[A-Za-z0-9]+)?$'
        return bool(re.match(pattern, identifier))

