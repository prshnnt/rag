
# ----------------------------------------------------------------------------
# FILE 2: src/ingestion/pdf_chunker.py
# ----------------------------------------------------------------------------
"""
PDF-aware chunker that creates legal chunks from parsed PDF sections.
"""

from typing import List, Optional
import re
from loguru import logger

from core.chunker import LegalChunk
from config.legal_guardrails import SUPPORTED_LAWS
from ingestion.pdf_parser import PDFSection


class PDFLegalChunker:
    """
    Chunker for PDF-parsed legal content.
    Converts PDFSection objects to LegalChunk objects.
    """
    
    def chunk_from_pdf_sections(self, sections: List[PDFSection], 
                                law_code: str) -> List[LegalChunk]:
        """
        Convert PDF sections to legal chunks.
        
        Args:
            sections: List of parsed PDF sections
            law_code: Law code (constitution, bns, etc.)
        
        Returns:
            List of LegalChunk objects
        """
        law_info = SUPPORTED_LAWS.get(law_code)
        if not law_info:
            logger.error(f"Unknown law code: {law_code}")
            return []
        
        chunks = []
        
        for section in sections:
            chunk = self._create_chunk(section, law_code, law_info)
            if chunk and chunk.validate_completeness():
                chunks.append(chunk)
            else:
                logger.warning(
                    f"Skipping incomplete chunk: {section.identifier_type} "
                    f"{section.identifier_number}"
                )
        
        logger.info(f"Created {len(chunks)} valid chunks from {len(sections)} sections")
        return chunks
    
    def _create_chunk(self, section: PDFSection, law_code: str, 
                     law_info: Dict) -> Optional[LegalChunk]:
        """Create a LegalChunk from a PDFSection."""
        
        # Extract proviso if present
        proviso = self._extract_proviso(section.text)
        
        # Extract explanation if present
        explanation = self._extract_explanation(section.text)
        
        # Clean main text (remove proviso and explanation)
        main_text = section.text
        if proviso:
            main_text = main_text.replace(proviso, '').strip()
        if explanation:
            main_text = main_text.replace(explanation, '').strip()
        
        # Generate chunk ID
        chunk_id = (
            f"{law_code}_{section.identifier_type.lower()}_"
            f"{section.identifier_number}"
        )
        
        return LegalChunk(
            law_code=law_code,
            law_name=law_info["name"],
            identifier_type=section.identifier_type,
            identifier_number=section.identifier_number,
            clause=None,  # Could extract clauses in future
            title=section.title,
            text=main_text,
            proviso=proviso,
            explanation=explanation,
            source_url=law_info["source"],
            chunk_id=chunk_id
        )
    
    def _extract_proviso(self, text: str) -> Optional[str]:
        """Extract proviso from section text."""
        # Look for "Proviso:" or "Provided that"
        patterns = [
            r'Proviso[:\s]+(.+?)(?=Explanation|$)',
            r'Provided\s+that\s+(.+?)(?=Explanation|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return None
    
    def _extract_explanation(self, text: str) -> Optional[str]:
        """Extract explanation from section text."""
        # Look for "Explanation:" or "Explanation.—"
        patterns = [
            r'Explanation[:\s\.—]+(.+?)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return None
