# src/core/chunker.py
from typing import List, Dict , Optional
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel
import re
from config.legal_guardrails import SUPPORTED_LAWS

# src/core/chunker.py (Enhanced version)

class LegalChunk(BaseModel):
    """Legal-aware chunk with mandatory metadata."""
    law_code: str
    law_name: str
    identifier_type: str  # Article or Section
    identifier_number: str
    clause: Optional[str] = None
    title: Optional[str] = None
    text: str
    proviso: Optional[str] = None
    explanation: Optional[str] = None
    source_url: str
    chunk_id: str
    
    def validate_completeness(self) -> bool:
        """Validate that chunk has all mandatory metadata."""
        required = [
            self.law_code,
            self.law_name,
            self.identifier_number,
            self.text,
            self.source_url
        ]
        return all(required) and len(self.text.strip()) > 0


class LegalChunker:
    """
    Legal-aware chunking engine.
    NO token-based splitting - respects legal structure.
    """
    
    def chunk_document(self, raw_html: str, law_code: str) -> List[LegalChunk]:
        """
        Chunk legal document by Article/Section boundaries.
        Preserves legal structure completely.
        """
        
        soup = BeautifulSoup(raw_html, 'lxml')
        law_info = SUPPORTED_LAWS[law_code]
        
        logger.info(f"Parsing HTML for {law_code}...")
        
        # Different parsing strategies for different laws
        if law_code == "constitution":
            chunks = self._chunk_constitution(soup, law_code, law_info)
        else:
            chunks = self._chunk_statute(soup, law_code, law_info)
        
        logger.info(f"Extracted {len(chunks)} raw chunks")
        
        # Validate all chunks
        valid_chunks = [c for c in chunks if c.validate_completeness()]
        logger.info(f"Validated {len(valid_chunks)} complete chunks")
        
        return valid_chunks
    
    def _chunk_constitution(self, soup: BeautifulSoup, law_code: str, law_info: Dict) -> List[LegalChunk]:
        """Chunk Constitution by Articles."""
        chunks = []
        
        # Find all article sections
        # This is a generic parser - adjust selectors based on actual HTML structure
        article_sections = soup.find_all(['div', 'section'], class_=re.compile(r'article|section', re.I))
        
        if not article_sections:
            # Fallback: look for headings
            article_sections = soup.find_all(['h2', 'h3', 'h4'], string=re.compile(r'article\s+\d+', re.I))
        
        for section in article_sections:
            chunk = self._extract_chunk_from_section(section, law_code, law_info, "Article")
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_statute(self, soup: BeautifulSoup, law_code: str, law_info: Dict) -> List[LegalChunk]:
        """Chunk statutes by Sections."""
        chunks = []
        
        # Find all section elements
        section_elements = soup.find_all(['div', 'section'], class_=re.compile(r'section', re.I))
        
        if not section_elements:
            # Fallback: look for section headings
            section_elements = soup.find_all(['h2', 'h3', 'h4'], string=re.compile(r'section\s+\d+', re.I))
        
        for section in section_elements:
            chunk = self._extract_chunk_from_section(section, law_code, law_info, "Section")
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _extract_chunk_from_section(self, element: Tag, law_code: str, 
                                    law_info: Dict, identifier_type: str) -> Optional[LegalChunk]:
        """Extract structured chunk from HTML section."""
        
        # Extract identifier number
        identifier = self._extract_identifier(element, identifier_type)
        if not identifier:
            return None
        
        # Extract title
        title = self._extract_title(element)
        
        # Extract main text
        text = self._extract_text(element)
        if not text or len(text.strip()) < 10:
            return None
        
        # Extract proviso
        proviso = self._extract_proviso(element)
        
        # Extract explanation
        explanation = self._extract_explanation(element)
        
        # Generate chunk ID
        chunk_id = f"{law_code}_{identifier_type.lower()}_{identifier}"
        
        return LegalChunk(
            law_code=law_code,
            law_name=law_info["name"],
            identifier_type=identifier_type,
            identifier_number=identifier,
            title=title,
            text=text,
            proviso=proviso,
            explanation=explanation,
            source_url=law_info["source"],
            chunk_id=chunk_id
        )
    
    def _extract_identifier(self, element: Tag, identifier_type: str) -> Optional[str]:
        """Extract Article/Section number."""
        
        # Look for identifier in text
        text = element.get_text()
        pattern = rf'{identifier_type}\s+(\d+[A-Za-z]?(?:-[A-Za-z0-9]+)?)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            return match.group(1)
        
        # Try to find in specific elements
        identifier_elem = element.find(['span', 'strong', 'b'], class_=re.compile(r'number|identifier', re.I))
        if identifier_elem:
            num_match = re.search(r'\d+[A-Za-z]?', identifier_elem.get_text())
            if num_match:
                return num_match.group(0)
        
        return None
    
    def _extract_title(self, element: Tag) -> Optional[str]:
        """Extract section/article title."""
        title_elem = element.find(['h3', 'h4', 'h5', 'strong'], class_=re.compile(r'title|heading', re.I))
        if title_elem:
            title = title_elem.get_text(strip=True)
            # Remove "Section X:" prefix if present
            title = re.sub(r'^(Article|Section)\s+\d+[A-Za-z]?[:\-\s]+', '', title, flags=re.I)
            return title if title else None
        return None
    
    def _extract_text(self, element: Tag) -> str:
        """Extract main provision text."""
        # Get all paragraphs
        paragraphs = element.find_all(['p', 'div'], class_=re.compile(r'text|content|provision', re.I))
        
        if paragraphs:
            text_parts = []
            for p in paragraphs:
                # Skip provisos and explanations
                if re.search(r'proviso|explanation', p.get('class', []) if p.get('class') else '', re.I):
                    continue
                text_parts.append(p.get_text(strip=True))
            return ' '.join(text_parts)
        
        # Fallback: get all text
        return element.get_text(strip=True)
    
    def _extract_proviso(self, element: Tag) -> Optional[str]:
        """Extract proviso if present."""
        proviso_elem = element.find(['p', 'div'], class_=re.compile(r'proviso', re.I))
        if proviso_elem:
            return proviso_elem.get_text(strip=True)
        
        # Look for "Proviso:" in text
        text = element.get_text()
        match = re.search(r'Proviso[:\s]+(.+?)(?=Explanation|$)', text, re.DOTALL | re.I)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _extract_explanation(self, element: Tag) -> Optional[str]:
        """Extract explanation if present."""
        expl_elem = element.find(['p', 'div'], class_=re.compile(r'explanation', re.I))
        if expl_elem:
            return expl_elem.get_text(strip=True)
        
        # Look for "Explanation:" in text
        text = element.get_text()
        match = re.search(r'Explanation[:\s]+(.+?)$', text, re.DOTALL | re.I)
        if match:
            return match.group(1).strip()
        
        return None
