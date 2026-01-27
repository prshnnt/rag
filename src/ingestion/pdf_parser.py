# ----------------------------------------------------------------------------
# FILE 1: src/ingestion/pdf_parser.py
# ----------------------------------------------------------------------------
"""
PDF parser for Indian legal documents.
Extracts structured content from Constitution and BNS PDFs.
"""

import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import pypdf
from loguru import logger


@dataclass
class PDFSection:
    """Represents a parsed section/article from PDF."""
    page_number: int
    identifier_type: str  # "Article" or "Section"
    identifier_number: str
    title: Optional[str]
    text: str
    raw_text: str


class LegalPDFParser:
    """Parser for Indian legal PDFs (Constitution, BNS, etc.)."""
    
    def __init__(self):
        self.current_law_code = None
        self.identifier_type = None
    
    def parse_pdf(self, pdf_path: Path, law_code: str) -> List[PDFSection]:
        """
        Parse legal PDF and extract sections/articles.
        
        Args:
            pdf_path: Path to PDF file
            law_code: Law code (constitution, bns, etc.)
        
        Returns:
            List of parsed sections
        """
        self.current_law_code = law_code
        self.identifier_type = "Article" if law_code == "constitution" else "Section"
        
        logger.info(f"Parsing PDF: {pdf_path}")
        logger.info(f"Law code: {law_code}, Identifier type: {self.identifier_type}")
        
        # Extract text from PDF
        reader = pypdf.PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        logger.info(f"Total pages: {total_pages}")
        
        # Extract text page by page
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                pages_text.append((i + 1, text))
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i+1}: {e}")
                pages_text.append((i + 1, ""))
        
        # Parse sections from combined text
        sections = self._extract_sections(pages_text)
        
        logger.info(f"Extracted {len(sections)} sections from PDF")
        return sections
    
    def _extract_sections(self, pages_text: List[Tuple[int, str]]) -> List[PDFSection]:
        """Extract individual sections/articles from page texts."""
        
        sections = []
        
        if self.current_law_code == "constitution":
            sections = self._extract_constitution_articles(pages_text)
        else:
            sections = self._extract_statute_sections(pages_text)
        
        return sections
    
    def _extract_constitution_articles(self, pages_text: List[Tuple[int, str]]) -> List[PDFSection]:
        """Extract Articles from Constitution PDF."""
        
        sections = []
        
        # Pattern for Article headings
        # Examples: "Article 1", "Article 14A", "ARTICLE 21"
        article_pattern = re.compile(
            r'^(?:ARTICLE|Article)\s+(\d+[A-Z]?)(?:\s*[\.:\-—]?\s*(.+?))?$',
            re.MULTILINE | re.IGNORECASE
        )
        
        for page_num, page_text in pages_text:
            # Clean text
            page_text = self._clean_text(page_text)
            
            # Find all article starts
            matches = list(article_pattern.finditer(page_text))
            
            for i, match in enumerate(matches):
                article_num = match.group(1)
                title = match.group(2).strip() if match.group(2) else None
                
                # Extract text until next article or end of page
                start_pos = match.end()
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(page_text)
                
                article_text = page_text[start_pos:end_pos].strip()
                
                # Clean up the text
                article_text = self._clean_article_text(article_text)
                
                if article_text:  # Only add if we have content
                    sections.append(PDFSection(
                        page_number=page_num,
                        identifier_type="Article",
                        identifier_number=article_num,
                        title=title,
                        text=article_text,
                        raw_text=page_text[match.start():end_pos]
                    ))
        
        return sections
    
    def _extract_statute_sections(self, pages_text: List[Tuple[int, str]]) -> List[PDFSection]:
        """Extract Sections from statute PDFs (BNS, BNSS, etc.)."""
        
        sections = []
        
        # Pattern for Section headings
        # Examples: "Section 1", "1.", "Section 302A"
        section_pattern = re.compile(
            r'^(?:SECTION|Section)?\s*(\d+[A-Z]?)(?:\s*[\.:\-—]?\s*(.+?))?$',
            re.MULTILINE | re.IGNORECASE
        )
        
        for page_num, page_text in pages_text:
            # Clean text
            page_text = self._clean_text(page_text)
            
            # Find all section starts
            matches = list(section_pattern.finditer(page_text))
            
            for i, match in enumerate(matches):
                section_num = match.group(1)
                title = match.group(2).strip() if match.group(2) else None
                
                # Extract text until next section or end of page
                start_pos = match.end()
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(page_text)
                
                section_text = page_text[start_pos:end_pos].strip()
                
                # Clean up the text
                section_text = self._clean_section_text(section_text)
                
                if section_text:  # Only add if we have content
                    sections.append(PDFSection(
                        page_number=page_num,
                        identifier_type="Section",
                        identifier_number=section_num,
                        title=title,
                        text=section_text,
                        raw_text=page_text[match.start():end_pos]
                    ))
        
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove header/footer patterns
        text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        return text.strip()
    
    def _clean_article_text(self, text: str) -> str:
        """Clean article text specifically."""
        # Remove numbering artifacts
        text = re.sub(r'^\s*\d+\s*', '', text)
        # Remove common artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove editorial notes
        return text.strip()
    
    def _clean_section_text(self, text: str) -> str:
        """Clean section text specifically."""
        # Similar to article cleaning
        text = re.sub(r'^\s*\d+\s*', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        return text.strip()