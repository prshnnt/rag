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