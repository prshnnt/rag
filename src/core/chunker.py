# src/core/chunker.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import re
from config.legal_guardrails import SUPPORTED_LAWS

# src/core/chunker.py (Enhanced version)

class LegalChunk(BaseModel):
    """
    Legal-aware chunk schema compatible with various Indian laws 
    (Constitution, IPC, BNS, BNSS, IT Act, etc.).
    """
    law_code: str
    law_name: str
    
    # Hierarchy
    part_number: Optional[str] = None      # For Constitution (Part III)
    chapter_number: Optional[str] = None   # For Acts (Chapter V)
    chapter_title: Optional[str] = None    # e.g., "Fundamental Rights"
    
    # Primary Identifier
    identifier_type: str  # Article, Section, Rule, Order
    identifier_number: str
    
    # Sub-divisions
    sub_section: Optional[str] = None
    clause: Optional[str] = None
    
    # Content
    title: Optional[str] = None
    text: str
    
    # Specific Legal Components
    proviso: Optional[str] = None
    explanation: Optional[str] = None
    illustration: Optional[str] = None
    
    # Metadata
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    chunk_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def validate_completeness(self) -> bool:
        """Validate that chunk has minimum viable metadata."""
        required = [
            self.law_code,
            self.law_name,
            self.identifier_number,
            self.text
        ]
        return all(required) and len(self.text.strip()) > 0