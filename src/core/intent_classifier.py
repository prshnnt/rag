# src/core/intent_classifier.py
from typing import Dict, List
from pydantic import BaseModel
import re

class LegalIntent(BaseModel):
    domain: str  # constitutional, criminal, civil, cyber
    law_type: str  # constitution, ipc, bns, crpc, bnss, cpc, it_act
    specific_sections: List[str] = []
    keywords: List[str] = []
    query_type: str  # definition, penalty, procedure, rights

class IntentClassifier:
    """Classifies legal queries into domains and law types."""
    
    DOMAIN_KEYWORDS = {
        "constitutional": ["article", "fundamental rights", "directive principles", "constitution"],
        "criminal": ["ipc", "bns", "offense", "punishment", "cognizable", "bailable", "section"],
        "procedure_criminal": ["crpc", "bnss", "arrest", "bail", "investigation", "trial"],
        "civil": ["cpc", "suit", "decree", "appeal", "civil procedure"],
        "cyber": ["it act", "cyber", "electronic", "digital signature", "hacking"],
    }
    
    SECTION_PATTERNS = {
        "article": r"article\s+(\d+[a-z]?)",
        "section": r"section\s+(\d+[a-z]?)",
        "ipc": r"ipc\s+(\d+[a-z]?)",
        "bns": r"bns\s+(\d+[a-z]?)",
        "crpc": r"crpc\s+(\d+[a-z]?)",
        "bnss": r"bnss\s+(\d+[a-z]?)",
    }
    
    def classify(self, query: str) -> LegalIntent:
        """Classify user query into legal intent."""
        query_lower = query.lower()
        
        # Extract specific sections
        sections = self._extract_sections(query_lower)
        
        # Determine domain
        domain = self._detect_domain(query_lower, sections)
        
        # Map to specific law
        law_type = self._map_to_law(domain, query_lower, sections)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Determine query type
        query_type = self._classify_query_type(query_lower)
        
        return LegalIntent(
            domain=domain,
            law_type=law_type,
            specific_sections=sections,
            keywords=keywords,
            query_type=query_type
        )
    
    def _extract_sections(self, query: str) -> List[str]:
        sections = []
        for pattern_type, pattern in self.SECTION_PATTERNS.items():
            matches = re.findall(pattern, query)
            sections.extend([f"{pattern_type}_{m}" for m in matches])
        return sections
    
    def _detect_domain(self, query: str, sections: List[str]) -> str:
        # Check sections first
        if any("article" in s for s in sections):
            return "constitutional"
        if any(s.startswith(("ipc_", "bns_")) for s in sections):
            return "criminal"
        if any(s.startswith(("crpc_", "bnss_")) for s in sections):
            return "procedure_criminal"
        
        # Check keywords
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                return domain
        
        return "general"
    
    def _map_to_law(self, domain: str, query: str, sections: List[str]) -> str:
        mapping = {
            "constitutional": "constitution",
            "criminal": "bns" if "bns" in query or any("bns_" in s for s in sections) else "ipc",
            "procedure_criminal": "bnss" if "bnss" in query or any("bnss_" in s for s in sections) else "crpc",
            "civil": "cpc",
            "cyber": "it_act",
        }
        return mapping.get(domain, "bns")
    
    def _extract_keywords(self, query: str) -> List[str]:
        legal_keywords = [
            "bailable", "cognizable", "non-bailable", "non-cognizable",
            "imprisonment", "fine", "punishment", "offense",
            "arrest", "warrant", "summons", "bail",
        ]
        return [kw for kw in legal_keywords if kw in query]
    
    def _classify_query_type(self, query: str) -> str:
        if any(word in query for word in ["what is", "define", "meaning"]):
            return "definition"
        if any(word in query for word in ["punishment", "penalty", "fine", "imprisonment"]):
            return "penalty"
        if any(word in query for word in ["procedure", "process", "how to"]):
            return "procedure"
        if any(word in query for word in ["right", "can i", "am i allowed"]):
            return "rights"
        return "general"