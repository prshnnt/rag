 
import re
from typing import List, Dict        

class AnswerValidator:
    """Validates LLM answers for legal compliance."""
    
    @staticmethod
    def validate(answer: str, retrieved_chunks: List[Dict]) -> Dict:
        """Validate answer structure and citations."""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "confidence": "high"
        }
        
        # Check for required sections
        required_sections = ["Legal Position:", "Relevant Provisions:", "Disclaimer:"]
        for section in required_sections:
            if section not in answer:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required section: {section}")
        
        # Check for section/article citations
        has_citations = bool(re.search(r'(Article|Section)\s+\d+', answer))
        if not has_citations:
            validation_result["valid"] = False
            validation_result["errors"].append("No Article/Section citations found")
        
        # Check for source URLs
        has_urls = bool(re.search(r'https?://\S+', answer))
        if not has_urls:
            validation_result["valid"] = False
            validation_result["errors"].append("No source URLs found")
        
        # Check for speculative language
        speculative_phrases = [
            "i think", "probably", "maybe", "might be",
            "could be interpreted", "in my opinion"
        ]
        answer_lower = answer.lower()
        for phrase in speculative_phrases:
            if phrase in answer_lower:
                validation_result["warnings"].append(f"Speculative language detected: {phrase}")
                validation_result["confidence"] = "medium"
        
        # Check if answer references provided chunks
        chunk_ids_in_context = {c['chunk_id'] for c in retrieved_chunks}
        referenced_sections = re.findall(r'(Article|Section)\s+(\d+[a-z]?)', answer)
        
        if not referenced_sections:
            validation_result["confidence"] = "low"
        
        return validation_result

