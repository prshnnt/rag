# src/config/legal_guardrails.py
"""
Strict legal guardrails for the RAG system.
"""

LEGAL_SYSTEM_PROMPT = """You are a precise Indian legal information system. Your ONLY role is to provide factual legal positions from Indian statutes.

STRICT RULES:
1. Answer ONLY from the provided legal context
2. ALWAYS cite specific Article/Section numbers
3. ALWAYS include official source URLs
4. NEVER provide legal advice or opinions
5. NEVER speculate or infer beyond explicit text
6. If context is insufficient, say "Insufficient legal provisions found"
7. Use exact legal terminology from statutes
8. Flag any interpretive statements clearly

OUTPUT FORMAT (mandatory):
**Legal Position:**
[Factual statement from statute]

**Relevant Provisions:**
- [Law Name] - Article/Section [Number]: [Summary]
  Source: [Official URL]

**Notes:**
[Any provisos, exceptions, or related sections]

**Disclaimer:** This is legal information from statutes, not legal advice. Consult a qualified lawyer for specific situations.
"""

SUPPORTED_LAWS = {
    "constitution": {
        "name": "Constitution of India",
        "source": "https://legislative.gov.in/constitution-of-india/",
        "identifier_type": "Article",
        "active": True,
    },
    "ipc": {
        "name": "Indian Penal Code, 1860",
        "source": "https://legislative.gov.in/actsofparliamentfromtheyear/indian-penal-code-1860",
        "identifier_type": "Section",
        "active": True,
        "superseded_by": "bns",
        "superseded_date": "2024-07-01",
    },
    "bns": {
        "name": "Bharatiya Nyaya Sanhita, 2023",
        "source": "https://legislative.gov.in/actsofparliamentfromtheyear/bharatiya-nyaya-sanhita-2023",
        "identifier_type": "Section",
        "active": True,
        "replaces": "ipc",
    },
    "crpc": {
        "name": "Code of Criminal Procedure, 1973",
        "source": "https://legislative.gov.in/actsofparliamentfromtheyear/code-criminal-procedure-1973",
        "identifier_type": "Section",
        "active": True,
        "superseded_by": "bnss",
        "superseded_date": "2024-07-01",
    },
    "bnss": {
        "name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "source": "https://legislative.gov.in/actsofparliamentfromtheyear/bharatiya-nagarik-suraksha-sanhita-2023",
        "identifier_type": "Section",
        "active": True,
        "replaces": "crpc",
    },
    "cpc": {
        "name": "Code of Civil Procedure, 1908",
        "source": "https://legislative.gov.in/actsofparliamentfromtheyear/code-civil-procedure-1908",
        "identifier_type": "Section",
        "active": True,
    },
    "it_act": {
        "name": "Information Technology Act, 2000",
        "source": "https://legislative.gov.in/actsofparliamentfromtheyear/information-technology-act-2000",
        "identifier_type": "Section",
        "active": True,
    },
}

# Versioning strategy: Support both old and new laws with clear dating
LAW_VERSION_STRATEGY = {
    "default_criminal_substantive": "bns",  # Post July 2024
    "fallback_criminal_substantive": "ipc",  # Pre July 2024
    "default_criminal_procedure": "bnss",
    "fallback_criminal_procedure": "crpc",
    "civil_procedure": "cpc",
    "constitutional": "constitution",
    "cyber": "it_act",
}