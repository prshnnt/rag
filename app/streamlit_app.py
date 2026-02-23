
# app/streamlit_app.py
import streamlit as st
from pathlib import Path
import sys
import json
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import Settings
from core.intent_classifier import IntentClassifier
from core.retriever import HybridRetriever
from core.reranker import LegalReranker
from core.llm_handler import LegalLLMHandler
from indexing.vector_store import VectorStore
from indexing.keyword_index import KeywordIndex
from validation.answer_validator import AnswerValidator
from orchestration.workflow import LegalRAGWorkflow

st.set_page_config(
    page_title="Legal RAG System",
    page_icon="⚖️",
    layout="wide"
)

@st.cache_resource
def load_system():
    """Load and cache all system components."""
    settings = Settings()
    
    # Load indices
    index_dir = Path(settings.index_dir)
    if not (index_dir / "faiss.index").exists():
        raise FileNotFoundError(f"FAISS index not found at {index_dir}. Please run the ingestion script first.")
        
    vector_store = VectorStore(settings.embedding_model)
    vector_store.load(settings.index_dir)
    
    keyword_index = KeywordIndex()
    keyword_index.load(settings.index_dir)
    
    # Initialize components
    intent_classifier = IntentClassifier()
    retriever = HybridRetriever(vector_store, keyword_index)
    reranker = LegalReranker(settings.reranker_model)
    
    # Use key from settings, fallback to env if needed, but settings handles .env loading
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in settings or environment")
        
    llm_handler = LegalLLMHandler(api_key, model="openai/gpt-oss-120b") # Using a valid Groq model name

    validator = AnswerValidator()
    
    # Build workflow
    workflow = LegalRAGWorkflow(
        intent_classifier,
        retriever,
        reranker,
        llm_handler,
        validator
    )
    
    return workflow, settings

def main():
    st.title("⚖️ Legal RAG System")
    st.markdown("**Factual legal information from your documents**")
    
    # Load system
    try:
        workflow, settings = load_system()
    except Exception as e:
        st.error(f"System initialization error: {e}")
        st.info("Please ensure indices are built. Run: `python main.py`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Indexed Documents")
        stats_path = Path(settings.index_dir) / "build_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            processed_files = stats.get("files", {})
            if processed_files:
                for filename, info in processed_files.items():
                    if info['status'] == 'success':
                        st.markdown(f"✓ {filename}")
            else:
                st.markdown("No documents indexed yet.")
        else:
            st.warning("No build stats found. Run the ingestion script.")

        st.divider()
        st.header("Settings")
        show_context = st.checkbox("Show Retrieved Context", value=False)
        show_metadata = st.checkbox("Show Metadata", value=False)
    
    # Main query interface
    query = st.text_area(
        "Enter your legal query:",
        placeholder="Example: What is the punishment for theft?",
        height=100
    )
    
    if st.button("Search Legal Provisions", type="primary"):
        if not query:
            st.warning("Please enter a query")
            return
        
        with st.spinner("Searching legal database..."):
            try:
                result = workflow.run(query)
                
                # Display results
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
                    return
                
                # Validation status
                validation = result.get("validation", {})
                if validation.get("valid"):
                    st.success(f"✓ Valid Answer (Confidence: {validation.get('confidence', 'unknown')})")
                else:
                    st.error("⚠️ Answer validation failed")
                    for error in validation.get("errors", []):
                        st.error(f"- {error}")
                
                # Display answer
                st.markdown("### Legal Position")
                st.markdown(result.get("answer", "No answer generated"))
                
                # Warnings
                for warning in validation.get("warnings", []):
                    st.warning(warning)
                
                # Metadata
                if show_metadata:
                    with st.expander("Query Metadata"):
                        st.json(result.get("intent", {}))
                
                # Retrieved context
                if show_context:
                    with st.expander("Retrieved Legal Provisions"):
                        for i, chunk in enumerate(result.get("final_chunks", []), 1):
                            st.markdown(f"**{i}. {chunk['law_name']} - {chunk['identifier_type']} {chunk['identifier_number']}**")
                            st.markdown(f"Score: {chunk.get('rerank_score', 0):.3f}")
                            st.markdown(chunk['text'][:500] + "...")
                            st.divider()
                
            except Exception as e:
                st.error(f"Query processing error: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Footer
    st.divider()
    st.caption("⚠️ This system provides legal information from statutes, NOT legal advice. Consult a qualified lawyer for specific situations.")

if __name__ == "__main__":
    main()
