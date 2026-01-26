import streamlit as st
import os
from pathlib import Path

# Import custom modules
from utils.llm_config import get_llm
from utils.vector_store import VectorStoreManager
from tools.search_tools import create_search_tool
from agents.rag_agent import RAGAgent

# Page config
st.set_page_config(
    page_title="RAG Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

if "agent" not in st.session_state:
    try:
        llm = get_llm()
        search_tool = create_search_tool(st.session_state.vector_store)
        st.session_state.agent = RAGAgent(llm, [search_tool])
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.stop()

# Sidebar for document management
with st.sidebar:
    st.title("📚 Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload documents (PDF or TXT)",
        type=["pdf", "txt"],
        help="Upload documents to add to the knowledge base"
    )
    
    if uploaded_file:
        # Save uploaded file
        data_dir = Path("data/documents")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = data_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Add to vector store
        with st.spinner("Processing document..."):
            try:
                num_chunks = st.session_state.vector_store.add_documents(str(file_path))
                st.success(f"✅ Added {num_chunks} chunks to knowledge base!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Info
    st.markdown("### ℹ️ About")
    st.markdown("""
    This is a RAG-powered AI agent that can:
    - Answer questions using uploaded documents
    - Maintain conversation context
    - Search through your knowledge base
    
    **How to use:**
    1. Upload documents (PDF/TXT)
    2. Ask questions in the chat
    3. The agent will search documents when needed
    """)

# Main chat interface
st.title("🤖 RAG Agent Chat")
st.markdown("Ask me anything! I can search through your uploaded documents to provide accurate answers.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.run(
                    prompt,
                    st.session_state.messages[:-1]
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

