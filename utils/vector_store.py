import os
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize or load existing vector store"""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
    
    def load_documents(self, file_path):
        """Load documents from file"""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        documents = loader.load()
        return documents
    
    def add_documents(self, file_path):
        """Add documents to vector store"""
        documents = self.load_documents(file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        self.vector_store.add_documents(splits)
        return len(splits)
    
    def search(self, query, k=3):
        """Search vector store"""
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def get_retriever(self, k=3):
        """Get retriever for RAG"""
        return self.vector_store.as_retriever(search_kwargs={"k": k})