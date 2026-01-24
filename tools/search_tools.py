from langchain.tools import Tool

def create_search_tool(vector_store_manager):
    """Create a search tool for the agent"""
    
    def search_documents(query: str) -> str:
        """Search through uploaded documents for relevant information"""
        try:
            results = vector_store_manager.search(query, k=3)
            
            if not results:
                return "No relevant documents found."
            
            context = "\n\n".join([doc.page_content for doc in results])
            return f"Found relevant information:\n\n{context}"
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    return Tool(
        name="search_documents",
        func=search_documents,
        description="Search through uploaded documents to find relevant information. Use this when you need to answer questions based on the document knowledge base."
    )