import os
from typing import List
from langchain_core.documents import Document
from langchain_core.tools import tool

def load_text_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Loads and returns the entire text content of a file.
    
    Args:
        file_path: Path to the text file.
        encoding: Text file encoding, defaults to 'utf-8'.
        
    Returns:
        The content of the text file as a string.
        
    Raises:
        FileNotFoundError: If the file_path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found at: {file_path}")
        
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()

def load_text_as_documents(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """
    Loads a text file and converts it into a single LangChain Document object.
    
    Args:
        file_path: Path to the text file.
        encoding: Text file encoding, defaults to 'utf-8'.
        
    Returns:
        A list containing a single Document object with the file content and metadata.
        
    Raises:
        FileNotFoundError: If the file_path does not exist.
    """
    content = load_text_file(file_path, encoding=encoding)
    filename = os.path.basename(file_path)
    return [
        Document(
            page_content=content,
            metadata={
                "source": filename
            }
        )
    ]

@tool
def load_text_file_tool(file_path: str) -> str:
    """
    Load a plain text file (e.g. .txt, .md, .py) from the given file path and return its content.
    
    Use this tool when you need to read the contents of a text file.
    
    Args:
        file_path: The local absolute or relative path to the text file.
        
    Returns:
        The content of the text file.
    """
    try:
        return load_text_file(file_path)
    except Exception as e:
        return f"Error loading text file: {str(e)}"
