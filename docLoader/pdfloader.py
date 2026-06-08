import os
from typing import List, Dict, Any, Optional
import pypdf
from langchain_core.documents import Document
from langchain_core.tools import tool

def load_pdf_as_pages(
    pdf_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    page_numbers: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file with optional page filtering.
    
    Args:
        pdf_path: Path to the PDF file.
        start_page: 1-indexed page number to start extraction from.
        end_page: 1-indexed page number to end extraction (inclusive).
        page_numbers: A list of specific 1-indexed page numbers to extract (e.g. [1, 5, 10]).
                      If page_numbers is provided, start_page and end_page are ignored.
        
    Returns:
        A list of dictionaries, where each dictionary represents a page with keys:
        - "page": The 1-indexed page number.
        - "text": The extracted text content from that page.
        - "metadata": A dictionary containing metadata such as "source", "page", and "total_pages".
        
    Raises:
        FileNotFoundError: If the pdf_path does not exist.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    # Determine target page numbers (1-indexed)
    target_pages = []
    if page_numbers is not None:
        target_pages = [p for p in page_numbers if 1 <= p <= total_pages]
    else:
        start = start_page if start_page is not None else 1
        end = end_page if end_page is not None else total_pages
        # Clamp bounds
        start = max(1, start)
        end = min(total_pages, end)
        if start <= end:
            target_pages = list(range(start, end + 1))
            
    pages = []
    filename = os.path.basename(pdf_path)
    for p_num in target_pages:
        page = reader.pages[p_num - 1]
        page_text = (page.extract_text() or "").strip()
        pages.append({
            "page": p_num,
            "text": page_text,
            "metadata": {
                "source": filename,
                "page": p_num,
                "total_pages": total_pages
            }
        })
    return pages

def load_pdf_as_text(
    pdf_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    page_numbers: Optional[List[int]] = None
) -> str:
    """
    Extracts and returns the text content of a PDF file combined as a single string,
    with optional page filtering.
    
    Args:
        pdf_path: Path to the PDF file.
        start_page: 1-indexed page number to start extraction from.
        end_page: 1-indexed page number to end extraction (inclusive).
        page_numbers: A list of specific 1-indexed page numbers to extract.
        
    Returns:
        The extracted text content combined from the selected pages.
    """
    pages = load_pdf_as_pages(
        pdf_path, start_page=start_page, end_page=end_page, page_numbers=page_numbers
    )
    return "\n".join(page["text"] for page in pages)

def load_pdf_as_documents(
    pdf_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    page_numbers: Optional[List[int]] = None
) -> List[Document]:
    """
    Converts selected PDF pages into LangChain Document objects.
    
    Args:
        pdf_path: Path to the PDF file.
        start_page: 1-indexed page number to start extraction from.
        end_page: 1-indexed page number to end extraction (inclusive).
        page_numbers: A list of specific 1-indexed page numbers to extract.
        
    Returns:
        A list of langchain_core.documents.Document objects.
    """
    pages = load_pdf_as_pages(
        pdf_path, start_page=start_page, end_page=end_page, page_numbers=page_numbers
    )
    return [
        Document(
            page_content=page["text"],
            metadata=page["metadata"]
        )
        for page in pages
    ]

@tool
def load_pdf_file_tool(
    pdf_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    page_numbers: Optional[List[int]] = None
) -> str:
    """
    Load specific pages or page ranges from a PDF file and return their combined text content.
    
    Use this tool to read the contents of a PDF document. For large PDFs, always prefer 
    supplying `start_page` and `end_page` or specific `page_numbers` to avoid exceeding 
    the token limits or context window.
    
    Args:
        pdf_path: The local absolute or relative path to the PDF file.
        start_page: The 1-indexed page number to start loading from (e.g. 1).
        end_page: The 1-indexed page number to stop loading at (inclusive, e.g. 10).
        page_numbers: A list of specific 1-indexed page numbers to retrieve (e.g. [1, 5, 12]).
                      If page_numbers is provided, start_page and end_page are ignored.
        
    Returns:
        The text content extracted from the selected pages of the PDF.
    """
    try:
        return load_pdf_as_text(
            pdf_path, start_page=start_page, end_page=end_page, page_numbers=page_numbers
        )
    except Exception as e:
        return f"Error loading PDF file: {str(e)}"
