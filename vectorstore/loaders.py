from typing import List
import opendataloader_pdf as odlp
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_core.documents import Document

class Convert:
    @staticmethod
    def pdf_to_markdown(path:str|List[str],output_dir="output/"):
        """
        Convert PDF(s) into the requested output format(s).
        
            Args:
                input_path: One or more input PDF file paths or directories
                output_dir: Directory where output files are written. Default: input file directory
        """
        odlp.convert(input_path=path,output_dir=output_dir,format="markdown")
    @staticmethod
    def langchain_load_pdf(path:str|List[str],format="text"):
        loader = OpenDataLoaderPDFLoader(
            file_path=path,
            format=format,
            quiet=True,
            # extra_metadata=extra_metadata
        )
        documents = loader.load()
        return documents

def document_to_lists(docs:List[Document]):
    content = []
    metadata = []
    for doc in docs:
        content.append(doc.page_content)
        metadata.append(doc.metadata)
    return [content,metadata]
