from pydantic import BaseModel , Field

class DocumentMetadata(BaseModel):
    user_id:int = Field(description="Primary key of user who uploaded document")
    document_id:int = Field(description="a unique id of document by which it's detail are stored in db.")
    document_name:str = Field(description="name of document file")
    document_type:str = Field(description="type of document : pdf , docx , txt , xlsx , pptx ")
    chunk_id:int = Field(description="a unique id of each chunk")
    chunk_index:int = Field(description="index of the chunk by which it is stored in db to easily find the chunk before or after it.")