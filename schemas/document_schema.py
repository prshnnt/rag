from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

DocumentType = Literal["pdf", "docx", "txt", "xlsx", "pptx"]


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UUID = Field(description="Primary key of user who uploaded document")
    document_id: UUID = Field(description="Unique id of document stored in db")
    document_name: str = Field(description="Name of document file")
    document_type: DocumentType = Field(description="Type of document: pdf, docx, txt, xlsx, pptx")
    chunk_id: UUID = Field(description="Unique id of each chunk")
    chunk_index: int = Field(ge=0, description="Index of chunk in document for ordering")
