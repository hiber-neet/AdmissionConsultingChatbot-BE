from fastapi import APIRouter, HTTPException
from app.models.schemas import DocumentUpload, DocumentResponse
from app.services.langchain_service import langchain_service
from typing import List

router = APIRouter()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(document: DocumentUpload):
    try:
        metadata = document.metadata or {}
        ids = langchain_service.add_documents(
            texts=[document.content],
            metadatas=[metadata]
        )

        return DocumentResponse(
            id=ids[0],
            message="Document uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-batch", response_model=List[DocumentResponse])
async def upload_documents_batch(documents: List[DocumentUpload]):
    try:
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]

        ids = langchain_service.add_documents(texts=texts, metadatas=metadatas)

        return [
            DocumentResponse(id=doc_id, message="Document uploaded successfully")
            for doc_id in ids
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
