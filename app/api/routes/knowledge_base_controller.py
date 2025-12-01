from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, APIRouter
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
from app.models.database import init_db, get_db
from app.models.schemas import TrainingQuestionRequest, TrainingQuestionResponse, KnowledgeBaseDocumentResponse
from app.models import entities
from app.services.training_service import TrainingService
from app.utils.document_processor import documentProcessor
from app.core.security import get_current_user, has_permission
from pathlib import Path
from sqlalchemy.orm import Session
import os

router = APIRouter()

def check_view_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user has permission to view training questions (Admin, Consultant, or Admission)"""
    if not current_user:
        raise HTTPException(status_code=403, detail="Not authenticated")

    try:
        user_perms_list = [p.permission_name.lower() for p in current_user.permissions] 
    except AttributeError:
        user_perms_list = [p.lower() for p in current_user.permissions]

    is_admin_or_consultant = "admin" in user_perms_list or "consultant" in user_perms_list
    is_admission_related = any("admission" in p for p in user_perms_list)

    if not (is_admin_or_consultant or is_admission_related):
        raise HTTPException(
            status_code=403,
            detail="Admin, Consultant, or Admission permission required"
        )
    
    return current_user
@router.post("/upload/document")
async def upload_document(
    intend_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # STEP 1: VALIDATE FILE
    try:
        is_valid, error_msg = documentProcessor.validate_file(file.filename, file.content_type)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # STEP 2: READ FILE
    try:
        file_content = await file.read()
        
        # Check file size (max 50MB)
        max_size = 50 * 1024 * 1024
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {max_size / 1024 / 1024}MB"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # STEP 3: EXTRACT TEXT FROM DOCUMENT
    try:
        content_text = documentProcessor.extract_text(
            file_content,
            file.filename,
            file.content_type
        )
        
        if not content_text or len(content_text.strip()) == 0:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from document. File may be empty or corrupted."
            )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Document processing failed: {str(e)}")
    
    # STEP 4: SAVE TO DATABASE
    # try:
    #     document = KnowledgeBaseDocument(
    #         title=title,
    #         content=content_text,
    #         document_type=file.content_type,
    #         file_path=f"/uploads/{file.filename}",
    #         created_by=current_user.id,
    #         metadata=json.dumps({
    #             "original_filename": file.filename,
    #             "file_extension": Path(file.filename).suffix.lower(),
    #             "upload_size_bytes": len(file_content),
    #             "extracted_size_bytes": len(content_text)
    #         })
    #     )
    #     db.add(document)
    #     db.commit()
    #     db.refresh(document)
    
    # except Exception as e:
    #     db.rollback()
    #     raise HTTPException(status_code=500, detail=f"Failed to save document: {str(e)}")
    
    # STEP 5: CHUNK + EMBED + STORE IN QDRANT
    try:
        print(type(content_text), content_text)
        service = TrainingService()
        chunk_ids = service.add_document(
            1,
            content_text,
            intend_id,
            {
                
                "type": file.content_type,
                "filename": file.filename
            }
        )
    
    except Exception as e:
        # Cleanup if chunking fails
        # db.delete(document)
        # db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    
    # STEP 6: SAVE CHUNK REFERENCES
    # try:
    #     for i, chunk_id in enumerate(chunk_ids):
    #         chunk = DocumentChunk(
    #             document_id=document.id,
    #             chunk_index=i,
    #             embedding_id=chunk_id
    #         )
    #         db.add(chunk)
        
    #     db.commit()
    
    # except Exception as e:
    #     db.rollback()
    #     raise HTTPException(status_code=500, detail=f"Failed to save chunks: {str(e)}")
    
    # SUCCESS
    return {
        "message": "Document uploaded and indexed successfully",
        "document_id": 1,
        "filename": file.filename,
        "file_type": Path(file.filename).suffix.lower(),
        "chunks_created": len(chunk_ids),
        "original_size_kb": len(file_content) / 1024,
        "extracted_text_length": len(content_text)
    }

@router.post("/upload/training_question")
async def upload_training_question(payload: TrainingQuestionRequest, db: Session = Depends(get_db), current_user_id: int = 1):
    service = TrainingService()
    result = service.add_training_qa(
        db=db,
        intent_id=payload.intent_id,
        question_text=payload.question,
        answer_text=payload.answer,
        
    )
    return {"message": "Training Q&A added successfully", "result": result}

@router.get("/training_questions", response_model=List[TrainingQuestionResponse])
def get_all_training_questions(
    db: Session = Depends(get_db), 
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Get all training questions in the system.
    Requires Admin, Consultant, or Admission permission.
    """
    training_questions = db.query(entities.TrainingQuestionAnswer).all()
    
    # Convert to response format
    result = []
    for tqa in training_questions:
        result.append({
            "question_id": tqa.question_id,
            "question": tqa.question,
            "answer": tqa.answer,
            "intent_id": tqa.intent_id
        })
    
    return result

@router.get("/documents", response_model=List[KnowledgeBaseDocumentResponse])
def get_all_documents(
    db: Session = Depends(get_db), 
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Get all documents in the knowledge base.
    Requires Admin, Consultant, or Admission permission.
    """
    documents = db.query(entities.KnowledgeBaseDocument).all()
    
    # Convert to response format
    result = []
    for doc in documents:
        result.append({
            "document_id": doc.document_id,
            "title": doc.title,
            "file_path": doc.file_path,
            "category": doc.category,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "created_by": doc.created_by
        })
    
    return result

@router.get("/documents/{document_id}/download")
def download_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Download a specific document by its ID.
    Requires Admin, Consultant, or Admission permission.
    """
    # Find the document in database
    document = db.query(entities.KnowledgeBaseDocument).filter(
        entities.KnowledgeBaseDocument.document_id == document_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if file exists on disk
    file_path = document.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    # Return the file for download
    return FileResponse(
        path=file_path,
        filename=document.title,
        media_type='application/octet-stream'
    )

@router.get("/documents/{document_id}/view")
def view_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    View/preview a specific document by its ID in browser.
    Requires Admin, Consultant, or Admission permission.
    """
    # Find the document in database
    document = db.query(entities.KnowledgeBaseDocument).filter(
        entities.KnowledgeBaseDocument.document_id == document_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if file exists on disk
    file_path = document.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    # Determine media type based on file extension
    file_extension = Path(file_path).suffix.lower()
    media_type_mapping = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif'
    }
    
    media_type = media_type_mapping.get(file_extension, 'application/octet-stream')
    
    # Return the file for viewing in browser
    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={"Content-Disposition": "inline"}  # This makes it open in browser instead of download
    )