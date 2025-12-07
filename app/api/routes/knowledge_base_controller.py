from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, APIRouter, Form
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
import json
import uuid
from datetime import datetime

router = APIRouter()

def check_view_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user has permission to view training questions (Admin, Consultant, or Admission Official)"""
    if not current_user:
        raise HTTPException(status_code=403, detail="Not authenticated")

    try:
        user_perms_list = [p.permission_name.lower() for p in current_user.permissions] 
    except AttributeError:
        user_perms_list = [p.lower() for p in current_user.permissions]

    # Check for Admin, Consultant, or Admission Official permissions
    is_admin = "admin" in user_perms_list
    is_consultant = "consultant" in user_perms_list
    is_admission_official = any(
        "admission" in p or "admission official" in p 
        for p in user_perms_list
    )

    if not (is_admin or is_consultant or is_admission_official):
        raise HTTPException(
            status_code=403,
            detail="Admin, Consultant, or Admission Official permission required"
        )
    
    return current_user
@router.post("/upload/document")
async def upload_document(
    intent_id: int,
    file: UploadFile = File(...),
    title: str = Form(None),
    category: str = Form(None),
    current_user_id: int = Form(1),
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
    
    # STEP 4: SAVE FILE TO DISK
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = upload_dir / unique_filename
        
        # Save file to disk
        with open(file_path, "wb") as f:
            f.write(file_content)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # STEP 5: SAVE TO DATABASE
    try:
        document = entities.KnowledgeBaseDocument(
            title=title or file.filename,
            file_path=str(file_path),
            category=category or "general",
            created_by=current_user_id
        )
        db.add(document)
        db.commit()
        db.refresh(document)
    
    except Exception as e:
        db.rollback()
        # Clean up file if database save fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save document to database: {str(e)}")

    # STEP 6: CHUNK + EMBED + STORE IN QDRANT
    # STEP 6: CHUNK + EMBED + STORE IN QDRANT
    try:
        service = TrainingService()
        chunk_ids = service.add_document(
            current_user_id,
            content_text,
            intent_id,
            {
                
                "type": file.content_type,
                "filename": file.filename,
                "document_id": document.document_id
            }
        )
    
    except Exception as e:
        # Cleanup if chunking fails
        db.delete(document)
        db.commit()
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    # STEP 7: SAVE CHUNK REFERENCES (if needed)
    try:
        for i, chunk_id in enumerate(chunk_ids):
            chunk = entities.DocumentChunk(
                document_id=document.document_id,
                chunk_text=f"Chunk {i+1}",  # You might want to store actual chunk text
                embedding_vector=str(chunk_id)  # Store the Qdrant vector ID
            )
            db.add(chunk)
        
        db.commit()
    
    except Exception as e:
        db.rollback()
        # Note: We don't delete the document here as it's already useful
        print(f"Warning: Failed to save chunk references: {str(e)}")

    # SUCCESS
    return {
        "message": "Document uploaded and indexed successfully",
        "document_id": document.document_id,
        "filename": file.filename,
        "title": document.title,
        "file_type": Path(file.filename).suffix.lower(),
        "chunks_created": len(chunk_ids),
        "original_size_kb": round(len(file_content) / 1024, 2),
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