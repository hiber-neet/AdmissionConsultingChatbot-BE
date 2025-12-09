from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, APIRouter, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
import os
import json
import uuid
from datetime import datetime

from app.models.database import init_db, get_db
from app.models.schemas import TrainingQuestionRequest, TrainingQuestionResponse, KnowledgeBaseDocumentResponse
from app.models import entities
from app.services.training_service import TrainingService
from app.utils.document_processor import documentProcessor
from app.core.security import get_current_user, has_permission

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

def get_document_or_404(document_id: int, db: Session) -> entities.KnowledgeBaseDocument:
    """Helper function to get document by ID or raise 404"""
    document = db.query(entities.KnowledgeBaseDocument).filter(
        entities.KnowledgeBaseDocument.document_id == document_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document

def resolve_file_path(relative_path: str) -> Path:
    """
    Resolve file path relative to project root, ensuring compatibility across machines.
    Handles both absolute and relative paths stored in database.
    """
    path = Path(relative_path)
    
    # If it's already an absolute path, extract just the relative part from 'uploads/'
    if path.is_absolute():
        # Find 'uploads' in the path and take everything from there
        parts = path.parts
        try:
            uploads_index = parts.index('uploads')
            relative_path = str(Path(*parts[uploads_index:]))
            path = Path(relative_path)
        except ValueError:
            # 'uploads' not in path, use as-is
            pass
    
    # If path is not absolute, resolve it relative to current working directory
    if not path.is_absolute():
        path = Path.cwd() / path
    
    return path

def check_file_exists(file_path: str) -> Path:
    """
    Helper function to check if file exists on disk or raise 404.
    Returns the resolved absolute path.
    """
    resolved_path = resolve_file_path(file_path)
    if not resolved_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"File not found on server. Looking for: {resolved_path}"
        )
    return resolved_path
@router.post("/approve/training_question/{qa_id}")
def api_approve_training_qa(
    qa_id: int,
    db: Session = Depends(get_db),
    reviewer_id: int = 1
):
    service = TrainingService()

    result = service.approve_training_qa(
        db=db,
        qa_id=qa_id,
        reviewer_id=reviewer_id
    )

    return {
        "message": "Training QA approved",
        **result
    }
@router.post("/upload/training_question")
def api_create_training_qa(
    payload: TrainingQuestionRequest,
    db: Session = Depends(get_db),
    current_user_id: int = 1
):
    service = TrainingService()

    qa = service.create_training_qa(
        db=db,
        intent_id=payload.intent_id,
        question=payload.question,
        answer=payload.answer,
        created_by=current_user_id
    )

    return {
        "message": "Training QA created as draft",
        "qa_id": qa.question_id,
        "status": qa.status
    }
@router.post("/upload/document")
async def upload_document(
    intent_id: int,
    file: UploadFile = File(...),
    title: str = Form(None),
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

        if len(file_content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # STEP 3: EXTRACT TEXT
    try:
        extracted_text = documentProcessor.extract_text(
            file_content,
            file.filename,
            file.content_type
        )
        if not extracted_text:
            raise HTTPException(
                status_code=422,
                detail="Cannot extract content from the file"
            )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Extract error: {str(e)}")

    # STEP 4: SAVE FILE TO DISK
    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = upload_dir / unique_filename
        with open(file_path, "wb") as f:
            f.write(file_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # STEP 5: SAVE DATABASE ONLY (NO QDRANT)
    try:
        service = TrainingService()

        doc = service.create_document(
            db=db,
            title=title or file.filename,
            file_path=str(file_path),       # <-- file text chứ không phải file gốc
            intend_id=intent_id,
            created_by=current_user_id
        )

        # save extracted text for approval stage
        temp_store_path = f"uploads/temp_text_{doc.document_id}.txt"
        with open(temp_store_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

    return {
        "message": "Document uploaded as draft. Waiting for approval.",
        "document_id": doc.document_id,
        "intend_id": doc.intend_id,
        "status": doc.status
    }
@router.post("/document/approve/{document_id}")
def api_approve_document(
    document_id: int,
    intent_id: int,
    db: Session = Depends(get_db),
    reviewer_id: int = 1
):
    service = TrainingService()

    result = service.approve_document(
        db=db,
        document_id=document_id,
        reviewer_id=reviewer_id,
        intent_id=intent_id
    )

    return {
        "message": "Document approved and indexed",
        "document_id": result.get("document_id"),
        "status": result.get("status")
    }
# @router.post("/upload/document")
# async def upload_document(
#     intent_id: int,
#     file: UploadFile = File(...),
#     title: str = Form(None),
#     category: str = Form(None),
#     current_user_id: int = Form(1),
#     db: Session = Depends(get_db)
# ):
#     # STEP 1: VALIDATE FILE
#     try:
#         is_valid, error_msg = documentProcessor.validate_file(file.filename, file.content_type)
#         if not is_valid:
#             raise HTTPException(status_code=400, detail=error_msg)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
    
#     # STEP 2: READ FILE
#     try:
#         file_content = await file.read()
        
#         # Check file size (max 50MB)
#         max_size = 50 * 1024 * 1024
#         if len(file_content) > max_size:
#             raise HTTPException(
#                 status_code=413,
#                 detail=f"File too large. Max size: {max_size / 1024 / 1024}MB"
#             )
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
#     # STEP 3: EXTRACT TEXT FROM DOCUMENT
#     try:
#         content_text = documentProcessor.extract_text(
#             file_content,
#             file.filename,
#             file.content_type
#         )
        
#         if not content_text or len(content_text.strip()) == 0:
#             raise HTTPException(
#                 status_code=422,
#                 detail="Could not extract text from document. File may be empty or corrupted."
#             )
    
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=422, detail=f"Document processing failed: {str(e)}")
    
#     # STEP 4: SAVE FILE TO DISK
#     try:
#         # Create uploads directory if it doesn't exist
#         upload_dir = Path("uploads")
#         upload_dir.mkdir(exist_ok=True)
        
#         # Generate unique filename to avoid conflicts
#         unique_filename = f"{uuid.uuid4()}_{file.filename}"
#         file_path = upload_dir / unique_filename
        
#         # Save file to disk
#         with open(file_path, "wb") as f:
#             f.write(file_content)
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

#     # STEP 5: SAVE TO DATABASE
#     try:
#         document = entities.KnowledgeBaseDocument(
#             title=title or file.filename,
#             file_path=str(file_path),
#             category=category or "general",
#             created_by=current_user_id,
#             status='draft'  # New documents start as draft, need review
#         )
#         db.add(document)
#         db.commit()
#         db.refresh(document)
    
#     except Exception as e:
#         db.rollback()
#         # Clean up file if database save fails
#         if file_path.exists():
#             file_path.unlink()
#         raise HTTPException(status_code=500, detail=f"Failed to save document to database: {str(e)}")

#     # STEP 6: CHUNK + EMBED + STORE IN QDRANT
#     try:
#         service = TrainingService()
#         chunk_ids = service.add_document(
#             current_user_id,
#             content_text,
#             intent_id,
#             {
#                 "type": file.content_type,
#                 "filename": file.filename,
#                 "document_id": document.document_id
#             }
#         )
    
#     except Exception as e:
#         # Cleanup if chunking fails
#         db.delete(document)
#         db.commit()
#         if file_path.exists():
#             file_path.unlink()
#         raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

#     # STEP 7: SAVE CHUNK REFERENCES (if needed)
#     try:
#         for i, chunk_id in enumerate(chunk_ids):
#             chunk = entities.DocumentChunk(
#                 document_id=document.document_id,
#                 chunk_text=f"Chunk {i+1}",  # You might want to store actual chunk text
#                 embedding_vector=str(chunk_id)  # Store the Qdrant vector ID
#             )
#             db.add(chunk)
        
#         db.commit()
    
#     except Exception as e:
#         db.rollback()
#         # Note: We don't delete the document here as it's already useful
#         print(f"Warning: Failed to save chunk references: {str(e)}")

#     # SUCCESS
#     return {
#         "message": "Document uploaded and indexed successfully",
#         "document_id": document.document_id,
#         "filename": file.filename,
#         "title": document.title,
#         "file_type": Path(file.filename).suffix.lower(),
#         "chunks_created": len(chunk_ids),
#         "original_size_kb": round(len(file_content) / 1024, 2),
#         "extracted_text_length": len(content_text)
#     }

# @router.post("/upload/training_question")
# async def upload_training_question(payload: TrainingQuestionRequest, db: Session = Depends(get_db), current_user_id: int = 1):
#     service = TrainingService()
#     result = service.add_training_qa(
#         db=db,
#         intent_id=payload.intent_id,
#         question_text=payload.question,
#         answer_text=payload.answer,
        
#     )
#     return {"message": "Training Q&A added successfully", "result": result}

@router.get("/training_questions", response_model=List[TrainingQuestionResponse])
def get_all_training_questions(
    status: Optional[str] = Query(None, description="Filter by status: draft, approved, rejected, deleted"),
    db: Session = Depends(get_db), 
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Get all training questions in the system.
    Requires Admin, Consultant, or Admission permission.
    
    - Regular users (Consultant) only see approved questions
    - Consultant Leader and Admin can see all statuses
    - Use ?status= query parameter to filter by specific status
    """
    # Check if user is admin or consultant leader
    is_admin = current_user.role in ['Admin', 'ConsultantLeader']
    
    # Build query
    query = db.query(entities.TrainingQuestionAnswer)
    
    # If not admin/leader, only show approved questions
    if not is_admin:
        query = query.filter(entities.TrainingQuestionAnswer.status == 'approved')
    # If admin/leader and status filter provided, apply it
    elif status:
        query = query.filter(entities.TrainingQuestionAnswer.status == status)
    
    training_questions = query.all()
    
    # Convert to response format
    result = []
    for tqa in training_questions:
        result.append({
            "question_id": tqa.question_id,
            "question": tqa.question,
            "answer": tqa.answer,
            "intent_id": tqa.intent_id,
            "status": tqa.status,
            "created_at": tqa.created_at,
            "approved_at": tqa.approved_at,
            "created_by": tqa.created_by,
            "approved_by": tqa.approved_by
        })
    
    return result

@router.get("/documents", response_model=List[KnowledgeBaseDocumentResponse])
def get_all_documents(
    status: Optional[str] = Query(None, description="Filter by status: draft, approved, rejected, deleted"),
    db: Session = Depends(get_db), 
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Get all documents in the knowledge base.
    Requires Admin, Consultant, or Admission permission.
    
    - Regular users (Consultant) only see approved documents
    - Consultant Leader and Admin can see all statuses
    - Use ?status= query parameter to filter by specific status
    """
    # Check if user is admin or consultant leader
    is_admin = current_user.role in ['Admin', 'ConsultantLeader']
    
    # Build query
    query = db.query(entities.KnowledgeBaseDocument)
    
    # If not admin/leader, only show approved documents
    if not is_admin:
        query = query.filter(entities.KnowledgeBaseDocument.status == 'approved')
    # If admin/leader and status filter provided, apply it
    elif status:
        query = query.filter(entities.KnowledgeBaseDocument.status == status)
    
    documents = query.all()
    
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
            "created_by": doc.created_by,
            "status": doc.status,
            "reviewed_by": doc.reviewed_by,
            "reviewed_at": doc.reviewed_at
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
    document = get_document_or_404(document_id, db)
    resolved_path = check_file_exists(document.file_path)
    
    return FileResponse(
        path=str(resolved_path),
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
    document = get_document_or_404(document_id, db)
    check_file_exists(document.file_path)
    
    # Determine media type based on file extension
    file_extension = Path(document.file_path).suffix.lower()
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
    
    return FileResponse(
        path=document.file_path,
        media_type=media_type,
        headers={"Content-Disposition": "inline"}
    )

@router.get("/documents/{document_id}", response_model=KnowledgeBaseDocumentResponse)
def get_document_by_id(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Get a specific document's metadata by its ID.
    Requires Admin, Consultant, or Admission permission.
    """
    document = get_document_or_404(document_id, db)
    
    return {
        "document_id": document.document_id,
        "title": document.title,
        "file_path": document.file_path,
        "category": document.category,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "created_by": document.created_by,
        "status": document.status,
        "reviewed_by": document.reviewed_by,
        "reviewed_at": document.reviewed_at
    }


# ==================== REVIEW WORKFLOW ENDPOINTS ====================

def check_leader_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user is Admin or ConsultantLeader"""
    if current_user.role not in ['Admin', 'ConsultantLeader']:
        raise HTTPException(status_code=403, detail="Only Admin or Consultant Leader can review content")
    return current_user


@router.get("/documents/pending-review", response_model=List[KnowledgeBaseDocumentResponse])
def get_pending_documents(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Get all documents pending review (status=draft).
    Only Admin or ConsultantLeader can access this endpoint.
    """
    documents = db.query(entities.KnowledgeBaseDocument).filter(
        entities.KnowledgeBaseDocument.status == 'draft'
    ).all()
    
    return documents


@router.post("/documents/{document_id}/submit-review")
def submit_document_for_review(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Submit a document for review by changing status to 'draft'.
    Any consultant can submit their own documents for review.
    """
    document = get_document_or_404(document_id, db)
    
    # Check if user owns this document
    if document.created_by != current_user.user_id and current_user.role not in ['Admin', 'ConsultantLeader']:
        raise HTTPException(status_code=403, detail="You can only submit your own documents for review")
    
    document.status = 'draft'
    db.commit()
    
    return {"message": "Document submitted for review successfully", "document_id": document_id}


@router.post("/documents/{document_id}/approve")
def approve_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Approve a document for production use.
    Only Admin or ConsultantLeader can approve documents.
    """
    document = get_document_or_404(document_id, db)
    
    document.status = 'approved'
    document.reviewed_by = current_user.user_id
    document.reviewed_at = datetime.now().date()
    db.commit()
    
    return {
        "message": "Document approved successfully",
        "document_id": document_id,
        "reviewed_by": current_user.user_id
    }


@router.post("/documents/{document_id}/reject")
def reject_document(
    document_id: int,
    reason: str = Form(..., description="Reason for rejection"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Reject a document with a reason.
    Only Admin or ConsultantLeader can reject documents.
    """
    document = get_document_or_404(document_id, db)
    
    document.status = 'rejected'
    document.reviewed_by = current_user.user_id
    document.reviewed_at = datetime.now().date()
    db.commit()
    
    # TODO: Consider adding a rejection_reason field to the entity or notification system
    
    return {
        "message": "Document rejected",
        "document_id": document_id,
        "reason": reason,
        "reviewed_by": current_user.user_id
    }


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Soft delete a document by setting status to 'deleted'.
    Only Admin or ConsultantLeader can delete documents.
    """
    document = get_document_or_404(document_id, db)
    
    document.status = 'deleted'
    db.commit()
    
    return {"message": "Document deleted successfully", "document_id": document_id}


# ==================== TRAINING Q&A REVIEW WORKFLOW ====================

def get_training_qa_or_404(question_id: int, db: Session) -> entities.TrainingQuestionAnswer:
    """Helper function to get training Q&A or raise 404"""
    qa = db.query(entities.TrainingQuestionAnswer).filter(
        entities.TrainingQuestionAnswer.question_id == question_id
    ).first()
    
    if not qa:
        raise HTTPException(status_code=404, detail=f"Training Q&A with id {question_id} not found")
    
    return qa


@router.get("/training_questions/pending-review", response_model=List[TrainingQuestionResponse])
def get_pending_training_questions(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Get all training Q&A pending review (status=draft).
    Only Admin or ConsultantLeader can access this endpoint.
    """
    questions = db.query(entities.TrainingQuestionAnswer).filter(
        entities.TrainingQuestionAnswer.status == 'draft'
    ).all()
    
    return questions


@router.post("/training_questions/{question_id}/submit-review")
def submit_training_qa_for_review(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Submit a training Q&A for review by changing status to 'draft'.
    Any consultant can submit their own Q&A for review.
    """
    qa = get_training_qa_or_404(question_id, db)
    
    # Check if user owns this Q&A
    if qa.created_by != current_user.user_id and current_user.role not in ['Admin', 'ConsultantLeader']:
        raise HTTPException(status_code=403, detail="You can only submit your own Q&A for review")
    
    qa.status = 'draft'
    db.commit()
    
    return {"message": "Training Q&A submitted for review successfully", "question_id": question_id}


@router.post("/training_questions/{question_id}/approve")
def approve_training_qa(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Approve a training Q&A for use in chatbot training.
    Only Admin or ConsultantLeader can approve Q&A.
    """
    qa = get_training_qa_or_404(question_id, db)
    
    qa.status = 'approved'
    qa.approved_by = current_user.user_id
    qa.approved_at = datetime.now().date()
    db.commit()
    
    return {
        "message": "Training Q&A approved successfully",
        "question_id": question_id,
        "approved_by": current_user.user_id
    }


@router.post("/training_questions/{question_id}/reject")
def reject_training_qa(
    question_id: int,
    reason: str = Form(..., description="Reason for rejection"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Reject a training Q&A with a reason.
    Only Admin or ConsultantLeader can reject Q&A.
    """
    qa = get_training_qa_or_404(question_id, db)
    
    qa.status = 'rejected'
    qa.approved_by = current_user.user_id
    qa.approved_at = datetime.now().date()
    db.commit()
    
    # TODO: Consider adding a rejection_reason field or notification system
    
    return {
        "message": "Training Q&A rejected",
        "question_id": question_id,
        "reason": reason,
        "rejected_by": current_user.user_id
    }


@router.delete("/training_questions/{question_id}")
def delete_training_qa(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Soft delete a training Q&A by setting status to 'deleted'.
    Only Admin or ConsultantLeader can delete Q&A.
    """
    qa = get_training_qa_or_404(question_id, db)
    
    qa.status = 'deleted'
    db.commit()
    
    return {"message": "Training Q&A deleted successfully", "question_id": question_id}