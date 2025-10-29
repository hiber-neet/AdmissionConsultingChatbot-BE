from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, APIRouter
from sqlalchemy.orm import Session
from app.models.database import init_db, get_db
from app.services.training_service import TrainingService
from app.utils.document_processor import documentProcessor
from pathlib import Path
from sqlalchemy.orm import Session

router = APIRouter()
@router.post("/upload/document")
async def upload_document(
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
        chunk_ids = TrainingService.add_document(
            1,
            content_text,
            {
                "title": "policy",
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
async def upload_training_question(intent_id: int, question_text: str, answer_text: str, db: Session = Depends(get_db), current_user_id: int = 1):
    result = TrainingService.add_training_qa(
        db=db,
        intent_id=intent_id,
        question_text=question_text,
        answer_text=answer_text,
        created_by=current_user_id
    )
    return {"message": "Training Q&A added successfully", "result": result}