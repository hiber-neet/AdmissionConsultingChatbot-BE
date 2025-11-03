from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
import uuid
import asyncio
import json
from app.services.training_service import TrainingService
from pathlib import Path

router = APIRouter()
# @router.post("/chat")
# async def chat(
#         message: str
# ):
#     doc_results = langchain_service.search_documents(message, top_k=5)
#             # Build context từ documents
#     context = "\n\n".join([r.payload.get("chunk_text", "") for r in doc_results])
            
#             # Generate response using LLM
#     generated_response = langchain_service.generate_response_from_context(message, context)
            
#     return {
#             "response": generated_response,
#             "response_source": "document",
#             "confidence": doc_results[0].score,
#             "top_match": None,
#             "intent_id": None,
#             "sources": [r.payload.get("document_id") for r in doc_results]
#             }
#thêm 2 tầng check chat
@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    service = TrainingService()
    await websocket.accept()
    greeting_chunks = ["Chào bạn! 👋 Mình là Chatbot tư vấn tuyển sinh của trường XYZ.", " Rất vui được đồng hành cùng bạn!\nMình có thể giúp bạn:","\n\n 1️⃣ Giới thiệu các ngành học, chương trình đào tạo và đặc điểm nổi bật của từng ngành.\n\n", "2️⃣ Tư vấn lộ trình học tập, cơ hội nghề nghiệp và kỹ năng cần có cho từng ngành.\n\n", "3️⃣ Cung cấp thông tin tuyển sinh: điều kiện, hồ sơ, mốc thời gian quan trọng.\n\n", "4️⃣ Hướng dẫn tham gia các hoạt động trải nghiệm, câu lạc bộ, thực tập và học bổng.\n\n", "5️⃣ Giải đáp thắc mắc về cơ sở vật chất, ký túc xá, và các dịch vụ hỗ trợ sinh viên.\n\nBạn muốn bắt đầu tìm hiểu về lĩnh vực hay ngành học nào trước? 😄"]
    for chunk in greeting_chunks:
        await websocket.send_text(json.dumps({"event": "chunk", "content": chunk}))
        await asyncio.sleep(0.01)  # delay ngắn để client hiển thị mượt
    
    await websocket.send_text(json.dumps({"event": "done", "sources": [], "confidence": 1.0}))
    
    try:
        while True:
            # Nhận tin nhắn từ client
            data = await websocket.receive_text()
            message = json.loads(data).get("message", "")
        
            
            # Tìm context liên quan
            # doc_results = TrainingService.search_documents(message, top_k=5)
           
            result  = service.hybrid_search(message)
            
            tier_source = result.get("response_source")
            confidence = result.get("confidence", 0.0)

            # === TIER 1: training_qa - score > 0.8 ===
            if tier_source == "training_qa" and confidence > 0.8:
                print("floor 1")
                response_text = result["response_official_answer"]
                async for chunk in service.stream_response_from_qa(message, response_text):
                    # chunk có thể là str hoặc object tuỳ model → ép về text
                    content = getattr(chunk, "content", None) or str(chunk)
                    
                    # Gửi JSON để client dễ parse
                    await websocket.send_text(json.dumps({
                        "event": "chunk",
                        "content": content
                    }))

            # Gửi tín hiệu kết thúc khi hoàn tất
                try:
                    await websocket.send_text(json.dumps({
                        "event": "done",
                        "sources": result.get("sources", []),
                        "confidence": confidence
                    }))
                except Exception:
                    print("Không thể gửi event done vì client đã ngắt.")
                    break
                continue
            # === TIER 2: hybrid (0.7 < score <= 0.8) ===
            # elif result["response_source"] == "training_qa" and result["confidence"] > 0.7:
            #     print("floor 2")
            #     async for chunk in service.stream_response_from_hybrid(
            #         query=message,
            #         official_answer=result["response_official_answer"],
            #         additional_context=result.get("additional_context", "")
            #     ):
            #         await websocket.send_text(json.dumps({
            #             "event": "chunk",
            #             "content": getattr(chunk, "content", str(chunk))
            #         }))
            #     try:
            #         await websocket.send_text(json.dumps({
            #             "event": "done",
            #             "confidence": confidence
            #         }))
            #     except Exception:
            #         print("Không thể gửi event done vì client đã ngắt.")
            #         break
            # === TIER 3: document-only (no QA match) ===
            else:
                print("floor 3")
                context = "\n\n".join([r.payload.get("chunk_text", "") for r in result["response"]])               
                # Stream phản hồi từng phần
                async for chunk in service.stream_response_from_context(message, context):
                    # chunk có thể là str hoặc object tuỳ model → ép về text
                    content = getattr(chunk, "content", None) or str(chunk)
                    
                    # Gửi JSON để client dễ parse
                    await websocket.send_text(json.dumps({
                        "event": "chunk",
                        "content": content
                    }))

            # Gửi tín hiệu kết thúc khi hoàn tất
                try:
                    await websocket.send_text(json.dumps({
                        "event": "done",
                        "sources": result.get("sources", []),
                        "confidence": confidence
                    }))
                except Exception:
                    print("Không thể gửi event done vì client đã ngắt.")
                    break
    except WebSocketDisconnect:
        print("Client disconnected")


            











# router.post("/chat")
# async def chat(
#     message: str,
#     # current_user: Users = Depends(get_current_user),
#     # db: Session = Depends(get_db)
# ):
#     """
#     Main Chat Endpoint - Hybrid RAG Pipeline
    
#     FLOW:
#     1. Session Management: Get/create session
#     2. Hybrid Search: Training Q&A > Documents > Fallback
#     3. Intent Detection: Track conversation topics
#     4. Database Logging: Save interaction + stats
#     5. Response Return: With confidence & source info
    
#     Response Source Types:
#     - "training_qa": Direct answer từ consultant (confidence >= 0.7)
#     - "document": LLM-generated từ documents (confidence >= 0.6)
#     - "fallback": Generic answer (no match found)
    
#     Confidence Interpretation:
#     - >= 0.8: Rất tự tin, có thể show trực tiếp
#     - 0.7-0.8: Tự tin, nhưng nên offer live chat option
#     - 0.6-0.7: Bình thường, definitely show sources
#     - < 0.6: Không tự tin, suggest live chat
#     """
    
#     # STEP 1: SESSION MANAGEMENT
#     session_id = message.session_id or str(uuid.uuid4())
    
#     # Get or create session
#     # session_type = "chatbot" (not live chat)
#     chat_session = db.query(AdmissionOfficialChatSession).filter(
#         AdmissionOfficialChatSession.session_id == session_id
#     ).first()
    
#     if not chat_session:
#         chat_session = AdmissionOfficialChatSession(
#             session_id=session_id,
#             user_id=current_user.id,
#             session_type="chatbot"  # Start as chatbot, can convert to live
#         )
#         db.add(chat_session)
#         db.commit()
    
#     # STEP 2: HYBRID RAG SEARCH
#     # Orchestrator function gọi:
#     # - search_training_qa() → Tier 1
#     # - search_documents() → Tier 2-3
#     # - generate_response_from_context() → Tier 2-3
#     # Tự động fallback nếu không tìm được
#     # search_result = rag_service.hybrid_search(message.message)
#     search_result = langchain_service.hybrid_search(message.message)

#     response_text = search_result.get("response")
#     response_source = search_result.get("response_source")  # training_qa/document/fallback
#     confidence = search_result.get("confidence", 0.0)
#     intent_id = search_result.get("intent_id")
#     question_id = search_result.get("question_id")
#     sources = search_result.get("sources", [])
    
#     # STEP 3: SAVE INTERACTION
#     # Log mỗi interaction để:
#     # - Build conversation history
#     # - Analytics/improvement
#     # - User feedback tracking
#     interaction = ChatInteraction(
#         session_id=session_id,
#         user_id=current_user.id,
#         user_message=message.message,
#         bot_response=response_text,
#         intent_detected=None,  # Will fill from intent table if found
#         confidence_score=confidence,
#         response_source=response_source,
#         sources_used=json.dumps(sources),
#         training_qa_id=question_id
#     )
#     db.add(interaction)
    
#     # STEP 4: UPDATE FAQ STATISTICS
#     # Track: Câu hỏi nào được hỏi nhiều nhất?
#     # Dùng để identify missing Q&A, improve knowledge base
#     if intent_id:
#         # Get intent name từ database
#         intent_obj = db.query(Intent).filter(Intent.intent_id == intent_id).first()
#         if intent_obj:
#             interaction.intent_detected = intent_obj.name
            
#             # Update FAQ stats
#             faq_stat = db.query(FaqStatistics).filter(
#                 FaqStatistics.intent_id == intent_id
#             ).first()
            
#             if faq_stat:
#                 faq_stat.count += 1
#                 faq_stat.last_asked = datetime.utcnow()
#             else:
#                 faq_stat = FaqStatistics(
#                     intent_id=intent_id,
#                     question_text=message.message,
#                     count=1
#                 )
#                 db.add(faq_stat)
    
#     db.commit()
    
#     # STEP 5: RETURN RESPONSE
#     return ChatResponse(
#         response=response_text,
#         intent=interaction.intent_detected,
#         confidence=confidence,
#         response_source=response_source,
#         sources=[{"id": s, "type": "document"} for s in sources],
#         session_id=session_id
#     )