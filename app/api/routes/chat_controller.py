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
#thêm 2 tầng check chat
@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    # session_id = 1
    # user_id = 1
    
    service = TrainingService()
    await websocket.accept()
    
    # 1️⃣ Nhận thông tin user và session trước
    data = await websocket.receive_json()
    user_id = data.get("user_id")
    session_id = data.get("session_id")
   
    if not user_id:
        await websocket.send_json({"event": "error", "message": "Missing user_id"})
        await websocket.close()
        return

    if not session_id:
        session_id = service.create_chat_session(user_id, "chatbot")
        await websocket.send_json({
            "event": "session_created",
            "session_id": session_id
        })

    # 2️⃣ Sau khi nhận xong → gửi lời chào
    greeting_chunks = [
        "Chào bạn! 👋 Mình là Chatbot tư vấn tuyển sinh của trường XYZ.",
        "Rất vui được đồng hành cùng bạn!\nMình có thể giúp bạn:",
        "\n\n1️⃣ Giới thiệu ngành học, chương trình đào tạo.",
        "\n\n2️⃣ Tư vấn lộ trình học tập và cơ hội nghề nghiệp.",
        "\n\n3️⃣ Cung cấp thông tin tuyển sinh, học bổng, ký túc xá.",
        "\n\nBạn muốn bắt đầu tìm hiểu về lĩnh vực nào trước? 😄"
    ]
    for chunk in greeting_chunks:
        await websocket.send_json({"event": "chunk", "content": chunk})
        await asyncio.sleep(0.05)

    await websocket.send_json({"event": "done", "sources": [], "confidence": 1.0})
 
    try:
        while True:
            # Nhận tin nhắn từ client
            raw_data = await websocket.receive_json()
            message = raw_data.get("message", "").strip()
            if not message:
                continue

             # enrich_query — tạo truy vấn "đầy đủ" dựa vào hội thoại cũ
            enriched_query = await service.enrich_query(session_id, message)
            print(f"👉 enriched_query: {enriched_query}")

             # Nếu enrich_query rỗng, nghĩa là user nói lan man → không cần RAG
            if not enriched_query:
                await websocket.send_json({
                    "event": "chunk",
                    "content": "Mình chưa rõ ý bạn lắm, bạn có thể nói rõ hơn được không?"
                })
                await websocket.send_json({"event": "done", "sources": [], "confidence": 0.0})
                continue


            # Tìm context liên quan
            # doc_results = TrainingService.search_documents(message, top_k=5)
           
            # Hybrid search (cả training QA và document)
            result = service.hybrid_search(enriched_query)
            tier_source = result.get("response_source")
            confidence = result.get("confidence", 0.0)

            # === TIER 1: training_qa - score > 0.8 ===
            if tier_source == "training_qa" and confidence > 0.8:
                print("floor 1")
                top = result["top_match"]
                q_text = top.payload.get("question_text")
                a_text = top.payload.get("answer_text")
                relevance_ok = await service.llm_relevance_check(enriched_query, q_text, a_text)

                if relevance_ok:
                    print("✅ floor 1: training QA valid")
                    async for chunk in service.stream_response_from_qa(enriched_query, a_text):
                        await websocket.send_text(json.dumps({
                            "event": "chunk",
                            "content": getattr(chunk, "content", str(chunk))
                        }))
                    await websocket.send_json({
                        "event": "done",
                        "sources": [q_text],
                        "confidence": confidence
                    })
                    continue
                else:
                    print("⚠️ QA not relevant → fallback xuống document")
                    # Chạy document search lại
                    doc_results = service.search_documents(enriched_query, top_k=5)
                    result = {
                        "response": doc_results,
                        "response_source": "document",
                        "confidence": doc_results[0].score if doc_results else 0.0,
                        "sources": [r.payload.get("document_id") for r in doc_results]
                    }
                    tier_source = "document"

            # === TIER 2: document-only (no QA match) ===
            if tier_source == "document" or confidence < 0.75:
                print("🔍 floor 3: using document context")
                context_chunks = result["response"]
                context = "\n\n".join([
                    r.payload.get("chunk_text", "") for r in context_chunks
                ])
                is_recommendation = await service.llm_recommendation_check(enriched_query, context)
                if is_recommendation:
                    async for chunk in service.stream_response_from_context(
                        enriched_query, context, session_id, user_id
                    ):
                        await websocket.send_text(json.dumps({
                            "event": "chunk",
                            "content": getattr(chunk, "content", str(chunk))
                        }))
                    # Gửi tín hiệu kết thúc khi hoàn tất
                    try:
                        await websocket.send_json({
                            "event": "done",
                            "sources": result.get("sources", []),
                            "confidence": confidence
                        })
                        continue
                    except Exception:
                        print("Không thể gửi event done vì client đã ngắt.")
                        break
                else: 
                    tier_source = "recommendation"

                # === TIER 3: recommedation ===
            if tier_source == "recommendation":
                print("🔍 floor 4: using recommendation layer")
                   
                async for chunk in service.stream_response_from_recommendation(
                    user_id, session_id, enriched_query
                ):
                    await websocket.send_text(json.dumps({
                        "event": "chunk",
                        "content": getattr(chunk, "content", str(chunk))
                    }))
                    # Gửi tín hiệu kết thúc khi hoàn tất
                try:
                    await websocket.send_json({
                        "event": "done",
                        "sources": result.get("sources", []),
                        "confidence": confidence
                    })
                    continue
                except Exception:
                    print("Không thể gửi event done vì client đã ngắt.")
                    break


            # 🧯 6️⃣ fallback cuối cùng
            await websocket.send_json({
                "event": "chunk",
                "content": "Xin lỗi, hiện tại mình chưa có thông tin chính xác cho câu hỏi này. \
Bạn vui lòng liên hệ với chuyên viên tư vấn để biết thêm thông tin chi tiết"
            })
            await websocket.send_json({
                "event": "done",
                "sources": [],
                "confidence": 0.0
            })
    except WebSocketDisconnect:
        # memory_manager.remove_memory(session_id)
        print("Client disconnected")


            











