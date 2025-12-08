import asyncio
import json
from fastapi import APIRouter, Request, WebSocket, Response
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from app.models.entities import ChatInteraction, LiveChatQueue
from app.services.livechat_service import LiveChatService
from app.models.database import SessionLocal

# Create a singleton instance of the LiveChatService
live_chat_service = LiveChatService()

router = APIRouter(prefix="/livechat", tags=["Live Chat"])

@router.post("/live-chat/join_queue")
async def join_queue(customer_id: int, official_id: int):
    return await live_chat_service.customer_join_queue(customer_id, official_id)

#xem trang thai 
@router.get("customer/queue/status/{customer_id}")
async def get_my_queue_status(customer_id: int):
    return live_chat_service.get_my_status(customer_id)


@router.delete("live-chat/queue/{queue_id}")
async def delete_queue(queue_id: int):
    return {"deleted": live_chat_service.delete_queue_item(queue_id)}

#xem tin nhan trong session live chat
@router.get("/session/{session_id}/messages")
async def get_messages(session_id: int):
    return live_chat_service.get_messages(session_id)

# @router.get("/sessions/user/{user_id}")
# async def list_sessions(user_id: int):
#     db = SessionLocal()
#     sessions = db.query(ParticipateChatSession)\
#         .filter_by(user_id=user_id)\
#         .all()
#     return sessions

# @router.get("/session/{session_id}")
# async def get_session(session_id: int):
#     db = SessionLocal()
#     return db.query(ChatSession).filter_by(chat_session_id=session_id).first()

#admission official xem danh sach cac customer co trong hang doi
@router.get("/admission_official/queue/list/{official_id}")
async def get_queue(official_id: int):
    return live_chat_service.get_queue_list(official_id)

#admission official xem danh sach cac session dang hoat dong
@router.get("/admission_official/active_sessions/{official_id}")
async def get_active_sessions(official_id: int):
    return await live_chat_service.get_active_sessions(official_id)

#admission offcial accept 1 queue(1 customer)
@router.post("/admission_official/accept")
async def accept_request(official_id: int, queue_id: int):
    return await live_chat_service.official_accept(official_id, queue_id)


@router.post("/admission_official/reject")
async def reject_request(official_id: int, queue_id: int, reason: str):
    return await live_chat_service.official_reject(official_id, queue_id, reason)

#ket thuc session live chat
@router.post("/live-chat/end")
async def end_session(session_id: int, ended_by: int):
    return await live_chat_service.end_session(session_id, ended_by)


#Server-Sent Events: sẽ trả về bên phía customer cho mấy cái như thông báo tin nhắn tới hay thông báo hàng đợi của mình được chấp nhận hay render theo thời gian thực
# { "event": "queue_updated",  "data": { "queue_id": 5 } }
# { "event": "accepted",  "data": { "queue_id": 2 } }
@router.get("/sse/customer/{customer_id}")
async def customer_sse(request: Request, customer_id: int):

    queue = asyncio.Queue()

    async def send_event(data: dict):
        await queue.put(data)

    # Đăng ký callback vào service
    live_chat_service.register_customer_sse(customer_id, send_event)

    async def event_stream():
        try:
            while True:
                data = await queue.get()

                yield {
                    "event": data.get("event", "update"),
                    "data": data
                }

                if await request.is_disconnected():
                    break

        finally:
            live_chat_service.unregister_customer_sse(customer_id, send_event)

    return EventSourceResponse(event_stream())

#Server-Sent Events: sẽ trả về cho bên phía admission official cho mấy cái như thông báo tin nhắn tới hay thông báo hàng đợi của mình được chấp nhận hay render theo thời gian thực
# Handle CORS preflight for SSE
@router.options("/sse/official/{official_id}")
async def sse_preflight(official_id: int):
    """Handle CORS preflight request for SSE endpoint"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "3600"
        }
    )

# { "event": "queue_updated",  "data": { "queue_id": 5 } }
# { "event": "accepted",  "data": { "queue_id": 2 } }
 
@router.get("/sse/official/{official_id}")
async def admission_official_sse(request: Request, official_id: int):
    
    queue = asyncio.Queue()

    async def send_event(data: dict):
        await queue.put(data)

    live_chat_service.register_official_sse(official_id, send_event)

    async def event_stream():
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'event': 'connected', 'message': 'SSE connection established'})}\n\n"
            
            while True:
                try:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break
                    
                    # Wait for data with timeout to allow disconnect checking
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Send the actual event
                    event_data = {
                        "event": data.get("event", "update"),
                        "data": data
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send periodic heartbeat to keep connection alive
                    yield f"data: {json.dumps({'event': 'ping', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
                    continue
                    
        except Exception as e:
            # Log error but don't expose details to client
            pass
        finally:
            live_chat_service.unregister_official_sse(official_id, send_event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Content-Type": "text/event-stream"
        }
    )


#live chat
@router.websocket("/chat/{session_id}")
async def chat_socket(websocket: WebSocket, session_id: int):
    await websocket.accept()
    await live_chat_service.join_chat(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_json()
            await live_chat_service.broadcast_message(
                session_id=session_id,
                sender_id=data["sender_id"],
                message=data["message"]
            )
    finally:
        # Always clean up the WebSocket connection when it ends
        await live_chat_service.leave_chat(websocket, session_id)
