import asyncio
from fastapi import APIRouter, Request, WebSocket
from sse_starlette.sse import EventSourceResponse
from app.models.entities import ChatInteraction, LiveChatQueue
from app.services.livechat_service import LiveChatService
from app.models.database import SessionLocal

router = APIRouter(prefix="/livechat", tags=["Live Chat"])

@router.post("/live-chat/join_queue")
async def join_queue(customer_id: int, official_id: int):
    return await LiveChatService.customer_join_queue(customer_id, official_id)

#xem trang thai 
@router.get("customer/queue/status/{customer_id}")
async def get_my_queue_status(customer_id: int):
    return LiveChatService.get_my_status(customer_id)


@router.delete("live-chat/queue/{queue_id}")
async def delete_queue(queue_id: int):
    return {"deleted": LiveChatService.delete_queue_item(queue_id)}

#xem tin nhan trong session live chat
@router.get("/session/{session_id}/messages")
async def get_messages(session_id: int):
    return LiveChatService.get_messages(session_id)

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
    return LiveChatService.get_queue_list(official_id)

#admission offcial accept 1 queue(1 customer)
@router.post("/admission_official/accept")
async def accept_request(official_id: int, queue_id: int):
    return await LiveChatService.official_accept(official_id, queue_id)


@router.post("/admission_official/reject")
async def reject_request(official_id: int, queue_id: int, reason: str):
    return await LiveChatService.official_reject(official_id, queue_id, reason)

#ket thuc session live chat
@router.post("/live-chat/end")
async def end_session(session_id: int, ended_by: int):
    return await LiveChatService.end_session(session_id, ended_by)


#Server-Sent Events: sẽ trả về bên phía customer cho mấy cái như thông báo tin nhắn tới hay thông báo hàng đợi của mình được chấp nhận hay render theo thời gian thực
# { "event": "queue_updated",  "data": { "queue_id": 5 } }
# { "event": "accepted",  "data": { "queue_id": 2 } }
@router.get("/sse/customer/{customer_id}")
async def customer_sse(request: Request, customer_id: int):

    queue = asyncio.Queue()

    async def send_event(data: dict):
        await queue.put(data)

    # Đăng ký callback vào service
    LiveChatService.register_customer_sse(customer_id, send_event)

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
            LiveChatService.unregister_customer_sse(customer_id, send_event)

    return EventSourceResponse(event_stream())

#Server-Sent Events: sẽ trả về cho bên phía admission official cho mấy cái như thông báo tin nhắn tới hay thông báo hàng đợi của mình được chấp nhận hay render theo thời gian thực
# { "event": "queue_updated",  "data": { "queue_id": 5 } }
# { "event": "accepted",  "data": { "queue_id": 2 } }
 
@router.get("/sse/official/{official_id}")
async def admission_official_sse(request: Request, official_id: int):

    queue = asyncio.Queue()

    async def send_event(data: dict):
        await queue.put(data)

    LiveChatService.register_official_sse(official_id, send_event)

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
            LiveChatService.unregister_official_sse(official_id, send_event)

    return EventSourceResponse(event_stream())


#live chat
@router.websocket("/chat/{session_id}")
async def chat_socket(websocket: WebSocket, session_id: int):
    await websocket.accept()
    await LiveChatService.join_chat(websocket, session_id)

    while True:
        data = await websocket.receive_json()
        await LiveChatService.broadcast_message(
            session_id=session_id,
            sender_id=data["sender_id"],
            message=data["message"]
        )
