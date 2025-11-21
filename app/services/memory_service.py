
from langchain_classic.memory import ConversationBufferMemory

class MemoryManager:
    def __init__(self):
        # in-memory map: session_id -> ConversationBufferMemory
        self._map = {}

    def get_memory(self, session_id: str):
        if session_id not in self._map:
            self._map[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")
        return self._map[session_id]

    def remove_memory(self, session_id: str):
        if session_id in self._map:
            del self._map[session_id]