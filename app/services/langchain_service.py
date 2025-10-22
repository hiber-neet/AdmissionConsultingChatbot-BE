from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
import asyncio
class LangChainService:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.7
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=self.gemini_api_key
        )
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.training_qa_collection = "training_qa"
        self.documents_collection = "knowledge_base_documents"
        self._init_collections()

    def _init_collections(self):
        try:
            self.qdrant_client.create_collection(
                collection_name=self.training_qa_collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
        except:
            pass
            
        try:
            self.qdrant_client.create_collection(
                collection_name=self.documents_collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
        except:
            pass

    # def add_documents(self, texts: List[str], metadatas: List[dict] = None) -> List[str]:
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,
    #         chunk_overlap=200
    #     )

    #     documents = []
    #     for i, text in enumerate(texts):
    #         chunks = text_splitter.split_text(text)
    #         for chunk in chunks:
    #             metadata = metadatas[i] if metadatas else {}
    #             documents.append(Document(page_content=chunk, metadata=metadata))

    #     ids = self.vector_store.add_documents(documents)
    #     return ids

    async def stream_response_from_context(self, query: str, context: str):
        """Stream phản hồi từ Gemini, từng chunk một."""
        prompt = f"""Bạn là một chatbot tư vấn tuyển sinh chuyên nghiệp...
        === THÔNG TIN THAM KHẢO ===
        {context}
        === CÂU HỎI ===
        {query}
        === HƯỚNG DẪN ===
        - Trả lời bằng tiếng Việt
        - Dựa vào thông tin trên, không bịa đặt
        """

        async for chunk in self.llm.astream(prompt):
            yield chunk
            await asyncio.sleep(0)  # Nhường event loop

    def add_document(self, document_id: int, content: str, metadata: dict = None):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Size optimal cho Vietnamese
            chunk_overlap=200     # Overlap to preserve context
        )
        chunks = text_splitter.split_text(content)
        
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            # Embed chunk
            embedding = self.embeddings.embed_query(chunk)
            point_id = str(uuid.uuid4())
            
            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name="knowledge_base_documents",
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "document_id": document_id,
                            "chunk_index": i,
                            "chunk_text": chunk,
                            "metadata": metadata or {},
                            "type": "document"
                        }
                    )
                ]
            )
            chunk_ids.append(point_id)
        
        return chunk_ids
    def search_documents(self, query: str, top_k: int = 5):
        """
        Search documents (Fallback)
        
        Fallback path: Tìm document chunks khi training Q&A không match
        - Query → Embed → Search documents collection
        - Return top_k chunks
        - LLM sẽ synthesize answer từ chunks
        
        Args:
            query: User question
            top_k: Số chunks (lower score → fallback)
        
        Returns:
            List of document chunks
        """
        
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.qdrant_client.search(
            collection_name=self.documents_collection,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return results

    def generate_response_from_context(self, query: str, context: str):
        """
        Generate response từ document context (LLM-based)
        
        Dùng cho: Fallback path khi không có training Q&A match
        - Input: query + document chunks
        - Output: Natural language answer via Gemini
        - Risk: Có thể hallucinate nếu context không đủ
        - Mitigation: Confidence score thấp, suggest live chat
        
        Args:
            query: User question
            context: Document chunks concatenated
        
        Returns:
            Generated response
        """
        
        # Prompt engineering: Context + Query + Instructions
        prompt = f"""Bạn là một chatbot tư vấn tuyển sinh chuyên nghiệp. 
Dựa trên thông tin dưới đây, hãy trả lời câu hỏi của sinh viên:

=== THÔNG TIN THAM KHẢO ===
{context}

=== CÂU HỎI CỦA SINH VIÊN ===
{query}

=== HƯỚNG DẪN ===
- Trả lời bằng tiếng Việt
- Thân thiện, chuyên nghiệp
- Dựa vào thông tin được cung cấp
- Nếu không tìm thấy thông tin, hãy nói rõ và gợi ý liên hệ trực tiếp
- Không bịa thêm thông tin ngoài context

=== TRẢ LỜI ===
"""
        
        response = self.llm.invoke(prompt)
        return response

langchain_service = LangChainService()
