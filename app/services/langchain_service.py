from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid

class LangChainService:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-pro",
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


    

langchain_service = LangChainService()
