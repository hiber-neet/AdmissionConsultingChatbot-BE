from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.core.config import settings
from app.core.qdrant_client import get_qdrant_client, initialize_collection
from typing import List, Tuple

class LangChainService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )
        self.qdrant_client = get_qdrant_client()
        initialize_collection(
            self.qdrant_client,
            settings.QDRANT_COLLECTION_NAME
        )
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=self.embeddings
        )

    def add_documents(self, texts: List[str], metadatas: List[dict] = None) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        documents = []
        for i, text in enumerate(texts):
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                metadata = metadatas[i] if metadatas else {}
                documents.append(Document(page_content=chunk, metadata=metadata))

        ids = self.vector_store.add_documents(documents)
        return ids

    def query(self, question: str, k: int = 4) -> Tuple[str, List[str]]:
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": question})

        answer = result["result"]
        sources = [doc.page_content[:100] + "..." for doc in result["source_documents"]]

        return answer, sources

    def chat(self, message: str) -> str:
        response = self.llm.invoke(message)
        return response.content

langchain_service = LangChainService()
