from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
url=os.getenv("QDRANT_URL"),
api_key=os.getenv("QDRANT_API_KEY"),
embeddings=GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"  
        )
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120  # Tăng timeout lên 120 giây
)
vector_store=QdrantVectorStore(
    client=client,
    collection_name="embedding_data",
    embedding=embeddings
)
def retrieval_vdb(input,top_k=2,score_threshold=0.7):
    retriever=vector_store.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={'k':top_k, 'score_threshold':score_threshold}
    )
    docs=retriever.invoke(input)
    return docs

