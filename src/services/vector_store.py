from langchain_qdrant import QdrantVectorStore
from src.services.embeddings import embeddings_model_langchain

### LANGCHAIN
qdrant_langchain = QdrantVectorStore.from_existing_collection(
    embedding=embeddings_model_langchain,
    collection_name="langchain_index",
    url="http://localhost:6333",
)
