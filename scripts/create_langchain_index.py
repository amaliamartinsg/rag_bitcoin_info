import os
import sys
from typing import List
from uuid import uuid4


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

from config.project_config import SETTINGS

qdrant_url = SETTINGS.qdrant_url
collection_name = SETTINGS.qdrant_collection
threshold = SETTINGS.threshold
qdrant_client = SETTINGS.qdrant_client
k_docs = SETTINGS.k_docs



def load_txt_chunks_from_processed(processed_dir, source_tag=None):
    documents = []
    for filename in os.listdir(processed_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(processed_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            base_name = os.path.splitext(filename)[0]
            metadata = {
                "source": source_tag or base_name,
                "filename": filename,
            }
            documents.append(Document(page_content=text, metadata=metadata))
    return documents


def load_price_documents(prices_dir, exclude_today=True):
    import datetime
    documents = []
    today = datetime.date.today().strftime('%Y-%m-%d')
    for filename in os.listdir(prices_dir):
        if exclude_today and today in filename:
            continue
        file_path = os.path.join(prices_dir, filename)
        if filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            metadata = {
                "source": "daily_price_json",
                "filename": filename,
            }
            documents.append(Document(page_content=content, metadata=metadata))
        elif filename.endswith('.pdf'):
            metadata = {
                "source": "daily_price_pdf",
                "filename": filename,
            }
            documents.append(Document(page_content=f"PDF_PATH:{file_path}", metadata=metadata))
    return documents

# --- Función reutilizable para indexar documentos ---
def index_documents(documents: List[Document], recreate_collection: bool = False):
    qdrant_client = SETTINGS.qdrant_client

    if recreate_collection:
        try:
            qdrant_client.get_collection(SETTINGS.qdrant_collection)
            qdrant_client.delete_collection(SETTINGS.qdrant_collection)
        except Exception:
            pass
        qdrant_client.create_collection(
            collection_name=SETTINGS.qdrant_collection,
            vectors_config=VectorParams(size=SETTINGS.vector_size, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=SETTINGS.qdrant_collection,
        embedding=SETTINGS.embeddings_model
    )

    for document in documents:
        doc_id = str(uuid4())
        vector_store.add_documents(documents=[document], ids=[doc_id])

    print(f"Subidos {len(documents)} documentos a Qdrant en la colección '{SETTINGS.qdrant_collection}'.")

# --- Ingesta inicial: reindexa todo ---


def ingest_initial_documents():
    info_processed = os.path.join(BASE_DIR, "data", "info", "processed")
    prices_dir = os.path.join(BASE_DIR, "data", "prices")

    docs_info = load_txt_chunks_from_processed(info_processed, source_tag="info")
    docs_prices = load_price_documents(prices_dir, exclude_today=True)

    all_docs = docs_info + docs_prices
    print(f"Documentos a indexar: {len(all_docs)}")
    index_documents(all_docs, recreate_collection=True)

# --- Ingesta incremental: agrega uno o varios documentos ---
def ingest_new_documents(new_docs: List[Document]):
    index_documents(new_docs, recreate_collection=False)


if __name__ == "__main__":
    # Por defecto, ejecuta la ingesta inicial
    ingest_initial_documents()