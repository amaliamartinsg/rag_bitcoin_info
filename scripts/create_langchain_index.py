import os
import sys
import json
from datetime import datetime
from typing import List
from uuid import uuid4

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from langchain_core.documents import Document
from qdrant_client.models import Distance, VectorParams

from config.project_config import SETTINGS
from src.services.vector_store import qdrant_langchain  # ajusta a src.services... si lo necesitas


# === Helpers para cargar documentos ======================================

def load_txt_chunks_from_processed(
    processed_dir: str,
    source_tag: str = "info",
) -> List[Document]:
    """
    Lee todos los .txt en data/info/processed (o el dir que pases)
    y los convierte en Document para LangChain.
    """
    docs: List[Document] = []

    if not os.path.isdir(processed_dir):
        print(f"[WARN] Directorio de procesados no existe: {processed_dir}")
        return docs

    for root, _, files in os.walk(processed_dir):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue

            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"[ERROR] Leyendo {full_path}: {e}")
                continue

            if not text.strip():
                continue

            metadata = {
                "_id": str(uuid4()),
                "_collection_name": SETTINGS.qdrant_collection,
                "source": source_tag,
                "filename": fname,
                "path": os.path.relpath(full_path, BASE_DIR),
            }

            docs.append(Document(page_content=text, metadata=metadata))

    print(f"[INFO] Cargados {len(docs)} chunks .txt desde {processed_dir}")
    return docs


def _price_json_to_text(data: dict) -> str:
    """
    Convierte el JSON de precios a un texto razonable para el LLM.
    Si no conoces el esquema exacto, esta versión genérica funciona igualmente.
    """
    # Si tu JSON tiene un campo concreto (ej. "Time Series (Digital Currency Daily)")
    # aquí podrías formatearlo de forma más bonita.
    return json.dumps(data, ensure_ascii=False, indent=2)


def load_price_documents(
    prices_dir: str,
    exclude_today: bool = True,
    source_tag: str = "precios",
) -> List[Document]:
    """
    Carga los JSON de data/prices/ como Document.
    Por defecto excluye el fichero de hoy (para diferenciarlo del precio “en vivo”
    que ya usas en la cadena).
    """
    docs: List[Document] = []

    if not os.path.isdir(prices_dir):
        print(f"[WARN] Directorio de precios no existe: {prices_dir}")
        return docs

    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    for fname in os.listdir(prices_dir):
        if not fname.lower().endswith(".json"):
            continue

        if exclude_today and today_str in fname:
            # por ejemplo: prices_2025-11-11.json
            print(f"[INFO] Saltando fichero de hoy: {fname}")
            continue

        full_path = os.path.join(prices_dir, fname)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Leyendo JSON {full_path}: {e}")
            continue

        text = _price_json_to_text(data)
        if not text.strip():
            continue

        metadata = {
            "_id": str(uuid4()),
            "_collection_name": SETTINGS.qdrant_collection,
            "source": source_tag,
            "filename": fname,
            "path": os.path.relpath(full_path, BASE_DIR),
        }

        docs.append(Document(page_content=text, metadata=metadata))

    print(f"[INFO] Cargados {len(docs)} documentos de precios desde {prices_dir}")
    return docs


# === Indexación en Qdrant =================================================

def index_documents(documents: List[Document], recreate_collection: bool = False) -> int:
    """
    Indexa una lista de Document en Qdrant.
    - recreate_collection=True: borra y recrea la colección antes de indexar.
    Devuelve el número de documentos indexados.
    """
    if not documents:
        print("[INFO] No hay documentos para indexar.")
        return 0

    client = SETTINGS.qdrant_client
    collection_name = SETTINGS.qdrant_collection

    if recreate_collection:
        # Calculamos la dimensión del embedding usando el modelo ya configurado
        from src.services.embeddings import vector_size

        print(
            f"[INFO] Re-creando colección '{collection_name}' con tamaño de vector {vector_size}"
        )
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    # Añadimos los documentos usando el vector store de LangChain ya configurado
    print(f"[INFO] Indexando {len(documents)} documentos en '{collection_name}'...")
    qdrant_langchain.add_documents(documents)
    print("[INFO] Indexación completada.")

    return len(documents)


# === Ingesta inicial (glosario + históricos) ==============================

def ingest_initial_documents() -> int:
    """
    Ingesta inicial:
      - Chunks de info desde data/info/processed
      - JSON de precios desde data/prices
    Siempre fuerza recrear la colección.
    """
    info_processed = os.path.join(BASE_DIR, "data", "info", "processed")
    prices_dir = os.path.join(BASE_DIR, "data", "prices")

    docs_info = load_txt_chunks_from_processed(info_processed, source_tag="info")
    docs_prices = load_price_documents(prices_dir, exclude_today=True)

    all_docs = docs_info + docs_prices
    print(f"[INFO] Documentos totales a indexar: {len(all_docs)}")

    index_documents(all_docs, recreate_collection=True)
    return len(all_docs)


# === Ingesta incremental (nuevos documentos) ==============================

def ingest_new_documents(new_docs: List[Document]) -> int:
    """
    Ingesta incremental: agrega uno o varios Document a la colección ya existente.
    NO recrea la colección.
    """
    return index_documents(new_docs, recreate_collection=False)


if __name__ == "__main__":
    # Por defecto, ejecuta la ingesta inicial desde la línea de comandos
    ingest_initial_documents()
