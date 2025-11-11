import os
import sys
import json
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from langchain_core.documents import Document
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from config.project_config import SETTINGS
from src.services.vector_store import qdrant_langchain  # ajusta si tu path cambia

logger = logging.getLogger(__name__)

# === Helpers comunes =======================================================

def compute_doc_id(text: str, filename: str = "") -> str:
    """
    Calcula un identificador estable (hash) para el documento/chunk.
    Usa contenido + nombre de fichero para minimizar colisiones.
    """
    base = f"{filename}|{text}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _extract_date_from_filename(fname: str) -> str | None:
    """
    Intenta extraer una fecha YYYY-MM-DD del nombre del fichero.
    Devuelve None si no encuentra nada.
    """
    m = re.search(r"\d{4}-\d{2}-\d{2}", fname)
    if not m:
        return None
    return m.group(0)


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
        logger.warning(f"Directorio de procesados no existe: {processed_dir}")
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
                logger.error(f"Leyendo {full_path}: {e}")
                continue

            if not text.strip():
                continue

            doc_id = compute_doc_id(text=text, filename=fname)

            metadata = {
                "_collection_name": SETTINGS.qdrant_collection,
                "source": source_tag,
                "filename": fname,
                "path": os.path.relpath(full_path, BASE_DIR),
                "doc_id": doc_id,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }

            docs.append(Document(page_content=text, metadata=metadata))

    logger.info(f"Cargados {len(docs)} chunks .txt desde {processed_dir}")
    return docs


def _price_json_to_text(data: dict) -> str:
    """
    Convierte el JSON de precios a un texto razonable para el LLM.
    De momento genérico (pretty JSON).
    """
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
        logger.warning(f"Directorio de precios no existe: {prices_dir}")
        return docs

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for fname in os.listdir(prices_dir):
        if not fname.lower().endswith(".json"):
            continue

        if exclude_today and today_str in fname:
            # por ejemplo: prices_2025-11-11.json
            logger.info(f"Saltando fichero de hoy en históricos: {fname}")
            continue

        full_path = os.path.join(prices_dir, fname)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Leyendo JSON {full_path}: {e}")
            continue

        text = _price_json_to_text(data)
        if not text.strip():
            continue

        doc_id = compute_doc_id(text=text, filename=fname)
        data_date = _extract_date_from_filename(fname)

        metadata = {
            "_collection_name": SETTINGS.qdrant_collection,
            "source": source_tag,
            "filename": fname,
            "path": os.path.relpath(full_path, BASE_DIR),
            "doc_id": doc_id,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        if data_date:
            metadata["data_date"] = data_date  # útil para políticas de obsolescencia

        docs.append(Document(page_content=text, metadata=metadata))

    logger.info(f"Cargados {len(docs)} documentos de precios desde {prices_dir}")
    return docs


# === Indexación en Qdrant =================================================

def _upsert_by_doc_id(documents: List[Document]) -> None:
    """
    Elimina de Qdrant todos los puntos cuyo doc_id coincida con los
    documents que vamos a insertar, para evitar duplicados.
    Solo se usa cuando NO recreamos la colección.
    """
    client = SETTINGS.qdrant_client
    collection_name = SETTINGS.qdrant_collection

    doc_ids = {
        doc.metadata.get("doc_id")
        for doc in documents
        if doc.metadata.get("doc_id") is not None
    }

    if not doc_ids:
        return

    logger.info(f"Upsert por doc_id: {len(doc_ids)} documentos lógicos a reemplazar")

    for doc_id in doc_ids:
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
            logger.debug(f"Borrados puntos previos con doc_id={doc_id}")
        except Exception as e:
            logger.error(f"Error borrando por doc_id={doc_id}: {e}")


def index_documents(
    documents: List[Document],
    recreate_collection: bool = False,
) -> int:
    """
    Indexa una lista de Document en Qdrant.
    - recreate_collection=True: borra y recrea la colección antes de indexar.
    - recreate_collection=False: hace upsert por doc_id (evita duplicados).
    Devuelve el número de documentos indexados.
    """
    if not documents:
        logger.info("No hay documentos para indexar.")
        return 0

    client = SETTINGS.qdrant_client
    collection_name = SETTINGS.qdrant_collection

    # Aseguramos que todos tienen doc_id (por si vienen de otro sitio)
    for doc in documents:
        if "doc_id" not in doc.metadata:
            filename = doc.metadata.get("filename", "")
            doc.metadata["doc_id"] = compute_doc_id(doc.page_content, filename)

        if "ingested_at" not in doc.metadata:
            doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()

    if recreate_collection:
        # Calculamos la dimensión del embedding usando el valor configurado
        from src.services.embeddings import vector_size

        logger.info(
            f"Re-creando colección '{collection_name}' con tamaño de vector {vector_size}"
        )
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
)
    else:
        # Ingesta incremental: upsert por doc_id para evitar duplicados
        _upsert_by_doc_id(documents)

    # Añadimos los documentos usando el vector store de LangChain ya configurado
    logger.info(f"Indexando {len(documents)} documentos en '{collection_name}'...")
    qdrant_langchain.add_documents(documents)
    logger.info("Indexación completada.")

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
    logger.info(f"Documentos totales a indexar (inicial): {len(all_docs)}")

    index_documents(all_docs, recreate_collection=True)
    return len(all_docs)


# === Ingesta incremental (nuevos documentos) ==============================

def ingest_new_documents(new_docs: List[Document]) -> int:
    """
    Ingesta incremental: agrega uno o varios Document a la colección ya existente.
    NO recrea la colección.
    Hace upsert por doc_id para evitar duplicados.
    """
    return index_documents(new_docs, recreate_collection=False)


if __name__ == "__main__":
    # Por defecto, ejecuta la ingesta inicial desde la línea de comandos
    ingest_initial_documents()
