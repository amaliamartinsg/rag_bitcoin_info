from fastapi import APIRouter
from datetime import datetime

from api.schema import RAGRequest, QueryResponse, SourceInfo
from processes.langchain_chain.chain import rag_chain, get_sources_info
from services.vector_store import qdrant_langchain

from config.project_config import SETTINGS

router = APIRouter()

collection_name = SETTINGS.qdrant_collection
qdrant_client = SETTINGS.qdrant_client


@router.get("/stats")
async def get_stats():
    """
    Endpoint para obtener estadísticas de la base de datos vectorial.
    """
    try:
        stats = qdrant_client.get_collection(collection_name)
        return {
            "status": "ok",
            "collection_name": collection_name,
            "vectors_count": stats.vectors_count if stats.vectors_count is not None else stats.indexed_vectors_count,
            "points_count": stats.points_count,
            "segments_count": stats.segments_count,
            "optimizer_status": str(stats.optimizer_status.value) if hasattr(stats.optimizer_status, "value") else str(stats.optimizer_status),
            "collection_status": str(stats.status.value) if hasattr(stats.status, "value") else str(stats.status),
        }
    except Exception:
        return {
            "status": "error",
            "message": f"Error al obtener estadísticas"
        }
        
@router.get("/health")
async def health_check():
    """
    Endpoint to check the health of the API and database connection.
    """
    try:
        # Intentamos consultar la colección en Qdrant
        try:
            qdrant_client.get_collection(collection_name)
            db_status = True
        except Exception:
            db_status = False

        return {
            "status": "ok" if db_status else "error",
            "database": "connected" if db_status else "disconnected",
            "message": "API is healthy" if db_status else "API is running but cannot reach Qdrant"
        }
    except Exception as e:
        return {
            "status": "error",
            "database": "disconnected",
            "message": f"Health check failed: {str(e)}"
        }

@router.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: RAGRequest):
    """
    Endpoint para la consulta RAG utilizando LangChain.
    """
    k = request.k_docs if request.k_docs is not None else SETTINGS.k_docs
    threshold = request.threshold if request.threshold is not None else SETTINGS.threshold

    result = await rag_chain.ainvoke({
        "question": request.question,
        "k_docs": k,
        "threshold": threshold
    })

    sources = [
        SourceInfo(source=result["source"].selection, reason=result["source"].reason)
    ] if result.get('source') else []

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        timestamp=datetime.now()
    )

@router.post("/search")
async def search(request: RAGRequest):
    
    # si no se especifica, usar valores por defecto
    k = request.k_docs if request.k_docs is not None else SETTINGS.k_docs
    threshold = request.threshold if request.threshold is not None else SETTINGS.threshold

    # obtenemos la
    sources_info = get_sources_info(request.question, k=k, threshold=threshold)
    
    return sources_info

