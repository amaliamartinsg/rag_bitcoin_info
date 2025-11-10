from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

# Add routers
from api.router_langchain import router as langchain_rag_router


logger = logging.getLogger('uvicorn')
# Placeholder for lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models and other resources
    logger.info("Starting up...")
    yield
    # Clean up the ML models and other resources
    logger.info("Shutting down...")

app = FastAPI(
    title="RAG and Semantic Search API",
    description="An API for RAG and Semantic Search with LangChain.",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/")
async def read_root():
    return {"message": "Bienvenido a la API de Búsqueda Semántica y Recuperación de Respuestas (RAG)"}


app.include_router(langchain_rag_router, tags=["Bitcoin RAG"])
