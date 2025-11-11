from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import time

# Add routers
from api.router_langchain import router as langchain_rag_router

# app.py
import logging
from logging.handlers import RotatingFileHandler
import os

# === Configuración general ===
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rag.log")

# Rotación: 5 MB por fichero, hasta 5 ficheros antiguos guardados
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
)

# Handler para consola (por si quieres mantenerlo también)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
)

# Configuración del logger raíz del proyecto
logger = logging.getLogger("rag-bitcoin")  # logger raíz del proyecto
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


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

# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger = logging.getLogger("rag-bitcoin.api")

    start_time = time.time()
    response = await call_next(request)
    process_time = (start_time - time.time()) * -1

    logger.info(
        f"Request {request.method} {request.url.path} "
        f"status={response.status_code} duration_ms={process_time*1000:.2f}"
    )

    return response


app.include_router(langchain_rag_router, tags=["Bitcoin RAG"])