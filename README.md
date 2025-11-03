
# Proyecto: RAG y Búsqueda Semántica con Qdrant y LangChain

Este repositorio implementa un sistema de búsqueda semántica y generación aumentada por recuperación (RAG) usando Qdrant como base de datos vectorial y LangChain con modelos de Google GenAI. El objetivo es procesar documentos, generar embeddings, almacenarlos en Qdrant y exponer una API para consultas inteligentes.

## Estructura del proyecto

- `config/config.yaml`: Configuración de Qdrant (puertos, parámetros HNSW, etc).
- `docker-compose.yaml`: Levanta Qdrant en Docker.
- `requirements.txt`: Dependencias principales del proyecto.
- `data/`: Datos de entrada y salida, incluyendo fragmentos y resúmenes de PDFs.
- `qdrant_data/`: Persistencia de Qdrant (ignorada en git).
- `scripts/`: Scripts para procesamiento, indexación y generación de resúmenes.
  - `preprocessing.py`: Fragmenta y procesa documentos PDF, Word y TXT.
  - `create_langchain_index.py`: Crea el índice vectorial en Qdrant usando embeddings de Google GenAI.
  - `routing_generation.py`: Genera resúmenes de los documentos procesados.
- `src/`: Código fuente principal (API, servicios y procesos).
  - `app.py`, `main.py`: API FastAPI para consultas RAG y búsqueda semántica.
  - `api/`: Endpoints para RAG y búsqueda.
  - `services/`: Adaptadores para embeddings, LLMs y Qdrant.
  - `processes/`: Cadenas RAG, selección de fuentes y prompts.

## Instalación y requisitos

- Python 3.10+
- Docker y docker-compose

Instalación rápida (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

## Variables de entorno

Crea un archivo `.env` en la raíz con tu clave de Google GenAI:
```
GOOGLE_API_KEY=tu_clave_aqui
```

## Flujo de trabajo

1. **Preprocesar documentos:**
   ```powershell
   python scripts/preprocessing.py
   ```
   Esto fragmenta los documentos y genera archivos en `data/info/processed/` y resúmenes en JSON.

2. **Levantar Qdrant:**
   ```powershell
   docker-compose up -d
   ```

3. **Crear el índice vectorial:**
   ```powershell
   python scripts/create_langchain_index.py
   ```

4. **Generar resúmenes de los documentos:**
   ```powershell
   python scripts/routing_generation.py
   ```

5. **Levantar la API FastAPI:**
   ```powershell
   python src/main.py
   ```
   La API estará disponible en `http://localhost:8000`.

## Endpoints principales

- `POST /langchain/rag`: Consulta RAG sobre los documentos indexados.
- `POST /langchain/search`: Búsqueda semántica en el índice vectorial.

## Troubleshooting

- Si Qdrant no responde, verifica que el contenedor esté corriendo y los puertos estén libres.
- Si hay errores con Google GenAI, revisa la variable de entorno y la versión del paquete.

## Notas adicionales

- El directorio `qdrant_data/` debe estar en `.gitignore`.
- Puedes adaptar los scripts para otros tipos de documentos o modelos de embeddings.

## Próximos pasos sugeridos

- Añadir versiones fijas en `requirements.txt`.
- Incluir ejemplos end-to-end y notebooks de uso.


