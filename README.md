# Proyecto VectorDB

## Descripción General
Este proyecto tiene como objetivo principal la creación de un sistema de recuperación de información basado en vectores para datos relacionados con el mercado de criptomonedas. Utiliza tecnologías como Qdrant para el almacenamiento vectorial, LangChain para la creación de cadenas de procesamiento, y Python para la implementación de servicios y scripts.

## Estructura del Proyecto
El proyecto está organizado de la siguiente manera:

```
Proyecto_vectordb/
├── config/
│   ├── config.yaml                # Configuración general del proyecto
│   ├── project_config.py          # Archivo Python para manejar configuraciones
├── data/
│   ├── info/
│   │   ├── processed/             # Archivos de texto procesados
│   │   ├── raw/                   # Datos crudos
│   ├── prices/
│   │   ├── daily_price_*.json     # Precios diarios de criptomonedas
│   │   ├── raw/                   # Datos crudos de precios
├── logs/                          # Archivos de logs
├── qdrant_config/
│   ├── config.yaml                # Configuración de Qdrant
├── qdrant_data/                   # Datos internos de Qdrant
├── scripts/
│   ├── create_langchain_index.py  # Script para crear índices con LangChain
│   ├── preprocessing.py           # Script para preprocesar datos
├── src/
│   ├── app.py                     # Archivo principal para ejecutar la aplicación
│   ├── main.py                    # Punto de entrada principal
│   ├── api/
│   │   ├── router_langchain.py    # Rutas para la API
│   │   ├── schema.py              # Esquema de datos para la API
│   ├── processes/
│   │   ├── langchain_chain/       # Procesos relacionados con LangChain
│   │   │   ├── chain.py           # Definición de cadenas
│   │   │   ├── prompts.py         # Prompts utilizados
│   │   │   ├── structures.py      # Estructuras de datos
│   ├── services/
│   │   ├── current_price_btc.py   # Servicio para obtener el precio actual de BTC
│   │   ├── embeddings.py          # Generación de embeddings
│   │   ├── llms.py                # Servicios relacionados con modelos de lenguaje
│   │   ├── vector_store.py        # Interacción con el almacenamiento vectorial
├── docker-compose.yaml            # Configuración para contenedores Docker
├── requirements.txt               # Dependencias del proyecto
├── README.md                      # Documentación del proyecto
```

## Requisitos Previos

1. **Python**: Asegúrate de tener Python 3.8 o superior instalado.
2. **Entorno Virtual**: Se recomienda usar un entorno virtual para manejar las dependencias.
3. **Docker**: Necesario para ejecutar Qdrant y otros servicios relacionados.

## Instalación y Configuración

1. Clona el repositorio:
   ```bash
   git clone https://github.com/amaliamartinsg/rag_bitcoin_info.git
   cd rag_bitcoin_info
   ```

2. Crea y activa un entorno virtual:
   ```bash
   python -m venv .venv_project
   .\.venv_project\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configura los servicios de Docker:
   ```bash
   docker-compose up -d
   ```

5. Verifica que los servicios estén corriendo correctamente:
   ```bash
   docker ps
   ```

## Ejecución

### Crear el Índice con LangChain
Ejecuta el script `create_langchain_index.py` para procesar los datos y crear el índice:
```bash
python scripts/create_langchain_index.py
```

### Iniciar la Aplicación
Ejecuta el archivo principal para iniciar la API:
```bash
python src/main.py
```

La API estará disponible en `http://localhost:8000`.

## Uso

### Endpoints Principales
- **/search**: Permite realizar búsquedas en el índice vectorial.
- **/current_price**: Devuelve el precio actual de Bitcoin.

### Ejemplo de Búsqueda
Puedes realizar una búsqueda utilizando herramientas como `curl` o Postman:
```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "¿Qué es Bitcoin?"}'
```

## Detalle del Funcionamiento del Proyecto

### 1. Procesamiento de Datos
El proyecto comienza con el procesamiento de datos crudos, que pueden incluir documentos en formato TXT, JSON u otros. Estos datos se encuentran en el directorio `data/info/raw/` y `data/prices/raw/`. El script `preprocessing.py` realiza las siguientes tareas:
- Fragmentación de documentos largos en partes más pequeñas.
- Limpieza de texto para eliminar caracteres innecesarios.
- Almacenamiento de los datos procesados en `data/info/processed/`.

### 2. Generación de Embeddings
Una vez que los datos están procesados, el script `create_langchain_index.py` utiliza modelos de lenguaje (LLMs) para generar embeddings vectoriales. Estos embeddings son representaciones matemáticas de los textos que capturan su significado semántico. Los pasos principales son:
- Cargar los datos procesados.
- Generar embeddings utilizando un modelo preentrenado (por ejemplo, Google GenAI).
- Almacenar los embeddings en Qdrant, una base de datos vectorial de alto rendimiento.

### 3. Almacenamiento Vectorial con Qdrant
Qdrant es el núcleo del sistema de recuperación de información. Los embeddings generados se almacenan en colecciones dentro de Qdrant. Cada colección representa un conjunto de datos relacionados, como definiciones de conceptos o precios históricos. Qdrant permite realizar búsquedas rápidas y precisas basadas en similitud vectorial.

### 4. API para Consultas
El proyecto incluye una API desarrollada con FastAPI que expone varios endpoints para interactuar con el sistema:
- **/search**: Permite realizar búsquedas semánticas en el índice vectorial. Por ejemplo, un usuario puede buscar "¿Qué es Bitcoin?" y obtener resultados relevantes.
- **/current_price**: Devuelve el precio actual de Bitcoin consultando datos en tiempo real.

### 5. Flujo de Consultas
Cuando un usuario realiza una consulta a través del endpoint `/search`, el flujo es el siguiente:
1. La consulta se convierte en un embedding utilizando el mismo modelo que se usó para indexar los datos.
2. Qdrant busca en su colección el embedding más cercano al de la consulta.
3. Se devuelven los resultados más relevantes al usuario.

### 6. Generación Aumentada por Recuperación (RAG)
El sistema también soporta RAG, donde los resultados de la búsqueda se utilizan como contexto para generar respuestas más completas y precisas. Esto se logra mediante cadenas de procesamiento definidas en `src/processes/langchain_chain/`.

### 7. Servicios Adicionales
El proyecto incluye varios servicios auxiliares:
- **current_price_btc.py**: Consulta el precio actual de Bitcoin desde una API externa.
- **embeddings.py**: Maneja la generación de embeddings.
- **llms.py**: Interactúa con modelos de lenguaje para tareas como generación de texto.
- **vector_store.py**: Proporciona una capa de abstracción para interactuar con Qdrant.

### 8. Configuración Personalizable
El archivo `config/config.yaml` permite personalizar varios aspectos del proyecto, como:
- Parámetros de Qdrant (puertos, configuración de índices).
- Rutas de entrada y salida de datos.
- Configuración de modelos de lenguaje.

### 9. Escalabilidad
Gracias al uso de Docker y Qdrant, el sistema es altamente escalable. Se pueden añadir más nodos de Qdrant para manejar mayores volúmenes de datos o consultas concurrentes.


---

¡Gracias por usar este proyecto! Si tienes alguna pregunta, no dudes en abrir un issue en el repositorio.


