### 0. Contexto del proyecto

El sistema RAG que he diseñado está orientado a responder preguntas sobre Bitcoin (precio actual, evolución histórica, conceptos financieros y técnicos asociados, etc.).

El flujo está montado para poder clasificar cada consulta entrante en dos categorías posibles. Esta clasificación se realiza antes de buscar en la base de datos vectorial, y determina **qué cadena de procesamiento se ejecuta** para obtener la respuesta final.

  **precio_actual** 
    - Pretende ofrecer información numérica actualizada sobre el valor de Bitcoin.
    - El sistema **NO consulta la base vectorial (Qdrant)**, ya que esta categoría corresponde a información dinámica, sino que se llama al servicio `current_price_btc.py`, que obtiene el último precio disponible desde la API de Alpha Vantage. 
    - La información se transforma en texto legible parseando los datos que vienen en el JSON obtenido y ese texto se pasa al modelo generativo como contexto y se genera una respuesta natural para el usuario.


  **informacion_conceptos**
    - El sistema **sí consulta la base vectorial (Qdrant)**.
    - Se convierte la pregunta en embedding y se realiza una búsqueda semántica sobre los documentos indexados (`data/info/processed` y `data/prices`).
    - Se seleccionan los fragmentos más relevantes (*chunks*) mediante un umbral de similitud (threshold).
    - El texto recuperado se combina y se envía al modelo de lenguaje junto con la pregunta para generar una respuesta final basada en ese contexto.


La información se actualiza con frecuencia (el precio cambia constantemente), y se trabaja principalmente con **texto estructurado y semiestructurado** (JSON de precios, documentos de texto, PDFs).

Por tanto, necesitaba un modelo de *embedding* que cumpla:

- Buena **calidad semántica** para preguntas en lenguaje natural sobre finanzas/cripto.
- **Latencia y coste razonables**, porque la ingesta puede ser frecuente.
- Soporte **multilingüe**, especialmente para preguntas en español, pero documentación e información en inglés.
- Integración sencilla con LangChain y Qdrant.

---

### 1. Selección y justificación del modelo de *embedding*

#### 1.1. Modelo seleccionado

He optado por utilizar el modelo *text-embedding-3-small* de OpenAI porque es en el que en otros proyectos me ha dado mejores resultados.

Este modelo se emplea tanto para:

- Indexar los documentos (glosario, informes, históricos de precios).
- Generar el vector de la consulta del usuario antes de buscar en Qdrant.

El modelo está entrenado para **búsqueda semántica y clasificación de texto**, lo que permite tener un buen desempeño con uso en un RAG.

Al usar un modelo generalista y estable, tan ampliamente utilizado, reduce el riesgo de tener que re-entrenar embeddings cuando cambie el tipo de preguntas.

---

#### 1.2. Justificación de la elección

**a) Tamaño y coste**

- `text-embedding-3-small` genera vectores de tamaño **1536 dimensiones**, lo que ofrece un buen equilibrio entre:
  - Capacidad de representar matices semánticos complejos.
  - Tamaño razonable para almacenamiento y búsquedas rápidas en la base vectorial.
- La versión “large”, por supuesto, daba muy buen rendimiento, pero en el proyecto actual la versión reducida cumplía perfectamente. Al ser la versión “small”, el coste por llamada y el tiempo de cálculo son menores que en modelos más grandes, algo importante ya que la ingesta es recurrente y el proyecto está pensado para poder escalarse fácilmente, sin temer que el coste se dispare.

**b) Soporte multilingüe**

- Aunque la información base sobre Bitcoin pueda estar parcialmente en inglés (pendiente implementar nuevas fuentes oficiales -- que estarían en inglés), las consultas de los usuarios se esperan en español. Permite así poder montarlo todo en un único pipeline sin importar el idioma de la fuente o de la pregunta del usuario.

**c) Adecuación a la naturaleza dinámica de la información**

- Es lo bastante ligero como para recalcular embeddings sin problema en cada nueva ingesta.
- Es lo bastante expresivo como para capturar relaciones entre conceptos técnicos (blockchain, hash, halving, volatilidad, etc.) y financieros (precio, volumen, capitalización).

**d) No necesitaba un modelo multimodal**

- En este diseño concreto no he optado por un modelo mulimodal porque:

    - Toda la información que gestiono y consulto (precios, descripciones, glosario, informes) es texto o datos estructurados convertidos a texto.
    - Queda pendiente si en el futuro se integrasen nuevas fuentes de documentación como pueden ser:
        . Imágenes de gráficos.
        . Vídeos.
        . Capturas de pantalla de velas o dashboards.

- Por tanto, al no tenerlo implementado acutalmente, no tiene sentido la complejidad adicional frente al valor real que aporta al proyecto. El modelo elegido simplifica la arquitectura, reduce coste y hace más sencillo el mantenimiento.

---

#### 1.3. Fine-tuning del modelo de embedding

**¿Consideraría hacer fine-tuning del modelo de embedding?**

No considero necesario hacer fine-tuning del modelo de embedding. Los motivos son:

Los modelos de embedding generalistas que he probado ya implementan un buen rendimiento en textos técnicos y financieros.


---
---


### 2. Diseño y justificación de la base de datos vectorial

#### 2.1. Elección de la base de datos vectorial

**BD vectorial elegida:** Qdrant

**Justificación:**

- **Escalabilidad y rendimiento:** Qdrant está pensado para producción, con soporte nativo para índices aproximados (HNSW) y filtros por metadatos, lo que permite crecer en volumen de documentos sin rediseñar el sistema.
- **Velocidad de búsqueda:** ofrece búsquedas muy rápidas sobre vectores de alta dimensión, incluso con filtros (por ejemplo, por fecha o tipo de documento).
- **Facilidad de uso:** integración directa con LangChain, cliente Python sencillo y API clara para crear colecciones, insertar, borrar y filtrar.
- **Coste y despliegue:** puede ejecutarse como servicio Docker propio (sin coste por uso adicional), lo que encaja bien con el enfoque del proyecto.
- **Actualización de información:** permite borrar e insertar puntos fácilmente a partir de metadatos (`doc_id`, `data_date`), algo clave para evitar duplicados y gestionar datos dinámicos.
- **Usada en otros proyectos:** es la BBDD vectorial que hemos visto durante el máster y la que hemos usado en otros proyectos previos, por lo que la curva de aprendizaje es menor que en otras BBDDs.

---

#### 2.2. Tipo de índice de búsqueda

**Índice utilizado inicialmente:**  
- Índice aproximado (ANN) basado en **HNSW**, que es el que Qdrant usa por defecto.

- El volumen de datos puede crecer en grna medida, por lo que un índice aproximado ofrece un buen equilibrio entre **velocidad** y **calidad de resultados**. Qdrant está optimizado para HNSW; no hace falta implementar ni mantener una solución propia de *brute-force*.

- No planteo cambiar el tipo de índice en índice en el futuro. Un índice *brute-force* solo tendría sentido en una fase muy inicial con poquísimos documentos; con Qdrant no aporta ventajas reales frente a HNSW.

---

#### 2.3. Organización de los datos en la base vectorial

Utilizo **una colección principal** (por ejemplo `bitcoin_docs_index`) y organizo la información mediante **metadatos**, de forma que la búsqueda y las actualizaciones sean eficientes:

- **Metadatos principales por chunk/documento:**
  - `source`: tipo de origen (`"info"` para glosario/conceptos, `"precios"` para históricos, etc.).
  - `filename`: nombre del fichero original.
  - `doc_id`: hash estable del documento/chunk, usado para evitar duplicados y hacer *upsert* (borrar la versión anterior y reinsertar) -- antes se creaba un uuid aleatorio, por lo que no permitía identificar si dos documentos eran realmente iguales, por lo que se podían dar duplicados.
  - `data_date` (si aplica): fecha a la que corresponde la información (por ejemplo, fecha del precio).
  - `ingested_at`: fecha/hora de ingesta.

- **Ventajas de esta organización:**
  - Puedo **filtrar búsquedas**, haciéndolas más eficientes en el futuro a la hora de ir a buscar en BBDD (por ejemplo, solo conceptos, solo precios recientes, etc.).
  - Puedo **actualizar o reemplazar documentos** borrando por `doc_id` antes de reindexar, evitando duplicidades.
  - Puedo aplicar políticas de **obsolescencia**, eliminando documentos antiguos según `data_date` o `ingested_at` -- en este proyecto los precios muy antiguos pueden no tener sentido conservarlos. Además, la información está en constante actualización, por lo que puede ser conveniente borrar archivos antiguos que puedan no ser relvantes a día de hoy.

De esta forma, la BBDD busca tener una estructura organizada y con sentido de los documentos que se van subiendo, haciendo más eficiente la búsqueda.


---
---


### 3. Estrategia para mantener la información actualizada

#### 3.1. Adquisición y procesamiento de nuevas fuentes

- **Fuentes principales:**
  - Datos de precios de Bitcoin (ej. API externa y ficheros JSON diarios en `data/prices`).
  - Documentos explicativos (glosario, informes, PDFs/TXTs subidos por el usuario en `data/info`).

- **Procesamiento:**
  - **Precios:** se convierten de JSON a texto legible (resumen con fecha, precio de cierre, máximo, mínimo, volumen, etc.).
  - **Documentos:** se procesan con un script de preprocesado:
    - extracción de texto (TXT/PDF/DOCX),
    - limpieza,
    - división en *chunks* (para mejorar la recuperación en el RAG), **muy necesario para posibles nuevas fuentes de datos mucho más extensas (libros, revistas, etc.)
    - asignación de metadatos (origen, filename, fechas).

Este flujo está automatizado mediante scripts de ingesta y expuesto también vía endpoints de la API para subir nuevos documentos.

---

#### 3.2. Frecuencia de actualización de la base vectorial

- A día de hoy, por cada consulta que se hace a la API para traerse el precio actualizado, se genera (o sobreescribe si existe) un archivo JSON con la información obtenida. La idea es poder automatizar la creación del fichero JSON diario con la información a hora de cierre, para tener siempre la información más actualizada. 

- Además, se pretende, por cada día, buscar una fuente de información con informes diarios donde expliquen con mayor detalle la evolución del precio del Bitcoin; o, si no se consiguiera, generar un fichero PDF con la información obtenida en la API de Alpha Vantage para poder ingestar esa información en la BBDD vectorial.

- Cuando se consiguieran nuevas fuentes de datos, como por ejemplo nuevos glosarios o definiciones de conceptos asociados, se pueden hacer cargas esporádicas (se puede hacer a través de endpoints desarrollados en la API).

---

#### 3.3. Generación de embeddings para la nueva información

- Para cada nuevo texto o *chunk*:
  - Se llama al modelo `text-embedding-3-small` para obtener el vector de 1536 dimensiones.
  - Se adjuntan metadatos relevantes (especificados previamente).
- Embeddings y metadatos se generan tanto:
  - en la ingesta inicial (batch),
  - como en las ingestas incrementales (nuevos precios o documentos subidos).

---

#### 3.4. Integración de nuevos embeddings en la base vectorial

- **Ingesta inicial (`/ingest/initial` / script):**
  - Se recrea la colección en Qdrant (borrado + creación).
  - Se insertan todos los documentos (glosario + históricos), generando embeddings desde cero.

- **Ingesta incremental (`/ingest/documents` y nuevos precios):**
  - Se construye una lista de `Document` con embeddings y metadatos.
  - Antes de insertar, se hace un *upsert* por `doc_id`:
    - se eliminan de Qdrant los puntos con el mismo `doc_id`,
    - se insertan los nuevos puntos (versión actualizada del documento).

Esto garantiza que las actualizaciones se reflejan sin arrastrar versiones antiguas del mismo contenido.

---

#### 3.5. Evitar duplicación y obsolescencia de datos

**Evitar duplicados:**

- Cada documento/chunk tiene un **`doc_id` determinista**, calculado como hash del contenido (y el nombre de fichero).
- En la ingesta incremental:
  - se borran los puntos existentes con ese `doc_id`,
  - y luego se insertan los nuevos.
- Subir el mismo documento varias veces no genera copias; lo reemplaza.

**Gestionar obsolescencia:**

- Los documentos relacionados con datos de mercado incluyen un metadato `data_date`.
- Esto permite:
  - aplicar políticas de **retención** (por ejemplo, borrar precios con `data_date` demasiado antiguos),
  - **filtrar** en las búsquedas para priorizar información reciente (solo últimos N días),
  - mantener el índice enfocado en la información más relevante para el RAG.

En conjunto, esta estrategia mantiene la base vectorial:

- libre de duplicados innecesarios,
- alineada con los datos más recientes,
- y preparada para limpiar información vieja según se defina la política de retención.


---
---


### 4. Integración con el Modelo de Lenguaje Grande (LLM)

#### 4.1. LLM utilizado y justificación

**LLM utilizado:** GPT-4o-mini (compatible con LangChain).

He utilizado este modelo por lo siguiente:

- **Calidad de generación:** ofrece un rendimiento sólido en razonamiento y explicación, adecuado para responder preguntas sobre Bitcoin y finanzas.
- **Multilingüe:** maneja eficazmente el español, permitiendo comprender preguntas y generar respuestas naturales.
- **Integración sencilla:** soporte directo en LangChain, facilitando la construcción de cadenas RAG (prompts, routing, etc.).
- **Eficiencia:** combina respuestas numéricas (precio actual) con explicaciones contextualizadas, optimizando recursos y tiempo de respuesta.

---

#### 4.2. Flujo completo del sistema RAG

1. **Pregunta del usuario**  
   - El usuario envía una pregunta al endpoint `/rag`.

2. **Clasificación de la pregunta**  
   - Se invoca una cadena de clasificación que decide si la pregunta es:
     - `precio_actual`
     - `informacion_conceptos`

3. **Rama A – `precio_actual`**
   - No se consulta la base vectorial.
   - Se obtiene el precio más reciente de Bitcoin desde los datos actualizados (JSON diario / API).
   - Se prepara un texto corto con la información relevante (precio, fecha, etc.).
   - El LLM recibe:
     - la pregunta,
     - el texto con el precio,
     - y un prompt que le indica que responda de forma clara y directa.
   - El LLM genera la respuesta final para el usuario.

4. **Rama B – `informacion_conceptos`**
   - Se genera el *embedding* de la pregunta con `text-embedding-3-small`.
   - Se consulta Qdrant para obtener los *chunks* más relevantes (glosario, informes, históricos).
   - Se construye un **contexto** concatenando esos fragmentos.
   - El LLM recibe:
     - la pregunta del usuario,
     - el contexto recuperado de la base vectorial,
     - y un prompt que le indica que:
       - se base únicamente en ese contexto,
       - no invente información fuera de él.
   - El LLM genera una respuesta explicativa apoyándose en ese contexto.

5. **Respuesta al usuario**
   - El backend devuelve un objeto con:
     - `answer`: respuesta del LLM,
     - `sources`: decisión de la fuente (`precio_actual` o `informacion_conceptos` + razón),
     - `timestamp` y `question`.

---

#### 4.3. Optimización de la calidad del contexto recuperado

Para mejorar la calidad de las respuestas del LLM, el sistema cuida especialmente **qué contexto se recupera** y **cómo se le pasa al modelo**:

1. **Buen *chunking***  
   - Los documentos se dividen en fragmentos de tamaño razonable (ni demasiado pequeños ni demasiado grandes).
   - Cada *chunk* incluye metadatos (`source`, `filename`, `doc_id`, `data_date`) que ayudan a filtrar y entender de dónde viene la información.

2. **Filtros por metadatos**  
   - Según el tipo de pregunta, se pueden priorizar o filtrar:
     - solo `source = "info"` para conceptos,
     - solo `source = "precios"` y `data_date` reciente para preguntas temporales/ históricas.
   - Esto evita recuperar contexto irrelevante.

3. **Selección y límite de documentos**  
   - Se recupera un número limitado de *chunks* (k reducido) para no “ahogar” al LLM con demasiado texto.
   - Se aplica un umbral de similitud para descartar resultados poco relacionados.

4. **Prompt de RAG bien diseñado**  
   - El prompt del LLM incluye instrucciones claras:
     - “Responde únicamente usando la información del contexto”.
     - “Si el contexto no contiene la respuesta, dilo explícitamente”.
   - Esto reduce alucinaciones y obliga al modelo a apoyarse en el contenido recuperado.

En conjunto, estos pasos aseguran que el LLM reciba **menos ruido y más señal**, mejorando la precisión y la utilidad de las respuestas generadas.




---
---


### 5. Arquitectura del Sistema

Se adjunta `diagrama.png` en esta misma carpeta.
