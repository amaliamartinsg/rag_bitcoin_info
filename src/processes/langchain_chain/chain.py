import json
import requests
from datetime import datetime
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch

from src.services.llms import llm_langchain
from src.services.vector_store import qdrant_langchain
from src.services.current_price_btc import parse_current_price
from src.processes.langchain_chain.prompts import source_selection_prompt, rag_prompt, none_selection_prompt
from src.processes.langchain_chain.structures import SourceModel

classifier_chain = source_selection_prompt | llm_langchain.with_structured_output(SourceModel)

answer_generation_chain = rag_prompt | llm_langchain | StrOutputParser()

def format_docs(input_dict) -> str:
    """Formatea los documentos recuperados en una sola cadena de contexto."""
    docs = input_dict["source_context"]

    # Caso 1: ya es un string (por ejemplo, el texto del precio actual de BTC)
    if isinstance(docs, str):
        return docs

    # Caso 2: lista (pueden ser Document de LangChain o diccionarios)
    if isinstance(docs, list):
        # Lista de diccionarios con clave "section" (lo que devuelve get_sources_info)
        if all(isinstance(d, dict) and "section" in d for d in docs):
            return "\n\n".join(d["section"] for d in docs if d.get("section"))

        # Lista de Document de LangChain
        if all(hasattr(d, "page_content") for d in docs):
            return "\n\n".join(d.page_content for d in docs)

        # Cualquier otra cosa: último recurso
        return "\n\n".join(str(d) for d in docs)

    # Formato no esperado
    return "No se pudo procesar el formato de los documentos."

# Función para obtener información conceptual desde la base de datos
def get_sources_info(question: str, k: int = None, threshold: float = None) -> list:
    """Extrae la información de las fuentes de los documentos desde Qdrant."""

    results = qdrant_langchain.similarity_search_with_score(question, k=k)
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Si no hay threshold, no filtramos por score
    if threshold is not None:
        filtered_results = [(doc, score) for doc, score in results if score >= threshold]
    else:
        filtered_results = results

    if results:
        print(
            f"{len(filtered_results)}/{len(results)} resultados sobre umbral {threshold} "
            f"para consulta '{question}'"
        )
    else:
        print(f"No se encontraron resultados para '{question}'")

    docs_filtered = []
    if not filtered_results:
        print(f"Ningún documento supera el umbral de {threshold} para la consulta '{question}'")

    for doc, score in filtered_results:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        docs_filtered.append({
            "score": score,
            "chunk_id": metadata.get("_id"),
            "page": None,  # No hay info de página
            "section": doc.page_content[:300] if hasattr(doc, "page_content") else "",
            "source": metadata.get("source"),
            "filename": metadata.get("filename"),
            "collection_name": metadata.get("_collection_name"),
        })

    return docs_filtered


def check_if_source_exists(input_dict):
    if input_dict["source"].selection == 'none':
        return False
    return True

rag_with_source_chain = (
    RunnablePassthrough.assign(
        source_context=RunnableLambda(
            lambda input_dict: (
                parse_current_price() if input_dict['source'].selection == 'precio_actual' else
                get_sources_info(
                    input_dict['question'],
                    k=input_dict.get('k_docs'),
                    threshold=input_dict.get('threshold')
                )
            )
        )
    )
    .assign(context=RunnableLambda(format_docs))
    .assign(answer=RunnableLambda(
        lambda input_dict: (
            input_dict['source_context'] if input_dict['source'].selection == 'precio_actual' else
            answer_generation_chain.invoke(input_dict)
        )
    ))
)

no_source_found_chain = RunnablePassthrough.assign(
    answer=none_selection_prompt | llm_langchain | StrOutputParser()
)

rag_chain = (
    RunnablePassthrough.assign(
        source=(itemgetter("question") | classifier_chain)
    )
    | RunnableBranch(
        (check_if_source_exists, rag_with_source_chain),
        no_source_found_chain,
    )
).with_types(input_type=dict, output_type=dict)
