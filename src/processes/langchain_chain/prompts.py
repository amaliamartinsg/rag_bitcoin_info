from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Ajuste de prompts para reflejar las categorías definidas

source_selection_prompt = PromptTemplate.from_template(
    template='''
    Eres un sistema experto encargado de seleccionar la mejor fuente de conocimiento para responder preguntas relacionadas con Bitcoin.
    Puedes responder preguntas sobre las siguientes categorías:
    - precio_actual: Información sobre el precio actual de Bitcoin.
    - informacion_conceptos: Información sobre los conceptos asociados a Bitcoin.

    Esta es la pregunta para tu selección de fuente de conocimiento:
    {question}
    '''
)

none_selection_prompt = PromptTemplate.from_template(
    template='''
    Eres un sistema de chat especializado en Bitcoin. Solo puedes responder preguntas relacionadas con las siguientes categorías:
    - precio_actual: Información sobre el precio actual de Bitcoin.
    - informacion_conceptos: Información sobre los conceptos asociados a Bitcoin.

    Si la pregunta no está relacionada con estas categorías, responde recordándole al usuario que solo puedes responder preguntas sobre Bitcoin.

    Esta es la pregunta del usuario:
    {question}
    '''
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Eres un asistente experto en Bitcoin que responde preguntas basándose en el contexto proporcionado.\n"
            "Puedes proporcionar información sobre el precio actual de Bitcoin o su evolución histórica.\n"
            "Contexto:\n{context}"
        ),
        HumanMessagePromptTemplate.from_template(
            "Pregunta: {question}"
        )
    ]
)