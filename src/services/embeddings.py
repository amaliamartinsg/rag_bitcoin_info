from config.project_config import SETTINGS
from langchain_openai import OpenAIEmbeddings

MODEL_NAME = SETTINGS.embedding_model_name

embeddings_model = OpenAIEmbeddings(model=MODEL_NAME)