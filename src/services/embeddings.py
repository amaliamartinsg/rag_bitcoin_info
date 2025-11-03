from langchain_google_genai import GoogleGenerativeAIEmbeddings

MODEL_NAME = 'models/gemini-embedding-001'

embeddings_model_langchain = GoogleGenerativeAIEmbeddings(model=MODEL_NAME)