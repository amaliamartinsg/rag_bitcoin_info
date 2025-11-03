from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

llm_langchain = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_retries=3
)
