from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

llm=ChatGroq(groq_api_key=os.getenv("Groq_API_KEY"),model_name='llama3-70b-8192')