""" LLM Setup- single source of trouth for the model and agent uses"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()   

def get_llm(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.2):
    """Returns configured Groq Chat Model instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    return ChatGroq(model=model_name, temperature=temperature, groq_api_key=api_key)


# Quick smoke test
if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("In one line, who is CEO of Impetus Technologies Inc?")
    print(response.content)