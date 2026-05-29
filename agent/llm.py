    
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
llm = init_chat_model(
    "gpt-4o-mini",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

