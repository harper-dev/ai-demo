from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
llm = init_chat_model(
     "llama3.1",
    model_provider="ollama",
    temperature=0.7,
)

