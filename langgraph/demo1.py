from asyncio import graph
from langgraph.graph import StateGraph, MessageGraph,MessagesState,START,END
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world "}]}

load_dotenv()

llm = init_chat_model(
    "gpt-4o-mini",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

grpah = StateGraph(MessagesState)    
grpah.add_node("mock_llm",mock_llm)
grpah.add_edge(START,"mock_llm")
grpah.add_edge("mock_llm",END)
graph = grpah.compile()

print(graph.invoke({"messages":[{"role":"user","content":"hi!"}]}))
