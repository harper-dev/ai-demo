from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph,MessagesState,START,END
from base.llm import llm
def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world "}]}

graph = StateGraph(MessagesState) 
graph.set_node_defaults(retry_policy=RetryPolicy(max_attempts=3))
graph.add_node("mock_llm",mock_llm)
graph.add_edge(START,"mock_llm")
graph.add_edge("mock_llm",END)
graph = graph.compile()

print(graph.invoke({"messages":[{"role":"user","content":"hi!"}]}))
