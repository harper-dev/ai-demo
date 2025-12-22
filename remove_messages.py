from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
load_dotenv()

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old message to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earlier message
        return {"messages:": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

agent = create_agent(
    "gpt-4o-mini",
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

for event in agent.stream(
    {"messages":[{"role":"user","content":"Hi! I'm bob!"}]},
    config,
    stream_mode="values"
):
    print([(message.type, message.content) for message in event["messages"]])

print("-----------------------")
for event in agent.stream(
    {"messages":[{"role":"user","content":"what's my name?"}]},
    config,
    stream_mode="values"
):
    print([(message.type, message.content) for message in event["messages"]])