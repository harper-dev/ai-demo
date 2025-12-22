from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_sellection(request: ModelRequest, handler) -> ModelResponse :
    """Choose model based on conversation complexity."""
    message_count = len(request.messages)
    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model
    return handler(request.override(model))

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[dynamic_model_sellection]
)