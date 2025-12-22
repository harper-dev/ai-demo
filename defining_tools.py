from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@tool
def search(query: str) -> str:
    """Search for information on the internet."""
    return f"Results for : {query}"

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"Weather in {location} is sunny, 72"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool executing errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )

agent = create_agent(model, [search, get_weather], middleware=[handle_tool_errors], verbose=True)
