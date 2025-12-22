import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv(verbose=True)
model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

@tool
def get_weather_for_location(city: str) -> str:
    """get weather for a given city."""
    return f"It's always sunny in {city}!"
# Batch
# response = model.batch_as_completed([
#     "Why do parrots have colorful feathers?",
#     "How do airplanes fly?",
#     "What is quantum computing?"
# ])
# for response in response:
#     print(response)

# model_with_tools = model.bind_tools([get_weather_for_location])
# response = model_with_tools.invoke("What is the weather in Florida?")
# print(response)
# for tool_call in response.tool_calls:
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")

tool = {"type":"web_search"}
model_with_tools = model.bind_tools([tool])
response = model_with_tools.invoke("What was a positive new story from today?")
print(response.content_blocks)