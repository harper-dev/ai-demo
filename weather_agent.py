from langchain.agents import create_agent


SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""


def get_weather(city: str) -> str:
    """get_weather retrieves the current weather for a given city."""
    return f"It's always sunny in {city}!"

agent =  create_agent(
        model="gpt-4o-mini",
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
)

agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather like in sf?"}]
})
    
    