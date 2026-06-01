from base import llm
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tavily import TavilyClient
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import tool
import asyncio
import os
load_dotenv(override=True)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
async def search(query: str) -> str:
    """Search for information on the internet."""
    return await asyncio.to_thread(tavily_client.search, query)

tools = [search]

today = datetime.now().strftime("%Y-%m-%d")
system_prompt = f"""
    You are a helpful assistant that helps users find news articles about recent events using search tool. Today is {today}.
    """
# model = init_chat_model("gpt-4o-mini", temperature=0, timeout=10, max_tokens=1000)
agent = create_agent(
    model=llm,
    system_prompt=system_prompt,
    tools=tools,
)