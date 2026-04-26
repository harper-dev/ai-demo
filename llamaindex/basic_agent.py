import asyncio
from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context

load_dotenv(override=True)

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)
ctx = Context(agent)

async def main():
    # Run the agent
    # result = await agent.run("What is 1234*4567?")
    # print(result)
    response = await agent.run("My name is bob", ctx=ctx)
    print(response)
    response = await agent.run("What is my name?", ctx=ctx)
    print(response)

if __name__  == "__main__":
    asyncio.run(main())