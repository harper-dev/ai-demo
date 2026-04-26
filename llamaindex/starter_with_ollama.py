from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import asyncio


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

# Create an agent workflow with our calculator tool
# agent = FunctionAgent(
#     tools=[multiply],
#     llm=Ollama(model="gpt-oss"),
#     request_timeout=360,
#     # embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2"),
#     system_prompt="You are a helpful assistant that can multiply two numbers.",
#     context_window=8000,
# )


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="gpt-oss",
                      context_window=8000,
                      request_timeout=360)

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)

agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[multiply, search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)
ctx = Context(agent)

async def main():
    # Run the agent
    result = await agent.run("What did the author do in college? Also, what's 7 * 8?", ctx=ctx)
    print(result)
    response = await agent.run("My name is Logan", ctx=ctx)
    print(response)
    response = await agent.run("What is my name?", ctx=ctx)

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
