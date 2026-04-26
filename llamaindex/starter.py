from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)

#Create a RAG tool using LlamaIndex
# documents = SimpleDirectoryReader('data').load_data()
# index = VectorStoreIndex.from_documents(documents)
# index.storage_context.persist(persist_dir="./index_storage")
# load from storage
storage_context = StorageContext.from_defaults(persist_dir="./index_storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)


# Create an enhanced workflow with both tools
agent = FunctionAgent(
    tools=[multiply, search_documents],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)


# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run(
        "What did the author do in college? Also, what's 7 * 8?"
    )
    print(response)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())