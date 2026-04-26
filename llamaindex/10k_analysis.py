import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import Settings
from dotenv import load_dotenv


load_dotenv(override=True)
Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")
nest_asyncio.apply()

# lyft_docs = SimpleDirectoryReader('./data/10k/lyft_2021.pdf').load_data()
lyft_docs = SimpleDirectoryReader(input_files=['./data/10k/lyft_2021.pdf']).load_data()
# uber_docs = SimpleDirectoryReader('./data/10k/uber_2021.pdf').load_data()
uber_docs = SimpleDirectoryReader(input_files=['./data/10k/uber_2021.pdf']).load_data()
lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)
lyft_engine = VectorStoreIndex.as_query_engine(lyft_index)
uber_engine = VectorStoreIndex.as_query_engine(uber_index)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="Lyft 2021 Annual Report",
            description="Useful for answering questions about Lyft's 2021 annual report.",
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="Uber 2021 Annual Report",
            description="Useful for answering questions about Uber's 2021 annual report.",
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
)
response = s_engine.query(
    "Compare and contrast the customer segments and geographies that grew the"
    " fastest"
)
print(response)
response = s_engine.query(
    "Compare revenue growth of Uber and Lyft from 2020 to 2021"
)
print(response)
