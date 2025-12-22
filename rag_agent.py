import bs4
from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import getpass
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest

load_dotenv(verbose=True)

embeddings = OllamaEmbeddings(model="llama3.1")
vector_store = InMemoryVectorStore(embeddings)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    )
)
docs = loader.load()
print(f"Loaded {len(docs)} documents from web page.")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")
# Index chunks in vector store
ids = vector_store.add_documents(documents = all_splits)

@tool
def retrieve_context(query: str):
    """Retrieve information to help answer the query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=3)
    doc_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_message = (
        "You are an helpful assistant. Use the following context in your response:\n\n"
        f"{doc_content}\n\n"
        "If the context does not contain the answer, respond with 'I don't know.'"
    )
    return system_message
tools = [retrieve_context]
agent = create_agent(
    model=init_chat_model("gpt-4o-mini", temperature=0, max_tokens=1000),
    middleware=[prompt_with_context],
    tools=[],
)

query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode = "values",
):
    step["messages"][-1].pretty_print()
# prompt = (
#     "You have access to a tool that retrieves context from a blog post."
#     "Use the tool to help answer user queries."
# )
# agent = create_agent(
#     model=init_chat_model("gpt-4o-mini", temperature=0, max_tokens=1000),
#     system_prompt=prompt,
#     tools=tools,
# )
#
# query = (
#     "What is the standard method for Task Decomposition?\n\n"
#     "Once you get the answer, look up common extensions of that method."
# )
#
# for event in agent.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     stream_mode = "values",
# ):
#     event["messages"][-1].pretty_print()