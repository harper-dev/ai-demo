from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

file_path = "doc/nke-10k-2023.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()
print(len(docs))
# print(f"{docs[0].page_content[:200]}\n")
# print(f"{docs[0].metadata}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))
print('--------------------------------0')
embeddings = OllamaEmbeddings(model="gpt-oss")
vector_store = InMemoryVectorStore(embeddings)

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)
print(f'Generated vectors of length {len(vector_1)} and {len(vector_2)}')
print(vector_1[:10])
ids = vector_store.add_documents(documents = all_splits)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])

print('--------------------------------1')
results_1=vector_store.similarity_search_with_score(
    "What was Nike's revenue in 2023"
)
doc, score = results_1[0]
print(f"score: {score}")
print(doc)
print('--------------------------------2')
embedding = embeddings.embed_query("How were Nike's margins impected in 2023")
results_2=vector_store.similarity_search_by_vector(embedding)
print(results_2[0])

print('--------------------------------3')

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=3)

batch_results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]
)
print("Batch results:")
for i, result in enumerate(batch_results):
    print(f"Query {i+1} result: {result[0].page_content[:200]}...")