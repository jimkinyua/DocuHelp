import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./data/chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_fn,
)

def add_chunks(doc_id: str, chunks: list[str]) -> int:
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def search_chunks(query: str, source: str = None, top_k: int = 4):
    if source:
        return collection.query(query_texts=[query], n_results=top_k, where={"source": source})
    return collection.query(query_texts=[query], n_results=top_k)

    