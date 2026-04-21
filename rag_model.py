import os
import uuid
import ollama
import chromadb
import pdfplumber
from pathlib import Path
from chromadb.config import Settings

# ── Configuration ────────────────────────────────────────────────────────────
EMBED_MODEL      = "paraphrase-multilingual"   # Ollama multilingual (Nepali support)
CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "rag_documents"
CHUNK_SIZE       = 500    # characters per chunk
CHUNK_OVERLAP    = 100    # overlap between chunks
TOP_K            = 5      # number of chunks to retrieve

# ── ChromaDB client (persistent) ─────────────────────────────────────────────
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Helpers ───────────────────────────────────────────────────────────────────
def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def _embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via Ollama for a list of texts."""
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract full text from PDF bytes using pdfplumber."""
    import io
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


# ── Public API ────────────────────────────────────────────────────────────────
def ingest_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """
    Extract text from PDF, chunk it, embed with Ollama, store in ChromaDB.
    Returns a summary of what was ingested.
    """
    collection = _get_collection()

    # 1. Extract text
    raw_text = _extract_text_from_pdf(pdf_bytes)
    if not raw_text.strip():
        return {"status": "error", "message": "No text could be extracted from the PDF."}

    # 2. Chunk
    chunks = _chunk_text(raw_text)

    # 3. Embed
    embeddings = _embed(chunks)

    # 4. Store in ChromaDB
    doc_id = str(uuid.uuid4())
    ids        = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas  = [{"filename": filename, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return {
        "status": "success",
        "filename": filename,
        "doc_id": doc_id,
        "chunks_indexed": len(chunks),
        "characters_extracted": len(raw_text),
    }


def query_rag(question: str, llm_model: str, top_k: int = TOP_K) -> dict:
    """
    Embed the question, retrieve top_k chunks from ChromaDB,
    build a context-aware prompt, and answer with Ollama LLM.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return {
            "answer": "No documents have been ingested yet. Please upload a PDF first.",
            "sources": [],
        }

    # 1. Embed the query
    query_embedding = _embed([question])[0]

    # 2. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = results["documents"][0]
    metadatas        = results["metadatas"][0]
    distances        = results["distances"][0]

    # 3. Build context
    context = "\n\n---\n\n".join(retrieved_chunks)

    # 4. Build prompt (supports Nepali input/output)
    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the question.
If the question is in Nepali, answer in Nepali. If in English, answer in English.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    # 5. Generate answer with Ollama
    response = ollama.chat(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
    )

    sources = [
        {
            "filename": m.get("filename", "unknown"),
            "chunk_index": m.get("chunk_index", -1),
            "relevance_score": round(1 - d, 4),   # cosine similarity
        }
        for m, d in zip(metadatas, distances)
    ]

    return {
        "answer": response["message"]["content"],
        "sources": sources,
    }


def list_documents() -> dict:
    """Return all unique documents stored in ChromaDB."""
    collection = _get_collection()
    if collection.count() == 0:
        return {"documents": [], "total_chunks": 0}

    results = collection.get(include=["metadatas"])
    seen, docs = set(), []
    for meta in results["metadatas"]:
        doc_id = meta.get("doc_id")
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append({"doc_id": doc_id, "filename": meta.get("filename")})

    return {"documents": docs, "total_chunks": collection.count()}


def delete_document(doc_id: str) -> dict:
    """Delete all chunks belonging to a document by its doc_id."""
    collection = _get_collection()
    results = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    ids_to_delete = results["ids"]
    if not ids_to_delete:
        return {"status": "error", "message": f"No document found with doc_id '{doc_id}'"}
    collection.delete(ids=ids_to_delete)
    return {"status": "success", "doc_id": doc_id, "chunks_deleted": len(ids_to_delete)}
