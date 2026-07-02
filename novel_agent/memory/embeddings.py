"""ChromaDB vector store for semantic chapter search."""

from pathlib import Path

import chromadb
from chromadb.config import Settings


class ChapterStore:
    """Vector store for chapter content using ChromaDB."""

    def __init__(self, persist_dir: Path):
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def get_or_create_collection(self, project_id: str):
        """Get or create a collection for a project."""
        name = _sanitize_collection_name(project_id)
        return self._client.get_or_create_collection(name=name)

    def delete_collection(self, project_id: str):
        """Delete a project's collection."""
        name = _sanitize_collection_name(project_id)
        try:
            self._client.delete_collection(name)
        except Exception:
            pass

    def index_chapter(
        self,
        project_id: str,
        chapter_number: int,
        content: str,
        metadata: dict | None = None,
    ):
        """Index a chapter into the vector store."""
        collection = self.get_or_create_collection(project_id)
        meta = metadata or {}
        meta["chapter_number"] = chapter_number

        chunk_size = 500
        chunks = _split_text(content, chunk_size=chunk_size, overlap=50)

        ids = [f"ch{chapter_number}_p{i}" for i in range(len(chunks))]
        metadatas = [{**meta, "chunk_index": i} for i in range(len(chunks))]

        # Remove existing chunks for this chapter
        existing = collection.get(
            where={"chapter_number": chapter_number}
        )
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        if chunks:
            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )

    def search(
        self,
        project_id: str,
        query: str,
        top_k: int = 5,
        chapter_range: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Semantic search across chapters."""
        collection = self.get_or_create_collection(project_id)
        where_filter = None
        if chapter_range:
            where_filter = {
                "$and": [
                    {"chapter_number": {"$gte": chapter_range[0]}},
                    {"chapter_number": {"$lte": chapter_range[1]}},
                ]
            }

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "chunk_id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                    if results["metadatas"] else {},
                    "distance": results["distances"][0][i]
                    if results["distances"] else None,
                })
        return output


def _sanitize_collection_name(name: str) -> str:
    """ChromaDB collection names must match [a-zA-Z0-9_-]+."""
    import re
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return sanitized[:63]  # ChromaDB limit


def _split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by paragraph boundaries."""
    paragraphs = text.split("\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap and len(current) > overlap else ""
        current += para + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]
