"""
Document Store
- Reads PDF and TXT files
- Splits into chunks
- Embeds using sentence-transformers (free, local)
- Does similarity search (cosine)
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path

class DocumentStore:
    def __init__(self, store_path="store"):
        self.store_path = store_path
        self.chunks = []       # List of {"text": ..., "source": ...}
        self.embeddings = None # numpy array of shape (N, 384)
        self.model = None
        os.makedirs(store_path, exist_ok=True)
        self._load_model()
        self._load_store()

    def _load_model(self):
        """Load the embedding model (downloads ~90MB on first run)."""
        try:
            from sentence_transformers import SentenceTransformer
            print("⏳ Loading embedding model (first run may take ~30 seconds)...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedding model ready.")
        except Exception as e:
            print(f"❌ Could not load embedding model: {e}")
            self.model = None

    def _load_store(self):
        """Load previously indexed chunks from disk."""
        chunks_path = os.path.join(self.store_path, "chunks.pkl")
        embeddings_path = os.path.join(self.store_path, "embeddings.npy")
        if os.path.exists(chunks_path) and os.path.exists(embeddings_path):
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            self.embeddings = np.load(embeddings_path)
            print(f"✅ Loaded {len(self.chunks)} chunks from previous index.")

    def _save_store(self):
        """Persist chunks and embeddings to disk."""
        with open(os.path.join(self.store_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        np.save(os.path.join(self.store_path, "embeddings.npy"), self.embeddings)

    def has_documents(self):
        return len(self.chunks) > 0

    def chunk_count(self):
        return len(self.chunks)

    def index_folder(self, folder_path: str) -> int:
        """Read all PDFs and TXT files in a folder and index them."""
        new_chunks = []
        folder = Path(folder_path)

        for file_path in folder.iterdir():
            if file_path.suffix.lower() == ".pdf":
                text = self._read_pdf(file_path)
            elif file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            else:
                continue

            if text.strip():
                file_chunks = self._split_into_chunks(text, source=file_path.name)
                new_chunks.extend(file_chunks)
                print(f"  📄 Indexed: {file_path.name} → {len(file_chunks)} chunks")

        if not new_chunks:
            print("⚠️  No valid documents found.")
            return 0

        # Add to existing chunks
        self.chunks.extend(new_chunks)

        # Embed all chunks
        print(f"⏳ Embedding {len(self.chunks)} total chunks...")
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        self._save_store()
        print(f"✅ Indexing complete. Total chunks: {len(self.chunks)}")
        return len(self.chunks)

    def search(self, query: str, top_k: int = 4) -> list:
        """Find the most relevant chunks for a query using cosine similarity."""
        if not self.has_documents() or self.model is None:
            return []

        query_embedding = self.model.encode([query])
        scores = self._cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": float(scores[idx])
            })
        return results

    def _read_pdf(self, file_path) -> str:
        """Extract text from a PDF file."""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception as e:
            print(f"  ⚠️  Could not read PDF {file_path.name}: {e}")
            return ""

    def _split_into_chunks(self, text: str, source: str, chunk_size=200, overlap=40) -> list:
        """Split text into overlapping chunks for better retrieval."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i: i + chunk_size]
            if len(chunk_words) < 20:  # Skip tiny trailing chunks
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "source": source
            })
        return chunks

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all chunks."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)
