
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalVectorStore:

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name=MODEL_NAME):

        self.model_name = model_name

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            self.model_name,
            device="cpu"
        )

        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        """
        Generate embeddings and build a FAISS index.
        """

        if not chunks:
            raise ValueError("No chunks provided.")

        # Remove empty chunks while preserving metadata.
        valid_chunks = []

        for chunk in chunks:

            text = chunk.get("text", "")

            if text and text.strip():

                valid_chunks.append({
                    **chunk,
                    "text": text.strip()
                })

        if not valid_chunks:
            raise ValueError("All provided chunks are empty.")

        texts = [
            chunk["text"]
            for chunk in valid_chunks
        ]

        print(f"Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # Create FAISS index using inner product.
        embedding_dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            embedding_dimension
        )

        # Add vectors to the index.
        self.index.add(embeddings)

        # Keep metadata in the same order as FAISS vectors.
        self.chunks = valid_chunks

        print("FAISS index created successfully!")

        return self.index.ntotal

    def save(self, directory="vector_db"):
        """
        Save FAISS index and corresponding metadata.
        """

        if self.index is None:
            raise ValueError("Build the index before saving.")

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        index_path = directory / "medical.index"
        metadata_path = directory / "metadata.json"

        # Save numerical vectors.
        faiss.write_index(
            self.index,
            str(index_path)
        )

        # Save original text and page metadata.
        metadata = {
            "model_name": self.model_name,
            "chunks": self.chunks
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("Vector database saved successfully!")

    def load(self, directory="vector_db"):
        """
        Load previously saved FAISS index and metadata.
        """

        directory = Path(directory)

        index_path = directory / "medical.index"
        metadata_path = directory / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        # Load FAISS index.
        self.index = faiss.read_index(
            str(index_path)
        )

        # Load metadata.
        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        if metadata["model_name"] != self.model_name:
            raise ValueError(
                "Embedding model mismatch. "
                "Use the same model that created this index."
            )

        self.chunks = metadata["chunks"]

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index and metadata have different sizes."
            )

        print("Vector database loaded successfully!")

    def search(self, query, top_k=3):
        """
        Retrieve the most relevant chunks for a question.
        """

        if self.index is None:
            raise ValueError(
                "Load or build the vector index before searching."
            )

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Convert user question into an embedding.
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        # Search FAISS index.
        actual_top_k = min(
            top_k,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            query_embedding,
            actual_top_k
        )

        results = []

        for score, index_position in zip(
            scores[0],
            indices[0]
        ):

            if index_position < 0:
                continue

            chunk = self.chunks[int(index_position)]

            results.append({

                **chunk,

                "similarity_score": float(score)

            })

        return results
