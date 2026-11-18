
from src.vector_store import MedicalVectorStore
from src.rag_pipeline import answer_medical_question


print("Loading medical vector database...")

# Initialize embedding model and vector store
vector_store = MedicalVectorStore()

# Load previously saved FAISS index
vector_store.load("vector_db")

print("\nVector database loaded successfully!")

# Ask a question
question = "What is the hemoglobin level?"

print("\nUser Question:", question)

# Generate answer using RAG
result = answer_medical_question(
    question=question,
    vector_store=vector_store,
    top_k=3
)

# Display answer
print("\n========== RAG ANSWER ==========")
print(result["answer"])

# Display source references
print("\n========== SOURCE REFERENCES ==========")

for source in result["sources"]:
    print("\nChunk ID:", source["chunk_id"])
    print("Page Number:", source["page_number"])
    print("Similarity Score:", source["similarity_score"])
    print("Retrieved Text:", source["text"])

print("\nRAG Pipeline Test Completed!")
