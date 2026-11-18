
from src.vector_store import MedicalVectorStore


# Synthetic medical report chunks
sample_chunks = [

    {
        "chunk_id": 1,
        "page_number": 1,
        "chunk_index": 0,
        "text": (
            "COMPLETE BLOOD COUNT\n"
            "Hemoglobin: 13.5 g/dL\n"
            "WBC: 7500 cells/mcL\n"
            "Platelets: 250000 cells/mcL"
        )
    },

    {
        "chunk_id": 2,
        "page_number": 1,
        "chunk_index": 1,
        "text": (
            "Red Blood Cell Count: 4.8 million cells/mcL\n"
            "Hematocrit: 40 percent\n"
            "MCV: 85 fL"
        )
    },

    {
        "chunk_id": 3,
        "page_number": 2,
        "chunk_index": 0,
        "text": (
            "LIPID PROFILE\n"
            "Total Cholesterol: 190 mg/dL\n"
            "HDL Cholesterol: 50 mg/dL\n"
            "LDL Cholesterol: 110 mg/dL\n"
            "Triglycerides: 140 mg/dL"
        )
    }

]


print("\nInitializing Medical Vector Store...")

# Initialize vector store
vector_store = MedicalVectorStore()


# Step 1: Generate embeddings and build FAISS index
print("\nBuilding FAISS index...")

total_vectors = vector_store.build_index(
    sample_chunks
)

print("Total vectors stored:", total_vectors)


# Step 2: Save index and metadata
print("\nSaving vector database...")

vector_store.save(
    directory="vector_db"
)


# Step 3: Simulate a fresh load from disk
print("\nTesting database loading...")

vector_store.index = None
vector_store.chunks = []

vector_store.load(
    directory="vector_db"
)


# Step 4: Search the medical report
query = "What is the hemoglobin level?"

print("\nUser Question:", query)

results = vector_store.search(
    query=query,
    top_k=2
)


# Step 5: Display search results
print("\nRetrieved Medical Information:")

for result in results:

    print("\nChunk ID:", result["chunk_id"])

    print("Page Number:", result["page_number"])

    print(
        "Similarity Score:",
        round(result["similarity_score"], 4)
    )

    print("Medical Text:")
    print(result["text"])


print("\nVector Store Test Completed Successfully!")
