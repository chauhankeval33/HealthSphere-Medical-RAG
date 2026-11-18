
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

# Load the embedding model on CPU
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Sample medical report information
medical_texts = [
    "Hemoglobin: 13.5 g/dL",
    "WBC: 7500 cells/mcL",
    "Platelets: 250000 cells/mcL"
]

print("\nGenerating embeddings...")

# Convert text into numerical vectors
embeddings = model.encode(
    medical_texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print("\nEmbeddings generated successfully!")

print("Number of input texts:", len(medical_texts))
print("Embedding shape:", embeddings.shape)
print("Data type:", embeddings.dtype)

# Display the first embedding
print("\nFirst embedding (first 10 values):")
print(embeddings[0][:10])

print("\nEmbedding test completed!")
