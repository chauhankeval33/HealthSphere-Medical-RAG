
from io import BytesIO

from src.document_loader import extract_text_from_pdf
from src.text_preprocessor import preprocess_medical_pages
from src.text_chunker import chunk_medical_pages
from src.vector_store import MedicalVectorStore
from src.rag_pipeline import answer_medical_question


# ---------------------------------------
# 1. Load actual PDF
# ---------------------------------------

pdf_path = "data/sample_reports/sample_blood_report.pdf"

print("\nLoading actual medical PDF...")

with open(pdf_path, "rb") as file:
    pdf_bytes = file.read()

pages = extract_text_from_pdf(BytesIO(pdf_bytes))

print("PDF extraction completed!")


# ---------------------------------------
# 2. Preprocess text
# ---------------------------------------

processed_pages = preprocess_medical_pages(pages)

print("Text preprocessing completed!")


# ---------------------------------------
# 3. Generate chunks
# ---------------------------------------

chunks = chunk_medical_pages(
    processed_pages,
    chunk_size=1000,
    overlap=150
)

print("Total chunks generated:", len(chunks))


# ---------------------------------------
# 4. Generate embeddings and FAISS index
# ---------------------------------------

print("\nBuilding vector database...")

vector_store = MedicalVectorStore()
vector_store.build_index(chunks)

print("Vector database created successfully!")


# ---------------------------------------
# 5. Ask questions using RAG
# ---------------------------------------

questions = [
    "What is the hemoglobin level?",
    "What is the platelet count?",
    "What is the total WBC count?"
]

for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)

    result = answer_medical_question(
        question=question,
        vector_store=vector_store,
        top_k=3
    )

    print("\nRAG ANSWER:")
    print(result["answer"])

    print("\nSOURCE REFERENCES:")

    for source in result["sources"]:
        print(
            f"Page: {source['page_number']} | "
            f"Chunk: {source['chunk_id']} | "
            f"Similarity: {source['similarity_score']:.4f}"
        )

print("\nReal PDF RAG Test Completed!")
