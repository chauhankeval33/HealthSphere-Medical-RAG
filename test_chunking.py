
from src.text_preprocessor import preprocess_medical_pages
from src.text_chunker import chunk_medical_pages


# Create sample medical report with two pages
sample_pages = [

    {
        "page_number": 1,
        "text": """
        COMPLETE BLOOD COUNT

        Hemoglobin: 13.5 g/dL
        WBC: 7500 cells/mcL
        Platelets: 250000 cells/mcL
        RBC: 4.8 million cells/mcL
        Hematocrit: 42%
        """ * 15
    },

    {
        "page_number": 2,
        "text": """
        LIPID PROFILE

        Total Cholesterol: 190 mg/dL
        HDL: 50 mg/dL
        LDL: 110 mg/dL
        Triglycerides: 140 mg/dL
        """ * 15
    }

]


# Step 1: Preprocess medical pages
processed_pages = preprocess_medical_pages(sample_pages)


# Step 2: Generate chunks
chunks = chunk_medical_pages(
    processed_pages,
    chunk_size=1000,
    overlap=150
)


# Step 3: Display results
print("\nTOTAL CHUNKS:", len(chunks))


for chunk in chunks:

    print("\n--------------------------------")

    print("Chunk ID:", chunk["chunk_id"])

    print("Page Number:", chunk["page_number"])

    print("Chunk Index:", chunk["chunk_index"])

    print("Characters:", len(chunk["text"]))

    print("\nChunk Content:\n")

    print(chunk["text"])


# Step 4: Basic validation
assert len(chunks) > 0

assert all(
    chunk["page_number"] in [1, 2]
    for chunk in chunks
)

assert all(
    chunk["text"].strip()
    for chunk in chunks
)

print("\nChunking test passed successfully!")
