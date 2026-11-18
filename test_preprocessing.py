
from src.text_preprocessor import (
    clean_medical_text,
    preprocess_medical_pages
)


sample_text = """
    COMPLETE BLOOD COUNT


    Hemoglobin:     13.5 g/dL


    WBC: 7500 cells/mcL


    Platelets: 250000 cells/mcL
"""


print("ORIGINAL TEXT:")
print(sample_text)


cleaned_text = clean_medical_text(sample_text)


print("\nCLEANED TEXT:")
print(cleaned_text)


sample_pages = [
    {
        "page_number": 1,
        "text": sample_text
    }
]


processed_pages = preprocess_medical_pages(sample_pages)


print("\nPROCESSED PAGES:")
print(processed_pages)
