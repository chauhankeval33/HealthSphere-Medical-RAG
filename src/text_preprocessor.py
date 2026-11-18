
import re


def clean_medical_text(text):

    """
    Clean extracted medical report text.

    Preserve:
    - Medical values
    - Units
    - Reference ranges
    - Medical terminology
    """

    if not text or not text.strip():
        return ""

    # 1. Normalize Windows and Unix line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 2. Remove unnecessary spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # 3. Remove spaces at the beginning and end of lines
    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    # 4. Remove unnecessary empty lines
    cleaned_lines = []

    for line in lines:

        if line:
            cleaned_lines.append(line)

    # 5. Reconstruct cleaned text
    cleaned_text = "\n".join(cleaned_lines)

    # 6. Remove excessive repeated blank lines
    cleaned_text = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_text
    )

    return cleaned_text.strip()


def preprocess_medical_pages(pages):

    """
    Clean extracted text page by page.
    Preserve page numbers for future RAG citations.
    """

    processed_pages = []

    for page in pages:

        cleaned_text = clean_medical_text(
            page["text"]
        )

        processed_pages.append({

            "page_number": page["page_number"],

            "text": cleaned_text

        })

    return processed_pages
