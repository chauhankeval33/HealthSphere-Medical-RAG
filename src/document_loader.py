      
import pymupdf

def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF file.

    Parameters:
        pdf_file: Uploaded PDF file from Streamlit.

    Returns:
        extracted_pages: List containing text and page numbers.
    """

    extracted_pages = []

    try:
        # Open PDF directly from uploaded file bytes
        pdf_bytes = pdf_file.getvalue()

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        # Extract text page by page
        for page_number, page in enumerate(document, start=1):

            page_text = page.get_text("text")

            extracted_pages.append({
                "page_number": page_number,
                "text": page_text.strip()
            })

        document.close()

        return extracted_pages

    except Exception as e:
        raise ValueError(f"PDF extraction failed: {str(e)}")


def get_full_text(extracted_pages):
    """
    Combine extracted text from all pages.
    """

    full_text = ""

    for page in extracted_pages:

        full_text += (
            f"\n\nPage {page['page_number']}\n"
            f"{page['text']}"
        )

    return full_text.strip()


