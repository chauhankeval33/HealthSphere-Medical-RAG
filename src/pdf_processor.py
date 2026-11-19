
from io import BytesIO

from pypdf import PdfReader


def extract_pdf_pages(pdf_file):
    """
    Extract text from an uploaded PDF using pypdf.

    Parameters:
        pdf_file: Streamlit UploadedFile or PDF bytes.

    Returns:
        List of dictionaries containing page numbers and extracted text.
    """

    try:
        # Get PDF bytes
        if hasattr(pdf_file, "getvalue"):
            pdf_bytes = pdf_file.getvalue()

        elif isinstance(pdf_file, bytes):
            pdf_bytes = pdf_file

        else:
            pdf_bytes = pdf_file.read()

        if not pdf_bytes:
            raise ValueError("The uploaded PDF is empty.")

        # Read PDF from memory
        reader = PdfReader(BytesIO(pdf_bytes))

        # Check encrypted PDF
        if reader.is_encrypted:
            raise ValueError(
                "Encrypted PDFs are not supported currently."
            )

        extracted_pages = []

        # Extract text page by page
        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text() or ""

            extracted_pages.append({
                "page_number": page_number,
                "text": page_text.strip()
            })

        return extracted_pages

    except Exception as e:
        raise ValueError(
            f"PDF extraction failed: {str(e)}"
        ) from e
