
from src.pdf_processor import extract_pdf_pages
from src.image_processor import load_medical_image
from src.image_quality import assess_image_quality
from src.ocr_processor import extract_text_from_image


SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png")


def get_ocr_warnings(confidence, extracted_text):
    """
    Generate warnings based on OCR confidence and extracted text.
    """

    warnings = []

    if not extracted_text.strip():
        warnings.append(
            "No readable text was detected. "
            "Try uploading a clearer image."
        )

    elif confidence < 50:
        warnings.append(
            "Low OCR confidence. Extracted text may contain "
            "significant recognition errors. Review carefully."
        )

    elif confidence < 70:
        warnings.append(
            "Moderate OCR confidence. Review extracted text "
            "carefully against the original report."
        )

    return warnings


def process_uploaded_documents(uploaded_files):
    """
    Process uploaded PDFs and medical report images.

    Supports:
    - PDF files
    - JPG / JPEG / PNG images
    - Multiple uploaded files

    Returns:
        extracted_pages:
            Unified page-wise extracted text with metadata.

        processing_details:
            Extraction, OCR and image-quality information.
    """

    if not uploaded_files:
        raise ValueError("Please upload a medical document.")

    extracted_pages = []
    processing_details = []

    current_page = 1

    for uploaded_file in uploaded_files:

        filename = getattr(
            uploaded_file,
            "name",
            "uploaded_document"
        )

        filename_lower = filename.lower()

        try:

            # ========================================
            # PDF PROCESSING
            # ========================================

            if filename_lower.endswith(".pdf"):

                pdf_pages = extract_pdf_pages(uploaded_file)

                if not pdf_pages:
                    raise ValueError(
                        "The PDF contains no pages."
                    )

                file_warnings = []

                for page in pdf_pages:

                    page_text = page["text"].strip()

                    extracted_pages.append({
                        "page_number": current_page,
                        "source_page_number": page["page_number"],
                        "text": page_text,
                        "source_file": filename,
                        "extraction_method": "PDF"
                    })

                    if not page_text:
                        file_warnings.append(
                            f"Page {page['page_number']} contains "
                            "no extractable text. It may be a scanned page."
                        )

                    current_page += 1

                processing_details.append({
                    "filename": filename,
                    "type": "PDF",
                    "pages": len(pdf_pages),
                    "warnings": file_warnings
                })

            # ========================================
            # IMAGE PROCESSING
            # ========================================

            elif filename_lower.endswith(SUPPORTED_IMAGE_FORMATS):

                # Load image once.
                image = load_medical_image(uploaded_file)

                # Assess image quality.
                quality = assess_image_quality(image)

                # Extract OCR text from the loaded image.
                ocr_result = extract_text_from_image(image)

                confidence = ocr_result["average_confidence"]

                extracted_text = ocr_result["text"]

                # Combine image-quality and OCR warnings.
                warnings = list(quality["warnings"])

                warnings.extend(
                    get_ocr_warnings(
                        confidence,
                        extracted_text
                    )
                )

                extracted_pages.append({
                    "page_number": current_page,
                    "source_page_number": 1,
                    "text": extracted_text,
                    "source_file": filename,
                    "extraction_method": "OCR"
                })

                processing_details.append({
                    "filename": filename,
                    "type": "Image",
                    "pages": 1,
                    "ocr_confidence": confidence,
                    "quality": quality,
                    "warnings": warnings
                })

                current_page += 1

            # ========================================
            # UNSUPPORTED FILE
            # ========================================

            else:

                raise ValueError(
                    f"Unsupported file format: {filename}. "
                    "Supported formats: PDF, JPG, JPEG and PNG."
                )

        except Exception as e:

            raise ValueError(
                f"Failed to process {filename}: {str(e)}"
            ) from e

    # ========================================
    # FINAL VALIDATION
    # ========================================

    if not extracted_pages:
        raise ValueError(
            "No pages could be extracted from the uploaded documents."
        )

    return extracted_pages, processing_details
