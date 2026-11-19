
import pytesseract

from src.image_processor import (
    load_medical_image,
    preprocess_medical_image
)


def extract_text_from_image(image_file):
    """
    Extract text and OCR confidence from a medical report image.

    Accepts:
    - Uploaded image file
    - Image bytes
    - File path
    - Already loaded OpenCV image (NumPy array)
    """

    try:

        # Reuse an already loaded image when provided
        if hasattr(image_file, "shape"):
            image = image_file
        else:
            image = load_medical_image(image_file)

        processed_image = preprocess_medical_image(image)

        ocr_config = "--oem 3 --psm 6"

        extracted_text = pytesseract.image_to_string(
            processed_image,
            lang="eng",
            config=ocr_config
        )

        ocr_data = pytesseract.image_to_data(
            processed_image,
            lang="eng",
            config=ocr_config,
            output_type=pytesseract.Output.DICT
        )

        confidence_values = []

        for confidence in ocr_data["conf"]:

            try:

                confidence = float(confidence)

                if confidence >= 0:
                    confidence_values.append(confidence)

            except (ValueError, TypeError):
                continue

        average_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )

        return {
            "text": extracted_text.strip(),
            "average_confidence": round(
                average_confidence, 2
            )
        }

    except Exception as e:

        raise ValueError(
            f"OCR extraction failed: {str(e)}"
        ) from e
