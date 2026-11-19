
import cv2
import numpy as np


def assess_image_quality(image):
    """
    Assess basic image quality before OCR processing.

    Checks:
    - Resolution
    - Blur
    - Brightness

    Returns quality metrics and warnings.
    """

    try:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image supplied.")

        if len(image.shape) == 3:
            grayscale = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )
        else:
            grayscale = image

        height, width = grayscale.shape

        # Resolution check
        resolution_ok = width >= 1000 and height >= 1000

        # Blur detection
        blur_score = cv2.Laplacian(
            grayscale, cv2.CV_64F
        ).var()

        # Brightness measurement
        brightness = float(np.mean(grayscale))

        warnings = []

        if not resolution_ok:
            warnings.append(
                "Image resolution is low. "
                "Use a higher-resolution image if possible."
            )

        if blur_score < 50:
            warnings.append(
                "Image may be blurry. "
                "Try capturing the report again with better focus."
            )

        if brightness < 60:
            warnings.append(
                "Image appears too dark. "
                "Improve lighting and capture again."
            )

        return {
            "width": width,
            "height": height,
            "resolution_ok": resolution_ok,
            "blur_score": round(float(blur_score), 2),
            "brightness": round(brightness, 2),
            "warnings": warnings,
            "is_acceptable": len(warnings) == 0
        }

    except Exception as e:
        raise ValueError(
            f"Image quality assessment failed: {str(e)}"
        ) from e
