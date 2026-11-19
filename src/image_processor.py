
import cv2
import numpy as np


def load_medical_image(image_file):
    """
    Load a medical report image using OpenCV.

    Supports:
        JPG, JPEG, PNG

    Parameters:
        image_file: Streamlit UploadedFile, file path,
                    or image bytes.

    Returns:
        OpenCV image in BGR format.
    """

    try:
        # Read image bytes
        if hasattr(image_file, "getvalue"):
            image_bytes = image_file.getvalue()

        elif isinstance(image_file, bytes):
            image_bytes = image_file

        elif isinstance(image_file, str):
            image = cv2.imread(image_file)

            if image is None:
                raise ValueError("Unable to read image file.")

            return image

        else:
            image_bytes = image_file.read()

        if not image_bytes:
            raise ValueError("Uploaded image is empty.")

        # Convert bytes into NumPy array
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        # Decode image
        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                "Invalid or unsupported image file."
            )

        return image

    except Exception as e:
        raise ValueError(
            f"Image loading failed: {str(e)}"
        ) from e


def preprocess_medical_image(image):
    """
    Preprocess a medical report image for OCR.

    Processing:
        1. Convert to grayscale.
        2. Resize small images.
        3. Apply noise reduction.
        4. Improve contrast.

    Returns:
        Processed grayscale image.
    """

    try:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image supplied.")

        # Convert BGR to grayscale
        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Resize small images for better OCR
        height, width = grayscale.shape

        if width < 1500:
            scale = 1500 / width

            grayscale = cv2.resize(
                grayscale,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        # Reduce image noise
        denoised = cv2.medianBlur(
            grayscale,
            3
        )

        # Improve contrast
        enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        ).apply(denoised)

        return enhanced

    except Exception as e:
        raise ValueError(
            f"Image preprocessing failed: {str(e)}"
        ) from e
