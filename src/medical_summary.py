
import os

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def generate_medical_summary(medical_text):

    if not medical_text or not medical_text.strip():
        raise ValueError("Medical report text is empty.")

    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment.")

    prompt = f"""
    You are HealthSphere, a medical document
    explanation assistant.

    Explain this medical report in simple language.

    Instructions:
    1. Explain the type of medical report.
    2. Explain medical terminology.
    3. Preserve all original values and units exactly.
    4. Mention reference ranges only when present in the report.
    5. Mention abnormal flags only when supported by the report.
    6. Never invent missing information or reference ranges.
    7. Do not silently correct unclear or malformed values.
    8. Do not diagnose diseases.
    9. Do not prescribe medication or treatment.
    10. Mention unclear or unreadable information.
    11. Recommend professional clinical interpretation.

    Structure:

    1. Report Overview
    2. Important Test Results
    3. Explanation in Simple Language
    4. Important Notes

    Medical Report:
    {medical_text}
    """

    client = genai.Client(api_key=API_KEY)

    try:
        print("Sending request to Gemini API...")

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "temperature": 0.2
            }
        )

        summary = response.text

        if not summary:
            raise ValueError("Gemini returned an empty response.")

        print("Medical summary generated successfully.")

        return summary

    finally:
        client.close()
