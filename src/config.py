
import os
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()


def get_config(key, default=None):
    """
    Read configuration from local environment
    or Streamlit Cloud secrets.
    """

    # First check local environment variables
    value = os.getenv(key)

    if value:
        return value

    # Then check Streamlit Cloud secrets
    try:
        import streamlit as st

        return st.secrets.get(key, default)

    except Exception:
        return default


# Gemini configuration
API_KEY = get_config("GEMINI_API_KEY")

MODEL = get_config(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)
