
import os

from dotenv import load_dotenv
from supabase import create_client

# Load local .env variables
load_dotenv()


def get_setting(name):
    """
    Read settings from Streamlit secrets first,
    then fall back to local environment variables.
    """

    try:
        import streamlit as st

        value = st.secrets.get(name)

        if value:
            return value

    except Exception:
        pass

    return os.getenv(name)


def get_supabase_client():
    """
    Create and return a Supabase client.
    """

    supabase_url = get_setting("SUPABASE_URL")
    supabase_key = get_setting("SUPABASE_PUBLISHABLE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase URL or Publishable Key is missing. "
            "Check your .env or Streamlit secrets."
        )

    return create_client(supabase_url, supabase_key)
