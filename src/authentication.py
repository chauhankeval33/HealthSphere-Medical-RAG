
import streamlit as st

from src.supabase_client import get_supabase_client


def login(email, password):
    """
    Authenticate a staff member using Supabase Auth.
    """

    email = email.strip()

    if not email or not password:
        raise ValueError("Email and password are required.")

    supabase = get_supabase_client()

    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    if not response.session or not response.user:
        raise RuntimeError("Authentication failed.")

    # Keep the authenticated client for this Streamlit session.
    st.session_state["supabase_client"] = supabase

    st.session_state["auth_user"] = {
        "id": str(response.user.id),
        "email": response.user.email
    }

    return st.session_state["auth_user"]


def get_authenticated_client():
    """
    Return the authenticated Supabase client.

    Returns None when nobody is logged in.
    """

    return st.session_state.get("supabase_client")


def get_current_user():
    """
    Return the currently logged-in staff member.
    """

    return st.session_state.get("auth_user")


def is_authenticated():
    """
    Check whether a staff member has logged in.
    """

    return (
        st.session_state.get("supabase_client") is not None
        and st.session_state.get("auth_user") is not None
    )


def logout():
    """
    Sign out and clear patient-specific application data.
    """

    supabase = get_authenticated_client()

    if supabase:
        try:
            supabase.auth.sign_out()
        finally:
            pass

    # Clear authentication and patient-specific data.
    session_keys = [
        "supabase_client",
        "auth_user",
        "selected_patient_id",
        "selected_report_id",
        "processed_pages",
        "full_text",
        "chunks",
        "vector_store",
        "document_ready",
        "review_complete",
        "medical_summary",
        "chat_history"
    ]

    for key in session_keys:
        st.session_state.pop(key, None)
