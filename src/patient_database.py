
from src.authentication import get_authenticated_client


def _get_client():
    """
    Return the Supabase client for the currently logged-in user.
    """
    supabase = get_authenticated_client()

    if supabase is None:
        raise RuntimeError(
            "Please log in before accessing patient records."
        )

    return supabase


def create_patient(name, age, gender):
    """
    Create a patient for the currently authenticated user.
    The database automatically assigns owner_id.
    """

    if not isinstance(name, str):
        raise ValueError("Patient name must be text.")

    name = name.strip()

    if len(name) < 2:
        raise ValueError(
            "Patient name must contain at least 2 characters."
        )

    age = int(age)

    if not 0 <= age <= 130:
        raise ValueError(
            "Patient age must be between 0 and 130."
        )

    allowed_genders = [
        "Male",
        "Female",
        "Other",
        "Prefer not to say"
    ]

    if gender not in allowed_genders:
        raise ValueError("Invalid gender selection.")

    supabase = _get_client()

    response = (
        supabase.table("patients")
        .insert({
            "name": name,
            "age": age,
            "gender": gender
        })
        .execute()
    )

    return response.data[0]


def get_all_patients():
    """
    Fetch patients visible to the authenticated user.
    RLS restricts results to that user's records.
    """

    supabase = _get_client()

    response = (
        supabase.table("patients")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def search_patients(search_text):
    """
    Search patients by name.
    """

    if not isinstance(search_text, str):
        search_text = ""

    search_text = search_text.strip()

    if not search_text:
        return get_all_patients()

    supabase = _get_client()

    response = (
        supabase.table("patients")
        .select("*")
        .ilike("name", f"%{search_text}%")
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def get_patient(patient_id):
    """
    Fetch one patient by ID.
    RLS ensures the patient belongs to the authenticated user.
    """

    supabase = _get_client()

    response = (
        supabase.table("patients")
        .select("*")
        .eq("id", patient_id)
        .maybe_single()
        .execute()
    )

    return response.data


def delete_patient(patient_id):
    """
    Delete a patient belonging to the authenticated user.
    Associated report records are deleted by database cascade.
    """

    supabase = _get_client()

    response = (
        supabase.table("patients")
        .delete()
        .eq("id", patient_id)
        .execute()
    )

    return response.data
