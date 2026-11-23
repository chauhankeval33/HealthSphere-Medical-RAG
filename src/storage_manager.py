
from pathlib import Path
from uuid import uuid4

from src.authentication import (
    get_authenticated_client,
    get_current_user
)


BUCKET_NAME = "medical-reports"

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_FILES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png"
}


def upload_report(patient_id, file_name, file_bytes, mime_type):
    """
    Upload a patient's medical report to private Supabase Storage
    and save its metadata in the reports PostgreSQL table.
    """

    # 1. Check authentication
    supabase = get_authenticated_client()
    user = get_current_user()

    if supabase is None or user is None:
        raise RuntimeError("Please log in before uploading reports.")

    owner_id = user["id"]

    # 2. Validate patient ownership
    patient_response = (
        supabase.table("patients")
        .select("id, owner_id")
        .eq("id", str(patient_id))
        .maybe_single()
        .execute()
    )

    patient = patient_response.data

    if not patient or patient["owner_id"] != owner_id:
        raise PermissionError("Patient not found or access denied.")

    # 3. Validate file
    if not file_name or not isinstance(file_name, str):
        raise ValueError("Invalid file name.")

    extension = Path(file_name).suffix.lower()

    if extension not in ALLOWED_FILES:
        raise ValueError("Only PDF, JPG, JPEG and PNG files are allowed.")

    if mime_type != ALLOWED_FILES[extension]:
        raise ValueError("File type does not match its extension.")

    if not isinstance(file_bytes, bytes) or not file_bytes:
        raise ValueError("File is empty or invalid.")

    file_size = len(file_bytes)

    if file_size > MAX_FILE_SIZE:
        raise ValueError("File exceeds the 10 MB upload limit.")

    # 4. Generate a unique private Storage path
    unique_filename = f"{uuid4()}{extension}"

    storage_path = (
        f"{owner_id}/{patient_id}/{unique_filename}"
    )

    # 5. Upload original file to Supabase Storage
    supabase.storage.from_(BUCKET_NAME).upload(
        storage_path,
        file_bytes,
        file_options={
            "content-type": mime_type,
            "upsert": "false"
        }
    )

    # 6. Save report metadata in PostgreSQL
    try:
        response = (
            supabase.table("reports")
            .insert({
                "patient_id": str(patient_id),
                "owner_id": owner_id,
                "filename": Path(file_name).name,
                "storage_path": storage_path,
                "mime_type": mime_type,
                "file_size": file_size,
                "status": "uploaded"
            })
            .execute()
        )

        if not response.data:
            raise RuntimeError("Report metadata was not saved.")

        return response.data[0]

    except Exception:
        # Remove uploaded file if database insertion fails
        try:
            supabase.storage.from_(BUCKET_NAME).remove(
                [storage_path]
            )
        except Exception:
            pass

        raise

def get_patient_reports(patient_id):
    """
    Retrieve all reports belonging to a patient.
    RLS ensures users can only access their own reports.
    """

    supabase = get_authenticated_client()
    user = get_current_user()

    if supabase is None or user is None:
        raise RuntimeError("Please log in before accessing reports.")

    response = (
        supabase.table("reports")
        .select("*")
        .eq("patient_id", str(patient_id))
        .eq("owner_id", user["id"])
        .order("uploaded_at", desc=True)
        .execute()
    )

    return response.data

def download_report(report_id):
    """
    Generate a temporary signed URL for an authorized report.
    """

    supabase = get_authenticated_client()
    user = get_current_user()

    if supabase is None or user is None:
        raise RuntimeError("Please log in before downloading reports.")

    # 1. Retrieve report metadata belonging to this user
    response = (
        supabase.table("reports")
        .select("id, owner_id, storage_path, filename")
        .eq("id", str(report_id))
        .eq("owner_id", user["id"])
        .maybe_single()
        .execute()
    )

    report = response.data

    if not report:
        raise PermissionError("Report not found or access denied.")

    # 2. Generate a temporary signed URL
    signed_response = (
        supabase.storage
        .from_(BUCKET_NAME)
        .create_signed_url(
            report["storage_path"],
            300
        )
    )

    # 3. Return URL
    if isinstance(signed_response, dict):
        signed_url = (
            signed_response.get("signedURL")
            or signed_response.get("signedUrl")
        )
    else:
        signed_url = getattr(signed_response, "signedURL", None)

    if not signed_url:
        raise RuntimeError("Could not generate signed download URL.")

    return {
        "filename": report["filename"],
        "signed_url": signed_url
    }

def delete_report(report_id):
    """
    Delete an authorized report from Supabase Storage
    and remove its metadata from PostgreSQL.
    """

    supabase = get_authenticated_client()
    user = get_current_user()

    if supabase is None or user is None:
        raise RuntimeError("Please log in before deleting reports.")

    # 1. Find report belonging to the logged-in user
    response = (
        supabase.table("reports")
        .select("id, owner_id, storage_path")
        .eq("id", str(report_id))
        .eq("owner_id", user["id"])
        .maybe_single()
        .execute()
    )

    report = response.data

    if not report:
        raise PermissionError("Report not found or access denied.")

    # 2. Delete original file from private Storage
    supabase.storage.from_(BUCKET_NAME).remove(
        [report["storage_path"]]
    )

    # 3. Delete report metadata from PostgreSQL
    delete_response = (
        supabase.table("reports")
        .delete()
        .eq("id", str(report_id))
        .eq("owner_id", user["id"])
        .execute()
    )

    if not delete_response.data:
        raise RuntimeError(
            "Storage file was deleted, but report metadata "
            "could not be deleted. Please investigate."
        )

    return True

def download_report_bytes(report_id):
    """
    Download the original report bytes from private Supabase Storage.

    Only the authenticated owner of the report can download it.

    Returns:
        Dictionary containing filename, mime_type and file_bytes.
    """

    supabase = get_authenticated_client()
    user = get_current_user()

    if not user:
        raise PermissionError("Please log in to access this report.")

    # Step 1: Retrieve report metadata belonging to this user
    response = (
        supabase.table("reports")
        .select("*")
        .eq("id", report_id)
        .eq("owner_id", user["id"])
        .maybe_single()
        .execute()
    )

    report = response.data

    if not report:
        raise ValueError(
            "Report not found or you do not have permission to access it."
        )

    # Step 2: Download original file from private Storage
    file_bytes = (
        supabase.storage
        .from_("medical-reports")
        .download(report["storage_path"])
    )

    if not file_bytes:
        raise ValueError("The stored report is empty or unavailable.")

    # Step 3: Return file information and original bytes
    return {
        "report_id": report["id"],
        "filename": report["filename"],
        "mime_type": report["mime_type"],
        "file_bytes": file_bytes
    }
