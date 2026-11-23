from io import BytesIO
import hashlib

import streamlit as st

from src.authentication import (
    login,
    logout,
    is_authenticated,
    get_current_user,
)
from src.patient_database import (
    create_patient,
    get_all_patients,
    search_patients,
)
from src.storage_manager import (
    upload_report,
    get_patient_reports,
    download_report,
    download_report_bytes,
    delete_report,
)
from src.document_processor import process_uploaded_documents
from src.medical_summary import generate_medical_summary
from src.text_preprocessor import preprocess_medical_pages
from src.text_chunker import chunk_medical_pages
from src.vector_store import MedicalVectorStore
from src.rag_pipeline import answer_medical_question


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="HealthSphere Medical AI",
    page_icon="🩺",
    layout="wide",
)


# =====================================================
# FILE WRAPPER FOR REPORTS DOWNLOADED FROM SUPABASE
# =====================================================

class NamedBytesIO(BytesIO):
    """Make downloaded bytes behave like a named uploaded file."""

    def __init__(self, file_bytes, filename):
        super().__init__(file_bytes)
        self.name = filename


# =====================================================
# SESSION STATE
# =====================================================

def initialize_state():
    defaults = {
        "chat_history": [],
        "selected_patient_id": None,
        "selected_patient": None,
        "loaded_saved_report_id": None,
        "loaded_saved_report_name": None,
        "pending_report_deletion": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


def clear_document_state():
    """Clear document and AI data when changing the active document/patient."""

    keys = [
        "extracted_pages",
        "processing_details",
        "processed_pages",
        "full_text",
        "medical_summary",
        "chunks",
        "vector_store",
        "document_ready",
        "document_hash",
        "review_complete",
        "loaded_saved_report_id",
        "loaded_saved_report_name",
        "loaded_saved_report_ids",
        "loaded_saved_report_names",
    ]

    for key in keys:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith("review_text_"):
            st.session_state.pop(key, None)

    st.session_state["chat_history"] = []


def calculate_document_hash(files):
    """Create a stable hash for the currently selected uploader files."""

    hasher = hashlib.sha256()

    for file in files or []:
        hasher.update(file.name.encode("utf-8"))
        hasher.update(file.getvalue())

    return hasher.hexdigest()


# =====================================================
# AUTHENTICATION
# =====================================================

if not is_authenticated():

    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.write("")
        st.write("")
        st.write("")

        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center; padding:10px 0 15px 0;">
                    <div style="font-size:48px;">🩺</div>
                    <h1 style="margin-bottom:0;">HealthSphere</h1>
                    <p style="color:gray; margin-top:5px;">
                        Medical Document AI Assistant
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.divider()
            st.markdown("### Welcome Back")
            st.caption("Sign in to access your patient workspace.")

            with st.form("login_form"):
                email = st.text_input(
                    "Email Address",
                    placeholder="Enter your email",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )

                submitted = st.form_submit_button(
                    "Sign In",
                    type="primary",
                    use_container_width=True,
                )

                if submitted:
                    try:
                        login(email, password)
                        st.rerun()
                    except Exception as error:
                        st.error(f"Login failed: {error}")

            st.divider()
            st.markdown(
                """
                <div style="text-align:center;">
                    <small style="color:gray;">
                        🔒 Secure access for authorized staff only
                    </small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.stop()


# =====================================================
# AUTHENTICATED USER
# =====================================================

current_user = get_current_user()


# =====================================================
# SIDEBAR: STAFF AND PATIENT MANAGEMENT
# =====================================================

with st.sidebar:
    st.title("🩺 HealthSphere")
    st.caption("Medical Document AI Assistant")
    st.divider()

    st.write("**Staff Account**")
    st.caption(current_user.get("email", "Unknown"))

    if st.button("Logout", use_container_width=True):
        logout()
        st.rerun()

    st.divider()
    st.subheader("👥 Patients")

    search_text = st.text_input(
        "Search Patients...",
        placeholder="Enter patient name",
        key="patient_search",
    )

    with st.expander("➕ Create New Patient", expanded=False):
        with st.form("create_patient_form"):
            patient_name = st.text_input("Patient Name")
            patient_age = st.number_input(
                "Age",
                min_value=0,
                max_value=130,
                value=25,
            )
            patient_gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female",
                    "Other",
                    "Prefer not to say",
                ],
            )

            create_submitted = st.form_submit_button(
                "Save Patient",
                type="primary",
                use_container_width=True,
            )

            if create_submitted:
                try:
                    new_patient = create_patient(
                        patient_name,
                        patient_age,
                        patient_gender,
                    )

                    clear_document_state()
                    st.session_state["selected_patient_id"] = new_patient["id"]
                    st.session_state["selected_patient"] = new_patient

                    st.success("Patient created successfully!")
                    st.rerun()

                except Exception as error:
                    st.error(f"Could not create patient: {error}")

    st.divider()

    try:
        if search_text.strip():
            patients = search_patients(search_text)
        else:
            patients = get_all_patients()
    except Exception as error:
        st.error(f"Unable to load patients: {error}")
        patients = []

    st.caption(f"Saved patients: {len(patients)}")

    if patients:
        patient_ids = [patient["id"] for patient in patients]
        current_index = 0
        if st.session_state.get("selected_patient_id") in patient_ids:
            current_index = patient_ids.index(
                st.session_state["selected_patient_id"]
            )

        patient_lookup = {patient["id"]: patient for patient in patients}
        if st.session_state.get("patient_selector") not in patient_ids:
            st.session_state["patient_selector"] = patient_ids[current_index]

        chosen_patient_id = st.selectbox(
            "Select Patient",
            options=patient_ids,
            index=current_index,
            format_func=lambda patient_id: (
                f"{patient_lookup[patient_id]['name']} · "
                f"{patient_lookup[patient_id]['age']} years · "
                f"{patient_lookup[patient_id]['gender']}"
            ),
            key="patient_selector",
        )
        chosen_patient = patient_lookup[chosen_patient_id]

        if chosen_patient_id != st.session_state.get("selected_patient_id"):
            clear_document_state()
            st.session_state["pending_report_deletion"] = None
            st.session_state["selected_patient_id"] = chosen_patient_id
            st.session_state["selected_patient"] = chosen_patient
            st.rerun()
    else:
        st.info("No patients found. Create a new patient to begin.")


# =====================================================
# SELECTED PATIENT / EMPTY STATE
# =====================================================

selected_patient = st.session_state.get("selected_patient")
selected_patient_id = st.session_state.get("selected_patient_id")

st.title("🩺 HealthSphere")
st.subheader("Medical Document AI Assistant")

if not selected_patient_id or not selected_patient:
    st.divider()
    st.info(
        "### No Patient Selected\n\n"
        "Select a patient from the sidebar or create a patient "
        "to start managing medical documents."
    )
    st.write(
        "Your medical document processing and AI tools "
        "will appear after selecting a patient."
    )
    st.caption("HealthSphere — Patient-centric Medical Document Assistant")
    st.stop()


# =====================================================
# PATIENT DASHBOARD
# =====================================================

st.header(selected_patient["name"])
st.caption(
    f"Gender: {selected_patient['gender']}  |  "
    f"Age: {selected_patient['age']} years"
)
st.divider()

st.info(
    "You are working in the selected patient's workspace. "
    "You can load all saved reports for this patient into one AI search system. "
    "The original files remain in private Supabase Storage; the combined AI index "
    "is held in this Streamlit session."
)


# =====================================================
# DOCUMENT UPLOAD
# =====================================================

st.header("📄 Upload Medical Reports")

if st.session_state.get("loaded_saved_report_ids"):
    loaded_names = st.session_state.get("loaded_saved_report_names", [])
    st.info(
        f"🧠 Patient-level AI workspace loaded with "
        f"{len(st.session_state['loaded_saved_report_ids'])} saved report(s): "
        + ", ".join(loaded_names)
    )

uploaded_files = st.file_uploader(
    "Upload PDF or medical report images",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"upload_{selected_patient_id}",
    help="Maximum size: 10 MB per file.",
)


# =====================================================
# PERSISTENT PATIENT REPORT MANAGEMENT
# =====================================================

st.divider()
st.header("📁 Saved Patient Reports")
st.caption(
    "Reports saved here are stored in your private Supabase Storage bucket."
)

# Save newly selected uploader files to Supabase.
if uploaded_files:
    if st.button(
        "☁️ Save Reports to Patient Records",
        type="primary",
        use_container_width=True,
    ):
        successful_uploads = 0

        for uploaded_file in uploaded_files:
            try:
                report = upload_report(
                    patient_id=selected_patient_id,
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    mime_type=uploaded_file.type,
                )
                successful_uploads += 1
                st.success(f"Saved: {report['filename']}")

            except Exception as error:
                st.error(
                    f"Upload failed for {uploaded_file.name}: {error}"
                )

        if successful_uploads:
            st.success(
                f"{successful_uploads} report(s) saved successfully!"
            )
            st.rerun()


# Retrieve and display all reports belonging to the selected patient.
try:
    saved_reports = get_patient_reports(selected_patient_id)

    if not saved_reports:
        st.info("No saved reports available for this patient.")

    else:
        st.success(f"{len(saved_reports)} saved report(s) found for {selected_patient['name']}.")

        if st.button(
            "🧠 Load All Patient Reports for AI",
            key=f"load_all_patient_reports_{selected_patient_id}",
            type="primary",
            use_container_width=True,
        ):
            try:
                clear_document_state()
                combined_pages = []
                combined_details = []
                loaded_ids = []
                loaded_names = []

                with st.spinner(
                    f"Downloading and extracting {len(saved_reports)} patient report(s)..."
                ):
                    for report in saved_reports:
                        saved_file = download_report_bytes(report["id"])
                        file_object = NamedBytesIO(
                            file_bytes=saved_file["file_bytes"],
                            filename=saved_file["filename"],
                        )

                        report_pages, report_details = process_uploaded_documents(
                            uploaded_files=[file_object]
                        )

                        if not report_pages:
                            combined_details.append({
                                "filename": saved_file["filename"],
                                "type": saved_file["mime_type"],
                                "pages": 0,
                                "warnings": ["No text or pages could be extracted from this report."],
                            })
                        else:
                            combined_pages.extend(report_pages)
                            combined_details.extend(report_details)

                        loaded_ids.append(saved_file["report_id"])
                        loaded_names.append(saved_file["filename"])

                if not combined_pages:
                    raise ValueError(
                        "No readable pages were extracted from this patient's saved reports. "
                        "Scanned PDFs may need OCR support before they can be searched."
                    )

                st.session_state["extracted_pages"] = combined_pages
                st.session_state["processing_details"] = combined_details
                st.session_state["loaded_saved_report_ids"] = loaded_ids
                st.session_state["loaded_saved_report_names"] = loaded_names
                st.session_state["review_complete"] = False

                st.success(
                    f"Loaded {len(loaded_ids)} saved report(s) and "
                    f"extracted {len(combined_pages)} page(s)."
                )
                st.rerun()

            except Exception as error:
                st.error(f"Could not load the patient's saved reports: {error}")

        st.markdown("#### Saved reports for this patient")

        for report in saved_reports:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**📄 {report['filename']}**")
                    st.caption(
                        f"Type: {report['mime_type']} | "
                        f"Size: {report['file_size'] / 1024:.2f} KB"
                    )
                    st.caption(f"Uploaded: {report['uploaded_at']}")
                    st.caption(f"Status: {report['status']}")

                with col2:
                    if st.button(
                        "⬇️ Download",
                        key=f"main_download_{report['id']}",
                        use_container_width=True,
                    ):
                        try:
                            result = download_report(report["id"])
                            st.link_button(
                                "Open / Download",
                                result["signed_url"],
                                use_container_width=True,
                            )
                        except Exception as error:
                            st.error(f"Download failed: {error}")

                    if st.button(
                        "🗑️ Delete",
                        key=f"main_delete_{report['id']}",
                        use_container_width=True,
                    ):
                        st.session_state["pending_report_deletion"] = report["id"]
                        st.rerun()

                if st.session_state.get("pending_report_deletion") == report["id"]:
                    st.warning(
                        "Permanently delete this report and its original file?"
                    )
                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:
                        if st.button(
                            "Confirm Delete",
                            key=f"confirm_main_{report['id']}",
                            type="primary",
                        ):
                            try:
                                delete_report(report["id"])

                                active_ids = st.session_state.get(
                                    "loaded_saved_report_ids", []
                                )
                                if report["id"] in active_ids or (
                                    st.session_state.get("loaded_saved_report_id")
                                    == report["id"]
                                ):
                                    clear_document_state()

                                st.session_state["pending_report_deletion"] = None
                                st.success("Report deleted successfully!")
                                st.rerun()

                            except Exception as error:
                                st.error(f"Deletion failed: {error}")

                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key=f"cancel_main_{report['id']}",
                        ):
                            st.session_state["pending_report_deletion"] = None
                            st.rerun()

except Exception as error:
    st.error(f"Unable to retrieve saved reports: {error}")

# =====================================================
# DOCUMENT HASH / CHANGE DETECTION
# =====================================================

has_documents = bool(uploaded_files)

# A saved report was loaded from Supabase. Do not let an empty uploader
# clear its extracted pages on the next Streamlit rerun.
saved_report_active = bool(
    st.session_state.get("loaded_saved_report_ids")
    or st.session_state.get("loaded_saved_report_id") is not None
)

if not saved_report_active:
    if has_documents:
        current_hash = calculate_document_hash(uploaded_files)

        if st.session_state.get("document_hash") != current_hash:
            clear_document_state()
            st.session_state["document_hash"] = current_hash

    elif st.session_state.get("document_hash") is not None:
        clear_document_state()


# =====================================================
# DISPLAY UPLOADED FILES
# =====================================================

if has_documents:
    st.subheader("Selected Documents")

    for file in uploaded_files:
        file_size_mb = file.size / (1024 * 1024)
        st.write(f"📄 **{file.name}** — {file_size_mb:.2f} MB")

        if file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            with st.expander(f"Preview: {file.name}"):
                st.image(
                    file.getvalue(),
                    caption=file.name,
                    use_container_width=True,
                )


# =====================================================
# STEP 1: DOCUMENT EXTRACTION FOR UPLOADER FILES
# =====================================================

if has_documents:
    if st.button("1. Extract Document Text", type="primary"):

        # If the user chooses the uploader workflow after loading a saved
        # report, switch the active document source to the uploader files.
        if (
            st.session_state.get("loaded_saved_report_ids")
            or st.session_state.get("loaded_saved_report_id") is not None
        ):
            clear_document_state()
            st.session_state["document_hash"] = (
                calculate_document_hash(uploaded_files)
            )

        oversized_files = [
            file.name
            for file in uploaded_files
            if file.size > 10 * 1024 * 1024
        ]

        if oversized_files:
            st.error(
                "Files exceed the 10 MB limit: "
                + ", ".join(oversized_files)
            )
        else:
            try:
                with st.spinner("Extracting medical documents..."):
                    extracted_pages, processing_details = (
                        process_uploaded_documents(
                            uploaded_files=uploaded_files
                        )
                    )

                    if not extracted_pages:
                        raise ValueError(
                            "No document pages were extracted."
                        )

                    st.session_state["extracted_pages"] = extracted_pages
                    st.session_state["processing_details"] = processing_details
                    st.session_state["review_complete"] = False

                st.success("Document extraction completed!")
                st.rerun()

            except Exception as error:
                st.error(f"Document extraction failed: {error}")


# =====================================================
# EXTRACTION RESULTS
# =====================================================

if "extracted_pages" in st.session_state:
    st.divider()
    st.header("Patient Report Extraction Results")

    extracted_pages = st.session_state["extracted_pages"]
    processing_details = st.session_state["processing_details"]

    col1, col2 = st.columns(2)
    col1.metric("Extracted Pages", len(extracted_pages))
    col2.metric("Processed Files", len(processing_details))

    st.subheader("Processing Information")

    for detail in processing_details:
        with st.expander(
            f"{detail['filename']} — {detail['type']}"
        ):
            st.write(f"Pages processed: {detail['pages']}")

            if "ocr_confidence" in detail:
                st.metric(
                    "Average OCR Confidence",
                    f"{detail['ocr_confidence']:.2f}%",
                )

            quality = detail.get("quality")
            if quality:
                st.write(
                    f"Image resolution: "
                    f"{quality['width']} × {quality['height']}"
                )
                st.write(f"Blur score: {quality['blur_score']}")
                st.write(f"Brightness: {quality['brightness']}")

            warnings = detail.get("warnings", [])
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success(
                    "No extraction or image-quality warnings."
                )


# =====================================================
# STEP 2: REVIEW EXTRACTED TEXT AND BUILD AI SEARCH
# =====================================================

if "extracted_pages" in st.session_state:
    st.divider()
    st.header("Review Patient Medical Records")
    st.write(
        "Review the extracted text from every loaded report. "
        "Use the source filename and page details to identify each record. "
        "Correct OCR errors before building the combined patient-level AI search system."
    )

    edited_pages = []

    for index, page in enumerate(st.session_state["extracted_pages"]):
        source_file = page.get("source_file", "Unknown document")
        source_page = page.get(
            "source_page_number",
            page["page_number"],
        )
        extraction_method = page.get(
            "extraction_method",
            "Unknown",
        )

        st.subheader(f"Page {page['page_number']}")
        st.caption(
            f"Source: {source_file} | "
            f"Original page: {source_page} | "
            f"Method: {extraction_method}"
        )

        edited_text = st.text_area(
            f"Review or correct extracted text — Page {page['page_number']}",
            value=page.get("text", ""),
            height=250,
            key=f"review_text_{index}",
        )

        edited_page = page.copy()
        edited_page["text"] = edited_text
        edited_pages.append(edited_page)

    if st.button("2. Build AI Search System", type="primary"):
        try:
            with st.spinner("Building medical AI search system..."):
                processed_pages = preprocess_medical_pages(edited_pages)

                text_parts = []
                for page in processed_pages:
                    if page["text"].strip():
                        text_parts.append(
                            f"Page {page['page_number']}\n{page['text']}"
                        )

                full_text = "\n\n".join(text_parts).strip()
                if not full_text:
                    raise ValueError("No readable text is available.")

                chunks = chunk_medical_pages(
                    processed_pages,
                    chunk_size=1000,
                    overlap=150,
                )
                if not chunks:
                    raise ValueError("Unable to generate document chunks.")

                vector_store = MedicalVectorStore()
                total_vectors = vector_store.build_index(chunks)

                st.session_state["processed_pages"] = processed_pages
                st.session_state["full_text"] = full_text
                st.session_state["chunks"] = chunks
                st.session_state["vector_store"] = vector_store
                st.session_state["document_ready"] = True
                st.session_state["review_complete"] = True
                st.session_state["chat_history"] = []
                st.session_state.pop("medical_summary", None)

            st.success("Medical AI search system created successfully!")
            st.success(f"Generated {total_vectors} searchable vectors.")
            st.rerun()

        except Exception as error:
            st.error(f"AI search system creation failed: {error}")


# =====================================================
# STEP 3: DOCUMENT INFORMATION
# =====================================================

if st.session_state.get("document_ready"):
    processed_pages = st.session_state["processed_pages"]
    full_text = st.session_state["full_text"]
    chunks = st.session_state["chunks"]

    st.divider()
    st.header("Document Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", len(processed_pages))
    col2.metric("Extracted Characters", len(full_text))
    col3.metric("Total Chunks", len(chunks))

    st.subheader("Processed Medical Report")

    for page in processed_pages:
        with st.expander(
            f"Page {page['page_number']} — "
            f"{page.get('source_file', 'Unknown document')}"
        ):
            st.text(page["text"] or "No text found.")

    with st.expander("View Complete Extracted Text"):
        st.text_area(
            "Complete Medical Report",
            full_text,
            height=300,
            disabled=True,
        )


# =====================================================
# STEP 4: MEDICAL SUMMARY
# =====================================================

if st.session_state.get("document_ready"):
    st.divider()
    st.header("AI Patient Records Summary")
    st.write("Generate an overview based on all reports loaded for this patient.")

    if st.button("Generate Medical Summary"):
        try:
            with st.spinner("Gemini AI is analyzing the report..."):
                summary = generate_medical_summary(
                    st.session_state["full_text"]
                )

                if summary and summary.strip():
                    st.session_state["medical_summary"] = summary
                    st.rerun()
                else:
                    st.error("Gemini AI returned an empty summary.")

        except Exception as error:
            st.error(f"Summary generation failed: {error}")

    if "medical_summary" in st.session_state:
        st.subheader("AI Generated Summary")
        st.markdown(st.session_state["medical_summary"])


# =====================================================
# STEP 5: MEDICAL RAG CHAT
# =====================================================

if st.session_state.get("document_ready"):
    st.divider()
    st.header("Ask Questions About This Patient's Records")
    st.write(
        "Ask questions across all reports loaded for the selected patient. "
        "Source references identify the report filename and page used for the answer."
    )

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Source References"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**File:** {source.get('source_file', 'Unknown')} "
                            f"| **Original Page:** "
                            f"{source.get('source_page_number', source.get('page_number', 'Unknown'))} "
                            f"| **Method:** {source.get('extraction_method', 'Unknown')} "
                            f"| **Chunk:** {source.get('chunk_id')} "
                            f"| **Similarity:** "
                            f"{source.get('similarity_score', 0):.4f}"
                        )
                        st.text(source["text"])

    user_question = st.chat_input(
        "Ask about this patient's medical history or saved reports..."
    )

    if user_question:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_question,
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching the medical report..."):
                    result = answer_medical_question(
                        question=user_question,
                        vector_store=st.session_state["vector_store"],
                        top_k=3,
                    )

                    answer = result["answer"]
                    st.markdown(answer)

                    with st.expander("View Source References"):
                        for source in result["sources"]:
                            st.markdown(
                                f"**File:** {source.get('source_file', 'Unknown')} "
                                f"| **Original Page:** "
                                f"{source.get('source_page_number', source.get('page_number', 'Unknown'))} "
                                f"| **Method:** {source.get('extraction_method', 'Unknown')} "
                                f"| **Chunk:** {source.get('chunk_id')} "
                                f"| **Similarity:** "
                                f"{source.get('similarity_score', 0):.4f}"
                            )
                            st.text(source["text"])

                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result["sources"],
                    })

        except (RuntimeError, ValueError) as error:
            st.warning(str(error))
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": f"⚠️ {str(error)}",
            })

        except Exception as error:
            st.error(
                "An unexpected error occurred while generating the answer."
            )
            print(f"RAG Error: {type(error).__name__}: {error}")


# =====================================================
# MEDICAL DISCLAIMER
# =====================================================

st.divider()
st.caption(
    "HealthSphere provides AI-generated explanations of uploaded "
    "documents for educational purposes only. It does not provide "
    "medical diagnosis or treatment. OCR and AI-generated content "
    "may contain errors. Always verify medical values against the "
    "original report and consult a qualified healthcare professional."
)
