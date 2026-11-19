
import hashlib
import streamlit as st

from src.document_processor import process_uploaded_documents
from src.medical_summary import generate_medical_summary
from src.text_preprocessor import preprocess_medical_pages
from src.text_chunker import chunk_medical_pages
from src.vector_store import MedicalVectorStore
from src.rag_pipeline import answer_medical_question


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HealthSphere Medical AI",
    page_icon="🩺",
    layout="wide"
)


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("HealthSphere")

st.subheader("Medical Document AI Assistant")

st.write(
    "Upload your medical report as a PDF or image, "
    "review extracted text, and ask questions using AI."
)

st.info(
    "Upload fictional or de-identified reports during testing. "
    "Review extracted medical values carefully before using AI."
)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def clear_document_state():
    """
    Clear previously processed document data.
    """

    keys_to_clear = [
        "extracted_pages",
        "processing_details",
        "processed_pages",
        "full_text",
        "medical_summary",
        "chunks",
        "vector_store",
        "document_ready",
        "document_hash",
        "review_complete"
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    # Remove previous OCR review widget values.
    for key in list(st.session_state.keys()):
        if key.startswith("review_text_"):
            st.session_state.pop(key, None)

    st.session_state["chat_history"] = []


# ==========================================
# DOCUMENT UPLOAD
# ==========================================

st.divider()

st.header("Upload Medical Report")

uploaded_files = st.file_uploader(
    "Upload PDF or medical report images",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help=(
        "You can upload multiple PDFs or images. "
        "Maximum size: 10 MB per file."
    )
)


# ==========================================
# DOCUMENT HASH / CHANGE DETECTION
# ==========================================

def calculate_document_hash(files):
    """
    Generate a unique hash for all uploaded documents.
    """

    hasher = hashlib.sha256()

    for file in files or []:

        hasher.update(file.name.encode("utf-8"))
        hasher.update(file.getvalue())

    return hasher.hexdigest()


has_documents = bool(uploaded_files)

if has_documents:

    current_hash = calculate_document_hash(uploaded_files)

    if st.session_state.get("document_hash") != current_hash:

        clear_document_state()

        st.session_state["document_hash"] = current_hash

else:

    if st.session_state.get("document_hash") is not None:
        clear_document_state()


# ==========================================
# DISPLAY UPLOADED FILES
# ==========================================

if has_documents:

    st.subheader("Selected Documents")

    for file in uploaded_files:

        file_size_mb = file.size / (1024 * 1024)

        st.write(
            f"📄 **{file.name}** — {file_size_mb:.2f} MB"
        )

        # Preview uploaded images.
        extension = file.name.lower()

        if extension.endswith((".jpg", ".jpeg", ".png")):

            with st.expander(f"Preview: {file.name}"):

                st.image(
                    file.getvalue(),
                    caption=file.name,
                    use_container_width=True
                )


# ==========================================
# STEP 1: EXTRACT DOCUMENT TEXT
# ==========================================

if has_documents:

    if st.button(
        "1. Extract Document Text",
        type="primary"
    ):

        # Validate individual file sizes.
        oversized_files = [
            file.name
            for file in uploaded_files
            if file.size > 10 * 1024 * 1024
        ]

        if oversized_files:

            st.error(
                "The following files exceed the 10 MB limit: "
                + ", ".join(oversized_files)
            )

        else:

            try:

                with st.spinner(
                    "Extracting text from your medical documents..."
                ):

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

                    st.session_state["processing_details"] = (
                        processing_details
                    )

                    st.session_state["review_complete"] = False

                    st.success(
                        "Document extraction completed!"
                    )

            except Exception as error:

                st.error(
                    f"Document extraction failed: {error}"
                )


# ==========================================
# DISPLAY EXTRACTION RESULTS AND WARNINGS
# ==========================================

if "extracted_pages" in st.session_state:

    st.divider()

    st.header("Document Extraction Results")

    extracted_pages = st.session_state["extracted_pages"]

    processing_details = st.session_state["processing_details"]

    col1, col2 = st.columns(2)

    col1.metric(
        "Extracted Pages",
        len(extracted_pages)
    )

    col2.metric(
        "Processed Files",
        len(processing_details)
    )

    st.subheader("Processing Information")

    for detail in processing_details:

        with st.expander(
            f"{detail['filename']} — {detail['type']}"
        ):

            st.write(
                f"Pages processed: {detail['pages']}"
            )

            if "ocr_confidence" in detail:

                st.metric(
                    "Average OCR Confidence",
                    f"{detail['ocr_confidence']:.2f}%"
                )

            quality = detail.get("quality")

            if quality:

                st.write(
                    f"Image resolution: "
                    f"{quality['width']} × {quality['height']}"
                )

                st.write(
                    f"Blur score: {quality['blur_score']}"
                )

                st.write(
                    f"Brightness: {quality['brightness']}"
                )

            warnings = detail.get("warnings", [])

            if warnings:

                for warning in warnings:
                    st.warning(warning)

            else:

                st.success(
                    "No extraction or image-quality warnings."
                )


# ==========================================
# STEP 2: REVIEW AND EDIT EXTRACTED TEXT
# ==========================================

if "extracted_pages" in st.session_state:

    st.divider()

    st.header("Review Extracted Medical Text")

    st.write(
        "Review the extracted text against your original report. "
        "You can manually correct OCR recognition errors before "
        "building the AI search system."
    )

    edited_pages = []

    for index, page in enumerate(
        st.session_state["extracted_pages"]
    ):

        source_file = page.get(
            "source_file",
            "Unknown document"
        )

        source_page = page.get(
            "source_page_number",
            page["page_number"]
        )

        extraction_method = page.get(
            "extraction_method",
            "Unknown"
        )

        st.subheader(
            f"Page {page['page_number']}"
        )

        st.caption(
            f"Source: {source_file} | "
            f"Original page: {source_page} | "
            f"Method: {extraction_method}"
        )

        edited_text = st.text_area(
            f"Review or correct extracted text — Page {page['page_number']}",
            value=page.get("text", ""),
            height=250,
            key=f"review_text_{index}"
        )

        edited_page = page.copy()

        edited_page["text"] = edited_text

        edited_pages.append(edited_page)

    st.divider()

    if st.button(
        "2. Build AI Search System",
        type="primary"
    ):

        try:

            with st.spinner(
                "Preprocessing text and building the medical AI search system..."
            ):

                # Step 1: Use reviewed text.
                processed_pages = preprocess_medical_pages(
                    edited_pages
                )

                # Step 2: Combine page text.
                text_parts = []

                for page in processed_pages:

                    if page["text"].strip():

                        text_parts.append(
                            f"Page {page['page_number']}\n"
                            f"{page['text']}"
                        )

                full_text = "\n\n".join(text_parts).strip()

                if not full_text:

                    raise ValueError(
                        "No readable text is available. "
                        "Please review the extracted pages."
                    )

                # Step 3: Generate chunks.
                chunks = chunk_medical_pages(
                    processed_pages,
                    chunk_size=1000,
                    overlap=150
                )

                if not chunks:

                    raise ValueError(
                        "Unable to generate document chunks."
                    )

                # Step 4: Build FAISS vector store.
                vector_store = MedicalVectorStore()

                total_vectors = vector_store.build_index(
                    chunks
                )

                # Step 5: Save processed data.
                st.session_state["processed_pages"] = processed_pages

                st.session_state["full_text"] = full_text

                st.session_state["chunks"] = chunks

                st.session_state["vector_store"] = vector_store

                st.session_state["document_ready"] = True

                st.session_state["review_complete"] = True

                st.session_state["chat_history"] = []

                st.session_state.pop(
                    "medical_summary",
                    None
                )

                st.success(
                    "Medical AI search system created successfully!"
                )

                st.success(
                    f"Generated {total_vectors} searchable vectors."
                )

        except Exception as error:

            st.error(
                f"AI search system creation failed: {error}"
            )


# ==========================================
# STEP 3: DOCUMENT INFORMATION
# ==========================================

if st.session_state.get("document_ready"):

    processed_pages = st.session_state["processed_pages"]

    full_text = st.session_state["full_text"]

    chunks = st.session_state["chunks"]

    st.divider()

    st.header("Document Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Pages",
        len(processed_pages)
    )

    col2.metric(
        "Extracted Characters",
        len(full_text)
    )

    col3.metric(
        "Total Chunks",
        len(chunks)
    )

    st.subheader("Processed Medical Report")

    for page in processed_pages:

        with st.expander(
            f"Page {page['page_number']} — "
            f"{page.get('source_file', 'Unknown document')}"
        ):

            st.text(
                page["text"] or "No text found."
            )

    with st.expander("View Complete Extracted Text"):

        st.text_area(
            "Complete Medical Report",
            full_text,
            height=300,
            disabled=True
        )


# ==========================================
# STEP 4: AI MEDICAL REPORT SUMMARIZATION
# ==========================================

if st.session_state.get("document_ready"):

    st.divider()

    st.header("AI Medical Report Explanation")

    st.write(
        "Generate a simple explanation of your uploaded "
        "medical report."
    )

    if st.button("Generate Medical Summary"):

        try:

            with st.spinner(
                "Gemini AI is analyzing your medical report..."
            ):

                summary = generate_medical_summary(
                    st.session_state["full_text"]
                )

                if summary and summary.strip():

                    st.session_state["medical_summary"] = summary

                else:

                    st.error(
                        "Gemini AI returned an empty summary."
                    )

        except Exception as error:

            st.error(
                f"Summary generation failed: {error}"
            )

    if "medical_summary" in st.session_state:

        st.subheader("AI Generated Summary")

        st.markdown(
            st.session_state["medical_summary"]
        )


# ==========================================
# STEP 5: MEDICAL DOCUMENT RAG CHAT
# ==========================================

if st.session_state.get("document_ready"):

    st.divider()

    st.header("Ask Questions About Your Medical Report")

    st.write(
        "Ask questions about the information contained "
        "in your uploaded medical documents."
    )

    # Display previous messages.
    for message in st.session_state["chat_history"]:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                with st.expander("View Source References"):

                    for source in message["sources"]:

                        st.markdown(
                            f"**File:** {source.get('source_file', 'Unknown document')} "
                            f"| **Original Page:** "
                            f"{source.get('source_page_number', source.get('page_number', 'Unknown'))} "
                            f"| **Method:** "
                            f"{source.get('extraction_method', 'Unknown')} "
                            f"| **Chunk:** {source.get('chunk_id')} "
                            f"| **Similarity:** "
                            f"{source.get('similarity_score', 0):.4f}"
                        )

                        st.text(source["text"])

    # Chat input.
    user_question = st.chat_input(
        "Ask something about your medical report..."
    )

    if user_question:

        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):

            st.markdown(user_question)

        try:

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching your report and generating an answer..."
                ):

                    result = answer_medical_question(
                        question=user_question,
                        vector_store=st.session_state["vector_store"],
                        top_k=3
                    )

                    answer = result["answer"]

                    st.markdown(answer)

                    with st.expander("View Source References"):

                        for source in result["sources"]:

                            st.markdown(
                                f"**File:** {source.get('source_file', 'Unknown document')} "
                                f"| **Original Page:** "
                                f"{source.get('source_page_number', source.get('page_number', 'Unknown'))} "
                                f"| **Method:** "
                                f"{source.get('extraction_method', 'Unknown')} "
                                f"| **Chunk:** {source.get('chunk_id')} "
                                f"| **Similarity:** "
                                f"{source.get('similarity_score', 0):.4f}"
                            )

                            st.text(source["text"])

                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result["sources"]
                    })

        except (RuntimeError, ValueError) as error:

            st.warning(str(error))

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": f"⚠️ {str(error)}"
            })

        except Exception as error:

            st.error(
                "An unexpected error occurred while "
                "generating your answer. Please try again."
            )

            print(
                f"RAG Error: {type(error).__name__}: {error}"
            )


# ==========================================
# MEDICAL DISCLAIMER
# ==========================================

st.divider()

st.caption(
    "HealthSphere provides AI-generated explanations of uploaded "
    "documents for educational purposes only. It does not provide "
    "medical diagnosis or treatment. OCR and AI-generated content "
    "may contain errors. Always verify medical values against the "
    "original report and consult a qualified healthcare professional."
)
