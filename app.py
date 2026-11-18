
import hashlib
import streamlit as st

from src.document_loader import (
    extract_text_from_pdf,
    get_full_text
)

from src.medical_summary import generate_medical_summary

from src.text_preprocessor import preprocess_medical_pages

from src.text_chunker import chunk_medical_pages

from src.vector_store import MedicalVectorStore

from src.rag_pipeline import answer_medical_question


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="HealthSphere Medical AI",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("HealthSphere")

st.subheader("Medical Document AI Assistant")

st.write(
    "Upload your medical report, understand its contents, "
    "and ask questions using AI-powered document processing."
)


# --------------------------------
# INITIALIZE SESSION STATE
# --------------------------------

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# --------------------------------
# PDF UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your medical PDF",
    type=["pdf"],
    help="Upload a PDF medical report."
)


# --------------------------------
# DETECT NEW DOCUMENT
# --------------------------------

if uploaded_file is not None:

    uploaded_bytes = uploaded_file.getvalue()

    document_hash = hashlib.sha256(
        uploaded_bytes
    ).hexdigest()

    # Reset previous document data when a new PDF is uploaded
    if st.session_state.get("document_hash") != document_hash:

        keys_to_clear = [
            "extracted_pages",
            "full_text",
            "medical_summary",
            "processed_pages",
            "chunks",
            "vector_store",
            "chat_history"
        ]

        for key in keys_to_clear:
            st.session_state.pop(key, None)

        st.session_state["document_hash"] = document_hash
        st.session_state["chat_history"] = []

    st.success(
        f"Uploaded successfully: {uploaded_file.name}"
    )


# --------------------------------
# DOCUMENT PROCESSING
# --------------------------------

if uploaded_file is not None:

    if st.button("Process Medical Document", type="primary"):

        if uploaded_file.size > 10 * 1024 * 1024:

            st.error("File size must be below 10 MB.")

        else:

            try:

                with st.spinner("Processing medical document..."):

                    # Step 1: Extract PDF text
                    extracted_pages = extract_text_from_pdf(
                        uploaded_file
                    )

                    # Step 2: Preprocess extracted pages
                    processed_pages = preprocess_medical_pages(
                        extracted_pages
                    )

                    # Step 3: Combine cleaned text
                    full_text = get_full_text(
                        processed_pages
                    )

                    if not full_text.strip():

                        st.warning(
                            "No readable text found. "
                            "This PDF may be scanned. "
                            "OCR support will be added later."
                        )

                    else:

                        # Step 4: Generate chunks
                        chunks = chunk_medical_pages(
                            processed_pages,
                            chunk_size=1000,
                            overlap=150
                        )

                        if not chunks:

                            st.error(
                                "Unable to generate document chunks."
                            )

                        else:

                            # Step 5: Generate embeddings and build FAISS
                            vector_store = MedicalVectorStore()

                            total_vectors = vector_store.build_index(
                                chunks
                            )

                            # Step 6: Save everything in session state
                            st.session_state["extracted_pages"] = extracted_pages
                            st.session_state["processed_pages"] = processed_pages
                            st.session_state["full_text"] = full_text
                            st.session_state["chunks"] = chunks
                            st.session_state["vector_store"] = vector_store

                            st.success(
                                "Medical document processed successfully!"
                            )

                            st.success(
                                f"Generated {total_vectors} searchable vectors."
                            )

            except Exception as e:

                st.error(
                    f"Document processing failed: {str(e)}"
                )


# --------------------------------
# DISPLAY DOCUMENT INFORMATION
# --------------------------------

if "processed_pages" in st.session_state:

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

    # Display extracted content
    st.subheader("Extracted Medical Report")

    for page in processed_pages:

        with st.expander(
            f"Page {page['page_number']}"
        ):

            st.text(
                page["text"] or "No text found on this page."
            )

    with st.expander("View Complete Extracted Text"):

        st.text_area(
            "Document Content",
            full_text,
            height=300
        )


# --------------------------------
# AI MEDICAL REPORT SUMMARIZATION
# --------------------------------

if "full_text" in st.session_state:

    st.divider()

    st.header("AI Medical Report Explanation")

    st.write(
        "Generate a simple explanation of your uploaded medical report."
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

        except Exception as e:

            st.error(
                f"Summary generation failed: {str(e)}"
            )

    if "medical_summary" in st.session_state:

        st.subheader("AI Generated Summary")

        st.markdown(
            st.session_state["medical_summary"]
        )


# --------------------------------
# MEDICAL DOCUMENT RAG CHAT
# --------------------------------

if "vector_store" in st.session_state:

    st.divider()

    st.header("Ask Questions About Your Medical Report")

    st.write(
        "Ask questions about the information contained "
        "in your uploaded medical document."
    )

    # Display previous chat messages
    for message in st.session_state["chat_history"]:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # Display sources for assistant responses
            if message["role"] == "assistant" and "sources" in message:

                with st.expander("View Source References"):

                    for source in message["sources"]:

                        st.markdown(
                            f"**Page:** {source['page_number']} "
                            f"| **Chunk:** {source['chunk_id']} "
                            f"| **Similarity:** "
                            f"{source['similarity_score']:.4f}"
                        )

                        st.text(source["text"])

    # Chat input
    user_question = st.chat_input(
        "Ask something about your medical report..."
    )

    if user_question:

        # Display and store user question
        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate RAG answer
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
                                f"**Page:** {source['page_number']} "
                                f"| **Chunk:** {source['chunk_id']} "
                                f"| **Similarity:** "
                                f"{source['similarity_score']:.4f}"
                            )

                            st.text(source["text"])

                    # Save assistant response
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result["sources"]
                    })

        except RuntimeError as e:

            st.warning(str(e))

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": f"⚠️ {str(e)}"
            })

        except ValueError as e:

            st.warning(str(e))

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": f"⚠️ {str(e)}"
            })

        except Exception as e:

            st.error(
                "An unexpected error occurred while "
                "generating your answer. Please try again."
            )

            print(f"RAG Error: {type(e).__name__}: {e}")


# --------------------------------
# MEDICAL DISCLAIMER
# --------------------------------

st.divider()

st.caption(
    "HealthSphere provides AI-generated explanations of uploaded "
    "documents for educational purposes only. It does not provide "
    "medical diagnosis or treatment. Consult a qualified healthcare "
    "professional for clinical interpretation."
)
