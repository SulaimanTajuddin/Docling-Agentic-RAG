import os
from pathlib import Path

import streamlit as st

from rag import DocumentStore, SUPPORTED_EXTENSIONS, agentic_answer


st.set_page_config(page_title="Docling Agentic RAG", page_icon="📚")
st.title("Docling Agentic RAG")
st.caption("Research engineering documents with Docling and source-grounded retrieval.")

store = DocumentStore(Path(__file__).parent / "data" / "documents.db")
answer_generation_enabled = bool(os.getenv("OPENAI_API_KEY"))

with st.sidebar:
    st.header("Document library")
    uploads = st.file_uploader(
        "Upload documents",
        type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        help=(
            "Supported: PDF, DOCX, XLSX, CSV, TXT, and Markdown. Docling "
            "extracts structured documents and supports scanned PDFs."
        ),
    )
    if uploads and st.button("Index uploaded documents"):
        for upload in uploads:
            try:
                if store.add_document(upload.name, upload.getvalue()):
                    st.success(f"Indexed {upload.name}")
                else:
                    st.info(f"Already indexed: {upload.name}")
            except Exception as error:
                st.error(f"Could not index {upload.name}: {error}")

    st.divider()
    st.subheader("Indexed documents")
    documents = store.list_documents()
    if documents:
        for name, chunk_count in documents:
            st.caption(f"{name} · {chunk_count} passages")
    else:
        st.caption("Upload documents to get started.")

if answer_generation_enabled:
    st.info("Agentic answers are enabled. The model can search the local index more than once.")
else:
    st.info("Local retrieval is ready. Set OPENAI_API_KEY to enable generated answers; without it, results stay local.")

with st.form("question_form"):
    question = st.text_input(
        "What document or information are you looking for?",
        placeholder="e.g. What is the inspection interval for pump P-101?",
    )
    submitted = st.form_submit_button("Search documents", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question or document reference to search.")
    else:
        response = None
        if answer_generation_enabled:
            try:
                response = agentic_answer(question, store)
                results = response.sources
            except Exception as error:
                st.error(f"Agentic search failed: {error}")
                results = []
        else:
            try:
                results = store.search(question, limit=5)
            except Exception as error:
                st.error(f"Search failed: {error}")
                results = []

        if not results:
            st.warning(
                "No matching passages found. Try a document number, equipment "
                "tag, or another specific term."
            )
        else:
            if response:
                st.subheader("Answer")
                st.write(response.answer)
                if response.searches:
                    st.caption("Agent searches: " + " → ".join(response.searches))
            else:
                st.subheader("Retrieved evidence")
                st.write(
                    "Generated answers are disabled. Review these matching "
                    "source passages:"
                )

            st.subheader("Sources")
            for index, result in enumerate(results, start=1):
                with st.expander(
                    f"[S{index}] {result.document_name} — {result.location} "
                    f"(match {result.score:.2f})"
                ):
                    st.write(result.text)
