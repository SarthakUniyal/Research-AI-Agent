import streamlit as st
import os
import tempfile
from backend import run_pipeline, rag_answer, retrieve_chunks, get_embed_fn

st.set_page_config(page_title="Research PDF Intelligence", layout="wide")

# Elegant UI Header
st.markdown("""
<h1 style='text-align:center; color:#4A90E2;'>📘 Research PDF Intelligence</h1>
<p style='text-align:center; font-size:18px;'>
AI-powered Summary • Research Gaps • Slide Generator
</p>
<hr>
""", unsafe_allow_html=True)

st.write("Upload a research paper (PDF) and let the multi-agent system process it.")
uploaded_file = st.file_uploader("📂 Upload PDF File", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded successfully!")

    # RUN THE ORCHESTRATOR PIPELINE
    st.info("Running AI Agents on your PDF. Please wait...")

    with st.spinner("Extracting, chunking, embedding, analyzing..."):
        output = run_pipeline(pdf_path)

    st.success("Processing complete!")

    # DISPLAY SUMMARY
    st.subheader("📌 Research Paper Summary")
    st.write(output["summary"]["answer"])

    # DISPLAY GAPS
    st.subheader("⚠ Research Gaps Identified")
    st.write(output["gaps"]["analysis"])

    # DISPLAY PPT SLIDES
    st.subheader("📑 Auto-Generated PPT Slide Content")
    st.text(output["slides"])

    st.markdown("<hr>", unsafe_allow_html=True)

    # Question Answering Section
    st.subheader("💬 Ask a question about the PDF")
    user_query = st.text_input("Ask anything related to the uploaded paper:")
    if st.button("Get Answer"):
        with st.spinner("Retrieving information..."):
            embed_fn = get_embed_fn(False)
            retrieved = retrieve_chunks(user_query, output["index"], embed_fn, output["chunks"], k=5)
            ans = rag_answer(user_query, output["index"], embed_fn, output["chunks"], k=5, use_gemini_generation=True )
        st.write("### Answer:")
        st.write(ans["answer"])
