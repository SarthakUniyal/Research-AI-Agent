**📘 Research PDF Intelligence – Multi-Agent System**

**AI-powered Research Paper Understanding, Gap Detection & Slide Generation**


**1. Problem Statement**

Research papers contain dense technical knowledge that requires extensive time to read, understand, and summarize. Extracting insights, identifying missing experiments, and preparing presentations further increases the workload for students, scholars, and professionals. Traditional workflows are slow and often miss deeper insights because the entire process is manual.
There is currently no unified system that performs research-paper extraction, RAG-based summarization, gap detection, and slide generation in a single automated pipeline.

**2. Solution Overview**

Research PDF Intelligence provides a complete end-to-end pipeline powered by multiple coordinated AI agents.
The system automatically extracts text from PDFs, chunks it for semantic retrieval, builds vector embeddings, detects research gaps, generates summaries, and produces presentation-ready slide content using Gemini models.
This creates a streamlined research assistant capable of simplifying academic workflows.

**3. System Architecture**
The architecture is built around a sequence of specialized agents:

  1. VisionAgent performs PDF extraction using PDFPlumber and OCR.
  2. ChunkAgent converts long paper text into overlapping semantic chunks.
  3. EmbedAgent generates embeddings using either a local SentenceTransformer model or Gemini embeddings.
  4. RetrieverAgent builds a FAISS index and retrieves relevant content through semantic search.
  5. RAG Summarizer Agent produces structured and factual summaries using Gemini-2.5-Pro.
  6. Gap Finder Agent identifies missing experiments, limitations, assumptions, and improvement opportunities.
  7. Slide Generator Agent converts summaries and gaps into structured PPT content.
  9. Streamlit Frontend provides an interactive interface for uploading PDFs and generating results.

**4. Technology Stack**

  1. Python 3.10+ for backend processing
  2. PDFPlumber + PyMuPDF + Tesseract for extraction and OCR
  3. SentenceTransformer and Gemini embeddings for semantic understanding
  4. FAISS for fast vector search
  5. Gemini 2.5 flash for generation tasks
  6. Streamlit for the frontend
  7. NLTK for sentence tokenization

**5. Key Features**

  1. Automatic PDF text extraction with OCR fallback.
  2. Semantic chunking to preserve research context.
  3. Local or Gemini-based embedding generation.
  4. FAISS indexing for efficient semantic retrieval.
  5. RAG-based structured summarization.
  6. Research gap and limitation detection.
  7. AI-generated multi-slide presentation content.
  8. Fully interactive Streamlit UI for ease of use.

**6. Installation**

  1. Create a virtual environment.
  2. Install dependencies using pip install -r requirements.txt.
  3. Set your Gemini API key using environment variables.
  4. Run the backend using Python.
  5. Launch the frontend with streamlit run app.py.

**7. How to Use**

  1. Upload a research PDF in the web interface.
  2. Wait for extraction, chunking, and embedding generation.
  3. Enter a query or select automatic summarization.
  4. View generated output such as summary, research gaps, and slide content.
  5. Copy or export the generated information for academic or presentation use.

**8. Project Benefits**

  1. Eliminates manual reading time.
  2. Improves research quality by identifying hidden gaps.
  3. Automatically prepares structured presentation slides.
  4. Acts as a personal AI research assistant available anytime.
  5. Speeds up literature reviews and academic workflows.

**9. Architecture Diagram**

A complete architecture diagram has been generated and included in the project files for clear visualization of the entire multi-agent pipeline.

**10. License**

This project is distributed under the MIT License allowing academic and personal use.




