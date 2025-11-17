# backend.py
# Optimized backend adapted from your Kaggle notebook.
# - Use: from backend import run_pipeline
# - Entry: run_pipeline(pdf_path, use_gemini_embeddings=False, use_gemini_generation=False, top_k=6)

import os
import re
import tempfile
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
load_dotenv()

# PDF / OCR
import fitz  # pymupdf
import pdfplumber
from PIL import Image
import pytesseract

# Data / math
import numpy as np
import faiss

# Embeddings
from sentence_transformers import SentenceTransformer

# Optional Gemini client
try:
    import google.generativeai as genai
    GEMINI_CLIENT_AVAILABLE = True
except Exception:
    GEMINI_CLIENT_AVAILABLE = False

# sentence tokenizer
def sent_tokenize_custom(text):
    # Splits sentences using . ! or ?
    return re.split(r'(?<=[.!?])\s+', text.strip())

# ----------------------------
# Configuration / defaults
# ----------------------------
LOCAL_EMBED_MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL_NAME", "paraphrase-MiniLM-L3-v2")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
DEFAULT_GEMINI_GEN_MODEL = os.getenv("GEMINI_GEN_MODEL", "models/gemini-2.5-flash")

# ----------------------------
# Utilities: PDF -> pages
# ----------------------------
def extract_text_from_pdf(pdf_path: str, ocr_on_empty: bool = True) -> List[Dict]:
    """
    Extract text from a PDF using pdfplumber primarily, fallback to PyMuPDF.
    Returns list of {"page": int, "text": str}
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip() and ocr_on_empty:
                    try:
                        pil = page.to_image(resolution=150).original
                        text = pytesseract.image_to_string(pil)
                    except Exception:
                        text = ""
                pages.append({"page": i + 1, "text": text})
        return pages
    except Exception:
        # Fallback to PyMuPDF
        doc = fitz.open(pdf_path)
        for i, p in enumerate(doc):
            try:
                text = p.get_text("text") or ""
            except Exception:
                text = ""
            if not text.strip() and ocr_on_empty:
                try:
                    pix = p.get_pixmap(dpi=150)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                except Exception:
                    text = ""
            pages.append({"page": i + 1, "text": text})
        return pages

# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, max_words: int = 350, overlap_words: int = 50) -> List[str]:
    """
    Split text into sentence-aware overlapping chunks.
    """
    if not text or not text.strip():
        return []
    sentences = sent_tokenize_custom(text)
    chunks = []
    cur = []
    cur_words = 0
    for s in sentences:
        s_words = len(s.split())
        if cur and (cur_words + s_words > max_words):
            chunks.append(" ".join(cur))
            # create overlap
            overlap = []
            ov_count = 0
            for sent in reversed(cur):
                ov_count += len(sent.split())
                overlap.insert(0, sent)
                if ov_count >= overlap_words:
                    break
            cur = overlap.copy()
            cur_words = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_words += s_words
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def build_chunks_from_pages(pages: List[Dict], max_words: int = 350, overlap_words: int = 50) -> List[Dict]:
    """
    Convert pages [{"page":1,"text":"..."}] -> list of chunks with metadata
    """
    all_chunks = []
    for p in pages:
        page_chunks = chunk_text(p.get("text", ""), max_words=max_words, overlap_words=overlap_words)
        for i, c in enumerate(page_chunks):
            all_chunks.append({"page": p["page"], "chunk_id": f"{p['page']}_{i}", "text": c})
    return all_chunks

# ----------------------------
# Embeddings: local & optional Gemini
# ----------------------------
_local_model = None

def init_local_embed_model():
    global _local_model
    if _local_model is None:
        _local_model = SentenceTransformer(LOCAL_EMBED_MODEL_NAME)
    return _local_model

def embed_texts_local(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = init_local_embed_model()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=batch_size)
    return np.asarray(embs, dtype=np.float32)

def embed_texts_gemini(texts: List[str]) -> np.ndarray:
    if not GEMINI_CLIENT_AVAILABLE:
        raise RuntimeError("google.generativeai not installed (required for Gemini embeddings)")
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set environment variable GOOGLE_API_KEY to use Gemini embeddings.")
    genai.configure(api_key=key)
    vectors = []
    for t in texts:
        resp = genai.embed_content(model=GEMINI_EMBED_MODEL, content=t)
        # resp expected to contain "embedding"
        if isinstance(resp, dict) and "embedding" in resp:
            vectors.append(resp["embedding"])
        else:
            # try attribute access or fallback
            try:
                vectors.append(resp.embedding)
            except Exception:
                raise RuntimeError("Unexpected embed response format from Gemini.")
    return np.array(vectors, dtype=np.float32)

def get_embed_fn(use_gemini: bool):
    return embed_texts_gemini if use_gemini else embed_texts_local

# ----------------------------
# FAISS index builder
# ----------------------------
def build_faiss_index(embeddings: np.ndarray, normalize: bool = True):
    embeddings = embeddings.astype(np.float32)
    if normalize:
        faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def index_pipeline(all_chunks: List[Dict], embed_fn) -> Tuple[Any, np.ndarray]:
    texts = [c["text"] for c in all_chunks]
    embs = embed_fn(texts)
    embs = np.asarray(embs, dtype=np.float32)
    index = build_faiss_index(embs, normalize=True)
    return index, embs

# ----------------------------
# Retriever
# ----------------------------
def retrieve_chunks(query: str, index, embed_fn, all_chunks: List[Dict], k: int = 5) -> List[Dict]:
    q_emb = embed_fn([query]).astype(np.float32)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(all_chunks):
            continue
        item = all_chunks[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results

# ----------------------------
# RAG Summarizer (Gemini or local fallback)
# ----------------------------
def build_rag_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    prompt = ("You are a Research Analysis Agent. Use ONLY the provided context extracted from a research paper. "
              "Do NOT add outside information.\n\n")
    for i, c in enumerate(retrieved_chunks):
        prompt += f"### Context Chunk {i+1} (Page {c['page']}):\n{c['text']}\n\n"
    prompt += (f"### Task:\n{query}\n\n"
               "### Instructions:\n- Summarize concisely.\n- Provide structured sections.\n- Avoid repetition.\n- Use only information from context.\n\n"
               "### Answer:\n")
    return prompt

_gen_model = None
def init_gemini_gen_model(model_name: str = None):
    global _gen_model
    if _gen_model is not None:
        return _gen_model
    if not GEMINI_CLIENT_AVAILABLE:
        return None
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    genai.configure(api_key=key)
    model_name = model_name or DEFAULT_GEMINI_GEN_MODEL
    try:
        _gen_model = genai.GenerativeModel(model_name)
    except Exception:
        # try a safe fallback name
        try:
            _gen_model = genai.GenerativeModel("models/gemini-2.5-flash")
        except Exception:
            _gen_model = None
    return _gen_model

def rag_answer(query: str, index, embed_fn, all_chunks: List[Dict], k: int = 6, use_gemini_generation: bool = True) -> Dict:
    retrieved = retrieve_chunks(query, index, embed_fn, all_chunks, k)
    prompt = build_rag_prompt(query, retrieved)
    gen = init_gemini_gen_model() if use_gemini_generation else None
    if gen:
        resp = gen.generate_content(prompt)
        answer = resp.text
    else:
        chunks_text = " ".join([c["text"] for c in retrieved])
        sentences = re.split(r'\. |\n', chunks_text)
        sentences = sorted([s.strip() for s in sentences if s.strip()], key=lambda x: len(x.split()), reverse=True)
        answer = ". ".join(sentences[:6]) + ('.' if sentences else '')
    return {"query": query, "answer": answer, "context": retrieved}

# ----------------------------
# Gap Finder Agent
# ----------------------------
def build_gap_prompt(retrieved_chunks: List[Dict]) -> str:
    prompt = ("You are a Research Gap Analysis Agent.\nYour task is to read ONLY the provided context from the paper and:\n"
              "- Identify missing experiments\n- Highlight unclear assumptions\n- Detect limitations\n- Find potential improvements\n- Suggest future research directions\nDo NOT add information not present in the text.\n\n")
    for i, c in enumerate(retrieved_chunks):
        prompt += f"### Context Chunk {i+1} (Page {c['page']}):\n{c['text']}\n\n"
    prompt += ("### Output Format (VERY IMPORTANT):\n- Missing Experiments:\n- Limitations:\n- Unexplored Gaps:\n- Improvement Opportunities:\n- Future Research Directions:\n\n### Begin Analysis:\n")
    return prompt

def gap_finder(index, embed_fn, all_chunks, k=6, use_gemini_generation: bool = True) -> Dict:
    gap_query = "What limitations, missing experiments, or research gaps does the paper have?"
    retrieved = retrieve_chunks(gap_query, index, embed_fn, all_chunks, k)
    prompt = build_gap_prompt(retrieved)
    gen = init_gemini_gen_model() if use_gemini_generation else None
    if gen:
        resp = gen.generate_content(prompt)
        answer = resp.text
    else:
        combined = " ".join(c["text"] for c in retrieved)
        sentences = [s.strip() for s in re.split(r'\. |\n', combined) if s.strip()]
        if sentences:
            parts = []
            parts.append("Missing Experiments:\n- " + (sentences[0][:300] + "..."))
            if len(sentences) > 1:
                parts.append("\nLimitations:\n- " + (sentences[1][:300] + "..."))
            if len(sentences) > 2:
                parts.append("\nFuture Research:\n- " + (sentences[2][:300] + "..."))
            answer = "\n\n".join(parts)
        else:
            answer = "No clear issues found."
    return {"analysis": answer, "context": retrieved}

# ----------------------------
# Slide generator
# ----------------------------
def build_slide_prompt(summary_text: str, gap_text: str) -> str:
    prompt = f"""
You are a PPT Slide Generator Agent.
Use ONLY the following two sources:

Paper Summary:
{summary_text}

Research Gaps:
{gap_text}

INSTRUCTIONS:
Create slide content in the EXACT format:
SLIDE 1 — Title Slide:
- Title:
- Subtitle:
- Authors (optional):
- One-line summary:

SLIDE 2 — Problem Statement:
- Bullet 1
- Bullet 2
- Bullet 3

SLIDE 3 — Proposed Method / Approach:
- Bullet 1
- Bullet 2
- Bullet 3

SLIDE 4 — Key Findings:
- Bullet 1
- Bullet 2
- Bullet 3

SLIDE 5 — Limitations:
- Bullet 1
- Bullet 2
- Bullet 3

SLIDE 6 — Research Gaps:
- Bullet 1
- Bullet 2
- Bullet 3

SLIDE 7 — Future Scope:
- Bullet 1
- Bullet 2
- Bullet 3

SLIDE 8 — Conclusion:
- Bullet 1
- Bullet 2
- Bullet 3

Return only the text formatted as above.
"""
    return prompt

def generate_slides(summary_text: str, gap_text: str, use_gemini_generation: bool = True) -> str:
    gen = init_gemini_gen_model() if use_gemini_generation else None
    prompt = build_slide_prompt(summary_text, gap_text)
    if gen:
        resp = gen.generate_content(prompt)
        return resp.text
    else:
        # Simple fallback
        title = summary_text.split('\n')[0] if summary_text else "Paper"
        return (f"SLIDE 1 — Title Slide:\n- Title: {title}\n- Subtitle: \n- Authors: \n- One-line summary: {(summary_text[:200] if summary_text else '')}\n\n"
                "SLIDE 2 — Problem Statement:\n- " + (gap_text[:150] if gap_text else "N/A"))

# ----------------------------
# Orchestrator wrapper
# ----------------------------
class Orchestrator:
    def __init__(self, use_gemini_embeddings: bool = False, use_gemini_generation: bool = True):
        self.use_gemini_embeddings = use_gemini_embeddings
        self.use_gemini_generation = use_gemini_generation
        self.embed_fn = get_embed_fn(use_gemini_embeddings)

    def run_from_pdf(self, pdf_path: str, k=6) -> Dict[str, Any]:
        pages = extract_text_from_pdf(pdf_path)
        all_chunks = build_chunks_from_pages(pages)
        index, embs = index_pipeline(all_chunks, self.embed_fn)
        summary = rag_answer("Provide a detailed structured summary of the research paper.", index, self.embed_fn, all_chunks, k=k, use_gemini_generation=self.use_gemini_generation)
        gaps = gap_finder(index, self.embed_fn, all_chunks, k=k, use_gemini_generation=self.use_gemini_generation)
        slides = generate_slides(summary["answer"], gaps["analysis"], use_gemini_generation=self.use_gemini_generation)
        return {
            "pages": pages,
            "chunks": all_chunks,
            "index": index,
            "embeddings": embs,
            "summary": summary,
            "gaps": gaps,
            "slides": slides
        }

# Convenience entrypoint used by frontend
def run_pipeline(pdf_path: str, use_gemini_embeddings: bool = False, use_gemini_generation: bool = True, top_k: int = 6):
    orch = Orchestrator(use_gemini_embeddings=use_gemini_embeddings, use_gemini_generation=use_gemini_generation)
    return orch.run_from_pdf(pdf_path, k=top_k)

# Quick CLI debug
if __name__ == "__main__":
    # pick first PDF in current directory if present
    sample = None
    for f in os.listdir("."):
        if f.lower().endswith(".pdf"):
            sample = f
            break
    if sample:
        print("Running pipeline on", sample)
        out = run_pipeline(sample)
        print("SUMMARY:\n", out["summary"]["answer"])
        print("\nGAPS:\n", out["gaps"]["analysis"])
        print("\nSLIDES:\n", out["slides"])
    else:
        print("No PDF found in current folder. Call run_pipeline(pdf_path) from your frontend.")