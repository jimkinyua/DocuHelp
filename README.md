# DocuHelp

A document ingestion and question-answering service built with FastAPI, ChromaDB, and Retrieval-Augmented Generation (RAG). It lets you upload PDF, DOCX, or text files, parses and chunks them into vector embeddings, and answers user questions based strictly on the uploaded content.

---

## Architecture Overview

```
┌─────────────────────────────────┐
│     User Upload (PDF/DOCX/TXT)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       Text Extraction Engine    │
│    (pypdf / python-docx / UTF-8)│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Sliding Window Text Chunker   │
│   (300 words / 50 word overlap) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      ChromaDB Vector Store      │
│  (all-MiniLM-L6-v2 Embeddings)  │
└────────────────┬────────────────┘
                 │ Semantic Search
                 ▼
┌─────────────────────────────────┐
│       LLM Response Engine       │
│     (Groq / OpenAI / Gemini)    │
└─────────────────────────────────┘
```

---

## Features

- **Multi-Format Extraction**: Parses text from `.pdf`, `.docx`, and `.txt` files with encrypted PDF error handling.
- **Sliding-Window Chunker**: Splits long documents into 300-word segments with 50-word overlaps to preserve semantic context across chunk boundaries.
- **ChromaDB Vector Storage**: Generates dense vector embeddings using `sentence-transformers` and performs cosine similarity queries.
- **Pluggable LLM Providers**: Supports Groq (fast inference), OpenAI, and Google Gemini.
- **Built-In Web Interface**: Clean frontend served directly via FastAPI static files.

---

## Project Structure

```
DocuHelp/
├── app/
│   ├── ingestion.py       # File parsing and text chunking
│   ├── vectorstore.py     # ChromaDB collection and similarity retrieval
│   ├── rag.py             # Prompt formatting and LLM integration
│   └── main.py            # FastAPI routes (/upload, /query, /)
├── static/
│   └── index.html         # Frontend interface
├── pyproject.toml         # Project dependencies
├── requirements.txt       # Core requirements
├── .env.example           # Environment template
└── README.md
```

---

## Setup & Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/jimkinyua/DocuHelp.git
cd DocuHelp
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the application

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser to access the web interface, or visit `http://localhost:8000/docs` for the interactive Swagger API documentation.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `POST` | `/upload` | Ingests a file, chunks text, and stores embeddings in ChromaDB |
| `POST` | `/query` | Retrieves top document chunks and generates context-grounded answers |

---

## Technologies Used

- **FastAPI** — High-performance asynchronous web framework
- **ChromaDB** — Open-source vector database for embedding storage
- **Sentence-Transformers** — Embedding model (`all-MiniLM-L6-v2`)
- **PyPDF & python-docx** — Document parsing
- **Groq / OpenAI / Gemini** — LLM inference
