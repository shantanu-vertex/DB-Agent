# DB-Agent — JSON RAG Assistant

## Project Overview
A Streamlit app that lets you upload a JSON file, chunk its content, retrieve relevant chunks using TF-IDF similarity, and answer prompts via OpenAI (with a local fallback).

- **Language:** Python 3.11+
- **Framework:** Streamlit
- **Key Libraries:** scikit-learn (TF-IDF retrieval), openai, python-dotenv

## Setup

### Prerequisites
- Python 3.11+
- Virtual environment at `.venv/`

### Install Dependencies
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure OpenAI (optional)
Copy `.env.example` to `.env` and set your key:
```
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini
```

## Run
```bash
streamlit run app.py
```
Or use the VS Code task: **Run Streamlit App**

## Project Structure
```
DB-Agent/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env.example
├── mcp.json
├── docs/                   # Architecture and planning docs
│   ├── ARCHITECTURE.md
│   ├── PLAN.md
│   ├── architecture-multi-db.md
│   └── architecture-postgres-mcp.md
├── output/                 # Generated files (gitignored)
│   └── relationships.json
├── src/
│   ├── rag/                # All RAG logic
│   └── schema/             # DB connection + FK extraction
└── .github/
    └── copilot-instructions.md
```

## Development Guidelines
- Keep all RAG logic in `src/rag/`
- Keep DB connection and extraction logic in `src/schema/`
- `app.py` is the Streamlit entry point only
- Generated output files (e.g. `relationships.json`) go in `output/`
- Architecture and planning docs go in `docs/`
- Use `.env` for secrets — never commit real keys
- Work through each checklist item systematically
- Keep communication concise and focused
- Follow development best practices
