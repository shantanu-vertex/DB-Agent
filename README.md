# JSON RAG Assistant

A Streamlit app that lets you upload a JSON file, chunk its content, run retrieval, and answer prompts using retrieved context.

## Features

- Upload JSON from the UI
- Chunk JSON text with configurable chunk size and overlap
- Retrieve top matching chunks using TF-IDF similarity
- Generate grounded answers using OpenAI (if `OPENAI_API_KEY` is set)
- Fallback to extractive grounded response without an API key

## Requirements

- Python 3.11+
- pip (bundled with most Python installs)
- Visual Studio Code

## Install Prerequisites And Tools

1. Install Python 3.11 or newer from the official Python installer.
2. During Python setup, enable **Add Python to PATH**.
3. Install Visual Studio Code.
4. (Optional) Install Git if you want to clone and version-control the project.

Verify your installation:

```bash
python --version
pip --version
```

## Install VS Code Extensions

Install these extensions from the Extensions view in VS Code:

- Python (`ms-python.python`)
- Pylance (`ms-python.vscode-pylance`)

Or install from terminal:

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
```

## Setup

1. Create and activate a virtual environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: configure OpenAI key:

```bash
copy .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`.

## Run

```bash
streamlit run app.py
```

## How to use

1. Upload a JSON file.
2. Click **Index JSON**.
3. Enter a prompt.
4. Click **Get Answer**.

The app displays the answer and the retrieved chunks used for that answer.
