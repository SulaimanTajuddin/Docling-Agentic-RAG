# Engineering Document Controller

A local-first RAG assistant for finding information in engineering documents. Upload documents, ask a question in plain language, and review the matching passages with their source filenames and locations.

## Features

- Upload PDF, DOCX, XLSX, CSV, TXT, and Markdown files.
- Extract and chunk their text, then store the index locally in SQLite.
- Retrieve relevant passages with local TF-IDF search; document contents stay on your machine during indexing and search.
- Optionally generate a concise, source-grounded answer through an OpenAI-compatible chat-completions API.
- See source citations for each answer and inspect the retrieved excerpts.
- Avoid duplicate indexing when the same file is uploaded more than once.

## Run locally

Requires Python 3.10 or later.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Upload documents in the sidebar, then enter a question on the main page.

Scanned PDFs are OCR'd automatically when a page has no extractable text. OCR
requires the Tesseract executable to be installed separately and available on
`PATH`. If it is installed outside `PATH`, set `TESSERACT_CMD` to its executable
before starting the app:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
python -m streamlit run app.py
```

The SQLite database is created at `data/documents.db` and is not committed to version control.

## Optional answer generation

Without an API key, the app returns retrieved evidence excerpts and citations without generating an answer. To enable answer generation, set these environment variables before starting Streamlit:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
# Optional overrides for an OpenAI-compatible provider:
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
python -m streamlit run app.py
```

When enabled, only the user's question and the retrieved excerpts are sent to the configured API. Do not enable this for confidential documents unless your organization's policies permit sending those excerpts to that provider. The API key is read from the environment and is not stored in the database.

## Supported formats and limitations

- PDF: text-based PDFs; scanned/image-only files need OCR before upload.
- DOCX: paragraphs and tables.
- XLSX: cell values grouped by worksheet.
- CSV: rows and column values.
- TXT and Markdown: plain text.

This starter app is intended for a local prototype or small collection. It does not interpret CAD drawings, enforce document permissions, or replace a controlled document-management system. Retrieval is lexical TF-IDF rather than a trained semantic embedding model, so include specific terms such as document numbers, equipment tags, or revision codes in questions when possible.

## Tests

```powershell
python -m unittest discover -s tests
```
