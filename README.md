# Docling Agentic RAG

A local-first assistant for searching engineering documents. [Docling](https://github.com/docling-project/docling) converts PDFs, Word documents, and spreadsheets into Markdown; a lightweight local retriever finds evidence; and an optional tool-using LLM can refine its searches before answering with citations.

## Features

- Parse PDF, DOCX, and XLSX with Docling; read CSV, TXT, and Markdown directly.
- OCR-capable PDF parsing through Docling.
- Chunk and index documents in a local SQLite database.
- Search locally with TF-IDF; document contents stay on your machine unless you enable a remote answer provider.
- Let the assistant call the document-search tool up to three times to refine its retrieval.
- Review source excerpts and citations alongside each answer.
- Avoid duplicate indexing based on file content.

## Run locally

Requires Python 3.10 or later.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Upload documents in the sidebar, then ask a question. Docling may download its processing models on first use.

The SQLite database is created at `data/documents.db` and is not committed to version control.

## Enable agentic answers

Without an API key, the app performs local retrieval and displays matching passages. To enable tool-calling answers, configure an OpenAI-compatible chat-completions provider that supports function calling:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
# Optional overrides:
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
python -m streamlit run app.py
```

With agentic answers enabled, the assistant can search the indexed documents more than once, then answer from retrieved excerpts and cite sources such as `[S1]`. The API receives the question and retrieved excerpts. Do not enable this for confidential documents unless your organization's policies permit sending those excerpts to that provider. API keys are read from the environment and are not stored in the database.

## Supported formats

| Format | Processing |
| --- | --- |
| PDF | Docling, including OCR-capable processing |
| DOCX | Docling |
| XLSX | Docling |
| CSV | UTF-8 text parsing |
| TXT, Markdown | UTF-8 text parsing |

This prototype does not interpret CAD drawings, enforce document permissions, or replace a controlled document-management system. Retrieval uses local TF-IDF rather than a trained semantic embedding model; include specific document numbers, equipment tags, or revision codes in questions when possible.

## Tests

```powershell
python -m unittest discover -s tests
```
