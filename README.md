# Auxilium

Auxilium is a private knowledge assistant built to answer natural language questions over two types of internal sources:

- Microsoft Outlook emails ingested through Microsoft authentication
- local documents selected by the user

It combines a FastAPI backend, a Next.js frontend, and a local RAG pipeline based on embeddings and FAISS. The application is designed for practical, document-grounded assistance rather than generic chat alone.

![Main product screenshot placeholder](./docs/screenshots/main-product.png)

## Quick demo for recruiters

If you want to evaluate the product quickly, start with the recruiter guide:

- [Recruiter testing guide](./RECRUITER_GUIDE.md)

This guide explains:

- how to sign in
- how to use the demo Microsoft account
- how to configure email and local document ingestion
- when to use `Ingestion emails` versus `Reindex`
- which mode to use for document-grounded answers
- example questions to test the assistant

## Key features

Auxilium can:

- ingest emails from selected Outlook / Microsoft Graph folders
- flatten and index email content for retrieval
- index authorised local folders and supported file types
- answer questions using the local knowledge base
- switch between local assistance and general LLM answers depending on context
- persist chat history per authenticated user

## Architecture

The repository is split into two main applications:

- `Back/`: FastAPI backend, Microsoft auth integration, ingestion flows, RAG indexing, retrieval, and chat persistence
- `Front/`: Next.js frontend used to chat with the assistant and manage settings

Generated project data lives under `Back/data/`, including:

- cached emails
- flattened email text used for retrieval
- the local RAG index (`corpus.jsonl`, embeddings, FAISS index)
- tracked settings data used for the demo environment

## Tech stack

- Frontend: Next.js 15, React 19, TypeScript
- Backend: FastAPI, Uvicorn, Pydantic
- Retrieval: Sentence Transformers, FAISS, BM25
- Auth and mail access: Microsoft Graph / MSAL
- Utilities: pdfplumber, NumPy, PyTorch

## Running locally

Backend:

```bash
cd Auxilium/Back
uvicorn api:app --host 127.0.0.1 --port 8765 --reload
```

Frontend:

```bash
cd Auxilium/Front
npm install
npm run dev
```

Then open:

- frontend: `http://localhost:3000`
- backend API: `http://127.0.0.1:8765`

## Current scope

This repository is focused on a working local test version of the product:

- email ingestion and local-document indexing are handled on the local machine
- the local index is stored inside the project workspace
- demo data included in the repository is intentionally fictitious

## Notes

- Some project behaviour depends on Microsoft authentication and local configuration.
- The local RAG index must exist before document-grounded answering can be used.
- `Reindex` rebuilds the local index, while email ingestion updates the mail side of the knowledge base.
