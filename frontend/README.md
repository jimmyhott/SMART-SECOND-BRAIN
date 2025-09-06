# Smart Second Brain - Streamlit Frontend

A modern, clean web interface for the Smart Second Brain AI-powered knowledge management system built with Streamlit.

## Features

- **Compact Health Monitoring** - Tiny status bar showing system health
- **Tabbed Interface** - Separate tabs for document ingestion and querying
- **Multiple PDF Upload** - Drag-and-drop support for multiple PDF files with batch processing
- **Batch Metadata** - Add source, categories, author, and custom metadata for PDF batches
- **Real-time Processing** - Live progress updates and detailed processing results
- **Chatbot Interface** - Modern chat UI for querying the knowledge base
- **Bootstrap Styling** - Professional, responsive design
- **Real-time Updates** - Live chat and status updates

## Installation

```bash
cd frontend
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Requirements

- Python 3.8+
- Smart Second Brain backend running on port 8000
- All backend services healthy (Graph, Vectorstore, AI models)

## API Endpoints

The frontend connects to:
- Health Check: `http://localhost:8000/smart-second-brain/api/v1/graph/health`
- Document Ingestion: `http://localhost:8000/smart-second-brain/api/v1/graph/ingest`
- Multiple PDF Ingestion: `http://localhost:8000/smart-second-brain/api/v1/graph/ingest-pdfs`
- Knowledge Query: `http://localhost:8000/smart-second-brain/api/v1/graph/query`

## PDF Batch Processing

The frontend supports uploading multiple PDF files with batch metadata:

- **Source**: Required identifier for the document batch
- **Categories**: Optional comma-separated categories for tagging
- **Author/Organization**: Optional author or organization information
- **Additional Metadata**: Optional JSON metadata for custom fields

All PDFs in a batch will be processed together with the same metadata applied to each document.
