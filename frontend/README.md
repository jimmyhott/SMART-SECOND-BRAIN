# Smart Second Brain - Streamlit Frontend

A modern, clean web interface for the Smart Second Brain AI-powered knowledge management system built with Streamlit.

## Features

- **Compact Health Monitoring** - Tiny status bar showing system health
- **Tabbed Interface** - Separate tabs for document ingestion and querying
- **Multiple PDF Upload** - Drag-and-drop support for multiple PDF files
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
- Knowledge Query: `http://localhost:8000/smart-second-brain/api/v1/graph/query`
