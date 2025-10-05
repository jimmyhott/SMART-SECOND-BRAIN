# Smart Second Brain - Streamlit Frontend

A modern, clean web interface for the Smart Second Brain AI-powered knowledge management system built with Streamlit.

## Features

- **System Health Dashboard** - Real-time status monitoring of Graph, Vectorstore, and AI components
- **Three-Tab Interface** - PDF Ingestion, Text Ingestion, and Chat with Knowledge Base
- **Multiple PDF Upload** - Drag-and-drop support for multiple PDF files with batch processing
- **Text Content Ingestion** - Direct text input with categorization and knowledge type selection
- **Batch Metadata** - Add source, categories, author, and custom metadata for document batches
- **Real-time Processing** - Live progress updates and detailed processing results
- **Advanced Chat Interface** - Modern chat UI with conversation threads and feedback system
- **Human-in-the-Loop Feedback** - Approve, reject, or edit AI-generated answers
- **Smart IDK Interface** - Knowledge input when AI doesn't know something
- **Thread Management** - SQLite-backed conversation history with unique thread IDs
- **Professional Styling** - Clean, responsive design with Bootstrap components

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
streamlit run app.py --server.port 5173 --server.address 0.0.0.0
```

The app will open in your browser at `http://localhost:5173`

## Requirements

- Python 3.8+
- Smart Second Brain backend running on port 8000
- All backend services healthy (Graph, Vectorstore, AI models)

## API Endpoints

The frontend connects to:
- Health Check: `http://localhost:8000/smart-second-brain/api/v1/graph/health`
- Text Document Ingestion: `http://localhost:8000/smart-second-brain/api/v1/graph/ingest`
- Multiple PDF Ingestion: `http://localhost:8000/smart-second-brain/api/v1/graph/ingest-pdfs`
- Knowledge Query: `http://localhost:8000/smart-second-brain/api/v1/graph/query`
- Submit Feedback: `http://localhost:8000/smart-second-brain/api/v1/graph/feedback`
- Get Feedback Status: `http://localhost:8000/smart-second-brain/api/v1/graph/feedback/{thread_id}`
- Clear Vector Database: `http://localhost:8000/smart-second-brain/api/v1/graph/clear-vector-db` (⚠️ irreversible)

## Document Processing

### PDF Batch Processing

The frontend supports uploading multiple PDF files with batch metadata:

- **Source**: Required identifier for the document batch
- **Categories**: Optional comma-separated categories for tagging
- **Author/Organization**: Optional author or organization information
- **Additional Metadata**: Optional JSON metadata for custom fields

All PDFs in a batch will be processed together with the same metadata applied to each document.

### Text Content Processing

The frontend also supports direct text content ingestion:

- **Text Input**: Large text area for pasting or typing content
- **Source**: Required source identifier for the text content
- **Categories**: Optional comma-separated categories for tagging
- **Author**: Optional author information
- **Knowledge Type**: Selection between "reusable", "verified", or "temporary"

## Human-in-the-Loop Feedback

The chat interface includes a comprehensive feedback system:

- **Approve**: Mark AI-generated answers as approved
- **Reject**: Mark answers as rejected
- **Edit**: Provide edited versions of AI answers
- **Smart IDK Interface**: When AI doesn't know something, provide knowledge input instead of feedback
- **Status Display**: Real-time feedback status for each message
- **Thread Management**: Track feedback across conversation threads

### IDK Response Handling

When the AI indicates it doesn't know something (via `is_idk: true`):

1. **Detection**: System automatically detects IDK responses using state properties
2. **Knowledge Input**: Shows "Provide Knowledge" interface instead of standard feedback
3. **Storage**: User-provided knowledge is stored in the vector database
4. **Future Use**: Knowledge becomes available for future queries

## Conversation Management

- **Thread-based Conversations**: Each conversation has a unique thread ID
- **Conversation History**: SQLite-backed persistence of chat history
- **Message Types**: User messages, assistant responses, and system messages
- **Feedback Integration**: Seamless integration between chat and feedback systems
