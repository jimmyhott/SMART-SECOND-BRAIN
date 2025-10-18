# Smart Second Brain

An intelligent knowledge management platform with AI-powered document processing, semantic search, and LangGraph workflows for building a personal second brain.

## 🚀 Features

- **FastAPI Backend**: High-performance REST API with automatic documentation and health monitoring
- **LangGraph Integration**: Advanced AI workflows for document processing and knowledge extraction
- **AI-Powered Knowledge Management**: Intelligent document ingestion, chunking, and embedding
- **Semantic Search**: Vector-based retrieval using ChromaDB for accurate document search
- **Azure OpenAI Integration**: Full support for Azure OpenAI with automatic deployment detection
- **Smart Query Processing**: RAG (Retrieval-Augmented Generation) for intelligent question answering with hybrid retrieval, metadata filters, and reranking options
- **Auto Metadata Enrichment**: Optional preprocessing node that infers categories and keywords when callers do not supply them
- **Human-in-the-Loop (HITL)**: Feedback system for AI-generated answers with approval/rejection/editing
- **Intelligent IDK Detection**: State-based "I don't know" response detection with knowledge input interface
- **Thread-based Conversations**: SQLite-backed conversation memory for multi-turn conversations
- **Configurable Collections**: Support for multiple ChromaDB collections with consistent naming across operations
- **Cited Answers**: LLM prompt expects JSON responses with citation metadata for transparency
- **Streamlit Frontend**: Modern web interface with PDF and text ingestion, chat interface, and feedback system
- **Comprehensive Testing**: Extensive unit, API, and integration tests with per-test isolated LangGraph checkpoints
- **Centralized Logging**: Project-level logging configuration with file and console output
- **Automated Startup**: Shell scripts for easy system management with health monitoring

## 📋 Requirements

- Python 3.12+
- Azure OpenAI Service (for AI features)
- OpenAI API Key (for embeddings and LLM)
- SQLite (for conversation memory via LangGraph checkpoints)
- Optional but recommended: spaCy model `en_core_web_sm` (for lemmatization during auto preprocessing)
- Optional: `rank_bm25` package (`pip install rank-bm25`) to enable sparse keyword retrieval in hybrid mode

**Optional:**
- PostgreSQL 13+ (for future database features)

## 🚀 Quick Start with Startup Scripts

The easiest way to get started is using our automated startup scripts that handle all components:

### Using Startup Scripts (Recommended)

```bash
# Start the entire system (Backend API + Frontend)
./start.sh

# Stop all components
./stop.sh
```

**What the startup scripts do:**
- ✅ **Environment Validation** - Check Python version, required packages, directory structure
- 🚀 **Component Startup** - Start Backend API (port 8000) and Frontend (port 5173) in correct order
- 🔍 **Health Monitoring** - Wait for each service to become available before proceeding
- 📊 **Status Display** - Show real-time status of all components
- 🔄 **Auto-Recovery** - Automatically restart components if they crash
- 🛑 **Graceful Shutdown** - Handle Ctrl+C and other signals properly

**Prerequisites for startup scripts:**
```bash
pip install uvicorn fastapi streamlit requests
```

### Manual Setup

If you prefer to start components manually or need to troubleshoot:

## 🛠️ Installation

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jimmyhott/SMART-SECOND-BRAIN.git
   cd SMART-SECOND-BRAIN
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   python -m spacy download en_core_web_sm
   ```

4. **Environment Configuration**
   ```bash
   # Create .env file with your API credentials
   cp .env.example .env

   # Required environment variables:
   OPENAI_API_KEY=your-openai-api-key
   AZURE_OPENAI_ENDPOINT_URL=https://your-resource.openai.azure.com/
   ```

5. **SQLite is automatically handled**
   ```bash
   # SQLite database is automatically created and managed by LangGraph
   # No additional setup required - checkpoints stored in ./data/checkpoints.sqlite
   ```

6. **Start the API**
   ```bash
   # Using uvicorn directly
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

   # Or with conda environment
   # PYTHONPATH=/path/to/project /opt/miniconda3/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Quick Start Example

After installation, test the API:

```bash
# 1. Start the server (ensure spaCy en_core_web_sm is installed)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 2. In another terminal, test the health endpoint
curl "http://localhost:8000/smart-second-brain/api/v1/graph/health"

# 3. Ingest a document
curl -X POST "http://localhost:8000/smart-second-brain/api/v1/graph/ingest" \
  -H "Content-Type: application/json" \
  -d '{"document": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.", "source": "example", "auto_preprocess": true}'

# 4. Query the knowledge base
curl -X POST "http://localhost:8000/smart-second-brain/api/v1/graph/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

## 🧠 How Documents Are Processed

### Auto Metadata Preprocessing
- Controlled by the `auto_preprocess` flag on ingestion. When enabled, the system:
  - Runs spaCy (`en_core_web_sm`) to lemmatize the document, filtering out scikit-learn's English stopwords.
  - Extracts the most frequent lemmas (up to 10 keywords, top 5 for categories).
  - Merges inferred categories/keywords with any caller-supplied metadata.
  - Falls back to a simple regex tokenizer if spaCy is not available, so ingestion continues gracefully.

### Chunking
- Uses `RecursiveCharacterTextSplitter` with 500-character chunks and 50-character overlap.
- Each chunk receives deterministic identifiers (`{doc_id}::chunk_{index}`), character offsets, categories, keywords, knowledge type, ingestion timestamp, and source.
- This metadata makes downstream filtering, reranking, and citation mapping possible.

### Embedding & Storage
- Embeddings default to Azure/OpenAI's `text-embedding-3-small` deployment. Errors are logged and replaced with dummy vectors to keep workflows moving.
- Chunks + metadata are normalized (lists serialized to comma-separated strings) before being written to ChromaDB.
- After persistence, the same `Document` objects are fed to `SmartDocumentRetriever`, ensuring the semantic and BM25 retrievers stay synchronized. On startup the retriever re-hydrates its BM25 index directly from the persisted Chroma documents, so hybrid retrieval is immediately available even before new ingests.

### Retrieval
- Hybrid strategy (semantic + optional BM25) is used when the `rank_bm25` package is installed; otherwise, pure vector search runs.
- Metadata filters (source, categories, knowledge type, auto_preprocess flag) are forwarded to ChromaDB to narrow results.
- Retrieval logs capture scores, chunk IDs, and metadata so the frontend can display citations and debugging information.
- Optional reranking boosts chunks with richer metadata, longer content, or recent timestamps (simple heuristic).

### Answer Generation
- Retrieved chunks are packaged into a context prompt; conversation history is prepended.
- The LLM is instructed to respond with structured JSON `{answer, citations[], is_idk, confidence}`. Failures yield a fallback IDK JSON, keeping the API contract stable.
- Citations contain doc and chunk IDs so the Streamlit frontend can show inline references and an expandable citation list.

### Frontend Setup (Streamlit)

The frontend is built with Streamlit, a modern Python web framework:

```bash
cd frontend
pip install -e .  # Install in development mode
streamlit run app.py --server.port 5173 --server.address 0.0.0.0
```

**Frontend Features:**
- 🏥 **System Health Dashboard** - Real-time status of all components (Graph, Vectorstore, AI models)
- 📄 **PDF Ingestion** - Multi-file PDF upload with batch processing and metadata
- 📝 **Text Ingestion** - Direct text content input with categorization and knowledge type selection
- 🔍 **Knowledge Query** - Natural language search interface with conversation memory
- 💬 **Thread-based Conversations** - SQLite-backed conversation history with thread management
- 👍 **Human Feedback System** - Approve, reject, or edit AI-generated answers
- 🤔 **Smart IDK Interface** - Knowledge input when AI doesn't know something
- 📊 **Status Monitoring** - Live updates and component health
- 🎨 **Modern UI** - Clean, responsive interface built with Streamlit

## 🏗️ Project Structure

```
SMART-SECOND-BRAIN/
├── api/                           # FastAPI backend application
│   ├── __init__.py                # API package initialization
│   ├── main.py                    # FastAPI application entry point and server configuration
│   ├── core/                      # Core API configuration and utilities
│   │   ├── __init__.py            # Core package initialization
│   │   ├── config.py              # Application settings, environment variables, and configuration
│   │   ├── database.py            # Database connection and session management (future)
│   │   ├── security.py            # Authentication and authorization (future)
│   │   └── __pycache__/           # Python bytecode cache
│   └── routes/                    # API endpoint definitions and routing
│       ├── __init__.py            # Routes package initialization
│       └── v1/                    # API version 1 endpoints
│           ├── __init__.py        # V1 package initialization
│           ├── graph_api.py       # LangGraph integration endpoints (/ingest, /ingest-pdfs, /query, /health, /feedback)
│           ├── auth.py            # Authentication endpoints (future)
│           ├── crypto.py          # Cryptocurrency endpoints (future)
│           ├── portfolio.py       # Portfolio management endpoints (future)
│           └── users.py           # User management endpoints (future)
├── agentic/                       # AI Agent System and LangGraph workflows
│   ├── agents/                    # Individual AI agent implementations (future)
│   ├── core/                      # Core agent system components
│   │   ├── knowledge_state.py     # LangGraph state management and data structures
│   │   └── __pycache__/           # Python bytecode cache
│   ├── data/                      # Agent data storage and persistence (future)
│   ├── prompts/                   # AI prompt templates and configurations
│   │   └── answer_prompt.txt      # Answer generation prompt template
│   ├── schemas/                   # Data schemas and validation models (future)
│   ├── secrets/                   # Secure credential management (future)
│   └── workflows/                 # LangGraph workflow definitions
│       ├── master_graph_builder.py # Main knowledge processing workflow with chunking, embedding, and storage
│       └── document_retriever.py   # Advanced document retrieval with semantic search, hybrid retrieval, and re-ranking
├── shared/                        # Shared utilities and common components
│   ├── config/                    # Shared configuration management (future)
│   ├── models/                    # Shared data models and schemas (future)
│   └── utils/                     # Shared utility functions
│       ├── logging_config.py      # Centralized logging configuration and setup
│       └── pathlib_example.py     # Path manipulation utilities example
├── frontend/                      # Streamlit Frontend application
│   ├── app.py                     # Main frontend application with health dashboard and chat interface
│   ├── pyproject.toml             # Frontend dependencies and configuration
│   ├── second-brain.jpeg          # Application icon
│   └── README.md                  # Frontend-specific documentation
├── tests/                         # Comprehensive test suite
│   ├── __init__.py                # Test package initialization
│   ├── conftest.py                # Test configuration, fixtures, and setup
│   ├── test_api/                  # API endpoint tests
│   │   ├── __init__.py            # API test package initialization
│   │   └── test_graph_api.py      # Graph API endpoint tests (47 tests total)
│   ├── test_services/             # Service and workflow tests
│   │   ├── test_master_graph_builder.py # AI workflow integration tests
│   │   ├── test_document_retriever.py   # Document retrieval tests
│   │   └── README.md              # Test documentation
│   └── test_utils/                # Utility function tests
├── infrastructure/                 # Infrastructure and deployment (future)
├── alembic/                       # Database migration system (future)
│   ├── versions/                  # Migration version files
│   ├── env.py                     # Alembic environment configuration
│   └── alembic.ini               # Alembic configuration file
├── logs/                          # Application log files
├── chroma_db/                     # Vector database storage for embeddings
├── .cursor/                       # Cursor IDE configuration
├── .idea/                         # PyCharm IDE configuration
├── .pytest_cache/                 # Pytest cache and temporary files
├── htmlcov/                       # Test coverage HTML reports
├── start_system.py                # Main system startup script with monitoring and auto-recovery
├── stop_system.py                 # System shutdown script with graceful process termination
├── start.sh                       # Shell wrapper for easy system startup
├── stop.sh                        # Shell wrapper for easy system shutdown
├── pyproject.toml                 # Project metadata, dependencies, and build configuration
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules and patterns
└── README.md                      # This comprehensive project documentation
```

## 🔧 Development

### Running Tests
```bash
# Run unit/API tests (no external services required)
pytest tests/test_api tests/test_services/test_document_retriever.py -v

# Integration workflow tests (require Azure/OpenAI credentials)
pytest tests/test_services/test_master_graph_builder.py -v

# Run with coverage
pytest --cov=api --cov=agentic --cov=shared --cov-report=html

# Run specific test file
pytest tests/test_services/test_master_graph_builder.py -v

# Run API tests (7 tests)
pytest tests/test_api/ -v

# Run service tests (40 tests)
pytest tests/test_services/ -v

# Run specific test class
pytest tests/test_api/test_graph_api.py::TestGraphAPI -v

# Run with detailed output
pytest tests/ -v --tb=long
```

### Code Formatting
```bash
black api/ agentic/ shared/ tests/
ruff check api/ agentic/ shared/ tests/
mypy api/ agentic/ shared/
```

### Starting the Development Server
```bash
# Using conda environment (recommended)
PYTHONPATH=/path/to/SMART-SECOND-BRAIN /opt/miniconda3/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or using pip virtual environment
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Database Migrations (Future Feature)
```bash
# Create new migration (when database is implemented)
alembic revision --autogenerate -m "Description of changes"

# Apply migrations (when database is implemented)
alembic upgrade head

# Rollback migration (when database is implemented)
alembic downgrade -1
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## 🌐 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Maintenance Scripts

#### **Clear Vector Database**
```bash
# Interactive mode with confirmation
python clear_vector_db.py

# Non-interactive mode (for automation)
python clear_vector_db.py --confirm

# Create backup before clearing
python clear_vector_db.py --backup --confirm

# Custom collection and directory
python clear_vector_db.py --collection "my_collection" --directory "./custom_chroma"
```

⚠️ **Warning**: This operation is irreversible and will delete ALL vector data.

### Core API Endpoints

#### **Text Document Ingestion**
```http
POST /smart-second-brain/api/v1/graph/ingest
Content-Type: application/json

{
  "document": "Your document content here...",
  "source": "webpage",
  "categories": ["research", "ai"],
  "metadata": {"author": "John Doe", "date": "2024-01-01"},
  "auto_preprocess": true,
  "collection_name": "smart_second_brain"
}
```

#### **Multiple PDF Ingestion**
```http
POST /smart-second-brain/api/v1/graph/ingest-pdfs
Content-Type: multipart/form-data

files: [PDF files]
source: "research_papers"
categories: "ai,research"
author: "Research Team"
metadata: {"project": "AI Research", "auto_preprocess": true}
```

#### **Knowledge Query**
```http
POST /smart-second-brain/api/v1/graph/query
Content-Type: application/json

{
  "query": "What are the main concepts discussed?",
  "thread_id": "conversation_123",
  "collection_name": "smart_second_brain"
}
```

#### Human-in-the-Loop (HITL) with Interrupts

The query workflow can pause at a human-review step using LangGraph interrupts. You can opt-in per request:

```http
POST /smart-second-brain/api/v1/graph/query
Content-Type: application/json

{
  "query": "Please summarize the content",
  "knowledge_type": "reusable",            // conversational | reusable | verified
  "require_human_review": true,             // if omitted, inferred from knowledge_type
  "collection_name": "smart_second_brain"
}
```

If a pause is required, the response includes a `thread_id` you will use to approve or edit the answer.

Approve to resume and optionally store:

```http
POST /smart-second-brain/api/v1/graph/feedback
Content-Type: application/json

{
  "thread_id": "query_1700000000",
  "feedback": "approved",                  // approved | rejected | edited
  "knowledge_type": "reusable",           // controls storage policy
  "collection_name": "smart_second_brain"
}
```

Edit the answer and resume:

```http
POST /smart-second-brain/api/v1/graph/feedback
Content-Type: application/json

{
  "thread_id": "query_1700000000",
  "feedback": "edited",
  "edits": "<your edited final answer>",
  "knowledge_type": "reusable",
  "collection_name": "smart_second_brain"
}
```

#### **Health Check**
```http
GET /smart-second-brain/api/v1/graph/health
```

#### **Submit Feedback (HITL Resume)**
```http
POST /smart-second-brain/api/v1/graph/feedback
Content-Type: application/json

{
  "thread_id": "<thread_id from query response>",
  "feedback": "approved" | "rejected" | "edited",
  "edits": "<required when feedback is 'edited'>",
  "knowledge_type": "conversational" | "reusable" | "verified",
  "collection_name": "smart_second_brain"
}
```

#### **Get Feedback Status**
```http
GET /smart-second-brain/api/v1/graph/feedback/{thread_id}?collection_name=smart_second_brain
```

#### **Clear Vector Database**
```http
DELETE /smart-second-brain/api/v1/graph/clear-vector-db
```

### Collection Configuration

The system supports multiple ChromaDB collections for organizing different types of knowledge. All API endpoints accept a `collection_name` parameter to specify which collection to use.

#### **Default Collection**
- **Name**: `smart_second_brain`
- **Purpose**: Default collection for general knowledge storage
- **Usage**: Used automatically when no collection name is specified

#### **Custom Collections**
You can create and use custom collections for different purposes:

```http
POST /smart-second-brain/api/v1/graph/ingest
Content-Type: application/json

{
  "document": "Research paper content...",
  "source": "academic_papers",
  "collection_name": "research_knowledge"
}
```

#### **Collection Isolation**
- Documents stored in one collection are not accessible from other collections
- Each collection maintains its own vector embeddings and metadata
- Query operations only search within the specified collection
- Perfect for organizing knowledge by domain, project, or user

#### **Frontend Configuration**
The Streamlit frontend provides a sidebar configuration where you can:
- Set the default collection name for all operations
- Switch between collections during your session
- Maintain consistent collection usage across all features

### API Response Format
```json
{
  "success": true,
  "thread_id": "ingest_1704067200",
  "result": {
    "status": "completed",
    "chunks_processed": 5,
    "embeddings_created": 5
  },
  "execution_time": 2.34,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🚀 System Management

### Startup Scripts

Our automated startup scripts provide the easiest way to manage your Smart Second Brain system:

#### **Quick Commands**
```bash
# Start everything
./start.sh

# Stop everything  
./stop.sh
```

#### **What Happens When You Start**
1. **Environment Check** - Validates Python version, packages, and directory structure
2. **Port Availability** - Ensures ports 8000 and 5173 are free
3. **Backend Start** - Launches FastAPI backend with uvicorn
4. **Health Check** - Waits for backend to respond to health endpoint
5. **Frontend Start** - Launches NiceGUI frontend
6. **Health Check** - Waits for frontend to become available
7. **Monitoring** - Starts background monitoring for auto-recovery
8. **Status Display** - Shows running status and access URLs

#### **What Happens When You Stop**
1. **Process Discovery** - Finds all Smart Second Brain processes by name and port
2. **Graceful Shutdown** - Sends SIGTERM to all processes
3. **Timeout Handling** - Waits up to 10 seconds for graceful shutdown
4. **Force Stop** - Sends SIGKILL if graceful shutdown fails
5. **Cleanup** - Removes process references and reports status

#### **Auto-Recovery Features**
- **Health Checks** - Continuously monitors component health
- **Auto-Restart** - Automatically restarts crashed components
- **Process Monitoring** - Tracks all running processes
- **Graceful Shutdown** - Handles system signals properly

#### **Troubleshooting**
```bash
# If you get port conflicts
./stop.sh
./start.sh

# Check what's running
ps aux | grep -E "(uvicorn|python app.py)"

# Check ports
lsof -i :8000
lsof -i :5173

# Manual process management
kill -TERM <PID>
kill -KILL <PID>  # Force kill if needed
```

### Manual Component Management

If you prefer to start components individually:

```bash
# Backend only
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend only  
cd frontend && python app.py

# Or with conda environment
PYTHONPATH=/path/to/project /opt/miniconda3/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 🔐 Environment Variables

Create a `.env` file with the following variables:

```env
# Required: OpenAI/Azure OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
AZURE_OPENAI_ENDPOINT_URL=https://your-resource.openai.azure.com/

# Optional: Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true
LOG_LEVEL=INFO

# Optional: CORS Configuration
ALLOWED_HOSTS=["*"]

# SQLite checkpointing is automatically configured
# Checkpoints stored in ./data/checkpoints.sqlite

# Optional: Future Database Configuration
# DATABASE_URL=postgresql://user:password@localhost/smart_second_brain
```

### Environment Variables Explanation

- **`OPENAI_API_KEY`**: Your OpenAI API key for embeddings and LLM access
- **`AZURE_OPENAI_ENDPOINT_URL`**: Your Azure OpenAI resource endpoint URL
- **`HOST` & `PORT`**: Server configuration (defaults to 0.0.0.0:8000)
- **`DEBUG`**: Enable debug mode for development
- **`ALLOWED_HOSTS`**: CORS allowed origins (defaults to all)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


