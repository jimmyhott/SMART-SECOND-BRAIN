# Smart Second Brain

An intelligent knowledge management platform with AI-powered document processing, semantic search, and LangGraph workflows for building a personal second brain.

## 🚀 Features

- **FastAPI Backend**: High-performance REST API with automatic documentation
- **LangGraph Integration**: Advanced AI workflows for document processing and knowledge extraction
- **AI-Powered Knowledge Management**: Intelligent document ingestion, chunking, and embedding
- **Semantic Search**: Vector-based retrieval using ChromaDB for accurate document search
- **Azure OpenAI Integration**: Full support for Azure OpenAI with automatic deployment detection
- **Smart Query Processing**: RAG (Retrieval-Augmented Generation) for intelligent question answering
- **Thread-based Conversations**: LangGraph checkpointing for multi-turn conversations
- **Modern Python**: Pathlib for cross-platform path handling, type hints, and async programming
- **Comprehensive Testing**: Unit tests with mocked components and integration tests with real APIs
- **Centralized Logging**: Project-level logging configuration with file and console output

## 📋 Requirements

- Python 3.12+
- Azure OpenAI Service (for AI features)
- OpenAI API Key (for embeddings and LLM)

**Optional:**
- PostgreSQL 13+ (for future database features)
- Redis 6+ (for future caching features)

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
   pip install -r requirements.txt
   # Or if using conda environment:
   # conda install -c conda-forge uvicorn fastapi pydantic pydantic-settings
   ```

4. **Environment Configuration**
   ```bash
   # Create .env file with your API credentials
   cp .env.example .env

   # Required environment variables:
   OPENAI_API_KEY=your-openai-api-key
   AZURE_OPENAI_ENDPOINT_URL=https://your-resource.openai.azure.com/
   ```

5. **Start the API**
   ```bash
   # Using uvicorn directly
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

   # Or with conda environment
   # PYTHONPATH=/path/to/project /opt/miniconda3/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Quick Start Example

After installation, test the API:

```bash
# 1. Start the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 2. In another terminal, test the health endpoint
curl "http://localhost:8000/smart-second-brain/api/v1/graph/health"

# 3. Ingest a document
curl -X POST "http://localhost:8000/smart-second-brain/api/v1/graph/ingest" \
  -H "Content-Type: application/json" \
  -d '{"document": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.", "source": "example"}'

# 4. Query the knowledge base
curl -X POST "http://localhost:8000/smart-second-brain/api/v1/graph/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

### Frontend Setup (Coming Soon)

```bash
cd frontend
npm install
npm run dev
```

## 🏗️ Project Structure

```
SMART-SECOND-BRAIN/
├── api/                           # FastAPI backend
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── core/                      # Core configuration
│   │   ├── __init__.py
│   │   └── config.py              # Settings and configuration
│   └── routes/                    # API routes and endpoints
│       ├── __init__.py
│       └── v1/                    # API version 1
│           ├── __init__.py
│           └── graph_api.py       # LangGraph integration endpoints
├── agentic/                       # AI Agent System
│   ├── core/                      # Core agent components
│   │   └── knowledge_state.py     # LangGraph state management
│   ├── workflows/                 # LangGraph workflows
│   │   ├── master_graph_builder.py # Main knowledge processing workflow
│   │   └── document_retriever.py   # Document retrieval system
│   ├── data/                      # Agent data storage
│   ├── prompts/                   # Agent prompts
│   ├── schemas/                   # Agent schemas
│   ├── secrets/                   # Secure credentials
│   ├── tools/                     # Agent tools
│   └── workflows/                 # Agent workflows
├── shared/                        # Shared utilities
│   ├── config/                    # Shared configuration
│   ├── models/                    # Shared models
│   └── utils/                     # Shared utilities
│       ├── logging_config.py      # Centralized logging configuration
│       └── pathlib_example.py     # Path utilities example
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Test configuration
│   ├── test_api/                  # API tests
│   ├── test_services/             # Service tests
│   │   └── test_master_graph_builder.py # AI workflow tests
│   └── test_utils/                # Utility tests
├── alembic/                       # Database migrations
│   ├── versions/
│   ├── env.py
│   └── alembic.ini
├── demo_document_retriever.py     # Document retriever demo
├── frontend/                      # Frontend application (coming soon)
├── pyproject.toml                 # Project configuration
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🔧 Development

### Running Tests
```bash
# Run all tests
pytest

# Run unit tests only (mocked components)
pytest -m unit

# Run integration tests only (real APIs)
pytest -m integration

# Run with coverage
pytest --cov=api --cov=agentic --cov=shared --cov-report=html

# Run specific test file
pytest tests/test_services/test_master_graph_builder.py

# Run API tests
pytest tests/test_api/

# Run service tests
pytest tests/test_services/
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

### Core API Endpoints

#### **Document Ingestion**
```http
POST /smart-second-brain/api/v1/graph/ingest
Content-Type: application/json

{
  "document": "Your document content here...",
  "source": "webpage",
  "categories": ["research", "ai"],
  "metadata": {"author": "John Doe", "date": "2024-01-01"}
}
```

#### **Knowledge Query**
```http
POST /smart-second-brain/api/v1/graph/query
Content-Type: application/json

{
  "query": "What are the main concepts discussed?",
  "thread_id": "conversation_123"
}
```

#### **Health Check**
```http
GET /smart-second-brain/api/v1/graph/health
```

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

# Optional: Future Database Configuration
# DATABASE_URL=postgresql://user:password@localhost/smart_second_brain
# REDIS_URL=redis://localhost:6379
```

### Environment Variables Explanation

- **`OPENAI_API_KEY`**: Your OpenAI API key for embeddings and LLM access
- **`AZURE_OPENAI_ENDPOINT_URL`**: Your Azure OpenAI resource endpoint URL
- **`HOST` & `PORT`**: Server configuration (defaults to 0.0.0.0:8000)
- **`DEBUG`**: Enable debug mode for development
- **`ALLOWED_HOSTS`**: CORS allowed origins (defaults to all)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


