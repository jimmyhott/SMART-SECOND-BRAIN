# Smart Second Brain

An intelligent knowledge management platform with AI-powered document processing, semantic search, and LangGraph workflows for building a personal second brain.

## 🚀 Features

- **FastAPI Backend**: High-performance REST API with automatic documentation
- **LangGraph Integration**: Advanced AI workflows for document processing and knowledge extraction
- **AI-Powered Knowledge Management**: Intelligent document ingestion, chunking, and embedding
- **Semantic Search**: Vector-based retrieval using ChromaDB for accurate document search
- **Azure OpenAI Integration**: Full support for Azure OpenAI with automatic deployment detection
- **Smart Query Processing**: RAG (Retrieval-Augmented Generation) for intelligent question answering
- **Thread-based Conversations**: Redis-backed conversation memory for multi-turn conversations
- **Modern Python**: Pathlib for cross-platform path handling, type hints, and async programming
- **Comprehensive Testing**: Unit tests with mocked components and integration tests with real APIs
- **Centralized Logging**: Project-level logging configuration with file and console output

## 📋 Requirements

- Python 3.12+
- Azure OpenAI Service (for AI features)
- OpenAI API Key (for embeddings and LLM)
- Redis 8+ (for conversation memory)

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
pip install uvicorn fastapi streamlit requests redis
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

5. **Start Redis**
   ```bash
   # On macOS with Homebrew
   brew install redis
   brew services start redis
   
   # On Ubuntu/Debian
   sudo apt install redis-server
   sudo systemctl start redis
   
   # Verify Redis is running
   redis-cli ping  # Should return PONG
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

### Frontend Setup (Streamlit)

The frontend is built with Streamlit, a modern Python web framework:

```bash
cd frontend
pip install -e .  # Install in development mode
streamlit run app.py --server.port 5173 --server.address 0.0.0.0
```

**Frontend Features:**
- 🏥 **System Health Dashboard** - Real-time status of all components
- 📄 **Document Ingestion** - Easy PDF upload and processing
- 🔍 **Knowledge Query** - Natural language search interface with conversation memory
- 💬 **Thread-based Conversations** - Redis-backed conversation history
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
│           ├── graph_api.py       # LangGraph integration endpoints (/ingest, /query, /health)
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
│   ├── prompts/                   # AI prompt templates and configurations (future)
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
│   ├── test_services/             # Service and workflow tests
│   │   ├── test_master_graph_builder.py # AI workflow integration tests
│   │   └── test_document_retriever.py   # Document retrieval tests
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

# Optional: Redis Configuration (defaults to localhost:6379)
# REDIS_URL=redis://localhost:6379

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


