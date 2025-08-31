# Smart Second Brain

A comprehensive AI-powered knowledge management platform with intelligent document processing, knowledge extraction, and modern web interface.

## 🚀 Features

- **FastAPI Backend**: High-performance REST API with automatic documentation
- **AI-Powered Knowledge Management**: Intelligent document processing and knowledge extraction
- **Smart Agents**: AI-powered agents for knowledge processing and insights using LangGraph
- **Knowledge Graph**: Intelligent connections and relationships between information
- **Azure OpenAI Integration**: Full support for Azure OpenAI with automatic deployment detection
- **Vector Database**: Chroma vector store for semantic search and document retrieval
- **Modern Python**: Pathlib for cross-platform path handling, type hints, and modern Python practices
- **Comprehensive Testing**: Unit tests with mocked components and integration tests with real APIs
- **Centralized Logging**: Project-level logging configuration with file and console output

## 📋 Requirements

- Python 3.12+
- PostgreSQL 13+
- Azure OpenAI Service (for AI features)
- Node.js 18+ (for frontend - coming soon)

**Optional:**
- Redis 6+ (for background tasks and caching - coming soon)

## 🛠️ Installation

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jimmyhott/smart-second-brain.git
   cd smart-second-brain
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your database and API credentials
   ```

5. **Environment Configuration**
   ```bash
   # Copy and configure environment variables
   cp .env.example .env
   
   # Required environment variables for AI features:
   # OPENAI_API_KEY=your-openai-api-key
   # AZURE_OPENAI_ENDPOINT_URL=https://your-resource.openai.azure.com/
   # LLM_MODEL=gpt-4o
   # API_VERSION=2024-12-01-preview
   # EMBEDDING_MODEL=text-embedding-ada-002
   ```

6. **Database Setup**
   ```bash
   # Create PostgreSQL database
   createdb smart_second_brain
   
   # Run migrations
   alembic upgrade head
   ```

6. **Start the API**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
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
│   ├── core/                      # Core configuration and utilities
│   │   ├── __init__.py
│   │   ├── config.py              # Settings and configuration
│   │   ├── database.py            # Database connection and session
│   │   └── security.py            # Authentication and security
│   ├── routes/                    # API routes and endpoints
│   │   ├── __init__.py
│   │   ├── v1/                    # API version 1
│   │   │   ├── __init__.py
│   │   │   ├── auth.py            # Authentication endpoints
│   │   │   ├── crypto.py          # Cryptocurrency data endpoints
│   │   │   ├── portfolio.py       # Portfolio management endpoints
│   │   │   └── users.py           # User management endpoints
│   │   └── deps.py                # Dependency injection
│   ├── models/                    # Database models
│   │   ├── __init__.py
│   │   ├── user.py                # User model
│   │   ├── crypto.py              # Cryptocurrency models
│   │   └── portfolio.py           # Portfolio models
│   ├── schemas/                   # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py                # User schemas
│   │   ├── crypto.py              # Cryptocurrency schemas
│   │   └── portfolio.py           # Portfolio schemas
│   ├── services/                  # Business logic
│   │   ├── __init__.py
│   │   ├── crypto_service.py      # Cryptocurrency data service
│   │   ├── auth_service.py        # Authentication service
│   │   └── portfolio_service.py   # Portfolio management service
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       └── helpers.py             # Helper functions
├── agentic/                       # AI Agent System
│   ├── agents/                    # AI agents
│   ├── core/                      # Core agent components
│   │   └── knowledge_state.py     # LangGraph state management
│   ├── workflows/                 # LangGraph workflows
│   │   └── master_graph_builder.py # Main knowledge processing workflow
│   ├── tools/                     # Agent tools
│   └── prompts/                   # Agent prompts
├── shared/                        # Shared utilities
│   ├── config/                    # Shared configuration
│   ├── models/                    # Shared models
│   └── utils/                     # Shared utilities
│       └── logging_config.py      # Centralized logging configuration
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

# Run API connection test
python test_openai_connection.py
```

### Code Formatting
```bash
black api/ agentic/ shared/ tests/
ruff check api/ agentic/ shared/ tests/
mypy api/ agentic/ shared/
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
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

## 🔐 Environment Variables

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/smart_second_brain

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis (Optional - for background tasks and caching)
# REDIS_URL=redis://localhost:6379

# AI/LLM Configuration
OPENAI_API_KEY=your-openai-api-key-here
AZURE_OPENAI_ENDPOINT_URL=https://your-resource.openai.azure.com/
LLM_MODEL=gpt-4o
API_VERSION=2024-12-01-preview
EMBEDDING_MODEL=text-embedding-ada-002

# External APIs
COINAPI_KEY=your-coinapi-key
BINANCE_API_KEY=your-binance-api-key
BINANCE_SECRET_KEY=your-binance-secret-key

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🧪 Testing

### Test Categories

- **Unit Tests**: Fast, isolated tests with mocked components
- **Integration Tests**: End-to-end tests with real Azure OpenAI APIs
- **API Tests**: FastAPI endpoint testing
- **Service Tests**: Business logic testing

### Test Configuration

Tests use pytest markers:
- `@pytest.mark.unit` - Unit tests (mocked)
- `@pytest.mark.integration` - Integration tests (real APIs)

### Logging

All tests use centralized logging configuration:
- Console output for immediate feedback
- File output in `logs/` directory for debugging
- Different log levels for different test types

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Email: support@smartsecondbrain.com
- Discord: [Join our community](https://discord.gg/smartsecondbrain)
