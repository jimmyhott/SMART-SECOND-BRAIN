# SmartSecondBrain

A comprehensive Smart Second Brain platform with AI-powered knowledge management and modern web interface.

## 🚀 Features

- **FastAPI Backend**: High-performance REST API with automatic documentation
- **AI-Powered Knowledge Management**: Intelligent document processing and knowledge extraction
- **Smart Agents**: AI-powered agents for knowledge processing and insights
- **Knowledge Graph**: Intelligent connections and relationships between information
- **Portfolio Tracking**: Personal portfolio management
- **Alerts & Notifications**: Price alerts and market notifications

## 📋 Requirements

- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Node.js 18+ (for frontend - coming soon)

## 🛠️ Installation

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jimmyhott/CryptoAnalyst.git
   cd CryptoAnalyst
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

5. **Database Setup**
   ```bash
   # Create PostgreSQL database
   createdb crypto_analyst
   
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
CryptoAnalyst/
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
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Test configuration
│   ├── test_api/                   # API tests
│   ├── test_services/              # Service tests
│   └── test_utils/                 # Utility tests
├── alembic/                        # Database migrations
│   ├── versions/
│   ├── env.py
│   └── alembic.ini
├── frontend/                       # Frontend application (coming soon)
├── pyproject.toml                  # Project configuration
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🔧 Development

### Running Tests
```bash
pytest
pytest --cov=src/crypto_analyst --cov-report=html
```

### Code Formatting
```bash
black src/ tests/
ruff check src/ tests/
mypy src/
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
DATABASE_URL=postgresql://user:password@localhost/crypto_analyst

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis
REDIS_URL=redis://localhost:6379

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

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Email: support@cryptoanalyst.com
- Discord: [Join our community](https://discord.gg/cryptoanalyst)
