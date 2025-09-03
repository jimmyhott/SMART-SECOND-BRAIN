# Smart Second Brain - NiceGUI Frontend

A modern, Python-based web interface for the Smart Second Brain AI-powered knowledge management system.

## 🚀 Features

- **System Health Monitoring** - Real-time status of all backend services
- **Document Ingestion** - Easy document upload and processing
- **Knowledge Querying** - AI-powered question answering
- **Real-time Updates** - Live status monitoring and auto-refresh
- **Modern UI** - Clean, responsive interface built with NiceGUI
- **No Build Tools** - Pure Python, no JavaScript compilation needed

## 🛠️ Technology Stack

- **NiceGUI** - Modern Python web framework
- **Python 3.12+** - Async/await support
- **HTTPX** - Async HTTP client for API communication
- **Pure Python** - No build tools or dependencies

## 📋 Prerequisites

- Python 3.12+
- Smart Second Brain backend running on port 8000
- All backend services healthy (Graph, Vectorstore, AI models)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd frontend
pip install -e .
```

### 2. Start the Frontend

```bash
python app.py
```

The frontend will open automatically in your browser at `http://localhost:5173`

### 3. Verify Backend Connection

- Check that the System Health section shows all green checkmarks
- Ensure your backend API is running on port 8000

## 🎯 Usage

### System Health
- **Auto-refresh**: Health status updates every 10 seconds
- **Manual refresh**: Click "🔄 Refresh Health" button
- **Status indicators**: Visual indicators for each service component

### Document Ingestion
1. **Paste content** into the Document Content textarea
2. **Add metadata**: Source, categories, author (optional)
3. **Click "Ingest Document"** to process
4. **View results** in Recent Ingestions section

### Knowledge Querying
1. **Enter your question** in the query textarea
2. **Add thread ID** for conversation continuity (optional)
3. **Click "Query Knowledge"** to get AI response
4. **View results** including AI answer and retrieved documents

## 🔧 Configuration

### API Endpoints
The frontend is configured to connect to:
- **Health Check**: `http://localhost:8000/smart-second-brain/api/v1/graph/health`
- **Document Ingestion**: `http://localhost:8000/smart-second-brain/api/v1/graph/ingest`
- **Knowledge Query**: `http://localhost:8000/smart-second-brain/api/v1/graph/query`

### Port Configuration
- **Frontend**: Port 5173 (configurable in `app.py`)
- **Backend**: Port 8000 (must match your backend configuration)

## 🎨 UI Components

### Header
- Application title and description
- Manual health refresh button
- Link to API documentation

### System Health Dashboard
- Overall system status
- Individual service indicators
- Last check timestamp

### Document Ingestion Panel
- Large textarea for content
- Metadata input fields
- Recent ingestions history

### Knowledge Query Panel
- Query input interface
- AI response display
- Retrieved documents view
- Query history

### Quick Start Guide
- Step-by-step instructions
- Visual workflow guide

## 🔍 Troubleshooting

### Frontend Won't Start
- Check Python version (3.12+ required)
- Verify NiceGUI installation: `pip show nicegui`
- Check port 5173 availability
- Install in development mode: `pip install -e .`

### Can't Connect to Backend
- Ensure backend is running on port 8000
- Check backend health endpoint manually
- Verify API endpoints in configuration

### Health Status Errors
- Check backend logs for errors
- Verify Azure OpenAI configuration
- Ensure all required environment variables are set

## 🚀 Development

### Adding New Features
1. **Modify `app.py`** - Add new UI components and functions
2. **Update API calls** - Add new endpoints as needed
3. **Test integration** - Verify with backend functionality

### Styling
- Uses NiceGUI's built-in styling system
- Custom CSS can be added via `ui.html()`
- Responsive design with column layouts

### State Management
- Global variables for system status
- Local state for recent activities
- Async functions for API communication

## 📱 Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Mobile**: Responsive design

## 🔒 Security Notes

- Frontend runs locally by default
- No authentication built-in (add as needed)
- API calls use HTTP (HTTPS recommended for production)

## 📞 Support

For issues or questions:
- Check backend logs first
- Verify API endpoints are accessible
- Ensure all dependencies are installed
- Check Python version compatibility
