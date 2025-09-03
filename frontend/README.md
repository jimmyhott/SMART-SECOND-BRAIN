# Smart Second Brain Frontend

A modern Vue 3 frontend application for the Smart Second Brain AI-powered knowledge management platform.

## 🚀 Features

- **Modern Vue 3** with Composition API and TypeScript
- **Beautiful UI** built with Tailwind CSS and Headless UI components
- **Real-time API Integration** with your Smart Second Brain backend
- **Responsive Design** that works on all devices
- **State Management** with Pinia for efficient data handling
- **Type Safety** with full TypeScript support

## 🛠️ Tech Stack

- **Vue 3.4+** - Progressive JavaScript framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Pinia** - State management for Vue
- **Vue Router** - Official router for Vue.js
- **Axios** - HTTP client for API communication
- **Headless UI** - Unstyled, accessible UI components
- **Heroicons** - Beautiful SVG icons

## 📋 Prerequisites

- **Node.js 18+** and npm
- **Smart Second Brain Backend** running on `http://localhost:8000`
- **OpenAI API Key** configured in your backend

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### 3. Build for Production
```bash
npm run build
```

### 4. Preview Production Build
```bash
npm run preview
```

## 🌐 API Configuration

The frontend is configured to connect to your Smart Second Brain backend at:
- **Base URL**: `http://localhost:8000`
- **API Endpoints**:
  - Health Check: `/smart-second-brain/api/v1/graph/health`
  - Document Ingestion: `/smart-second-brain/api/v1/graph/ingest`
  - Knowledge Query: `/smart-second-brain/api/v1/graph/query`

## 🎨 UI Components

### Core Components
- **`SystemHealth.vue`** - Displays backend system status
- **`DocumentIngestion.vue`** - Form for uploading documents
- **`KnowledgeQuery.vue`** - Interface for asking questions
- **`StatusIcon.vue`** - Reusable status indicator

### Layout
- **`HomeView.vue`** - Main dashboard with all features
- **`App.vue`** - Root application component
- **Responsive grid layout** that adapts to screen size

## 🔧 Development

### Project Structure
```
src/
├── components/          # Reusable Vue components
├── views/              # Page-level components
├── stores/             # Pinia state management
├── services/           # API communication layer
├── router/             # Vue Router configuration
├── style.css           # Global styles with Tailwind
└── main.ts            # Application entry point
```

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier
- `npm run type-check` - Run TypeScript type checking

### Styling
- **Tailwind CSS** for utility-first styling
- **Custom CSS components** in `style.css`
- **Responsive design** with mobile-first approach
- **Color scheme** using primary and secondary color palettes

## 🔌 API Integration

### Smart Brain API Service
The frontend communicates with your backend through the `smartBrainAPI` service:

```typescript
// Health check
await smartBrainAPI.getHealth()

// Document ingestion
await smartBrainAPI.ingestDocument({
  document: "Your content here...",
  source: "webpage",
  categories: ["ai", "research"]
})

// Knowledge query
await smartBrainAPI.queryKnowledge({
  query: "What is machine learning?",
  thread_id: "optional_thread_id"
})
```

### State Management
Uses Pinia store (`useSmartBrainStore`) for:
- System health status
- Recent ingestions and queries
- Loading states and error handling
- Thread ID management for conversations

## 🎯 Key Features

### 1. System Health Monitoring
- Real-time backend status checking
- Component health indicators
- Automatic refresh capabilities

### 2. Document Ingestion
- Rich text input for documents
- Metadata and categorization support
- JSON metadata input
- Success/error feedback

### 3. Knowledge Querying
- Natural language question input
- Thread-based conversation support
- AI-generated answers display
- Retrieved document snippets

### 4. Activity Dashboard
- Recent ingestion history
- Query history with results
- Execution time tracking
- Timestamp information

## 🚨 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure your backend is running on `http://localhost:8000`
   - Check that your OpenAI API key is configured
   - Verify the backend health endpoint is accessible

2. **Build Errors**
   - Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
   - Check Node.js version: `node --version` (should be 18+)

3. **Styling Issues**
   - Ensure Tailwind CSS is properly configured
   - Check that `style.css` is imported in `main.ts`

### Development Tips

- Use Vue DevTools for debugging
- Check browser console for API errors
- Monitor network tab for request/response details
- Use the health endpoint to verify backend status

## 📱 Responsive Design

The frontend is fully responsive and works on:
- **Desktop** (1024px+) - Full layout with side-by-side components
- **Tablet** (768px-1023px) - Adjusted grid layouts
- **Mobile** (<768px) - Stacked single-column layout

## 🔒 Security

- **No sensitive data** stored in frontend
- **API keys** managed by backend only
- **CORS** configured for local development
- **Input validation** on all forms

## 🤝 Contributing

1. Follow Vue 3 Composition API patterns
2. Use TypeScript for all new code
3. Follow Tailwind CSS utility-first approach
4. Maintain responsive design principles
5. Add proper error handling for API calls

## 📄 License

This frontend is part of the Smart Second Brain project and follows the same license terms.

---

**Happy coding! 🚀**

Your Smart Second Brain frontend is ready to help users interact with their AI-powered knowledge base!
