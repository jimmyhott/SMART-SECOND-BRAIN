import axios from 'axios'

// API base configuration
const API_BASE_URL = 'http://localhost:8000'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// API endpoints
export const API_ENDPOINTS = {
  HEALTH: '/smart-second-brain/api/v1/graph/health',
  INGEST: '/smart-second-brain/api/v1/graph/ingest',
  QUERY: '/smart-second-brain/api/v1/graph/query',
} as const

// Types
export interface IngestRequest {
  document: string
  source?: string
  categories?: string[]
  metadata?: Record<string, any>
}

export interface QueryRequest {
  query: string
  thread_id?: string
}

export interface WorkflowResponse {
  success: boolean
  thread_id: string
  result: Record<string, any>
  execution_time: number
  timestamp: string
}

export interface HealthResponse {
  status: string
  graph_initialized: boolean
  vectorstore_ready: boolean
  embedding_model_ready: boolean
  llm_ready: boolean
  timestamp: string
}

// API methods
export const smartBrainAPI = {
  // Health check
  async getHealth(): Promise<HealthResponse> {
    const response = await api.get(API_ENDPOINTS.HEALTH)
    return response.data
  },

  // Document ingestion
  async ingestDocument(data: IngestRequest): Promise<WorkflowResponse> {
    const response = await api.post(API_ENDPOINTS.INGEST, data)
    return response.data
  },

  // Knowledge query
  async queryKnowledge(data: QueryRequest): Promise<WorkflowResponse> {
    const response = await api.post(API_ENDPOINTS.QUERY, data)
    return response.data
  },
}

// Error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error)
    if (error.response?.status === 500) {
      throw new Error('Server error: Please check your API configuration')
    }
    throw error
  }
)

export default api
