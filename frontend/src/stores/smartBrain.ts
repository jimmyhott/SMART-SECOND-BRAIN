import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { smartBrainAPI, type IngestRequest, type QueryRequest, type WorkflowResponse, type HealthResponse } from '@/services/api'

export const useSmartBrainStore = defineStore('smartBrain', () => {
  // State
  const health = ref<HealthResponse | null>(null)
  const isHealthy = ref(false)
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const recentIngestions = ref<WorkflowResponse[]>([])
  const recentQueries = ref<WorkflowResponse[]>([])
  const currentThreadId = ref<string | null>(null)

  // Computed
  const systemStatus = computed(() => {
    if (!health.value) return 'unknown'
    return health.value.status
  })

  const canIngest = computed(() => {
    return health.value?.graph_initialized && health.value?.embedding_model_ready
  })

  const canQuery = computed(() => {
    return health.value?.graph_initialized && health.value?.llm_ready
  })

  // Actions
  const checkHealth = async () => {
    try {
      isLoading.value = true
      error.value = null
      health.value = await smartBrainAPI.getHealth()
      isHealthy.value = health.value.status === 'healthy'
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to check health'
      isHealthy.value = false
    } finally {
      isLoading.value = false
    }
  }

  const ingestDocument = async (data: IngestRequest) => {
    try {
      isLoading.value = true
      error.value = null
      const response = await smartBrainAPI.ingestDocument(data)
      recentIngestions.value.unshift(response)
      if (recentIngestions.value.length > 10) {
        recentIngestions.value = recentIngestions.value.slice(0, 10)
      }
      return response
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to ingest document'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const queryKnowledge = async (data: QueryRequest) => {
    try {
      isLoading.value = true
      error.value = null
      const response = await smartBrainAPI.queryKnowledge(data)
      recentQueries.value.unshift(response)
      if (recentQueries.value.length > 10) {
        recentQueries.value = recentQueries.value.slice(0, 10)
      }
      currentThreadId.value = response.thread_id
      return response
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to query knowledge'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const clearError = () => {
    error.value = null
  }

  const resetStore = () => {
    health.value = null
    isHealthy.value = false
    isLoading.value = false
    error.value = null
    recentIngestions.value = []
    recentQueries.value = []
    currentThreadId.value = null
  }

  return {
    // State
    health,
    isHealthy,
    isLoading,
    error,
    recentIngestions,
    recentQueries,
    currentThreadId,
    
    // Computed
    systemStatus,
    canIngest,
    canQuery,
    
    // Actions
    checkHealth,
    ingestDocument,
    queryKnowledge,
    clearError,
    resetStore,
  }
})
