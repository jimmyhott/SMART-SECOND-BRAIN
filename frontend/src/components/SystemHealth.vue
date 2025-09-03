<template>
  <div class="card">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-semibold text-secondary-900">System Health</h3>
      <button 
        @click="checkHealth" 
        :disabled="isLoading"
        class="btn-secondary text-sm"
      >
        <svg v-if="isLoading" class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        {{ isLoading ? 'Checking...' : 'Refresh' }}
      </button>
    </div>

    <div v-if="error" class="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
      <p class="text-red-700 text-sm">{{ error }}</p>
    </div>

    <div v-if="health" class="space-y-3">
      <!-- Overall Status -->
      <div class="flex items-center justify-between">
        <span class="text-sm font-medium text-secondary-700">Overall Status</span>
        <span 
          :class="{
            'px-2 py-1 text-xs font-medium rounded-full': true,
            'bg-green-100 text-green-800': systemStatus === 'healthy',
            'bg-yellow-100 text-yellow-800': systemStatus === 'degraded',
            'bg-red-100 text-red-800': systemStatus === 'unhealthy' || systemStatus === 'error'
          }"
        >
          {{ systemStatus }}
        </span>
      </div>

      <!-- Component Status -->
      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-sm text-secondary-600">Graph Initialized</span>
          <StatusIcon :status="health.graph_initialized" />
        </div>
        <div class="flex items-center justify-between">
          <span class="text-sm text-secondary-600">Vector Store</span>
          <StatusIcon :status="health.vectorstore_ready" />
        </div>
        <div class="flex items-center justify-between">
          <span class="text-sm text-secondary-600">Embedding Model</span>
          <StatusIcon :status="health.embedding_model_ready" />
        </div>
        <div class="flex items-center justify-between">
          <span class="text-sm text-secondary-600">LLM Service</span>
          <StatusIcon :status="health.llm_ready" />
        </div>
      </div>

      <!-- Last Updated -->
      <div class="pt-3 border-t border-secondary-200">
        <p class="text-xs text-secondary-500">
          Last updated: {{ formatTimestamp(health.timestamp) }}
        </p>
      </div>
    </div>

    <div v-else class="text-center py-8">
      <svg class="mx-auto h-12 w-12 text-secondary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
      <p class="mt-2 text-sm text-secondary-500">Click refresh to check system health</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useSmartBrainStore } from '@/stores/smartBrain'
import StatusIcon from './StatusIcon.vue'

const store = useSmartBrainStore()
const { health, isLoading, error, systemStatus, checkHealth } = store

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleString()
}

onMounted(() => {
  checkHealth()
})
</script>
