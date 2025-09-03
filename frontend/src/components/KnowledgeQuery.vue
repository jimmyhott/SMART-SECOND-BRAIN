<template>
  <div class="card">
    <h3 class="text-lg font-semibold text-secondary-900 mb-4">Knowledge Query</h3>
    
    <form @submit.prevent="handleSubmit" class="space-y-4">
      <div>
        <label class="block text-sm font-medium text-secondary-700 mb-2">
          Your Question *
        </label>
        <textarea
          v-model="form.query"
          rows="4"
          class="input-field"
          placeholder="Ask a question about your knowledge base..."
          required
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-secondary-700 mb-2">
          Thread ID (Optional)
        </label>
        <input
          v-model="form.thread_id"
          type="text"
          class="input-field"
          placeholder="Leave empty for new conversation or enter existing thread ID"
        />
        <p class="text-xs text-secondary-500 mt-1">
          Use thread ID to continue a previous conversation
        </p>
      </div>

      <div class="flex items-center justify-between">
        <button
          type="submit"
          :disabled="!canSubmit || isLoading"
          class="btn-primary"
        >
          <svg v-if="isLoading" class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          {{ isLoading ? 'Querying...' : 'Ask Question' }}
        </button>
        
        <button
          type="button"
          @click="resetForm"
          class="btn-secondary"
        >
          Reset
        </button>
      </div>
    </form>

    <div v-if="error" class="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
      <p class="text-red-700 text-sm">{{ error }}</p>
    </div>

    <div v-if="lastQuery" class="mt-6 space-y-4">
      <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 class="text-sm font-medium text-blue-800 mb-2">Query Result</h4>
        <div class="text-sm text-blue-700 space-y-2">
          <p><strong>Thread ID:</strong> {{ lastQuery.thread_id }}</p>
          <p><strong>Execution Time:</strong> {{ lastQuery.execution_time.toFixed(2) }}s</p>
          <p><strong>Timestamp:</strong> {{ formatTimestamp(lastQuery.timestamp) }}</p>
        </div>
      </div>

      <div v-if="lastQuery.result" class="p-4 bg-green-50 border border-green-200 rounded-lg">
        <h4 class="text-sm font-medium text-green-800 mb-2">AI Response</h4>
        <div class="text-sm text-green-700">
          <div v-if="lastQuery.result.generated_answer" class="mb-3">
            <strong>Answer:</strong>
            <p class="mt-1 p-3 bg-white rounded border">{{ lastQuery.result.generated_answer }}</p>
          </div>
          
          <div v-if="lastQuery.result.retrieved_docs && lastQuery.result.retrieved_docs.length > 0" class="mb-3">
            <strong>Retrieved Documents:</strong>
            <div class="mt-2 space-y-2">
              <div 
                v-for="(doc, index) in lastQuery.result.retrieved_docs" 
                :key="index"
                class="p-2 bg-white rounded border text-xs"
              >
                <p class="font-medium">{{ doc.content?.substring(0, 100) }}...</p>
                <p v-if="doc.metadata" class="text-gray-600 mt-1">
                  {{ JSON.stringify(doc.metadata) }}
                </p>
              </div>
            </div>
          </div>

          <div v-if="lastQuery.result.status" class="mb-3">
            <strong>Status:</strong> {{ lastQuery.result.status }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useSmartBrainStore } from '@/stores/smartBrain'
import type { QueryRequest } from '@/services/api'

const store = useSmartBrainStore()
const { isLoading, error, canQuery, queryKnowledge, currentThreadId, recentQueries } = store

const form = ref<QueryRequest>({
  query: '',
  thread_id: ''
})

const lastQuery = ref<any>(null)

const canSubmit = computed(() => {
  return form.value.query.trim().length > 0 && canQuery.value
})

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleString()
}

const handleSubmit = async () => {
  try {
    const response = await queryKnowledge(form.value)
    lastQuery.value = response
    // Don't reset form to allow follow-up questions
  } catch (err) {
    // Error is handled by the store
  }
}

const resetForm = () => {
  form.value = {
    query: '',
    thread_id: ''
  }
}

// Auto-fill thread ID if available
watch(currentThreadId, (newThreadId) => {
  if (newThreadId && !form.value.thread_id) {
    form.value.thread_id = newThreadId
  }
})

// Watch for new queries
watch(() => store.recentQueries, (newQueries) => {
  if (newQueries.length > 0) {
    lastQuery.value = newQueries[0]
  }
}, { immediate: true })
</script>
