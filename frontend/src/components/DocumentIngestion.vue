<template>
  <div class="card">
    <h3 class="text-lg font-semibold text-secondary-900 mb-4">Document Ingestion</h3>
    
    <form @submit.prevent="handleSubmit" class="space-y-4">
      <div>
        <label class="block text-sm font-medium text-secondary-700 mb-2">
          Document Content *
        </label>
        <textarea
          v-model="form.document"
          rows="6"
          class="input-field"
          placeholder="Paste or type your document content here..."
          required
        />
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium text-secondary-700 mb-2">
            Source
          </label>
          <input
            v-model="form.source"
            type="text"
            class="input-field"
            placeholder="e.g., webpage, document, email"
          />
        </div>
        
        <div>
          <label class="block text-sm font-medium text-secondary-700 mb-2">
            Categories
          </label>
          <input
            v-model="categoriesInput"
            type="text"
            class="input-field"
            placeholder="e.g., ai, research, technology"
          />
          <p class="text-xs text-secondary-500 mt-1">
            Separate multiple categories with commas
          </p>
        </div>
      </div>

      <div>
        <label class="block text-sm font-medium text-secondary-700 mb-2">
          Additional Metadata (JSON)
        </label>
        <textarea
          v-model="metadataInput"
          rows="3"
          class="input-field font-mono text-sm"
          placeholder='{"author": "John Doe", "date": "2024-01-01"}'
        />
        <p class="text-xs text-secondary-500 mt-1">
          Optional: Add custom metadata in JSON format
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
          {{ isLoading ? 'Ingesting...' : 'Ingest Document' }}
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

    <div v-if="lastIngestion" class="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
      <h4 class="text-sm font-medium text-green-800 mb-2">Last Ingestion Result</h4>
      <div class="text-sm text-green-700 space-y-1">
        <p><strong>Thread ID:</strong> {{ lastIngestion.thread_id }}</p>
        <p><strong>Status:</strong> {{ lastIngestion.result?.status || 'completed' }}</p>
        <p><strong>Execution Time:</strong> {{ lastIngestion.execution_time.toFixed(2) }}s</p>
        <p><strong>Timestamp:</strong> {{ formatTimestamp(lastIngestion.timestamp) }}</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useSmartBrainStore } from '@/stores/smartBrain'
import type { IngestRequest } from '@/services/api'

const store = useSmartBrainStore()
const { isLoading, error, canIngest, ingestDocument, recentIngestions } = store

const form = ref<IngestRequest>({
  document: '',
  source: '',
  categories: [],
  metadata: {}
})

const categoriesInput = ref('')
const metadataInput = ref('')
const lastIngestion = ref<any>(null)

const canSubmit = computed(() => {
  return form.value.document.trim().length > 0 && canIngest.value
})

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleString()
}

const handleSubmit = async () => {
  try {
    // Parse categories
    if (categoriesInput.value.trim()) {
      form.value.categories = categoriesInput.value
        .split(',')
        .map(cat => cat.trim())
        .filter(cat => cat.length > 0)
    }

    // Parse metadata
    if (metadataInput.value.trim()) {
      try {
        form.value.metadata = JSON.parse(metadataInput.value)
      } catch (e) {
        throw new Error('Invalid JSON in metadata field')
      }
    }

    const response = await ingestDocument(form.value)
    lastIngestion.value = response
    resetForm()
  } catch (err) {
    // Error is handled by the store
  }
}

const resetForm = () => {
  form.value = {
    document: '',
    source: '',
    categories: [],
    metadata: {}
  }
  categoriesInput.value = ''
  metadataInput.value = ''
}

// Watch for new ingestions
watch(() => store.recentIngestions, (newIngestions) => {
  if (newIngestions.length > 0) {
    lastIngestion.value = newIngestions[0]
  }
}, { immediate: true })
</script>
