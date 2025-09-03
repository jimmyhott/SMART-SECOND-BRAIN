<script setup lang="ts">
import { computed } from 'vue'
import { useSmartBrainStore } from '@/stores/smartBrain'
import SystemHealth from '@/components/SystemHealth.vue'
import DocumentIngestion from '@/components/DocumentIngestion.vue'
import KnowledgeQuery from '@/components/KnowledgeQuery.vue'

const store = useSmartBrainStore()
const { recentIngestions, recentQueries } = store

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleString()
}
</script>

<template>
  <div class="min-h-screen bg-secondary-50">
    <!-- Header -->
    <header class="bg-white shadow-sm border-b border-secondary-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center py-6">
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <svg class="h-8 w-8 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div class="ml-3">
              <h1 class="text-2xl font-bold text-secondary-900">Smart Second Brain</h1>
              <p class="text-sm text-secondary-600">AI-Powered Knowledge Management</p>
            </div>
          </div>
          
          <div class="flex items-center space-x-4">
            <a 
              href="http://localhost:8000/docs" 
              target="_blank" 
              class="btn-secondary text-sm"
            >
              API Docs
            </a>
            <a 
              href="https://github.com/jimmyhott/SMART-SECOND-BRAIN" 
              target="_blank" 
              class="btn-secondary text-sm"
            >
              GitHub
            </a>
          </div>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Welcome Section -->
      <div class="text-center mb-12">
        <h2 class="text-3xl font-bold text-secondary-900 mb-4">
          Welcome to Your Smart Second Brain
        </h2>
        <p class="text-lg text-secondary-600 max-w-3xl mx-auto">
          Upload documents, ask questions, and let AI help you build and navigate your personal knowledge base. 
          Powered by LangGraph and advanced AI workflows.
        </p>
      </div>

      <!-- System Status -->
      <div class="mb-8">
        <SystemHealth />
      </div>

      <!-- Main Features Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <DocumentIngestion />
        <KnowledgeQuery />
      </div>

      <!-- Recent Activity -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <!-- Recent Ingestions -->
        <div class="card">
          <h3 class="text-lg font-semibold text-secondary-900 mb-4">Recent Document Ingestions</h3>
          <div v-if="recentIngestions.length === 0" class="text-center py-8 text-secondary-500">
            <svg class="mx-auto h-12 w-12 text-secondary-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p>No documents ingested yet</p>
          </div>
          <div v-else class="space-y-3">
            <div 
              v-for="ingestion in recentIngestions.slice(0, 5)" 
              :key="ingestion.thread_id"
              class="p-3 bg-secondary-50 rounded-lg border border-secondary-200"
            >
              <div class="flex justify-between items-start">
                <div class="flex-1">
                  <p class="text-sm font-medium text-secondary-900">
                    {{ ingestion.result?.status || 'completed' }}
                  </p>
                  <p class="text-xs text-secondary-600 mt-1">
                    {{ formatTimestamp(ingestion.timestamp) }}
                  </p>
                </div>
                <span class="text-xs text-secondary-500">
                  {{ ingestion.execution_time.toFixed(2) }}s
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Recent Queries -->
        <div class="card">
          <h3 class="text-lg font-semibold text-secondary-900 mb-4">Recent Knowledge Queries</h3>
          <div v-if="recentQueries.length === 0" class="text-center py-8 text-secondary-500">
            <svg class="mx-auto h-12 w-12 text-secondary-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p>No queries made yet</p>
          </div>
          <div v-else class="space-y-3">
            <div 
              v-for="query in recentQueries.slice(0, 5)" 
              :key="query.thread_id"
              class="p-3 bg-secondary-50 rounded-lg border border-secondary-200"
            >
              <div class="flex justify-between items-start">
                <div class="flex-1">
                  <p class="text-sm font-medium text-secondary-900">
                    {{ query.result?.generated_answer?.substring(0, 50) || 'Query completed' }}...
                  </p>
                  <p class="text-xs text-secondary-600 mt-1">
                    {{ formatTimestamp(query.timestamp) }}
                  </p>
                </div>
                <span class="text-xs text-secondary-500">
                  {{ query.execution_time.toFixed(2) }}s
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Quick Start Guide -->
      <div class="card bg-gradient-to-r from-primary-50 to-blue-50 border-primary-200">
        <h3 class="text-lg font-semibold text-primary-900 mb-4">🚀 Quick Start Guide</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
          <div class="space-y-2">
            <div class="flex items-center">
              <span class="flex-shrink-0 w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-bold">1</span>
              <span class="ml-3 font-medium text-primary-800">Check System Health</span>
            </div>
            <p class="text-primary-700 ml-9">Ensure your API is running and all services are healthy</p>
          </div>
          
          <div class="space-y-2">
            <div class="flex items-center">
              <span class="flex-shrink-0 w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-bold">2</span>
              <span class="ml-3 font-medium text-primary-800">Ingest Documents</span>
            </div>
            <p class="text-primary-700 ml-9">Upload or paste content to build your knowledge base</p>
          </div>
          
          <div class="space-y-2">
            <div class="flex items-center">
              <span class="flex-shrink-0 w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-bold">3</span>
              <span class="ml-3 font-medium text-primary-800">Ask Questions</span>
            </div>
            <p class="text-primary-700 ml-9">Query your knowledge base with natural language</p>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>
