<template>
  <div class="flex items-center">
    <svg 
      v-if="status" 
      class="w-4 h-4 text-green-500" 
      fill="currentColor" 
      viewBox="0 0 20 20"
    >
      <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
    </svg>
    <svg 
      v-else 
      class="w-4 h-4 text-red-500" 
      fill="currentColor" 
      viewBox="0 0 20 20"
    >
      <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
    </svg>
    <span 
      :class="{
        'ml-2 text-xs font-medium': true,
        'text-green-700': status,
        'text-red-700': !status
      }"
    >
      {{ status ? 'Ready' : 'Not Ready' }}
    </span>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  status: boolean
}>()
</script>
