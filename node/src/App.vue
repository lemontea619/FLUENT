<template>
  <div class="min-h-screen bg-gray-100 p-8">
    <div class="max-w-4xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl font-bold text-gray-800">FLUENT Label Manager</h1>
        <p class="text-gray-600">Audio to Sensitivity Mapping</p>
      </header>

      <div class="bg-white rounded-lg shadow-md p-6 mb-8">
        <div class="flex overflow-x-auto gap-2 pb-4 mb-4 timeline-container">
          <button
            v-for="segment in segments" :key="segment" @click="selectSegment(segment)"
            class="px-4 py-2 rounded border whitespace-nowrap min-w-[120px] text-center flex items-center justify-center gap-2"
            :class="currentSegment === segment ? 'bg-blue-500 text-white border-blue-600' : 'bg-gray-50 border-gray-300'"
          >
            <span
              class="inline-block w-2 h-2 rounded-full flex-shrink-0"
              :class="{
                'bg-gray-400': segmentStatus(segment) === 'unlabeled',
                'bg-yellow-400': segmentStatus(segment) === 'pending',
                'bg-green-500': segmentStatus(segment) === 'trained',
              }"
            ></span>
            {{ segment }}
          </button>
        </div>
        <div class="flex gap-4 text-xs text-gray-500">
          <span><span class="inline-block w-2 h-2 rounded-full bg-gray-400 mr-1"></span>unlabeled</span>
          <span><span class="inline-block w-2 h-2 rounded-full bg-yellow-400 mr-1"></span>pending</span>
          <span><span class="inline-block w-2 h-2 rounded-full bg-green-500 mr-1"></span>trained</span>
        </div>

        <div v-if="currentSegment" class="mt-6 border-t pt-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-medium text-gray-800">Now Labeling: {{ currentSegment }}</h3>
            <audio controls :src="'/data/segments/' + currentSegment + '.wav'" class="h-10"></audio>
          </div>
          <form @submit.prevent="saveLabels" class="space-y-6">
            <div v-for="(field, index) in config" :key="field.id" class="p-4 bg-gray-50 rounded border">
              <label class="block text-sm font-medium text-gray-700 mb-2">{{ field.label }}</label>

              <div v-if="field.type === 'color-picker'">
                <input type="color" v-model="currentLabels[index]" class="h-10 w-full cursor-pointer rounded">
              </div>

              <div v-if="field.type === 'dropdown'">
                <select v-model="currentLabels[index]" class="w-full border-gray-300 rounded-md shadow-sm p-2 bg-white">
                  <option v-for="(opt, i) in field.options" :key="opt" :value="i">{{ opt }}</option>
                </select>
              </div>

              <div v-if="field.type === 'slider'">
                <input type="range" :min="field.min" :max="field.max" :step="field.step" v-model.number="currentLabels[index]" class="w-full">
              </div>

              <div v-if="field.type === 'checkboxes'" class="space-y-2">
                <div v-for="(opt, i) in field.options" :key="opt" class="flex items-center">
                  <input type="checkbox" v-model="currentLabels[index][i]" :id="field.id + '-' + i" class="h-4 w-4 text-blue-600 border-gray-300 rounded">
                  <label :for="field.id + '-' + i" class="ml-2 block text-sm text-gray-900">{{ opt }}</label>
                </div>
              </div>
            </div>

            <div class="flex gap-4 pt-4 items-center">
              <button type="submit" :disabled="isLoading" class="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 shadow disabled:opacity-50">Save Labels</button>
              <button type="button" @click="predict()" :disabled="isLoading" class="bg-indigo-600 text-white px-6 py-2 rounded hover:bg-indigo-700 shadow disabled:opacity-50">AI Suggestion</button>
              <span v-if="isLoading" class="text-blue-600 font-medium ml-4">Loading...</span>
            </div>
          </form>
        </div>
      </div>

      <div class="bg-white rounded-lg shadow-md p-6">
        <h2 class="text-lg font-medium text-gray-800 mb-4">Model Training</h2>
        <div class="flex items-center gap-4 mb-4">
          <label class="text-sm text-gray-700 whitespace-nowrap">Alpha: {{ alpha }}</label>
          <input type="range" min="0.001" max="0.1" step="0.001" v-model.number="alpha" class="w-48">
          <button @click="trainModel" :disabled="isTraining" class="bg-purple-600 text-white px-6 py-2 rounded hover:bg-purple-700 shadow disabled:opacity-50">
            {{ isTraining ? 'Training...' : 'Train Model' }}
          </button>
          <span v-if="pendingCount > 0 && !isTraining" class="text-yellow-600 text-sm">{{ pendingCount }} pending — retrain recommended</span>
          <span v-if="trainStatus === 'converged'" class="text-green-600 text-sm font-medium">Converged</span>
          <span v-else-if="trainStatus === 'error'" class="text-red-600 text-sm font-medium">Training failed</span>
        </div>
        <div v-if="trainLogs.length" class="font-mono text-xs bg-gray-50 rounded p-3 max-h-32 overflow-y-auto">
          <div v-for="log in trainLogs" :key="log.epoch">epoch {{ log.epoch }}: loss = {{ log.loss.toFixed(6) }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import { io } from 'socket.io-client'

const isLoading = ref(false)
const alpha = ref(0.01)
const isTraining = ref(false)
const trainLogs = ref([])
const trainStatus = ref('')
const config = ref([])
const segments = ref([])
const labelset = ref({ cols: 0, data: {} })
const trainedIds = ref([])
const currentSegment = ref(null)
const currentLabels = ref([])

const fetchConfig = async () => config.value = (await axios.get('/api/config')).data
const fetchSegments = async () => segments.value = (await axios.get('/api/segments')).data
const fetchLabels = async () => labelset.value = (await axios.get('/api/labels')).data
const fetchTrainMeta = async () => {
  const meta = (await axios.get('/api/train/meta')).data
  trainedIds.value = meta.trained_ids || []
}

const segmentStatus = (id) => {
  if (!labelset.value.data[id]) return 'unlabeled'
  if (trainedIds.value.includes(id)) return 'trained'
  return 'pending'
}

const pendingCount = computed(() =>
  segments.value.filter(id => segmentStatus(id) === 'pending').length
)

const decodeLabels = (flat, cfg) => {
  let idx = 0
  return cfg.map(field => {
    if (field.type === 'color-picker') {
      const dims = field.dims || 3
      const ch = Array.from({ length: dims }, () => flat[idx++])
      return '#' + (1 << 24 | Math.round(ch[0]*255) << 16 | Math.round(ch[1]*255) << 8 | Math.round(ch[2]*255)).toString(16).slice(1)
    } else if (field.type === 'dropdown') {
      return Math.round(flat[idx++] * (field.options.length - 1))
    } else if (field.type === 'checkboxes') {
      return Array.from({ length: field.options.length }, () => flat[idx++] >= 0.5)
    } else {
      return flat[idx++]
    }
  })
}

const encodeLabels = (uiValues, cfg) => {
  const flat = []
  cfg.forEach((field, i) => {
    const val = uiValues[i]
    if (field.type === 'color-picker') {
      const rgb = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(val)
      const dims = field.dims || 3
      ;[rgb[1], rgb[2], rgb[3]].slice(0, dims).forEach(h => flat.push(parseInt(h, 16) / 255))
    } else if (field.type === 'dropdown') {
      flat.push(val / (field.options.length - 1))
    } else if (field.type === 'checkboxes') {
      val.forEach(v => flat.push(v ? 1.0 : 0.0))
    } else {
      flat.push(val)
    }
  })
  return flat
}

const initLabels = () => {
  if (currentSegment.value && labelset.value.data[currentSegment.value]) {
    currentLabels.value = decodeLabels(labelset.value.data[currentSegment.value], config.value)
  } else {
    currentLabels.value = config.value.map(field => {
      if (field.type === 'color-picker') return '#000000';
      if (field.type === 'slider') return field.min || 0;
      if (field.type === 'checkboxes') return Array(field.options.length).fill(false);
      return null;
    });
  }
}

const hasLabels = (segment) => !!labelset.value.data[segment]
const selectSegment = async (segment) => {
  currentSegment.value = segment;
  initLabels();
  if (!hasLabels(segment)) {
    await predict(true);
  }
}

const saveLabels = async () => {
  try {
    isLoading.value = true
    const flatLabels = encodeLabels(currentLabels.value, config.value)
    await axios.post('/api/labels', { id: currentSegment.value, labels: flatLabels, cols: flatLabels.length })
    labelset.value.data[currentSegment.value] = flatLabels
  } finally {
    isLoading.value = false
  }
}

const predict = async (silent = false) => {
  try {
    if (!silent) isLoading.value = true
    const res = await axios.post('/api/predict', { id: currentSegment.value })
    if (res.data.prediction) {
      currentLabels.value = decodeLabels(res.data.prediction, config.value)
    }
  } catch (e) {
    if (!silent) console.error(e)
  } finally {
    if (!silent) isLoading.value = false
  }
}

onMounted(async () => {
  await fetchConfig(); await fetchSegments(); await fetchLabels(); await fetchTrainMeta()
  const socket = io()
  socket.on('train:progress', data => {
    trainLogs.value.push(data)
    if (data.status === 'converged') trainStatus.value = 'converged'
  })
  socket.on('train:done', async ({ success }) => {
    isTraining.value = false
    if (!success) trainStatus.value = 'error'
    else await fetchTrainMeta()
  })
})

const trainModel = async () => {
  trainLogs.value = []
  trainStatus.value = ''
  isTraining.value = true
  await axios.post('/api/train', { alpha: alpha.value })
}
</script>
