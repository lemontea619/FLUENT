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
            class="px-4 py-2 rounded border whitespace-nowrap min-w-[120px] text-center"
            :class="[currentSegment === segment ? 'bg-blue-500 text-white border-blue-600' : 'bg-gray-50 border-gray-300', hasLabels(segment) ? 'border-b-4 border-b-green-500' : '']"
          >
            {{ segment }}
          </button>
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
            </div>

            <div class="flex gap-4 pt-4">
              <button type="submit" class="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 shadow">Save Labels</button>
              <button type="button" @click="predict" class="bg-indigo-600 text-white px-6 py-2 rounded hover:bg-indigo-700 shadow">AI Suggestion</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const config = ref([])
const segments = ref([])
const labelset = ref({ cols: 0, data: {} })
const currentSegment = ref(null)
const currentLabels = ref([])

const fetchConfig = async () => config.value = (await axios.get('/api/config')).data
const fetchSegments = async () => segments.value = (await axios.get('/api/segments')).data
const fetchLabels = async () => labelset.value = (await axios.get('/api/labels')).data

const initLabels = () => {
  if (currentSegment.value && labelset.value.data[currentSegment.value]) {
    let flatIdx = 0;
    const labelsData = labelset.value.data[currentSegment.value];
    currentLabels.value = config.value.map(field => {
      if (field.type === 'color-picker') {
        const r = labelsData[flatIdx++], g = labelsData[flatIdx++], b = labelsData[flatIdx++];
        return "#" + (1 << 24 | Math.round(r*255) << 16 | Math.round(g*255) << 8 | Math.round(b*255)).toString(16).slice(1);
      } else if (field.type === 'dropdown') {
        return Math.round(labelsData[flatIdx++] * (field.options.length - 1));
      } else {
        return labelsData[flatIdx++];
      }
    });
  } else {
    currentLabels.value = config.value.map(field => field.type === 'color-picker' ? '#000000' : field.type === 'slider' ? field.min || 0 : null)
  }
}

const selectSegment = (segment) => { currentSegment.value = segment; initLabels() }
const hasLabels = (segment) => !!labelset.value.data[segment]

const saveLabels = async () => {
  let flatLabels = []
  config.value.forEach((field, i) => {
    let val = currentLabels.value[i]
    if (field.type === 'color-picker') {
      const rgb = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(val)
      flatLabels.push(parseInt(rgb[1], 16)/255, parseInt(rgb[2], 16)/255, parseInt(rgb[3], 16)/255)
    } else if (field.type === 'dropdown') {
      flatLabels.push(val / (field.options.length - 1))
    } else {
      flatLabels.push(val)
    }
  })
  await axios.post('/api/labels', { id: currentSegment.value, labels: flatLabels, cols: flatLabels.length })
  labelset.value.data[currentSegment.value] = flatLabels
}

const predict = async () => {
  try {
    const res = await axios.post('/api/predict', { id: currentSegment.value })
    if (res.data.prediction) {
       let flatIdx = 0
       config.value.forEach((field, i) => {
         if (field.type === 'color-picker') {
           const r = res.data.prediction[flatIdx++], g = res.data.prediction[flatIdx++], b = res.data.prediction[flatIdx++]
           currentLabels.value[i] = "#" + (1 << 24 | Math.round(r*255) << 16 | Math.round(g*255) << 8 | Math.round(b*255)).toString(16).slice(1)
         } else if (field.type === 'dropdown') {
           currentLabels.value[i] = Math.round(res.data.prediction[flatIdx++] * (field.options.length - 1))
         } else {
           currentLabels.value[i] = res.data.prediction[flatIdx++]
         }
       })
    }
  } catch (e) { console.error(e) }
}

onMounted(async () => {
  await fetchConfig(); await fetchSegments(); await fetchLabels()
})
</script>
