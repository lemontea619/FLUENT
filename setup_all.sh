#!/bin/bash
set -e

mkdir -p python node data/raw_audio data/segments

cat << 'REQ' > python/requirements.txt
librosa
scikit-learn
numpy
soundfile
scipy
REQ

cat << 'DUMMY' > python/generate_dummy_audio.py
import numpy as np
import soundfile as sf
import os

os.makedirs('data/raw_audio', exist_ok=True)
sr = 22050

t = np.linspace(0, 5, sr * 5)
low_freq = np.sin(2 * np.pi * 100 * t)
sf.write('data/raw_audio/low_freq.wav', low_freq, sr)

high_freq = np.sin(2 * np.pi * 8000 * t)
sf.write('data/raw_audio/high_freq.wav', high_freq, sr)

noise = np.random.randn(len(t))
sf.write('data/raw_audio/noise.wav', noise, sr)
DUMMY

cat << 'SEG' > python/segmenter.py
import os
import glob
import librosa
import soundfile as sf
import numpy as np

def segment_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(input_dir, '*.wav'))
    segment_id = 1

    for file_path in audio_files:
        y, sr = librosa.load(file_path, sr=None)
        hop_length = 512
        if len(y) < sr * 1:
             out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
             sf.write(out_path, y, sr)
             segment_id += 1
             continue
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
        k = max(2, len(y) // (sr * 2))
        try:
            boundaries = librosa.segment.agglomerative(mfcc, k)
            boundary_samples = librosa.frames_to_samples(boundaries, hop_length=hop_length)
            boundary_samples = np.unique(np.concatenate([[0], boundary_samples, [len(y)]]))
            for i in range(len(boundary_samples) - 1):
                start = boundary_samples[i]
                end = boundary_samples[i+1]
                if end - start > 0:
                    segment_y = y[start:end]
                    out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
                    sf.write(out_path, segment_y, sr)
                    segment_id += 1
        except Exception:
            out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
            sf.write(out_path, y, sr)
            segment_id += 1

if __name__ == "__main__":
    segment_audio('data/raw_audio', 'data/segments')
SEG

cat << 'EXT' > python/extractor.py
import os
import glob
import librosa
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)[1:]
    S, _ = librosa.magphase(librosa.stft(y))
    centroid = np.mean(librosa.feature.spectral_centroid(S=S))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_tuple = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = tempo_tuple[0][0] if isinstance(tempo_tuple[0], np.ndarray) else tempo_tuple[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key = np.argmax(chroma_mean)
    mode = 1.0 if chroma_mean[(key + 4) % 12] > chroma_mean[(key + 3) % 12] else 0.0
    features = np.concatenate([
        mfcc_mean,
        [centroid, flatness, rolloff, bandwidth],
        [tempo, key, mode],
        chroma_mean[:7]
    ])
    return features

def create_dataset(input_dir, output_json):
    audio_files = sorted(glob.glob(os.path.join(input_dir, '*.wav')))
    feature_dict = {}
    all_features = []
    file_ids = []
    for file_path in audio_files:
        file_id = os.path.basename(file_path).split('.')[0]
        features = extract_features(file_path)
        all_features.append(features)
        file_ids.append(file_id)
    all_features = np.array(all_features)
    if len(all_features) > 0:
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)
        for i, file_id in enumerate(file_ids):
            feature_dict[file_id] = all_features_scaled[i].tolist()
    dataset = {
        "cols": all_features.shape[1] if len(all_features) > 0 else 26,
        "data": feature_dict
    }
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    create_dataset('data/segments', 'data/dataset.json')
EXT

cat << 'TRAIN' > python/train.py
import json
import os
import sys
import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle

def train_model(dataset_path, labelset_path, model_path):
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)

    if not os.path.exists(labelset_path):
        print(f"Error: {labelset_path} not found. Start labeling first.")
        sys.exit(1)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    with open(labelset_path, 'r') as f:
        labelset = json.load(f)

    X = []
    y = []

    labeled_ids = list(labelset['data'].keys())

    for file_id in labeled_ids:
        if file_id in dataset['data']:
            X.append(dataset['data'][file_id])
            y.append(labelset['data'][file_id])

    if len(X) == 0:
        print("Error: No intersecting data between dataset and labelset.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)

    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        alpha=0.01,
        max_iter=1000,
        random_state=42,
        early_stopping=False
    )

    mlp.set_params(warm_start=True, max_iter=1)

    prev_loss = float('inf')
    patience = 10
    wait = 0

    for i in range(1000):
        mlp.fit(X, y)
        loss = mlp.loss_

        if abs(prev_loss - loss) < 1e-5:
            wait += 1
            if wait >= patience:
                print(f"Leveled out at epoch {i} with loss {loss:.6f}")
                break
        else:
            wait = 0

        prev_loss = loss

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(mlp, f)

if __name__ == "__main__":
    train_model('data/dataset.json', 'data/labelset.json', 'data/model.pkl')
TRAIN

cat << 'PRED' > python/predict.py
import json
import os
import sys
import numpy as np
import pickle

def predict_labels(dataset_path, model_path, output_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        mlp = pickle.load(f)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    predictions = {}

    for file_id, features in dataset['data'].items():
        X = np.array([features])
        y_pred = mlp.predict(X)[0]
        y_pred = np.clip(y_pred, 0.0, 1.0)
        predictions[file_id] = y_pred.tolist()

    output_data = {
        "cols": len(list(predictions.values())[0]) if len(predictions) > 0 else 0,
        "data": predictions
    }

    print(json.dumps(output_data))

if __name__ == "__main__":
    predict_labels('data/dataset.json', 'data/model.pkl', 'data/predictions.json')
PRED

cat << 'CONF' > data/config.json
[
  {
    "id": "color",
    "type": "color-picker",
    "label": "色彩"
  },
  {
    "id": "mood",
    "type": "dropdown",
    "options": ["Bright", "Dark", "Neutral"],
    "label": "印象"
  },
  {
    "id": "intensity",
    "type": "slider",
    "min": 0,
    "max": 1,
    "step": 0.1,
    "label": "強度"
  }
]
CONF

python3 -m venv venv
source venv/bin/activate
pip install -r python/requirements.txt
python python/generate_dummy_audio.py
python python/segmenter.py
python python/extractor.py

cat << 'LBL' > data/labelset.json
{
  "cols": 5,
  "data": {
    "example-001": [
      1,
      0,
      0,
      0.5,
      0.8
    ],
    "example-002": [
      0,
      1,
      0,
      1.0,
      0.2
    ]
  }
}
LBL

python python/train.py

npm create vite@latest node -- --template vue
cd node
npm install
npm install express cors multer concurrently axios
npm install -D tailwindcss@3 postcss autoprefixer
npx tailwindcss init -p

cat << 'TWC' > tailwind.config.js
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
TWC

cat << 'STY' > src/style.css
@tailwind base;
@tailwind components;
@tailwind utilities;
STY

cat << 'VITE' > vite.config.js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:3000',
      '/data': 'http://localhost:3000',
    }
  }
})
VITE

cat << 'SERVER' > server.js
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.join(__dirname, '../data');

const app = express();
app.use(cors());
app.use(express.json());

app.use('/data', express.static(DATA_DIR));

app.get('/api/config', (req, res) => {
  const config = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'config.json'), 'utf8'));
  res.json(config);
});

app.get('/api/segments', (req, res) => {
  const files = fs.readdirSync(path.join(DATA_DIR, 'segments'))
                  .filter(f => f.endsWith('.wav'))
                  .map(f => f.replace('.wav', ''));
  res.json(files.sort());
});

app.get('/api/labels', (req, res) => {
  const labelsetPath = path.join(DATA_DIR, 'labelset.json');
  if (!fs.existsSync(labelsetPath)) return res.json({ cols: 0, data: {} });
  res.json(JSON.parse(fs.readFileSync(labelsetPath, 'utf8')));
});

app.post('/api/labels', (req, res) => {
  const { id, labels, cols } = req.body;
  const labelsetPath = path.join(DATA_DIR, 'labelset.json');
  let labelset = { cols: cols || 0, data: {} };
  if (fs.existsSync(labelsetPath)) labelset = JSON.parse(fs.readFileSync(labelsetPath, 'utf8'));
  labelset.cols = cols || labelset.cols;
  labelset.data[id] = labels;
  fs.writeFileSync(labelsetPath, JSON.stringify(labelset, null, 2));
  res.json({ success: true });
});

app.post('/api/predict', (req, res) => {
  const { id } = req.body;
  const pythonPath = path.join(__dirname, '../venv/bin/python');
  const scriptPath = path.join(__dirname, '../python/predict.py');
  const process = spawn(pythonPath, [scriptPath]);
  let output = '';
  process.stdout.on('data', (data) => output += data.toString());
  process.on('close', (code) => {
    if (code !== 0) return res.status(500).json({ error: "Prediction failed", details: output });
    try {
      const predictions = JSON.parse(output);
      if (predictions.data && predictions.data[id]) res.json({ success: true, prediction: predictions.data[id] });
      else res.json({ success: false, message: "No prediction found" });
    } catch (e) {
      res.status(500).json({ error: "Invalid JSON" });
    }
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
SERVER

cat << 'APP' > src/App.vue
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
          <h3 class="text-lg font-medium text-gray-800 mb-4">Now Labeling: {{ currentSegment }}</h3>
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
    currentLabels.value = [...labelset.value.data[currentSegment.value]]
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
APP

# update package.json
sed -i 's/"dev": "vite"/"dev": "concurrently \\"vite\\" \\"node server.js\\""/' package.json

# build node
npm run build
