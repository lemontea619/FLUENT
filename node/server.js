import express from 'express';
import cors from 'cors';
import fs from 'fs';
import { readFile, writeFile, readdir } from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.join(__dirname, '../data');

const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : 'http://localhost:5173';

const app = express();
app.use(cors({ origin: ALLOWED_ORIGINS }));
app.use(express.json());

app.use('/data', express.static(DATA_DIR));

const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: { origin: ALLOWED_ORIGINS }
});

app.get('/api/config', async (req, res) => {
  try {
    const config = JSON.parse(await readFile(path.join(DATA_DIR, 'config.json'), 'utf8'));
    res.json(config);
  } catch (err) {
    res.status(500).json({ error: "Failed to read config" });
  }
});

app.get('/api/segments', async (req, res) => {
  try {
    const files = (await readdir(path.join(DATA_DIR, 'segments')))
                    .filter(f => f.endsWith('.wav'))
                    .map(f => f.replace('.wav', ''));
    res.json(files.sort());
  } catch (err) {
    res.status(500).json({ error: "Failed to list segments" });
  }
});

app.get('/api/labels', async (req, res) => {
  try {
    const labelsetPath = path.join(DATA_DIR, 'labelset.json');
    const content = await readFile(labelsetPath, 'utf8');
    res.json(JSON.parse(content));
  } catch (err) {
    if (err.code === 'ENOENT') return res.json({ cols: 0, data: {} });
    res.status(500).json({ error: "Failed to read labels" });
  }
});

app.post('/api/labels', async (req, res) => {
  try {
    const { id, labels, cols } = req.body;
    const labelsetPath = path.join(DATA_DIR, 'labelset.json');
    let labelset = { cols: cols || 0, data: {} };

    try {
      const content = await readFile(labelsetPath, 'utf8');
      labelset = JSON.parse(content);
    } catch (err) {
      if (err.code !== 'ENOENT') throw err;
    }

    labelset.cols = cols || labelset.cols;
    labelset.data[id] = labels;
    await writeFile(labelsetPath, JSON.stringify(labelset, null, 2));
    res.json({ success: true });
  } catch (err) {
    console.error('Error saving labels:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post('/api/predict', (req, res) => {
  const { id } = req.body;
  const pythonPath = path.join(__dirname, '../.venv/bin/python');
  const scriptPath = path.join(__dirname, '../python/predict.py');
  const process = spawn(pythonPath, [scriptPath], { cwd: path.join(__dirname, '..') });
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
