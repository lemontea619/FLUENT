import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import { constants } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.join(__dirname, '../data');

const app = express();
const allowedOrigins = (process.env.ALLOWED_ORIGINS || 'http://localhost:5173').split(',');
app.use(cors({
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  }
}));
app.use(express.json());

app.use('/data', express.static(DATA_DIR));

app.get('/api/config', async (req, res) => {
  try {
    const data = await fs.readFile(path.join(DATA_DIR, 'config.json'), 'utf8');
    res.json(JSON.parse(data));
  } catch (err) {
    res.status(500).json({ error: 'Failed to read config' });
  }
});

app.get('/api/segments', async (req, res) => {
  try {
    const files = await fs.readdir(path.join(DATA_DIR, 'segments'));
    const segments = files.filter(f => f.endsWith('.wav')).map(f => f.replace('.wav', ''));
    res.json(segments.sort());
  } catch (err) {
    res.status(500).json({ error: 'Failed to read segments' });
  }
});

app.get('/api/labels', async (req, res) => {
  const labelsetPath = path.join(DATA_DIR, 'labelset.json');
  try {
    await fs.access(labelsetPath, constants.F_OK);
    const data = await fs.readFile(labelsetPath, 'utf8');
    res.json(JSON.parse(data));
  } catch (err) {
    res.json({ cols: 0, data: {} });
  }
});

app.post('/api/labels', async (req, res) => {
  const { id, labels, cols } = req.body;
  const labelsetPath = path.join(DATA_DIR, 'labelset.json');
  let labelset = { cols: cols || 0, data: {} };
  try {
    await fs.access(labelsetPath, constants.F_OK);
    const data = await fs.readFile(labelsetPath, 'utf8');
    labelset = JSON.parse(data);
  } catch (err) {
    // file does not exist, use default
  }
  labelset.cols = cols || labelset.cols;
  labelset.data[id] = labels;
  try {
    await fs.writeFile(labelsetPath, JSON.stringify(labelset, null, 2));
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: 'Failed to save labels' });
  }
});

app.post('/api/predict', (req, res) => {
  const { id } = req.body;
  const pythonPath = path.join(__dirname, '../.venv/bin/python');
  const scriptPath = path.join(__dirname, '../python/predict.py');

  const process = spawn(pythonPath, [scriptPath], { cwd: path.join(__dirname, '..') });
  let output = '';
  let stderrOutput = '';

  process.stdout.on('data', (data) => output += data.toString());
  process.stderr.on('data', (data) => stderrOutput += data.toString());

  process.on('close', (code) => {
    if (code !== 0) return res.status(500).json({ error: "Prediction failed", details: stderrOutput || output });
    try {
      const predictions = JSON.parse(output);
      if (predictions.data && predictions.data[id]) {
        res.json({ success: true, prediction: predictions.data[id] });
      } else {
        res.json({ success: false, message: "No prediction found" });
      }
    } catch (e) {
      res.status(500).json({ error: "Invalid JSON from stdout", details: stderrOutput || output });
    }
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
