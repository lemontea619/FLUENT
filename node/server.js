import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import fs from 'fs';
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
httpServer.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
