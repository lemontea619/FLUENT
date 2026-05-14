// -----------------------------------------------------------
// server.js — Express API サーバー
//
// ブラウザ (public/index.html) と Python スクリプトの橋渡し役。
// 詳細は node/DEVELOPMENT.md を参照してください。
// -----------------------------------------------------------

// dotenv は他の import より先に実行する必要があるため、先頭に配置
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.join(__dirname, "../.env") });

import express from "express";
import fs from "fs/promises";
import { constants } from "fs";
import { spawn } from "child_process";
import cors from "cors";

// --- 環境変数（未設定の場合はデフォルト値を使用） ---
const DATA_DIR = path.resolve(__dirname, "..", process.env.DATA_DIR || "data");
const PYTHON_BIN = path.resolve(
  __dirname,
  "..",
  process.env.PYTHON_PATH || ".venv/bin/python",
);
const ROOT_DIR = path.resolve(__dirname, "..");
const PORT = process.env.PORT || 3000;
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "http://localhost:5173").split(",");

const app = express();

app.use(cors({ origin: ALLOWED_ORIGINS }));
app.use(express.json());

// data/ フォルダを静的配信（音声ファイルをブラウザから直接再生するため）
app.use("/data", express.static(DATA_DIR));

// public/ フォルダを静的配信（index.html を含む UI ファイル）
app.use(express.static(path.join(__dirname, "public")));

// -----------------------------------------------------------
// API エンドポイント
// -----------------------------------------------------------

// フォームの構成定義を返す。UI はこの内容をもとに動的にフォームを生成する
app.get("/api/config", async (req, res) => {
  try {
    res.json(
      JSON.parse(await fs.readFile(path.join(DATA_DIR, "config.json"), "utf8")),
    );
  } catch {
    res.status(500).json({ error: "Failed to read config" });
  }
});

// data/segments/ にある WAV ファイルの ID 一覧を返す（拡張子なし・ソート済み）
app.get("/api/segments", async (req, res) => {
  try {
    const files = await fs.readdir(path.join(DATA_DIR, "segments"));
    res.json(
      files
        .filter((f) => f.endsWith(".wav"))
        .map((f) => f.replace(".wav", ""))
        .sort(),
    );
  } catch {
    res.status(500).json({ error: "Failed to read segments" });
  }
});

// 保存済みラベルをすべて返す。まだ1件もなければ空の A-MAP 構造を返す
app.get("/api/labels", async (req, res) => {
  const labelsetPath = path.join(DATA_DIR, "labelset.json");
  try {
    await fs.access(labelsetPath, constants.F_OK);
    res.json(JSON.parse(await fs.readFile(labelsetPath, "utf8")));
  } catch {
    res.json({ cols: 0, data: {} });
  }
});

// ラベルを1件保存する。上書き前に data/backups/ へ自動バックアップを作成する
app.post("/api/labels", async (req, res) => {
  const { id, labels, cols } = req.body;
  const labelsetPath = path.join(DATA_DIR, "labelset.json");
  let labelset = { cols: cols || 0, data: {} };
  try {
    await fs.access(labelsetPath, constants.F_OK);
    const data = await fs.readFile(labelsetPath, "utf8");
    labelset = JSON.parse(data);
    // タイムスタンプ付きのファイル名でバックアップ
    const backupDir = path.join(DATA_DIR, "backups");
    await fs.mkdir(backupDir, { recursive: true });
    await fs.writeFile(
      path.join(
        backupDir,
        `labelset-${new Date().toISOString().replace(/[:.]/g, "-")}.json`,
      ),
      data,
    );
  } catch {}
  labelset.cols = cols || labelset.cols;
  labelset.data[id] = labels;
  try {
    await fs.writeFile(labelsetPath, JSON.stringify(labelset, null, 2));
    res.json({ success: true });
  } catch {
    res.status(500).json({ error: "Failed to save labels" });
  }
});

// -----------------------------------------------------------
// Python 呼び出しヘルパー
//
// exit code 0 → 成功
// exit code 2 → モデル未存在（「まだ学習していない」状態。エラーではない）
// それ以外   → reject（エラー扱い）
//
// ※ 学習（train.py）だけは進捗をリアルタイムでストリームする必要があるため
//   このヘルパーを使わず spawn を直接使っている（意図的な例外）
// -----------------------------------------------------------
const spawnPython = (script, args = []) =>
  new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [path.join(ROOT_DIR, script), ...args], {
      cwd: ROOT_DIR,
    });
    let out = "",
      err = "";
    proc.stdout.on("data", (d) => (out += d));
    proc.stderr.on("data", (d) => {
      err += d;
      console.error(`[${script}]`, d.toString().trim());
    });
    proc.on("close", (code) =>
      code === 0 || code === 2
        ? resolve({ out, code })
        : reject(new Error(err)),
    );
  });

// YouTube などから音声の一部をダウンロードし、データセットを更新する
app.post("/api/youtube", async (req, res) => {
  const { url, start, duration } = req.body;
  if (!url) return res.status(400).json({ error: "URL is required" });

  try {
    const dlArgs = ["--url", url, "--output", path.join(DATA_DIR, "segments")];
    if (start !== undefined) {
      dlArgs.push("--start", String(start));
    }
    if (duration !== undefined) {
      dlArgs.push("--duration", String(duration));
    }

    // 1. YouTube等からダウンロードして segments/ に保存
    await spawnPython("python/youtube_dl.py", dlArgs);

    // 2. 特徴量抽出を実行して dataset.json と scaler.pkl を更新
    await spawnPython("python/extractor.py", [
      "--input",
      path.join(DATA_DIR, "segments"),
      "--output",
      path.join(DATA_DIR, "dataset.json"),
    ]);

    res.json({ success: true });
  } catch (e) {
    res.status(500).json({ error: "Download or extraction failed", details: e.message });
  }
});

// 全セグメントを推論し、指定 ID の予測値を返す
// モデルが存在しない場合は 500 ではなく { success: false, message: "model_not_found" } を返す
app.post("/api/predict", async (req, res) => {
  const { id } = req.body;
  try {
    const args = [
      "--dataset", path.join(DATA_DIR, "dataset.json"),
      "--model", path.join(DATA_DIR, "model.pkl")
    ];
    if (id) args.push("--id", id);
    const { out, code } = await spawnPython("python/predict.py", args);
    if (code === 2)
      return res.json({ success: false, message: "model_not_found" });
    const predictions = JSON.parse(out);
    predictions.data?.[id]
      ? res.json({ success: true, prediction: predictions.data[id] })
      : res.json({ success: false, message: "No prediction found" });
  } catch (e) {
    res.status(500).json({ error: "Prediction failed", details: e.message });
  }
});

// 学習中のプロセスを保持する（同時に2つ走らせないための管理用）
let trainingProcess = null;

// 学習を開始する。レスポンスは SSE ストリームで進捗を配信する
// resume: true を渡すと既存モデルを読み込んで続きから学習する
app.post("/api/train", (req, res) => {
  if (trainingProcess)
    return res.status(409).json({ error: "Training already in progress" });

  const alpha = parseFloat(req.body.alpha) || 0.01;
  const resume = req.body.resume === true;
  const args = [
    "--alpha", String(alpha),
    "--dataset", path.join(DATA_DIR, "dataset.json"),
    "--labelset", path.join(DATA_DIR, "labelset.json"),
    "--model", path.join(DATA_DIR, "model.pkl")
  ];
  if (resume) args.push("--resume");

  const proc = spawn(
    PYTHON_BIN,
    [path.join(ROOT_DIR, "python/train.py"), ...args],
    { cwd: ROOT_DIR },
  );
  trainingProcess = proc;

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");

  // SSE 送信ヘルパー（クライアント切断後の write エラーを無視する）
  const send = (data) => {
    try {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    } catch {}
  };

  // train.py は各 epoch 後に JSON を1行 stdout へ出力する
  // 例: { "epoch": 5, "loss": 0.0123 } / { "status": "converged", ... }
  proc.stdout.on("data", (data) => {
    data
      .toString()
      .trim()
      .split("\n")
      .filter(Boolean)
      .forEach((line) => {
        try {
          JSON.parse(line);
          res.write(`data: ${line}\n\n`);
        } catch {}
      });
  });

  proc.stderr.on("data", (data) =>
    console.error("[train.py]", data.toString().trim()),
  );

  proc.on("close", (code) => {
    trainingProcess = null;
    send({ status: "done", success: code === 0 });
    res.end();
  });
});

// 学習を中断する。SIGTERM を受け取った train.py がその時点のモデルを保存して終了する
app.post("/api/train/stop", (req, res) => {
  if (!trainingProcess)
    return res.status(409).json({ error: "No training in progress" });
  trainingProcess.kill("SIGTERM");
  res.json({ success: true });
});

// モデルと学習履歴を削除する（リセット）
app.delete("/api/model", async (req, res) => {
  const modelPath = path.join(DATA_DIR, "model.pkl");
  try {
    await fs.rm(modelPath, { force: true });
    await fs.rm(path.join(DATA_DIR, "train_meta.json"), { force: true });
    res.json({ success: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// 最終学習のメタデータ（日時・対象セグメント）を返す
// UI はこの trained_ids でセグメントのステータスドット（緑/黄/グレー）を判定する
app.get("/api/train/meta", async (req, res) => {
  try {
    const meta = JSON.parse(
      await fs.readFile(path.join(DATA_DIR, "train_meta.json"), "utf8"),
    );
    res.json(meta);
  } catch {
    // ファイルが存在しない = まだ一度も学習していない
    res.json({ trained_at: null, trained_ids: [] });
  }
});

app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
