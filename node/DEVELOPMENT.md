# Module C 開発者ガイド

`node/` ディレクトリを担当する開発者向けの入門資料です。

---

## Module C の仕事

FLUENT は Python（音声解析・学習）と Node.js（UI・API）の2層構造です。
Module C は データにラベルをつける（正解データを作る）UIを提供します。

- 音声セグメントを試聴し、正解データラベルを設定・保存する
- 学習の開始・中断・リセットを操作し、進捗をリアルタイムで確認する
- 未ラベルのセグメントに対して AI の予測結果をプレビューする

```
ブラウザ (public/index.html — vanilla JS)
    ↕ HTTP / SSE
Express (server.js)
    ↕ child_process (stdout/stderr)
Python scripts (train.py, predict.py ...)
    ↕ ファイル読み書き
data/*.json / data/segments/*.wav
```

---

## まず動かす

```bash
# リポジトリルートで実行
cp .env.example .env      # 初回のみ
npm --prefix node install
npm --prefix node run dev
```

ブラウザで `http://localhost:3000` を開くと UI が表示されます。

---

## ファイル構成

```
node/
├── server.js          ← Express API。Python との橋渡し役
└── public/
    ├── favicon.svg
    ├── icons.svg
    └── index.html     ← UI のすべて（vanilla JS）
```

---

## 環境変数（`../.env`）

```
PORT=3000
DATA_DIR=data
PYTHON_PATH=.venv/bin/python
```

すべてデフォルト値があるので基本変更不要です。

---

## API 一覧

| メソッド | パス | 何をするか |
|---|---|---|
| GET | `/api/config` | `config.json` を返す（フォームの定義） |
| GET | `/api/segments` | セグメント ID の一覧を返す |
| GET | `/api/labels` | `labelset.json` を返す |
| POST | `/api/labels` | ラベルを1件保存（自動バックアップあり） |
| POST | `/api/predict` | 指定セグメントの AI 予測値を返す |
| POST | `/api/train` | 学習開始（SSE ストリームで進捗配信）。`resume: true` で継続学習 |
| POST | `/api/train/stop` | 学習を中断（途中のモデルを保存して終了） |
| DELETE | `/api/model` | モデルと学習履歴を削除（リセット） |
| GET | `/api/train/meta` | 最終学習の日時と対象セグメント一覧を返す |

### Python 呼び出し

`spawnPython` ヘルパーを使ってください。

```js
const { out, code } = await spawnPython("python/predict.py", ["--arg", "value"])
// exit code 0 → 成功
// exit code 2 → モデル未存在（エラーではない）
// それ以外   → reject
```

学習（`train.py`）のみ進捗を SSE でストリームするため `spawn` を直接使用しています。これは意図的な例外です。

### SSE（学習進捗）

`POST /api/train` はレスポンスを `text/event-stream` として返します。
各 `data:` 行は JSON です。

| 内容 | 例 |
|---|---|
| 各 epoch | `{ "epoch": 5, "loss": 0.0123 }` |
| 収束通知 | `{ "status": "converged", "epoch": 42, "loss": 0.0001 }` |
| 中断保存 | `{ "status": "interrupted", "saved": true }` |
| 終了 | `{ "status": "done", "success": true }` |

---

## フロントエンド（public/index.html）

### 画面の構成

- 上部：セグメント選択ボタン（ステータスドット付き）+ pending 再学習推奨メッセージ
- 中部：試聴プレイヤー + ラベル入力フォーム + Save / AI Suggestion ボタン
- 下部：学習パネル（Alpha スライダー・Train / Stop ボタン・ログ）

### 主要な変数

```js
let config          // config.json の内容
let segments        // セグメント ID の配列
let labelset        // 保存済みラベルの全データ
let trainedIds      // 最後の学習に含まれたセグメントの ID 一覧
let currentSegment  // 今選択中のセグメント
```

### セグメントのステータス

```js
function segmentStatus(id) {
  if (!labelset.data[id]) return 'unlabeled'              // グレー
  if (trainedIds.includes(id)) return 'trained'           // 緑
  return 'pending'                                         // 黄
}
```

---

## ラベルのシリアライズ（最重要）

`labelset.json` は Python（Module D）と共有します。値はすべて 0.0〜1.0 のフラット配列です。
UI の値 ↔ フラット配列 の変換を `encodeLabels` / `decodeLabels` が担います。

```
UI の値                    保存される値
─────────────────────────────────────────────────
"#ff8800"              →  [1.0, 0.533, 0.0]
dropdown: index 1      →  [1/(N-1)]      (index / (選択肢数 - 1))
slider: 0.7            →  [0.7]
checkboxes: [✓,✗,✓]   →  [1.0, 0.0, 1.0]
```

### 新しいフィールドタイプを追加するとき

`encodeLabels` と `decodeLabels` は必ずペアで更新してください。

#### 1. `data/config.json` にフィールド定義を追加

```json
{ "id": "YOUR_ID", "type": "YOUR_TYPE", "label": "表示名" }
```

#### 2. `index.html` の `buildForm()` に入力 UI を追加

```js
} else if (field.type === 'YOUR_TYPE') {
  // input 要素などを生成して wrap.appendChild()
}
```

#### 3. `encodeLabels` に変換ロジックを追加（UI 値 → 0.0〜1.0）

```js
} else if (field.type === 'YOUR_TYPE') {
  flat.push(/* val を 0.0〜1.0 に変換 */)
}
```

#### 4. `decodeLabels` に逆変換ロジックを追加（0.0〜1.0 → UI 値）

```js
} else if (field.type === 'YOUR_TYPE') {
  return /* flat[idx++] を UI 値に変換 */
}
```

#### 5. `defaultValues()` にデフォルト値を追加

```js
if (field.type === 'YOUR_TYPE') return /* 初期値 */
```

---

#### 実装例：5段階評価 `rating` を追加する

```json
// config.json
{ "id": "quality", "type": "rating", "max": 5, "label": "クオリティ" }
```

```js
// buildForm() — select 要素で実装する例
} else if (field.type === 'rating') {
  const sel = document.createElement('select')
  for (let n = 1; n <= field.max; n++) {
    const o = document.createElement('option')
    o.value = n; o.textContent = '★'.repeat(n)
    sel.appendChild(o)
  }
  sel.dataset.fieldId = field.id
  wrap.appendChild(sel)
}

// encodeLabels
} else if (field.type === 'rating') {
  flat.push((val - 1) / (field.max - 1))
}

// decodeLabels
} else if (field.type === 'rating') {
  return Math.round(flat[idx++] * (field.max - 1)) + 1
}

// defaultValues
if (field.type === 'rating') return Math.ceil(field.max / 2)
```

---

## 2つの主要フロー

### AI Suggestion（未ラベルセグメントを選んだとき）

```
セグメント選択（ラベルなし）
  → predict() が自動実行
  → POST /api/predict
  → predict.py が全セグメントを推論
  → 該当 ID の結果を decodeLabels で UI 値に変換
  → フォームにプレビュー表示（まだ保存はされていない）
```

モデルがまだない場合はプレビューをスキップするだけです。

### 学習（Train Model ボタンを押したとき）

```
POST /api/train → SSE ストリーム開始
  各 epoch: data: {"epoch":N,"loss":X}
  収束時:   data: {"status":"converged","epoch":N,"loss":X}
  終了時:   data: {"status":"done","success":true}
  → fetchTrainMeta() を呼び出してステータスドットを更新
```
