# Module C — 開発者ガイド

## このモジュールの役割

Module C は FLUENT システムの唯一のユーザー接点。
Python 側（Module A/B/D）が生成したデータを読み書きし、ラベリング作業と学習操作を担う。

```
[ブラウザ UI]  ←HTTP/WebSocket→  [Express server.js]  ←child_process→  [Python scripts]
     ↑                                    ↑
  App.vue                            data/*.json
 (Vue 3 SPA)                        (ファイル共有)
```

---

## ディレクトリ構成

```
node/
├── server.js          # Express API + Socket.io サーバー
├── src/
│   ├── App.vue        # UI 全体（単一コンポーネント構成）
│   ├── main.js        # Vue エントリーポイント
│   └── style.css      # Tailwind CSS インポート
├── index.html         # Vite が使う HTML テンプレート
├── vite.config.js     # Vite 設定（プロキシ含む）
├── tailwind.config.js # Tailwind 設定
├── package.json
└── DEVELOPMENT.md     # このファイル
```

---

## 技術スタック

| 役割 | ライブラリ | バージョン |
|---|---|---|
| フロントエンド | Vue 3 (Composition API) | ^3.5 |
| HTTP クライアント | axios | ^1.16 |
| リアルタイム通信 | socket.io-client | ^4.8 |
| API サーバー | Express | ^5.2 |
| リアルタイム通信(サーバー) | socket.io | ^4.8 |
| CSS | Tailwind CSS | ^3.4 |
| ビルド | Vite | ^8.0 |
| 並列起動 | concurrently | ^9.2 |

---

## 起動方法

```bash
# リポジトリルートで
cp .env.example .env     # 初回のみ
npm --prefix node install

# 開発サーバー（Vite + Express を同時起動）
npm --prefix node run dev
```

- Vite dev server: `http://localhost:5173`
- Express API server: `http://localhost:3000`

Vite は `/api`・`/data`・`/socket.io` を Express へプロキシするため、
フロントエンドからは同一オリジン扱いになる。

---

## 環境変数（../.env）

```
PORT=3000
DATA_DIR=data
PYTHON_PATH=.venv/bin/python
ALLOWED_ORIGINS=http://localhost:5173
```

すべて省略可能。デフォルト値が上記。

---

## サーバー (server.js)

### 起動シーケンス

```
1. dotenv で ../.env を読み込む
2. DATA_DIR / PYTHON_BIN / ROOT_DIR を path.resolve で絶対パス化
3. Express app + httpServer + Socket.io を構成
4. /data を静的ファイルとして公開（音声ファイルの直接配信）
5. API エンドポイントを登録
6. httpServer.listen(PORT)
```

### API エンドポイント一覧

#### GET /api/config
`data/config.json` を返す。UI のフォーム構成を定義する。

#### GET /api/segments
`data/segments/*.wav` のファイル名（拡張子なし）をソートして返す。

```json
["example-001", "example-002", "example-003"]
```

#### GET /api/labels
`data/labelset.json` を返す。ファイルが存在しない場合は `{ cols: 0, data: {} }` を返す。

#### POST /api/labels
ラベルを1件保存する。保存前に `data/backups/` へ自動バックアップを作成する。

```json
// リクエスト
{ "id": "example-001", "labels": [0.2, 0.6, 1.0, ...], "cols": 10 }

// レスポンス
{ "success": true }
```

#### POST /api/predict
全セグメントの予測を実行し、指定 ID の予測値を返す。
モデル未存在時は `{ success: false, message: "model_not_found" }` を返す（500 ではない）。

```json
// リクエスト
{ "id": "example-001" }

// レスポンス（成功時）
{ "success": true, "prediction": [0.3, 0.7, 0.5, ...] }
```

#### POST /api/train
学習を非同期で開始する。進捗は Socket.io で配信される。
学習中に再度呼ぶと 409 を返す。

```json
// リクエスト
{ "alpha": 0.01, "resume": false }

// レスポンス（受付時点）
{ "success": true }
```

#### POST /api/train/stop
学習プロセスへ SIGTERM を送る。Python 側がその時点のモデルを保存して終了する。

#### DELETE /api/model
`model.pkl` と `train_meta.json` を削除する。

#### GET /api/train/meta
最終学習のメタデータを返す。

```json
{ "trained_at": "2024-01-01T00:00:00+00:00", "trained_ids": ["example-001", ...] }
```

ファイルが存在しない場合は `{ trained_at: null, trained_ids: [] }` を返す。

### spawnPython ヘルパー

Python スクリプトを子プロセスで実行し、Promise を返す。

```js
const spawnPython = (script, args = []) => new Promise((resolve, reject) => { ... })
// resolve({ out, code }) — exit(0) または exit(2)
// reject(new Error(stderr)) — それ以外の終了コード
```

exit code 2 は「モデル未存在」を示す特別なコードで、正常系として扱う。

### Socket.io イベント（サーバー → クライアント）

| イベント名 | ペイロード | タイミング |
|---|---|---|
| `train:progress` | `{ epoch, loss }` または `{ status: "converged", epoch, loss }` | 学習の各 epoch 後 |
| `train:done` | `{ success: boolean }` | 学習プロセス終了時 |

---

## フロントエンド (App.vue)

### コンポーネント構成

現在は単一ファイルコンポーネント（SFC）。
UI の複雑化に合わせてコンポーネント分割を検討する場合は後述の指針を参照。

### 主要な state

```js
config        // config.json の内容（フォーム定義）
segments      // セグメント ID の配列
labelset      // labelset.json の内容 { cols, data }
trainedIds    // train_meta.json の trained_ids
currentSegment  // 選択中のセグメント ID
currentLabels   // 編集中のラベル値（UI 表現）
```

### セグメントのステータス判定

```js
const segmentStatus = (id) => {
  if (!labelset.value.data[id]) return 'unlabeled'  // ラベルなし
  if (trainedIds.value.includes(id)) return 'trained' // 学習済み
  return 'pending'  // ラベルあり・未学習
}
```

---

## config.json とラベルのエンコード

### config.json スキーマ

UI のフォームはすべて `config.json` から動的生成される。
フィールドタイプごとのスキーマは以下の通り。

```json
[
  { "id": "color",       "type": "color-picker", "dims": 3,    "label": "色彩" },
  { "id": "mood",        "type": "dropdown",     "options": ["Bright", "Dark", "Neutral"], "label": "印象" },
  { "id": "intensity",   "type": "slider",       "min": 0, "max": 1, "step": 0.1, "label": "強度" },
  { "id": "instruments", "type": "checkboxes",   "options": ["Piano", "Guitar", ...], "label": "楽器選択" }
]
```

### フラットベクトル変換（最重要）

`labelset.json` のラベルは A-MAP 規約に従い **0.0〜1.0 のフラット配列** で保存される。
UI 上の値（カラーコード・インデックス・真偽値）とフラット配列の変換は
`encodeLabels` / `decodeLabels` の2関数が担う。

```
UI 値                     フラット配列（labelset.json に保存）
─────────────────────────────────────────────────────────────
"#ff8800"              →  [1.0, 0.533, 0.0]          (RGB / 255)
dropdown index 1       →  [0.5]                       (index / (options.length - 1))
slider 0.7             →  [0.7]                       (そのまま)
checkboxes [T,F,T,F,F] →  [1.0, 0.0, 1.0, 0.0, 0.0] (true=1.0 / false=0.0)
```

`decodeLabels(flat, config)` — フラット配列 → UI 値（表示・編集用）
`encodeLabels(uiValues, config)` — UI 値 → フラット配列（保存・送信用）

フィールドを追加するときはこの2関数に対応するケースを追加する必要がある。

---

## 新しいフィールドタイプを追加する手順

1. `data/config.json` に新しいフィールド定義を追加する
2. `App.vue` テンプレートに `v-if="field.type === '新タイプ'"` ブロックを追加する
3. `encodeLabels` に変換ロジックを追加する（UI 値 → フラット配列）
4. `decodeLabels` に逆変換ロジックを追加する（フラット配列 → UI 値）
5. `initLabels` にデフォルト値を追加する

例: 数値入力 `number-input` を追加する場合

```js
// encodeLabels
} else if (field.type === 'number-input') {
  flat.push(Math.min(1, Math.max(0, val / field.max)))  // 正規化して push

// decodeLabels
} else if (field.type === 'number-input') {
  return flat[idx++] * field.max  // 元のスケールに戻す

// initLabels
if (field.type === 'number-input') return field.min || 0
```

---

## AI Suggestion の動作フロー

```
1. 未ラベルのセグメントを selectSegment で選択
2. hasLabels が false → predict(silent=true) を自動呼び出し
3. POST /api/predict { id }
4. server.js → spawnPython("python/predict.py")
5. predict.py が dataset.json の全セグメントを推論 → stdout に JSON 出力
6. レスポンスの prediction 配列を decodeLabels で UI 値に変換
7. currentLabels に反映（フォームにプレビュー表示）
8. ユーザーが内容を確認・修正して Save Labels
```

モデルが存在しない場合は `message: "model_not_found"` が返り、フォームは変更されない。

---

## 学習フロー（リアルタイム通信）

```
1. trainModel() → POST /api/train { alpha, resume }
2. server.js がトレーニングプロセスを spawn し即座に { success: true } を返す
3. train.py の stdout から1行ずつ JSON を読み取り socket.io で emit
   - 各 epoch: train:progress { epoch, loss }
   - 収束時:   train:progress { status: "converged", epoch, loss }
4. プロセス終了時: train:done { success }
5. success=true → fetchTrainMeta() でステータスドットを更新
```

---

## コンポーネント分割の指針

現在 `App.vue` は 1 ファイルだが、以下の単位で分割を検討できる。

| コンポーネント | 責務 |
|---|---|
| `SegmentList.vue` | セグメントボタン一覧・ステータスドット |
| `LabelForm.vue` | 動的フォーム・Save/AI Suggestion |
| `TrainingPanel.vue` | Alpha スライダー・学習ログ・Train ボタン |
| `FieldRenderer.vue` | フィールドタイプごとの入力 UI |

分割する場合、`encodeLabels` / `decodeLabels` は `src/utils/labels.js` に切り出す。

---

## コーディング規約

- state は `ref()` で宣言し `const` を使う
- 副作用を持つ関数は `async` + `try/finally` で isLoading を確実に解除する
- API 呼び出しはすべて axios 経由（fetch は使わない）
- サーバー側のパスは必ず `path.join` / `path.resolve` を使う（文字列結合禁止）
- Python 呼び出しは `spawnPython` ヘルパーを経由する（spawn を直接使わない）
  - ただし学習は進捗ストリーミングが必要なため `spawn` 直接使用（例外）

---

## よくあるミス

| ミス | 結果 | 対処 |
|---|---|---|
| `encodeLabels` / `decodeLabels` の片方だけ更新 | 保存・読み込みで値がずれる | 必ずペアで更新する |
| `config.json` の `dims` を省略 | color-picker が 3 次元固定になる | `field.dims \|\| 3` がデフォルト値 |
| `labelset.json` の `cols` を更新しない | A-MAP スキーマ不正 | `POST /api/labels` の `cols` に `flatLabels.length` を渡す |
| Socket.io 接続を `onMounted` 外で行う | SSR 環境で window 参照エラー | 必ず `onMounted` 内で `io()` を呼ぶ |
