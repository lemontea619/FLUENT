# FLUENT 実装ドキュメント

## ディレクトリ構成

```
FLUENT/
├── python/          # Module A, B, D（解析・学習）
├── node/            # Module C（ラベリング UI + Express API）
├── data/
│   ├── raw_audio/   # 元音源 WAV
│   ├── segments/    # 分割済みセグメント WAV
│   ├── config.json  # ラベル定義（Y の次元を決める）
│   ├── dataset.json # A-MAP 特徴量（X）
│   ├── labelset.json# A-MAP ラベル（Y）
│   ├── scaler.pkl   # StandardScaler パラメータ（joblib）
│   ├── model.pkl    # 学習済み MLP モデル（joblib）
│   └── train_meta.json # 最終学習メタデータ
├── .env             # 環境変数（.gitignore 対象）
├── .env.example     # .env のテンプレート
└── pyproject.toml   # Python 依存関係（uv 管理）
```

## 1. 環境構築

```bash
cp .env.example .env
bash setup_all.sh
```

`setup_all.sh` が以下を実行する：
- `uv sync` で Python 仮想環境を構築
- `npm install` で Node.js 依存関係をインストール
- ダミー音源生成・セグメント分割・特徴量抽出を実行

## 2. 実行手順

### Step 1: 音源準備と特徴量抽出（Module A & B）

独自の WAV を使う場合は `data/raw_audio/` に配置してから：

```bash
# 2秒単位で分割（デフォルト、推奨）
uv run python python/segmenter.py

# 粒度を変更する場合（例: 5秒単位 / 10秒単位）
uv run python python/segmenter.py --duration 5

# 特徴量抽出（26次元）
uv run python python/extractor.py
```

`data/dataset.json`（A-MAP 形式、26次元）が生成される。分割粒度（duration）を変更した場合は、過去の `labelset.json` との ID 整合性が失われるため、再学習が必要です。

### Step 2: ラベリング UI 起動（Module C）

```bash
cd node && npm run dev
```

`http://localhost:3000` にアクセス。

- セグメントを選択して試聴
- `config.json` の定義に従い動的生成されたフォームでラベル付け
- `Save Labels` で `data/labelset.json` に保存（自動バックアップあり）

### Step 3: 学習と推論（Module D）

UI の `Model Training` セクションから：

- Alpha スライダーで正則化強度を調整（0.001〜0.1）
- `Train Model` ボタンで学習開始
- epoch ごとの loss 値がリアルタイム表示され、収束時に `Converged` と通知
- ラベル未入力のセグメントを選択すると AI Suggestion が自動実行

## 3. 学習ライフサイクル

### セグメントのステータス

UI の各セグメントボタンにはドットでステータスが表示される。

| ドット色 | ステータス | 意味 |
|---|---|---|
| グレー | `unlabeled` | ラベルなし |
| 黄色 | `pending` | ラベルあり・未学習（モデルに未反映） |
| 緑 | `trained` | ラベルあり・学習済み |

`pending` のセグメントが存在する場合、Train Model ボタン横に "N pending — retrain recommended" と警告が表示される。

### train_meta.json

学習完了時に `data/train_meta.json` が書き出される。UI はこのファイルを参照してステータスを判定する。

```json
{
  "trained_at": "2024-01-01T00:00:00+00:00",
  "trained_ids": ["example-001", "example-002"]
}
```

### 学習の中断・再開・リセット

| 操作 | API | 動作 |
|---|---|---|
| 通常学習 | `POST /api/train` `{ alpha }` | ゼロから学習 |
| 途中から再開 | `POST /api/train` `{ alpha, resume: true }` | `model.pkl` をロードして続きから学習 |
| 中断（途中保存） | `POST /api/train/stop` | SIGTERM を送信。Python 側でその時点の重みを `model.pkl` に保存して終了 |
| リセット | `DELETE /api/model` | `model.pkl` と `train_meta.json` を削除 |

再開フロー：

```
POST /api/train/stop
  → SIGTERM → train.py が現在の mlp を model.pkl に保存 → exit(0)
  → train:done { success: true } がブロードキャストされ UI のステータスが更新

POST /api/train { resume: true }
  → train.py が model.pkl をロードして epoch ループを継続
```

注意：再開時は同じ `dataset.json` / `labelset.json` を使用する。音源を追加して `extractor.py` を再実行した場合は scaler が再フィットされるため、必ずゼロから学習し直すこと（`resume` は使用不可）。

## 4. スケーリング仕様

### 特徴量の標準化（X）

`extractor.py` は以下の 26 次元の物理特徴を抽出し、一括で `StandardScaler`（平均0、標準偏差1）に変換してから `dataset.json` に書き出します。

| カテゴリ | 次元数 | 内容 |
|---|---|---|
| **MFCC** | 12 | メル周波数ケプストラム係数（第0係数除外） |
| **Spectral** | 4 | Centroid, Flatness, Rolloff, Bandwidth |
| **Theory** | 3 | Tempo(BPM), Key(0-11), Mode(0-1) |
| **Chroma** | 7 | 色彩度（C, C#, D, D#, E, F, F#） |

スケーラーのパラメータは `data/scaler.pkl` に保存されます。

```
extractor.py → StandardScaler.fit_transform(全セグメント) → dataset.json（スケール済み）
                                                          → data/scaler.pkl
```

注意事項：
- `dataset.json` に保存される特徴量はスケール済みの値である
- 学習（`train.py`）は `dataset.json` をそのまま読み込むため、追加のスケール変換は不要
- 新規セグメントを後から単体で予測する場合は `data/scaler.pkl` をロードして同じ変換を適用すること（スケール係数とモデルの不整合を防ぐため）

### 推論結果の正規化（Y）

`predict.py` はモデル出力に `numpy.clip(y_pred, 0.0, 1.0)` を適用して出力を 0.0〜1.0 に収める。
これにより、モデルがトレーニングデータ外の値を外挿（Extrapolation）した場合でもアダプター互換性が保たれる。

```
MLPRegressor.predict(x) → np.clip(..., 0.0, 1.0) → JSON 出力
```

## 5. データ契約（A-MAP スキーマ）

### A-MAP とは

A-MAP は [FluCoMa（Fluid Corpus Manipulation）](https://www.flucoma.org/) のデータセット形式に準拠したデータ交換規約。
FluCoMa は Max/MSP・SuperCollider・Pure Data 向けの音響コーパス操作ツールキットであり、
その `fluid.dataset~` オブジェクトが以下の JSON を標準入出力として扱う。

FLUENT が生成する `dataset.json` / `labelset.json` はこの形式に準拠しているため、
FluCoMa パッチへそのまま渡せる互換性を持つ。

```json
// dataset.json（X: 物理特徴量）
{ "cols": 26, "data": { "example-001": [...] } }

// labelset.json（Y: 主観ラベル）
{ "cols": 10, "data": { "example-001": [...] } }
```

`config.json` でラベルの次元構成を定義する。現在の構成（cols=10）：

| フィールド | type | 次元数 |
|---|---|---|
| color | color-picker | 3（RGB） |
| mood | dropdown | 1 |
| intensity | slider | 1 |
| instruments | checkboxes | 5 |
