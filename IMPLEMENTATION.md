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
│   └── labelset.json# A-MAP ラベル（Y）
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
uv run python python/segmenter.py
uv run python python/extractor.py
```

`data/dataset.json`（A-MAP 形式、26次元）が生成される。

### Step 2: ラベリング UI 起動（Module C）

```bash
cd node && npm run dev
```

`http://localhost:5173` にアクセス。

- セグメントを選択して試聴
- `config.json` の定義に従い動的生成されたフォームでラベル付け
- `Save Labels` で `data/labelset.json` に保存（自動バックアップあり）

### Step 3: 学習と推論（Module D）

UI の `Model Training` セクションから：

- Alpha スライダーで正則化強度を調整（0.001〜0.1）
- `Train Model` ボタンで学習開始
- epoch ごとの loss 値がリアルタイム表示され、収束時に `Converged` と通知
- ラベル未入力のセグメントを選択すると AI Suggestion が自動実行

## 3. データ契約（A-MAP スキーマ）

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
