# FLUENT 実装ドキュメント

本ドキュメントは `AGENTS.md` の要件に基づき実装された FLUENT フレームワークの構成と使い方について説明します。

## ディレクトリ構成

システムは主に3つのディレクトリで構成されています。

- `/python`: Module A, B, D のための Python 解析・機械学習スクリプト
- `/node`: Module C のための Node.js (Express) + Vue.js (Vite) サーバーおよびUIアプリケーション
- `/data`: 音源データ (raw_audio), 分割済みデータ (segments), 設定 (config.json), および A-MAP JSON ファイル (dataset.json, labelset.json)

## 1. 環境構築 (Setup)

### Python 環境 (Module A, B, D)
```bash
uv venv
source .venv/bin/activate
uv pip install -r python/requirements.txt
```

### Node.js 環境 (Module C)
```bash
cd node
npm install
npm run build
```

## 2. システムの実行手順

### Step 1: 音源の準備と特徴量抽出 (Module A & B)
まず、`/data/raw_audio` ディレクトリに解析したい WAV ファイルを配置します。ダミー音声を用意する場合は以下のスクリプトを実行します。

```bash
source .venv/bin/activate
python python/generate_dummy_audio.py
```

次に、音源をセグメントに分割し、A-MAP JSON に変換します。

```bash
# セグメンテーション (Module A)
python python/segmenter.py

# 特徴量抽出と標準化 (Module B)
python python/extractor.py
```
これにより `/data/dataset.json` が生成されます。

### Step 2: ラベリングとUI操作 (Module C)
Node.js サーバーと Vue フロントエンドを起動します。

```bash
cd node
npm run dev
```

- ブラウザで `http://localhost:5173/` (または表示されたローカルポート) にアクセスします。
- `data/config.json` に定義された内容に基づいて、UI (色彩ピッカー、ドロップダウン、スライダー) が動的に生成されます。
- セグメントを選択して音声を試聴し、ラベリングを行った後 `Save Labels` で保存します（結果は `data/labelset.json` に書き込まれます）。

### Step 3: モデルの学習と推論 (Module D)
いくつかラベリングが完了したら、そのデータを使って MLP (多層パーセプトロン) を学習させます。

```bash
source .venv/bin/activate
python python/train.py
```
学習中、Loss値がログ出力され、収束するとモデルが `data/model.pkl` に保存されます。

学習済みのモデルがある状態であれば、UI 上で未ラベリングのセグメントを選択し `AI Suggestion` ボタンをクリックすることで、予測値が自動で入力されます。

## 3. 実装のポイントと A-MAP 規約の順守

- **状態のファイル管理:** モジュール間のデータのやり取りはすべて A-MAP JSON 形式 (`dataset.json`, `labelset.json`) を介して行っており、ステートレスな連携を実現しています。
- **UI の動的生成:** ラベルの種類 (色や印象など) をハードコードせず、`config.json` から動的にフォームを構成するようにしています。
- **データの前処理・後処理:** `extractor.py` にて、特徴量は `StandardScaler` によって必ず正規化 (平均0, 標準偏差1) されます。推論時は出力が 0.0 〜 1.0 にクリッピングされるように実装しています。
- **非同期 UI 連携:** Vue.js から Express の API へリクエストを投げ、バックエンドで `child_process.spawn` を使用して Python スクリプト (`predict.py`) を非同期に呼び出しています。