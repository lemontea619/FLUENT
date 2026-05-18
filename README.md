# FLUENT

音響物理特徴 X → 主観感性ラベル Y へのマッピングフレームワーク。

## 構成

```
python/   Module A/B/D  セグメンタ・特徴量抽出・学習・推論（Python）
node/     Module C      ラベリング UI + Express API（Node.js）
data/                   入出力データ（音源・JSON・モデル）
```

## クイックスタート

```bash
cp .env.example .env
bash setup_all.sh
```

`setup_all.sh` はダミー音源生成・セグメント分割・特徴量抽出まで自動で実行します。

### UI 起動

```bash
cd node && npm run dev
```

`http://localhost:3000` にアクセス。

## 手順

1. `data/raw_audio/` に WAV を配置して特徴量抽出（自前素材を使う場合）

```bash
# セグメント分割（--duration 2, 5, 10 などで粒度調整可能）
uv run python python/segmenter.py --duration 2

# 特徴量抽出（26次元物理特徴の生成）
uv run python python/extractor.py
```

2. ブラウザでセグメントを試聴 → ラベル入力 → Save Labels

3. UI の Train Model ボタンで学習。epoch ごとの loss がリアルタイム表示される

4. 未ラベルセグメントを選択すると AI Suggestion が自動表示される

## 詳細

- 実装仕様: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- Module C 開発ガイド: [node/DEVELOPMENT.md](node/DEVELOPMENT.md)
- 設計原則: [AGENTS.md](AGENTS.md)
