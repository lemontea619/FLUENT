"""
=============================================================================
Module D: Inference Engine (Predictor)
=============================================================================

学習済み MLP モデルを使用して、新規セグメント（または未ラベルセグメント）に対して
感性ラベルを予測し、A-MAP 形式の JSON で出力する

【処理フロー】
1. 学習済み MLP モデル（model.pkl）をロード
2. dataset.json から特徴量（X: 26次元）を取得
3. モデルで感性ラベル予測（Y: 10次元）を実行
4. 出力を 0.0～1.0 の範囲にクリップ（アダプター互換性確保）
5. 結果を JSON で出力

【推論パイプライン】
  特徴量（26次元）
    ↓
  MLP forward pass
    ↓
  予測値（10次元）
    ↓
  np.clip(y, 0.0, 1.0)
    ↓
  JSON 出力

【出力例】predict.json（またはstdout）
{
  "cols": 10,
  "data": {
    "example-001": [1.0, 0.0, 0.0, 0.8, 1.0, 1, 0, 0, 0, 0],
    "example-002": [0.0, 0.0, 1.0, 0.2, 0.5, 0, 0, 1, 0, 0]
  }
}

【重要】
  - モデルの学習に使用した dataset.json と同じスケーリング体系を使用
    （訓練時 StandardScaler でフィット済み）
  - 10次元出力: RGB(3) + Mood(1) + Intensity(1) + Instruments(5)
  - 出力は常に 0.0～1.0 クリップされる（推論が 0.0 より小さい、1.0 より大きい値を出力してもハンドル）
  - target_id を指定すると、そのセグメント ID のみ予測

【使用例】
  python predict.py
  → 全セグメントを予測し stdout に出力

  python predict.py --id example-001
  → example-001 のみ予測し stdout に出力

  python predict.py --output predictions.json
  → 全セグメントを predictions.json に保存

  python predict.py --id example-001 --output pred_001.json
  → example-001 のみを pred_001.json に保存（UI の AI Suggestion 想定）
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np


def predict_labels(
    dataset_path: str,
    model_path: str,
    target_id: str | None = None,
    output_path: str | None = None,
) -> None:
    """
    【推論関数】MLP モデルで感性ラベルを予測

    Args:
        dataset_path: 特徴量 JSON ファイルパス（data/dataset.json）
        model_path: 学習済み MLP モデルパス（data/model.pkl）
        target_id: 特定セグメント ID のみ予測する場合（例: "example-001"）
                   None なら全セグメント予測
        output_path: 結果を JSON ファイルに保存するパス
                     None なら stdout に JSON 出力

    Raises:
        SystemExit(2): model.pkl が見つからない場合
    """

    # ============================================================
    # Step 1: モデルファイルの存在確認
    # ============================================================
    if not os.path.exists(model_path):
        # モデルが見つからない場合は JSON エラーメッセージを出力して終了
        # （UI が JSON を期待しているため、例外ではなく JSON で返す）
        print(json.dumps({"error": "model_not_found"}))
        sys.exit(2)

    # ============================================================
    # Step 2: 学習済み MLP モデルをロード
    # ============================================================
    # pickle 形式で保存された scikit-learn の MLPRegressor/MLPClassifier
    with open(model_path, "rb") as f:
        mlp = pickle.load(f)

    # ============================================================
    # Step 3: 特徴量 JSON をロード
    # ============================================================
    # dataset.json は A-MAP 形式
    # {"cols": 26, "data": {"example-001": [...], "example-002": [...], ...}}
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # 予測結果を格納する辞書
    predictions = {}

    # ============================================================
    # Step 4: 各セグメントの予測を実行（バッチ処理）
    # ============================================================
    # フィルタリングされたデータセットを構築する
    file_ids = []
    features_list = []

    for file_id, features in dataset["data"].items():
        # ============================================================
        # Step 4-a: target_id フィルタ（指定された場合のみ）
        # ============================================================
        # 例）--id example-001 を指定した場合、example-001 のみ処理
        if target_id is not None and file_id != target_id:
            continue
        file_ids.append(file_id)
        features_list.append(features)

    # 推論対象がある場合のみ予測を実行する
    if features_list:
        # ============================================================
        # Step 4-b: 特徴量を MLP 入力形式（2次元配列）に整形
        # ============================================================
        # features は 1 次元リスト（長さ 26）のリスト
        # MLP.predict は (N_samples, 26) の 2 次元入力を期待するため
        # np.array(features_list) で (N_samples, 26) に一括変換
        X = np.array(features_list)

        # ============================================================
        # Step 4-c: モデルで予測実行（バッチ処理）
        # ============================================================
        # mlp.predict(X) → (N_samples, 10) の 2 次元配列を返す
        #   - N_samples: サンプル数
        #   - 10: 感性ラベル次元数
        y_preds = mlp.predict(X)

        # ============================================================
        # Step 4-d: 出力を 0.0～1.0 の範囲にクリップ（正規化）
        # ============================================================
        # 【重要】モデルが外挿（Extrapolation）した場合の対応
        # MLP は訓練データの範囲外の値を出力することがある
        # 例：訓練では [0, 1] 範囲だが、推論で [-0.3, 1.5] が出力された
        #
        # np.clip(y_preds, 0.0, 1.0) で強制的に [0.0, 1.0] に収める
        #
        # ロジック：
        #   - y < 0.0 → 0.0 に設定
        #   - 0.0 <= y <= 1.0 → そのまま
        #   - y > 1.0 → 1.0 に設定
        #
        # これにより、UI やシンセサイザーパラメータのアダプター互換性が保証される
        y_preds = np.clip(y_preds, 0.0, 1.0)

        # ============================================================
        # Step 4-e: 予測結果をリスト化して格納
        # ============================================================
        # tolist(): numpy 配列を Python リストに変換（JSON シリアライズ対応）
        for i, file_id in enumerate(file_ids):
            predictions[file_id] = y_preds[i].tolist()

    # ============================================================
    # Step 5: 出力データを A-MAP 形式で構成
    # ============================================================
    # cols: 感性ラベルの次元数（通常 10）
    # data: ID → 予測ラベルのマッピング
    #
    # 注意：predictions が空の場合（target_id が見つからない場合など）、
    #      cols は 0 になる可能性
    output_data = {
        "cols": len(list(predictions.values())[0]) if predictions else 0,
        "data": predictions,
    }

    # ============================================================
    # Step 6: 出力先に応じて JSON を書き出し
    # ============================================================
    if output_path:
        # ファイルに保存
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f)
    else:
        # stdout に出力（Node.js UI が受け取る）
        print(json.dumps(output_data))


if __name__ == "__main__":
    # ============================================================
    # コマンドラインオプション解析
    # ============================================================
    # 使用例:
    #   python predict.py
    #   → 全セグメント予測、stdout に出力
    #
    #   python predict.py --id example-001
    #   → example-001 のみ予測、stdout に出力
    #   → UI の AI Suggestion で使用（未ラベルセグメント1個を予測）
    #
    #   python predict.py --output predictions.json
    #   → 全セグメント予測、predictions.json に保存
    #
    #   python predict.py --id example-001 --output pred.json
    #   → example-001 のみ予測、pred.json に保存
    parser = argparse.ArgumentParser(description="Predict labels using MLP model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset.json",
        help="Path to dataset JSON file (input features)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/model.pkl",
        help="Path to trained MLP model (pickle format)",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Target segment ID to predict. If not specified, predict all segments",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path to save predictions. If not specified, print to stdout",
    )

    args = parser.parse_args()
    predict_labels(args.dataset, args.model, args.id, args.output)
