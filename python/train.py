"""
=============================================================================
Module D: Training Engine
=============================================================================

A-MAP 形式のデータセット（特徴量 X）とラベルセット（感性 Y）から、
多層パーセプトロン（MLP）を訓練し、26次元 → 10次元のマッピングモデルを構築する

【処理フロー】
1. dataset.json（26次元特徴量、スケール済み）をロード
2. labelset.json（10次元感性ラベル）をロード
3. 両者に存在するセグメント ID で X・Y を対応させる
4. MLPRegressor で訓練開始
5. epoch ごとに loss 値をリアルタイム出力（UI でプログレス表示）
6. 収束判定（loss 改善が停滞）またはユーザー中断で学習終了
7. モデルを model.pkl に保存、メタデータを train_meta.json に記録

【訓練パイプライン】
  dataset.json（X: 26次元）
    + labelset.json（Y: 10次元）
        ↓
  MLP初期化（隠れ層: 32→16→出力10）
        ↓
  epoch ループ（warm_start で増分訓練）
    - loss 出力（JSON）
    - 収束判定（loss 改善度 < 1e-5 が 10 epoch 連続）
        ↓
  SIGTERM で安全に中断（途中重み保存）
        ↓
  model.pkl + train_meta.json 出力

【MLP アーキテクチャ】
  入力層     : 26 次元（特徴量）
  隠れ層1    : 32 ユニット（ReLU活性化）
  隠れ層2    : 16 ユニット（ReLU活性化）
  出力層     : 10 次元（感性ラベル）

【重要なパラメータ】
  - alpha: L2 正則化強度（過学習抑制）
    - デフォルト 0.01（推奨値）
    - 大きい（0.1）: より強く過学習を抑制、ただし汎化性能低下の可能性
    - 小さい（0.001）: より複雑なパターン学習、過学習リスク上昇

  - warm_start: True
    - 増分訓練を有効化（既存モデルから続きから訓練）
    - resume フラグで既存 model.pkl から学習再開

  - early_stopping: False
    - FLUENT は独自の収束判定ロジックを使用
    - scikit-learn 組み込みの early stopping は使わない

【収束判定ロジック】
  - Loss 改善度が 1e-5 未満（ほぼ変化なし）が 10 epoch 連続
  - 続いた時点で「収束」と判定し、学習を自動停止

【UI への通信】
  各行は JSON で stdout に出力（Node.js が受信）：
  - {"epoch": i, "loss": 0.123}  毎 epoch（プログレス表示）
  - {"status": "converged", ...}  収束時
  - {"status": "interrupted", "saved": True}  SIGTERM 受信時

【使用例】
  python train.py
  → 全ラベル付きセグメントで訓練（デフォルト: alpha=0.01）

  python train.py --alpha 0.05
  → 正則化を強くして訓練（過学習を抑制）

  python train.py --resume
  → model.pkl があれば続きから訓練、なければゼロから

  python train.py --resume --alpha 0.02
  → 既存モデルから続き、alpha を 0.02 に変更して訓練

【重要な制限】
  - dataset.json と scaler.pkl は訓練時と推論時で一致が必須
    音源を追加して extractor.py を再実行した場合は、
    scaler が再フィットされるため、model.pkl は無効になる（ゼロから訓練し直す）

  - resume は同じ dataset.json / labelset.json 前提
    ラベル追加・変更があった場合は resume しない（ゼロから訓練）
"""

import argparse
import json
import os
import pickle
import signal
import sys
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# グローバルステート
# ============================================================
# SIGTERM ハンドラ内でモデル・パスにアクセスするため、グローバル変数で管理
# （Python では関数内の局所変数は SIGTERM 発火時にアクセス困難）
_state: dict[str, Any] = {"mlp": None, "model_path": None}


def _handle_sigterm(_sig: int, _frame: object) -> None:
    """
    【SIGTERM ハンドラ】中断時に安全にモデルを保存

    動作：
      1. Node.js UI が学習停止ボタンをクリック
      2. Python プロセスに SIGTERM を送信
      3. このハンドラが発火
      4. 現在の MLP 重みを model.pkl に保存
      5. JSON ステータスを stdout に出力
      6. プロセス終了

    【重要】途中保存により、resume フラグで続きから再開可能
    """
    mlp = _state["mlp"]
    model_path = _state["model_path"]
    if mlp is not None and model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(mlp, f)
        # UI に完了を通知（JSON で stdout に出力）
        print(json.dumps({"status": "interrupted", "saved": True}), flush=True)
    sys.exit(0)


# SIGTERM シグナルをハンドラに登録
signal.signal(signal.SIGTERM, _handle_sigterm)


def train_model(
    dataset_path: str,
    labelset_path: str,
    model_path: str,
    alpha: float = 0.01,
    resume: bool = False,
) -> None:
    """
    【訓練実行関数】

    Args:
        dataset_path: 特徴量 JSON ファイルパス（data/dataset.json）
        labelset_path: ラベル JSON ファイルパス（data/labelset.json）
        model_path: 保存先モデルパス（data/model.pkl）
        alpha: L2 正則化強度（デフォルト 0.01）
        resume: True なら model.pkl から続きから訓練
    """

    # ============================================================
    # Step 1: グローバルステートにモデルパスを記録
    # ============================================================
    # SIGTERM ハンドラがアクセスするため、事前に設定
    _state["model_path"] = model_path

    # ============================================================
    # Step 2: ファイルの存在確認
    # ============================================================
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(labelset_path):
        print(
            f"Error: {labelset_path} not found. Start labeling first.", file=sys.stderr
        )
        sys.exit(1)

    # ============================================================
    # Step 3: dataset.json と labelset.json をロード
    # ============================================================
    # dataset.json
    #   {"cols": 26, "data": {"example-001": [...], "example-002": [...], ...}}
    # labelset.json
    #   {"cols": 10, "data": {"example-001": [...], "example-002": [...], ...}}
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    with open(labelset_path, "r") as f:
        labelset = json.load(f)

    # ============================================================
    # Step 4: データの対応付け（交差セット）
    # ============================================================
    # labelset に存在するセグメント ID のみ訓練対象
    # （未ラベル ID は除外）
    x_arr = []  # 特徴量（26次元）のリスト
    y_arr = []  # ラベル（10次元）のリスト

    labeled_ids = list(labelset["data"].keys())

    for file_id in labeled_ids:
        # labelset に ID があり、かつ dataset にも ID がある場合のみ
        if file_id in dataset["data"]:
            x_arr.append(dataset["data"][file_id])
            y_arr.append(labelset["data"][file_id])

    # ============================================================
    # Step 5: 訓練データの検証
    # ============================================================
    # 交差セットが空の場合（labelset に ID があるが dataset に ID がない場合など）
    # → 訓練不可能なため終了
    if len(x_arr) == 0:
        print(
            "Error: No intersecting data between dataset and labelset.", file=sys.stderr
        )
        sys.exit(1)

    # ============================================================
    # Step 6: NumPy 配列に変換
    # ============================================================
    # x_arr: (N, 26) の 2 次元配列
    #   N = ラベル付きセグメント数
    #   26 = 特徴量次元数
    # y_arr: (N, 10) の 2 次元配列
    #   N = ラベル付きセグメント数
    #   10 = 感性ラベル次元数
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)

    # ============================================================
    # Step 7: モデルの初期化または復元
    # ============================================================
    if resume and os.path.exists(model_path):
        # ============================================================
        # Step 7-a: 既存モデルから続きから訓練（resume フロー）
        # ============================================================
        # pickle で保存された既存 MLP をロード
        with open(model_path, "rb") as f:
            mlp = pickle.load(f)

        # alpha（正則化強度）を指定値に変更
        # warm_start=True で増分訓練を有効化
        # max_iter=1 で毎回の fit() を 1 epoch として使用
        mlp.set_params(alpha=alpha, warm_start=True, max_iter=1)
        print(json.dumps({"status": "resumed"}), flush=True)
    else:
        # ============================================================
        # Step 7-b: 新規モデルの作成（ゼロから訓練）
        # ============================================================
        # MLPRegressor パラメータ：
        #
        # hidden_layer_sizes=(32, 16)
        #   隠れ層の構成：32ユニット → 16ユニット → 出力10次元
        #   理由：過学習を避けるため、層を深くしすぎず
        #        small data 想定の標準的サイズ
        #
        # activation="relu"
        #   ReLU（Rectified Linear Unit）活性化
        #   f(x) = max(0, x)：単純かつ効果的
        #
        # solver="adam"
        #   Adam オプティマイザ：適応的学習率、勾配下降より安定
        #
        # alpha=alpha
        #   L2 正則化強度（デフォルト 0.01）
        #   loss += alpha * (重み二乗和)
        #   大きいほど、モデルが単純化される（過学習抑制）
        #
        # max_iter=1000
        #   最大 epoch 数（早期停止がない場合の上限）
        #   実際には fit() が warm_start で 1 epoch ずつ進むため、
        #   ここでは上限値としてのみ機能
        #
        # random_state=42
        #   乱数シード固定（再現性確保）
        #
        # early_stopping=False
        #   scikit-learn 組み込みの early stopping は使用しない
        #   FLUENT は独自の収束判定ロジック（loss 改善度）を使用
        mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=alpha,
            max_iter=1000,
            random_state=42,
            early_stopping=False,
            tol=1e-5,
            n_iter_no_change=10,
            verbose=True,
        )

    # ============================================================
    # Step 8: グローバルステートに MLP を記録
    # ============================================================
    # SIGTERM ハンドラが中断時にモデルを保存するため
    _state["mlp"] = mlp

    # ============================================================
    # Step 9 & 10: 訓練ループと stdout のキャプチャ
    # ============================================================
    # scikit-learn の C 実装ループを活用し高速化
    # verbose=True の stdout をキャプチャし、UI 用に JSON で出力する
    class VerboseCapture:
        def __init__(self, original_stdout):
            self.original_stdout = original_stdout
            self.last_loss = None
            self.last_epoch = None

        def write(self, text):
            if text.startswith("Iteration "):
                # 例: "Iteration 1, loss = 0.73030389"
                try:
                    parts = text.split(", loss = ")
                    epoch = int(parts[0].replace("Iteration ", "")) - 1 # UI 用に 0-indexed に変換
                    loss = float(parts[1].strip())
                    self.last_epoch = epoch
                    self.last_loss = loss
                    self.original_stdout.write(json.dumps({"epoch": epoch, "loss": loss}) + "\n")
                    self.original_stdout.flush()
                except Exception:
                    pass

        def flush(self):
            self.original_stdout.flush()

    old_stdout = sys.stdout
    capture = VerboseCapture(old_stdout)
    sys.stdout = capture

    try:
        mlp.fit(x_arr, y_arr)
    finally:
        sys.stdout = old_stdout

    # 収束判定
    if capture.last_epoch is not None and capture.last_loss is not None:
        if mlp.n_iter_ < mlp.max_iter:
            # max_iter に到達する前に終了した場合、収束とみなす
            print(
                json.dumps(
                    {
                        "status": "converged",
                        "epoch": capture.last_epoch,
                        "loss": capture.last_loss,
                    }
                ),
                flush=True,
            )

    # ============================================================
    # Step 11: 学習済みモデルを保存
    # ============================================================
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(mlp, f)

    # ============================================================
    # Step 12: メタデータ（train_meta.json）を保存
    # ============================================================
    # UI はこのファイルを参照してセグメントのステータスを判定
    # - "trained_at": 訓練完了時刻（ISO 8601 フォーマット）
    # - "trained_ids": 訓練に使用したセグメント ID のリスト
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_ids": labeled_ids,
    }
    meta_path = os.path.join(os.path.dirname(model_path), "train_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {model_path}", file=sys.stderr)


if __name__ == "__main__":
    # ============================================================
    # コマンドラインオプション解析
    # ============================================================
    # 使用例:
    #   python train.py
    #   → 全ラベル付きセグメントで訓練（デフォルト: alpha=0.01）
    #
    #   python train.py --alpha 0.05
    #   → 正則化を強くして訓練（過学習を抑制）
    #
    #   python train.py --resume
    #   → model.pkl があれば続きから訓練、なければゼロから
    #
    #   python train.py --resume --alpha 0.02
    #   → 既存モデルから続き、alpha を 0.02 に変更して訓練
    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset.json",
        help="Path to dataset JSON (input features)",
    )
    parser.add_argument(
        "--labelset",
        type=str,
        default="data/labelset.json",
        help="Path to labelset JSON (ground truth labels)",
    )
    parser.add_argument(
        "--model", type=str, default="data/model.pkl", help="Path to save the model"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing model.pkl if it exists. Otherwise, train from scratch",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="L2 regularization alpha (default: 0.01). Larger value = stronger regularization, weaker overfitting but less complex patterns. Smaller value = weaker regularization, stronger overfitting risk but more flexible learning",
    )

    args = parser.parse_args()
    train_model(args.dataset, args.labelset, args.model, args.alpha, args.resume)
