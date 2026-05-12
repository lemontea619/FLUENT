import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import json
import os
import pickle
import signal
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

_state: dict[str, Any] = {"mlp": None, "model_path": None}


def _handle_sigterm(_sig: int, _frame: object) -> None:
    mlp = _state["mlp"]
    model_path = _state["model_path"]
    if mlp is not None and model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(mlp, f)
        print(json.dumps({"status": "interrupted", "saved": True}), flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)


def train_model(
    dataset_path: str,
    labelset_path: str,
    model_path: str,
    alpha: float = 0.01,
    resume: bool = False,
) -> None:
    _state["model_path"] = model_path
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(labelset_path):
        print(
            f"Error: {labelset_path} not found. Start labeling first.", file=sys.stderr
        )
        sys.exit(1)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    with open(labelset_path, "r") as f:
        labelset = json.load(f)

    x_arr = []
    y_arr = []

    labeled_ids = list(labelset["data"].keys())

    for file_id in labeled_ids:
        if file_id in dataset["data"]:
            x_arr.append(dataset["data"][file_id])
            y_arr.append(labelset["data"][file_id])

    if len(x_arr) == 0:
        print(
            "Error: No intersecting data between dataset and labelset.", file=sys.stderr
        )
        sys.exit(1)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)

    if resume and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            mlp = pickle.load(f)
        mlp.set_params(alpha=alpha, warm_start=True, max_iter=1)
        print(json.dumps({"status": "resumed"}), flush=True)
    else:
        mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=alpha,
            max_iter=1000,
            random_state=42,
            early_stopping=False,
        )
        mlp.set_params(warm_start=True, max_iter=1)

    _state["mlp"] = mlp
    prev_loss = float("inf")
    patience = 10
    wait = 0

    for i in range(1000):
        mlp.fit(x_arr, y_arr)
        loss = mlp.loss_

        print(json.dumps({"epoch": i, "loss": float(loss)}), flush=True)

        if abs(prev_loss - loss) < 1e-5:
            wait += 1
            if wait >= patience:
                print(
                    json.dumps(
                        {"status": "converged", "epoch": i, "loss": float(loss)}
                    ),
                    flush=True,
                )
                break
        else:
            wait = 0

        prev_loss = loss

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(mlp, f)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_ids": labeled_ids,
    }
    meta_path = os.path.join(os.path.dirname(model_path), "train_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model saved to {model_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument(
        "--dataset", type=str, default="data/dataset.json", help="Path to dataset JSON"
    )
    parser.add_argument(
        "--labelset",
        type=str,
        default="data/labelset.json",
        help="Path to labelset JSON",
    )
    parser.add_argument(
        "--model", type=str, default="data/model.pkl", help="Path to save the model"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing model.pkl"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="L2 regularization alpha"
    )

    args = parser.parse_args()
    train_model(args.dataset, args.labelset, args.model, args.alpha, args.resume)
