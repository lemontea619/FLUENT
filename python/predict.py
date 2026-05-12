import argparse
import json
import os
import pickle
import sys

import numpy as np


def predict_labels(dataset_path, model_path):
    if not os.path.exists(model_path):
        print(json.dumps({"error": "model_not_found"}))
        sys.exit(2)

    with open(model_path, "rb") as f:
        mlp = pickle.load(f)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    predictions = {}

    for file_id, features in dataset["data"].items():
        x = np.array([features])
        y_pred = mlp.predict(x)[0]
        y_pred = np.clip(y_pred, 0.0, 1.0)
        predictions[file_id] = y_pred.tolist()

    output_data = {
        "cols": len(list(predictions.values())[0]) if predictions else 0,
        "data": predictions,
    }

    print(json.dumps(output_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict labels using MLP model")
    parser.add_argument("--dataset", type=str, default="data/dataset.json")
    parser.add_argument("--model", type=str, default="data/model.pkl")

    args = parser.parse_args()
    predict_labels(args.dataset, args.model)
