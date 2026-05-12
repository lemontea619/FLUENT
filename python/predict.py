import json
import os
import sys
import numpy as np
import pickle
import argparse

def predict_labels(dataset_path, model_path, output_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.", file=sys.stderr)
        sys.exit(1)

    with open(model_path, 'rb') as f:
        mlp = pickle.load(f)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    predictions = {}

    for file_id, features in dataset['data'].items():
        X = np.array([features])
        y_pred = mlp.predict(X)[0]
        y_pred = np.clip(y_pred, 0.0, 1.0)
        predictions[file_id] = y_pred.tolist()

    output_data = {
        "cols": len(list(predictions.values())[0]) if len(predictions) > 0 else 0,
        "data": predictions
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(json.dumps(output_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict labels using MLP model")
    parser.add_argument('--dataset', type=str, default='data/dataset.json', help='Path to dataset JSON')
    parser.add_argument('--model', type=str, default='data/model.pkl', help='Path to saved model')
    parser.add_argument('--output', type=str, default='data/predictions.json', help='Path to output JSON')

    args = parser.parse_args()

    predict_labels(args.dataset, args.model, args.output)
