import json
import os
import sys
import numpy as np
import pickle

def predict_labels(dataset_path, model_path, output_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.")
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

    print(json.dumps(output_data))

if __name__ == "__main__":
    predict_labels('data/dataset.json', 'data/model.pkl', 'data/predictions.json')
