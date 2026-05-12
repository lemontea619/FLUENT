import json
import os
import sys
import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle
import argparse

def train_model(dataset_path, labelset_path, model_path):
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(labelset_path):
        print(f"Error: {labelset_path} not found. Start labeling first.", file=sys.stderr)
        sys.exit(1)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    with open(labelset_path, 'r') as f:
        labelset = json.load(f)

    X = []
    y = []

    labeled_ids = list(labelset['data'].keys())

    for file_id in labeled_ids:
        if file_id in dataset['data']:
            X.append(dataset['data'][file_id])
            y.append(labelset['data'][file_id])

    if len(X) == 0:
        print("Error: No intersecting data between dataset and labelset.", file=sys.stderr)
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)

    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        alpha=0.01,
        max_iter=1000,
        random_state=42,
        early_stopping=False
    )

    mlp.set_params(warm_start=True, max_iter=1)

    prev_loss = float('inf')
    patience = 10
    wait = 0

    for i in range(1000):
        mlp.fit(X, y)
        loss = mlp.loss_

        if abs(prev_loss - loss) < 1e-5:
            wait += 1
            if wait >= patience:
                print(f"Leveled out at epoch {i} with loss {loss:.6f}", file=sys.stderr)
                break
        else:
            wait = 0

        prev_loss = loss

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(mlp, f)
    print(f"Model saved to {model_path}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument('--dataset', type=str, default='data/dataset.json', help='Path to dataset JSON')
    parser.add_argument('--labelset', type=str, default='data/labelset.json', help='Path to labelset JSON')
    parser.add_argument('--model', type=str, default='data/model.pkl', help='Path to save the model')

    args = parser.parse_args()
    train_model(args.dataset, args.labelset, args.model)
