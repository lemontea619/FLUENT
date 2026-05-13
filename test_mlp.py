import sys
import json
from sklearn.neural_network import MLPRegressor
import numpy as np

class VerboseCapture:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, text):
        if text.startswith("Iteration "):
            # "Iteration 1, loss = 0.73030389"
            try:
                parts = text.split(", loss = ")
                epoch = int(parts[0].replace("Iteration ", "")) - 1 # 0-indexed
                loss = float(parts[1].strip())
                self.original_stdout.write(json.dumps({"epoch": epoch, "loss": loss}) + "\n")
                self.original_stdout.flush()
            except Exception as e:
                pass
        elif text.startswith("Training loss did not improve"):
            pass

    def flush(self):
        self.original_stdout.flush()

X = np.random.rand(100, 26)
y = np.random.rand(100, 10)

mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=20, verbose=True, random_state=42)

old_stdout = sys.stdout
sys.stdout = VerboseCapture(old_stdout)

mlp.fit(X, y)

sys.stdout = old_stdout
