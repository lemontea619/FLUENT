import time
import numpy as np
from sklearn.neural_network import MLPRegressor
import io

X = np.random.rand(1000, 26)
y = np.random.rand(1000, 10)

def train_manual_loop():
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.01,
        max_iter=1000,
        random_state=42,
        early_stopping=False,
    )
    mlp.set_params(warm_start=True, max_iter=1)

    prev_loss = float("inf")
    patience = 10
    wait = 0
    start_time = time.time()

    for i in range(1000):
        mlp.fit(X, y)
        loss = mlp.loss_
        if abs(prev_loss - loss) < 1e-5:
            wait += 1
            if wait >= patience:
                break
        else:
            wait = 0
        prev_loss = loss

    return time.time() - start_time, i, mlp.loss_

class DummyWriter(io.StringIO):
    def write(self, s):
        pass

def train_sklearn_builtin():
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.01,
        max_iter=1000,
        random_state=42,
        early_stopping=False,
        tol=1e-5,
        n_iter_no_change=10,
        verbose=True
    )
    import sys
    old_stdout = sys.stdout
    sys.stdout = DummyWriter()

    start_time = time.time()
    mlp.fit(X, y)
    end_time = time.time()

    sys.stdout = old_stdout

    return end_time - start_time, mlp.n_iter_, mlp.loss_

manual_times = []
builtin_times = []

# Warmup
train_manual_loop()
train_sklearn_builtin()

for _ in range(5):
    t, iters, loss = train_manual_loop()
    manual_times.append(t)

for _ in range(5):
    t, iters, loss = train_sklearn_builtin()
    builtin_times.append(t)

print(f"Manual Loop: {np.mean(manual_times):.4f} seconds (std: {np.std(manual_times):.4f})")
print(f"Built-in: {np.mean(builtin_times):.4f} seconds (std: {np.std(builtin_times):.4f})")
print(f"Speedup: {np.mean(manual_times)/np.mean(builtin_times):.2f}x")
