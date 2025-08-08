import sys
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from pathlib import Path

np.random.seed(0)

repo_root = Path(__file__).resolve().parents[1]
csv_path = repo_root / "Outputs" / "ball_trajectory.csv"
equation_path = repo_root / "Outputs" / "yx_equation_pixel.txt"
equation_path.parent.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    print(f"CSV not found: {csv_path}")
    sys.exit(1)

def run_symbolic_regression():
    df = pd.read_csv(csv_path)
    X = df[["X"]].values
    y = df["Y"].values

    model = PySRRegressor(
        model_selection="best",
        niterations=100,
        binary_operators=["+", "-", "*", "pow"],
        unary_operators=[],
        elementwise_loss="loss(x, y) = (x - y)^2",
        constraints={"pow": (1, 3)},
        maxsize=20,
        verbosity=0,
    )
    model.fit(X, y)

    expr = model.get_best()["sympy_format"]
    with open(equation_path, "w") as f:
        f.write(f"y = {expr}\n")
    print(f"Wrote equation to {equation_path}")

if __name__ == "__main__":
    run_symbolic_regression()
