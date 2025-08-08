import sys
import cv2
import pandas as pd
import numpy as np
from sympy import symbols, sympify, lambdify
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
video_path = base_dir / "Data" / "input_video.mp4"
csv_path = base_dir / "Outputs" / "ball_trajectory.csv"
equation_path = base_dir / "Outputs" / "yx_equation_pixel.txt"
output_path = base_dir / "Outputs" / "overlay_output.mp4"

missing = []
for p, name in [(video_path, "video"), (csv_path, "csv"), (equation_path, "equation")]:
    if not p.exists():
        missing.append((name, p))
if missing:
    for name, p in missing:
        print(f" Missing {name}: {p}")
    sys.exit(1)

with open(equation_path) as f:
    rhs = f.read().strip().split("=", 1)[1].strip()

x0 = symbols("x0")
expr = sympify(rhs)
f_sym = lambdify(x0, expr, modules=["numpy"])

df = pd.read_csv(csv_path)
df = df[df["Time"] >= 1.0].reset_index(drop=True)

x_vals = df["X"].values
try:
    y_vals = f_sym(x_vals)
    y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=0.0, neginf=0.0)
except Exception as e:
    print(f"Error evaluating symbolic equation:", e)
    y_vals = np.zeros_like(x_vals)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"Could not open video: {video_path}")
    sys.exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

start_frame = int(fps * 1.0)
frame_idx = 0
predicted_path = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx >= start_frame:
        pred_idx = frame_idx - start_frame
        if pred_idx < len(x_vals):
            px = int(x_vals[pred_idx])
            flipped_y = height - y_vals[pred_idx]
            py = int(np.clip(flipped_y, 0, height - 1))
            if 0 <= px < width and 0 <= py < height:
                predicted_path.append((px, py))

        for i in range(1, len(predicted_path)):
            cv2.line(frame, predicted_path[i - 1], predicted_path[i], (255, 0, 0), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"Overlay video saved to {output_path}")
