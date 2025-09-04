import sys
import cv2
import pandas as pd
import numpy as np
from sympy import symbols, sympify, lambdify
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
video_path = base_dir / "Data" / "input_video.mp4"
csv_path = base_dir / "Outputs" / "ball_trajectory.csv"
equation_path_orig = base_dir / "Outputs" / "yx_equation_pixel.txt"
equation_path_custom = base_dir / "Outputs" / "yx_equation_pixel_custom.txt"
output_path_orig = base_dir / "Outputs" / "overlay_output.mp4"
output_path_custom = base_dir / "Outputs" / "overlay_output_custom.mp4"

missing = []
for p, name in [
    (video_path, "video"),
    (csv_path, "csv"),
    (equation_path_orig, "equation_orig"),
    (equation_path_custom, "equation_custom"),
]:
    if not p.exists():
        missing.append((name, p))
if missing:
    for name, p in missing:
        print(f" Missing {name}: {p}")
    sys.exit(1)

with open(equation_path_orig) as f:
    rhs_orig = f.read().strip().split("=", 1)[1].strip()
with open(equation_path_custom) as f:
    rhs_custom = f.read().strip().split("=", 1)[1].strip()

x0 = symbols("x0")
expr_orig = sympify(rhs_orig)
expr_custom = sympify(rhs_custom)
f_orig = lambdify(x0, expr_orig, modules=["numpy"])
f_custom = lambdify(x0, expr_custom, modules=["numpy"])

df = pd.read_csv(csv_path)
df = df[df["Time"] >= 1.0].reset_index(drop=True)

x_vals = df["X"].values
try:
    y_orig = f_orig(x_vals)
    y_orig = np.nan_to_num(y_orig, nan=0.0, posinf=0.0, neginf=0.0)
except Exception:
    y_orig = np.zeros_like(x_vals)
try:
    y_custom = f_custom(x_vals)
    y_custom = np.nan_to_num(y_custom, nan=0.0, posinf=0.0, neginf=0.0)
except Exception:
    y_custom = np.zeros_like(x_vals)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"Could not open video: {video_path}")
    sys.exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_orig = cv2.VideoWriter(str(output_path_orig), fourcc, fps, (width, height))
out_custom = cv2.VideoWriter(str(output_path_custom), fourcc, fps, (width, height))

start_frame = int(fps * 1.0)
frame_idx = 0
path_orig = []
path_custom = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_o = frame.copy()
    frame_c = frame.copy()

    if frame_idx >= start_frame:
        pred_idx = frame_idx - start_frame
        if pred_idx < len(x_vals):
            px = int(x_vals[pred_idx])
            py_o = int(np.clip(height - y_orig[pred_idx], 0, height - 1))
            py_c = int(np.clip(height - y_custom[pred_idx], 0, height - 1))
            if 0 <= px < width and 0 <= py_o < height:
                path_orig.append((px, py_o))
            if 0 <= px < width and 0 <= py_c < height:
                path_custom.append((px, py_c))

        for i in range(1, len(path_orig)):
            cv2.line(frame_o, path_orig[i - 1], path_orig[i], (255, 0, 0), 2)
        for i in range(1, len(path_custom)):
            cv2.line(frame_c, path_custom[i - 1], path_custom[i], (0, 255, 0), 2)

    out_orig.write(frame_o)
    out_custom.write(frame_c)
    frame_idx += 1

cap.release()
out_orig.release()
out_custom.release()
print(f"Overlay video saved to {output_path_orig}")
print(f"Overlay video saved to {output_path_custom}")
