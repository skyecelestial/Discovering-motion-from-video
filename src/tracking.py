import cv2
import numpy as np
import pandas as pd
from pathlib import Path

SHOW = False

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "Data"
output_path = project_root / "Outputs"
output_path.mkdir(parents=True, exist_ok=True)

input_video = data_path / "input_video.mp4"
output_video = output_path / "tracked_output.mp4"
output_csv = output_path / "ball_trajectory.csv"

if not input_video.exists():
    print(f"❌ Video file not found at: {input_video}")
    exit(1)

cap = cv2.VideoCapture(str(input_video))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

positions = []
trail = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([1, 100, 60])
    upper_red = np.array([6, 180, 120])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(biggest)
        if r > 5:
            x, y = int(x), int(y)
            flipped_y = height - y
            time = frame_idx / fps
            positions.append((time, x, flipped_y))
            trail.append((x, y))
            cv2.circle(frame, (x, y), int(r), (0, 255, 0), 2)

    for i in range(1, len(trail)):
        cv2.line(frame, trail[i - 1], trail[i], (0, 0, 255), 2)

    if SHOW:
        cv2.imshow("tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
if SHOW:
    cv2.destroyAllWindows()

df = pd.DataFrame(positions, columns=["Time", "X", "Y"])
df.to_csv(output_csv, index=False)
