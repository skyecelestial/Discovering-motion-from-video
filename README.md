# Discovering Motion from Video

A Python project for tracking a coloured object in a video, learning a closed‑form equation of its trajectory via symbolic regression, and overlaying the predicted path back onto the original video.

## Features

- **Ball tracking** – Uses OpenCV to detect a red ball in each video frame and records its pixel coordinates and timestamps.
- **Symbolic regression** – Employs the PySR library to learn a mathematical relationship \(y = f(x)\) between the ball’s horizontal and vertical positions.
- **Overlay visualisation** – Overlays the predicted trajectory onto the original video, producing a combined output video.
- **End‑to‑end pipeline** – Run all steps automatically with a single command.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/skyecelestial/Discovering-motion-from-video.git
   cd Discovering-motion-from-video
   ```

2. Create a virtual environment (recommended) and activate it:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .\\.venv\\Scripts\\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

The project requires **Python 3.8** or later. The main dependencies are OpenCV, numpy, pandas, sympy and PySR.

## Usage

1. **Prepare a video:** Place your input video in the `Data` directory and name it `input_video.mp4`. The default scripts assume the object to track is red; adjust the colour thresholds in `src/tracking.py` if you are tracking a different colour.

2. **Run the tracking step:** This will detect the object and record its positions and create a video with the tracked path.

   ```bash
   python src/tracking.py
   ```

3. **Learn the equation:** Perform symbolic regression on the recorded positions to discover a closed‑form trajectory equation.

   ```bash
   python src/real_data.py
   ```

4. **Overlay the prediction:** Draw the predicted path on the original video.

   ```bash
   python src/overlay.py
   ```

5. **Run everything at once:** You can run all of the above steps with the pipeline script:

   ```bash
   python src/pipeline.py
   ```

After running the pipeline, you will find:

- `Outputs/ball_trajectory.csv` – CSV file with time, X and Y pixel positions.
- `Outputs/yx_equation_pixel.txt` – the learned equation in the form `y = f(x)`.
- `Outputs/overlay_output.mp4` – the original video with the predicted path drawn on top.

## Directory Structure

```
.
├── Data/                 # Input videos (place your video here as input_video.mp4)
├── Outputs/              # Generated outputs (CSV, equation, overlay video)
├── src/                  # Source code
│   ├── tracking.py       # Tracks the object and saves positions and a tracked video
│   ├── real_data.py      # Runs symbolic regression on the positions
│   ├── overlay.py        # Overlays the predicted path onto the original video
│   └── pipeline.py       # Runs the full pipeline
├── requirements.txt      # List of Python dependencies
└── README.md             # Project documentation
```

## How it works

1. **Tracking** (`src/tracking.py`): Using OpenCV, each frame is converted to HSV, a red mask is applied and the largest contour is used to determine the ball’s position and radius. The script records the time, X and flipped Y coordinates (origin at the bottom of the frame) and writes them to `Outputs/ball_trajectory.csv`. It also outputs a video showing the detected ball and its trail.
2. **Symbolic Regression** (`src/real_data.py`): The PySR library is used to fit a symbolic model `y = f(x)` using the recorded positions. The best expression is saved to `Outputs/yx_equation_pixel.txt`.
3. **Overlay** (`src/overlay.py`): The learned equation is converted to a SymPy expression, evaluated at each x‑coordinate and drawn on the frames of the original video to produce `Outputs/overlay_output.mp4`.

## Customising

- **Colour detection:** To track an object of a different colour, modify the `lower_red` and `upper_red` HSV bounds in `src/tracking.py`.
- **Symbolic model complexity:** You can adjust the number of iterations, allowed operators and constraints in `src/real_data.py` to control the complexity and quality of the learned equation.
- **Input/output paths:** By default the scripts assume `Data/input_video.mp4` and write outputs to `Outputs/`. For more flexibility, modify the paths or refactor the scripts to accept command‑line arguments.

## Example Output

Below is an example GIF showing the predicted path overlaid on the original video. The blue trajectory is the model’s prediction based on the learned equation:

![Predicted trajectory overlay](overlay_output.gif)

## Contributing

Contributions, issues and feature requests are welcome. If you find a bug or have a suggestion, please open an issue or submit a pull request.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full license text and details.
