# Ego-Trajectory from a Fixed Traffic Light

This solution estimates the autonomous car's 2-D ground-frame trajectory from the apparent motion of a fixed traffic light. It produces the required `trajectory.png` and `trajectory.mp4`, plus a frame-by-frame CSV for inspection.

## Method

For each bounding-box row, I sample a 9 x 9 XYZ patch around the traffic-light center. I discard NaN, zero, and robust range outliers, then take the component-wise median. Missing frames are linearly interpolated; a Hampel filter removes temporal spikes and a 15-frame (0.5-second) moving average reduces stereo-depth jitter. Both the documented `points` arrays `(H,W,3)` and the released `xyz` arrays `(H,W,4)` are supported; the unused fourth channel is ignored.

Camera coordinates are normalized to `(X forward, Y right, Z up)`. The released arrays behave as positive-left despite the written positive-right specification, so the implementation detects lateral sign from the relationship between pixel position and measured Y. It then flips lateral sign to obtain `(X forward, Y left)`, rotates all measurements so the initial car-to-light vector defines world `+X`, and negates the fixed landmark position to recover the car position relative to the traffic light at the origin.

The single-landmark formulation cannot independently recover camera yaw, so I assume heading changes are small during the 10-second sequence. This limitation is explicit rather than estimating unobservable yaw.

## Results

The released sequence contains 299 frames (9.97 s). Four zero bounding boxes were interpolated. The estimated car position begins 38.75 m from the traffic-light origin and ends at `(-7.29, 3.91)` m, following a smooth curved path of approximately 34.34 m. A 15-frame smoothing window reduced implausible stereo-depth velocity spikes while preserving the path shape.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python trajectory.py --dataset dataset --output outputs
```

Expected layout: `dataset/bbox_light.csv` (or `bboxes_light.csv`) and `dataset/xyz/*.npz`. Both documented CSV columns (`frame_id,x_min,y_min,x_max,y_max`) and the released dataset's columns (`frame,x1,y1,x2,y2`) are supported. Rows containing `0,0,0,0` are treated as missing detections and interpolated. The program also tolerates alternate filename prefixes and zero padding by matching the numeric frame suffix. `ffmpeg` must be installed for MP4 encoding (`brew install ffmpeg` on macOS).

Outputs are `outputs/trajectory.png`, `outputs/trajectory.mp4`, and `outputs/trajectory.csv`. Tests run with `pytest -q`.
