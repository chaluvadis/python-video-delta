# python-video-delta

A Python 3 script that compares two video files and detects visual differences using Structural Similarity Index (SSIM).

## Requirements

- Python 3.12 or higher
- [opencv-python](https://pypi.org/project/opencv-python/) ≥ 4.8
- [scikit-image](https://pypi.org/project/scikit-image/) ≥ 0.21
- [numpy](https://pypi.org/project/numpy/) ≥ 1.24

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python compare_videos.py <video_a> <video_b> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `video_a` | Path to the first (reference) video file |
| `video_b` | Path to the second (comparison) video file |
| `--output` | Output directory (default: `./output`) |
| `--fps` | Sampling rate in frames per second (default: `2`) |
| `--threshold` | SSIM threshold; frames below this are marked different (default: `0.90`) |
| `--blur` | Apply a small Gaussian blur before comparison to reduce noise |

### Example

```bash
python compare_videos.py reference.mp4 candidate.mp4 --output ./output --fps 2 --threshold 0.90 --blur
```

## Output

```
output/
├── frames/
│   ├── 000112_000_A.jpg      # Frame from video A at the differing timestamp
│   ├── 000112_000_B.jpg      # Corresponding frame from video B
│   └── 000112_000_DIFF.jpg   # Thresholded SSIM difference map
└── report.json               # JSON report of all detected differences
```

### report.json structure

```json
[
  {
    "timestamp": "00:01:12.000",
    "ssim_score": 0.712345,
    "frame_a_path": "output/frames/000112_000_A.jpg",
    "frame_b_path": "output/frames/000112_000_B.jpg",
    "diff_frame_path": "output/frames/000112_000_DIFF.jpg"
  }
]
```

## How it works

1. **Preprocessing** – Both videos are opened with OpenCV. Frames from video B are resized to match video A's resolution. Frames are sampled at the requested rate (default 2 FPS), converted to grayscale, and optionally Gaussian-blurred.
2. **Comparison** – SSIM is computed for each sampled frame pair. Any pair whose SSIM score falls below the threshold (default 0.90) is flagged as a difference.
3. **Diff image** – A thresholded SSIM difference map is generated and saved alongside the two original frames.
4. **Edge cases** – If one video ends before the other, the missing frames are treated as black (solid-zero) frames, which will almost certainly trigger a difference.

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```