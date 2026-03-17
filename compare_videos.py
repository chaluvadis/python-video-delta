"""
compare_videos.py - Compare two video files and detect visual differences.

Usage:
    python compare_videos.py <video_a> <video_b> [--output ./output] [--fps 2] [--threshold 0.90] [--blur]
"""

import argparse
import json
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_FPS = 2
DEFAULT_SSIM_THRESHOLD = 0.90
DEFAULT_OUTPUT_DIR = "./output"
FRAMES_SUBDIR = "frames"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frames_dir(output_dir: str) -> str:
    return os.path.join(output_dir, FRAMES_SUBDIR)


def ensure_output_dirs(output_dir: str) -> None:
    """Create output and frames directories if they do not exist."""
    os.makedirs(frames_dir(output_dir), exist_ok=True)


def frame_index_to_timestamp(frame_index: int, video_fps: float) -> str:
    """Convert a frame index (in the *original* video) to HH:MM:SS.sss."""
    total_ms = (frame_index / video_fps) * 1000
    hours = int(total_ms // 3_600_000)
    remaining = total_ms % 3_600_000
    minutes = int(remaining // 60_000)
    remaining %= 60_000
    seconds = int(remaining // 1_000)
    ms = int(remaining % 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def timestamp_to_file_prefix(timestamp: str) -> str:
    """
    Convert 'HH:MM:SS.sss' to a filename-safe prefix.

    Example: '00:01:12.000' -> '000112_000'
    """
    # Remove colons and dot separators
    clean = timestamp.replace(":", "").replace(".", "_")
    # clean is now 'HHMMSS_mmm', keep it as-is
    return clean


def build_frame_paths(timestamp: str, output_dir: str) -> tuple[str, str, str]:
    """Return (path_A, path_B, path_DIFF) for a given timestamp."""
    prefix = timestamp_to_file_prefix(timestamp)
    fd = frames_dir(output_dir)
    return (
        os.path.join(fd, f"{prefix}_A.jpg"),
        os.path.join(fd, f"{prefix}_B.jpg"),
        os.path.join(fd, f"{prefix}_DIFF.jpg"),
    )


def normalize_frame(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize frame to target_size (width, height) if needed."""
    h, w = frame.shape[:2]
    if (w, h) != target_size:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    return frame


def preprocess_frame(
    frame: np.ndarray,
    target_size: tuple[int, int],
    apply_blur: bool = False,
) -> np.ndarray:
    """Normalize, convert to grayscale, and optionally apply Gaussian blur."""
    frame = normalize_frame(frame, target_size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if apply_blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def compute_diff_image(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    """
    Generate a thresholded diff image using the SSIM difference map.

    Returns an 8-bit single-channel image where differences are white.
    """
    _, diff = ssim(gray_a, gray_b, full=True)
    # diff values are in [-1, 1]; convert to [0, 255]
    diff_uint8 = (np.clip(1.0 - diff, 0.0, 1.0) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(diff_uint8, 25, 255, cv2.THRESH_BINARY)
    return thresh


def save_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    diff_img: np.ndarray,
    path_a: str,
    path_b: str,
    path_diff: str,
) -> None:
    """Write frame images to disk."""
    cv2.imwrite(path_a, frame_a)
    cv2.imwrite(path_b, frame_b)
    cv2.imwrite(path_diff, diff_img)


# ---------------------------------------------------------------------------
# Black-frame generation for missing frames (different duration videos)
# ---------------------------------------------------------------------------

def black_frame(size: tuple[int, int]) -> np.ndarray:
    """Return a solid black BGR frame of given (width, height)."""
    w, h = size
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def compare_videos(
    video_a_path: str,
    video_b_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    ssim_threshold: float = DEFAULT_SSIM_THRESHOLD,
    apply_blur: bool = False,
) -> list[dict]:
    """
    Compare two video files frame-by-frame.

    Returns a list of report entries (dicts) for frames where SSIM < threshold.
    Also writes report.json to output_dir and saves frame images.
    """
    ensure_output_dirs(output_dir)

    cap_a = cv2.VideoCapture(video_a_path)
    cap_b = cv2.VideoCapture(video_b_path)

    if not cap_a.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_a_path}")
    if not cap_b.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_b_path}")

    fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 25.0
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 25.0

    total_frames_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))

    width_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use video A's resolution as the canonical size
    target_size = (width_a, height_a)

    # Sampling: step in frames (based on video A's FPS)
    step_a = max(1, round(fps_a / sample_fps))
    step_b = max(1, round(fps_b / sample_fps))

    # Duration in seconds
    duration_a = total_frames_a / fps_a if fps_a > 0 else 0
    duration_b = total_frames_b / fps_b if fps_b > 0 else 0
    max_duration = max(duration_a, duration_b)

    report: list[dict] = []

    sample_index = 0  # which sample we are on (0-based)

    while True:
        # Compute the position in original video frames for this sample
        frame_pos_a = sample_index * step_a
        frame_pos_b = sample_index * step_b

        time_a = frame_pos_a / fps_a if fps_a > 0 else 0

        # Stop if we have passed the longer video
        if time_a > max_duration:
            break

        # --- Read frame from video A ---
        if frame_pos_a < total_frames_a:
            cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_pos_a)
            ret_a, raw_a = cap_a.read()
            if not ret_a:
                raw_a = None
        else:
            raw_a = None

        # --- Read frame from video B ---
        if frame_pos_b < total_frames_b:
            cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_pos_b)
            ret_b, raw_b = cap_b.read()
            if not ret_b:
                raw_b = None
        else:
            raw_b = None

        # If both videos have ended, we are done
        if raw_a is None and raw_b is None:
            break

        # Handle one video ending earlier — treat as maximum difference
        if raw_a is None:
            raw_a = black_frame(target_size)
        if raw_b is None:
            raw_b = black_frame(target_size)

        # Build timestamp from video A's frame position
        timestamp = frame_index_to_timestamp(frame_pos_a, fps_a)

        # Preprocess
        gray_a = preprocess_frame(raw_a, target_size, apply_blur)
        gray_b = preprocess_frame(raw_b, target_size, apply_blur)

        # Compute SSIM
        ssim_score, _ = ssim(gray_a, gray_b, full=True)
        ssim_score = float(ssim_score)

        if ssim_score < ssim_threshold:
            path_a, path_b, path_diff = build_frame_paths(timestamp, output_dir)

            # Generate diff image
            diff_img = compute_diff_image(gray_a, gray_b)

            # Save frames (original colour frame for A and B)
            frame_a_display = normalize_frame(raw_a, target_size)
            frame_b_display = normalize_frame(raw_b, target_size)
            save_frames(frame_a_display, frame_b_display, diff_img, path_a, path_b, path_diff)

            report.append(
                {
                    "timestamp": timestamp,
                    "ssim_score": round(ssim_score, 6),
                    "frame_a_path": path_a,
                    "frame_b_path": path_b,
                    "diff_frame_path": path_diff,
                }
            )

        sample_index += 1

    cap_a.release()
    cap_b.release()

    # Write JSON report
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"Comparison complete. {len(report)} difference(s) found.")
    print(f"Report saved to: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two video files and detect visual differences."
    )
    parser.add_argument("video_a", help="Path to the first video file.")
    parser.add_argument("video_b", help="Path to the second video file.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_SAMPLE_FPS,
        help=f"Sampling rate in frames per second (default: {DEFAULT_SAMPLE_FPS}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SSIM_THRESHOLD,
        help=f"SSIM threshold below which a frame is marked different (default: {DEFAULT_SSIM_THRESHOLD}).",
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        help="Apply a small Gaussian blur before comparison to reduce noise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_videos(
        video_a_path=args.video_a,
        video_b_path=args.video_b,
        output_dir=args.output,
        sample_fps=args.fps,
        ssim_threshold=args.threshold,
        apply_blur=args.blur,
    )


if __name__ == "__main__":
    main()
