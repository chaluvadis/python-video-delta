"""
tests/test_compare_videos.py

Unit tests for compare_videos.py.

Synthetic video files are created in-memory using OpenCV so the tests run
without any external assets.
"""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compare_videos import (
    black_frame,
    build_frame_paths,
    compute_diff_image,
    compare_videos,
    ensure_output_dirs,
    frame_index_to_timestamp,
    frames_dir,
    normalize_frame,
    preprocess_frame,
    timestamp_to_file_prefix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_video(path: str, num_frames: int, fps: float, color: tuple[int, int, int], size=(64, 64)) -> None:
    """Write a solid-colour synthetic video to *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Unit tests – pure functions
# ---------------------------------------------------------------------------

class TestFrameIndexToTimestamp:
    def test_zero(self):
        assert frame_index_to_timestamp(0, 25.0) == "00:00:00.000"

    def test_one_second(self):
        assert frame_index_to_timestamp(25, 25.0) == "00:00:01.000"

    def test_one_minute(self):
        assert frame_index_to_timestamp(1500, 25.0) == "00:01:00.000"

    def test_one_hour(self):
        assert frame_index_to_timestamp(90000, 25.0) == "01:00:00.000"

    def test_fractional_seconds(self):
        # Frame 1 at 30 fps → 33.333… ms
        ts = frame_index_to_timestamp(1, 30.0)
        assert ts == "00:00:00.033"

    def test_format_length(self):
        ts = frame_index_to_timestamp(0, 25.0)
        assert len(ts) == 12  # HH:MM:SS.mmm


class TestTimestampToFilePrefix:
    def test_basic(self):
        assert timestamp_to_file_prefix("00:01:12.000") == "000112_000"

    def test_non_zero_ms(self):
        assert timestamp_to_file_prefix("01:23:45.678") == "012345_678"


class TestBuildFramePaths:
    def test_extensions(self, tmp_path):
        output_dir = str(tmp_path)
        pa, pb, pd = build_frame_paths("00:01:12.000", output_dir)
        assert pa.endswith("_A.jpg")
        assert pb.endswith("_B.jpg")
        assert pd.endswith("_DIFF.jpg")

    def test_prefix_in_path(self, tmp_path):
        output_dir = str(tmp_path)
        pa, pb, _ = build_frame_paths("00:01:12.000", output_dir)
        assert "000112_000" in pa
        assert "000112_000" in pb


class TestNormalizeFrame:
    def test_already_correct_size(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = normalize_frame(frame, (64, 64))
        assert result.shape == (64, 64, 3)

    def test_resize(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        result = normalize_frame(frame, (64, 64))
        assert result.shape == (64, 64, 3)


class TestPreprocessFrame:
    def test_output_is_grayscale(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = preprocess_frame(frame, (64, 64))
        assert result.ndim == 2

    def test_blur_does_not_change_shape(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = preprocess_frame(frame, (64, 64), apply_blur=True)
        assert result.shape == (64, 64)


class TestComputeDiffImage:
    def test_identical_frames_produce_minimal_diff(self):
        gray = np.full((64, 64), 128, dtype=np.uint8)
        diff = compute_diff_image(gray, gray)
        # Identical frames → near-zero diff image
        assert diff.max() == 0

    def test_different_frames_produce_nonzero_diff(self):
        gray_a = np.zeros((64, 64), dtype=np.uint8)
        gray_b = np.full((64, 64), 255, dtype=np.uint8)
        diff = compute_diff_image(gray_a, gray_b)
        assert diff.max() > 0

    def test_output_shape_matches_input(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        diff = compute_diff_image(gray, gray)
        assert diff.shape == (64, 64)


class TestBlackFrame:
    def test_shape(self):
        frame = black_frame((64, 64))
        assert frame.shape == (64, 64, 3)

    def test_all_zeros(self):
        frame = black_frame((32, 32))
        assert frame.sum() == 0


class TestEnsureOutputDirs:
    def test_creates_directories(self, tmp_path):
        output_dir = str(tmp_path / "new_output")
        ensure_output_dirs(output_dir)
        assert os.path.isdir(output_dir)
        assert os.path.isdir(frames_dir(output_dir))

    def test_idempotent(self, tmp_path):
        output_dir = str(tmp_path / "out")
        ensure_output_dirs(output_dir)
        ensure_output_dirs(output_dir)  # Should not raise


# ---------------------------------------------------------------------------
# Integration tests – full compare_videos() pipeline
# ---------------------------------------------------------------------------

class TestCompareVideos:
    """End-to-end tests using synthetic video files."""

    def test_identical_videos_no_differences(self, tmp_path):
        """Two identical videos should produce no difference entries."""
        video_path = str(tmp_path / "same.mp4")
        make_video(video_path, num_frames=10, fps=10.0, color=(100, 100, 100))
        output_dir = str(tmp_path / "output")

        report = compare_videos(
            video_a_path=video_path,
            video_b_path=video_path,
            output_dir=output_dir,
            sample_fps=2,
            ssim_threshold=0.90,
        )

        assert report == []

    def test_different_videos_detect_differences(self, tmp_path):
        """Two clearly different-coloured videos should produce differences."""
        video_a = str(tmp_path / "a.mp4")
        video_b = str(tmp_path / "b.mp4")
        make_video(video_a, num_frames=20, fps=10.0, color=(0, 0, 0))
        make_video(video_b, num_frames=20, fps=10.0, color=(255, 255, 255))
        output_dir = str(tmp_path / "output")

        report = compare_videos(
            video_a_path=video_a,
            video_b_path=video_b,
            output_dir=output_dir,
            sample_fps=2,
            ssim_threshold=0.90,
        )

        assert len(report) > 0

    def test_report_entry_schema(self, tmp_path):
        """Each report entry must contain the required keys."""
        video_a = str(tmp_path / "a.mp4")
        video_b = str(tmp_path / "b.mp4")
        make_video(video_a, num_frames=20, fps=10.0, color=(0, 0, 0))
        make_video(video_b, num_frames=20, fps=10.0, color=(200, 200, 200))
        output_dir = str(tmp_path / "output")

        report = compare_videos(
            video_a_path=video_a,
            video_b_path=video_b,
            output_dir=output_dir,
            sample_fps=2,
            ssim_threshold=0.90,
        )

        required_keys = {"timestamp", "ssim_score", "frame_a_path", "frame_b_path", "diff_frame_path"}
        for entry in report:
            assert required_keys.issubset(entry.keys())
            # ssim_score should be a float between -1 and 1
            assert isinstance(entry["ssim_score"], float)
            assert -1.0 <= entry["ssim_score"] <= 1.0

    def test_report_json_written(self, tmp_path):
        """report.json must be created in the output directory."""
        video_path = str(tmp_path / "v.mp4")
        make_video(video_path, num_frames=10, fps=10.0, color=(50, 50, 50))
        output_dir = str(tmp_path / "output")

        compare_videos(
            video_a_path=video_path,
            video_b_path=video_path,
            output_dir=output_dir,
            sample_fps=2,
        )

        report_path = os.path.join(output_dir, "report.json")
        assert os.path.isfile(report_path)

        with open(report_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)

    def test_frame_images_saved_on_difference(self, tmp_path):
        """Frame images (A, B, DIFF) must be saved when a difference is found."""
        video_a = str(tmp_path / "a.mp4")
        video_b = str(tmp_path / "b.mp4")
        make_video(video_a, num_frames=20, fps=10.0, color=(0, 0, 0))
        make_video(video_b, num_frames=20, fps=10.0, color=(255, 0, 0))
        output_dir = str(tmp_path / "output")

        report = compare_videos(
            video_a_path=video_a,
            video_b_path=video_b,
            output_dir=output_dir,
            sample_fps=2,
            ssim_threshold=0.90,
        )

        for entry in report:
            assert os.path.isfile(entry["frame_a_path"])
            assert os.path.isfile(entry["frame_b_path"])
            assert os.path.isfile(entry["diff_frame_path"])

    def test_different_duration_videos(self, tmp_path):
        """Shorter video ends earlier; remaining frames treated as differences."""
        video_a = str(tmp_path / "long.mp4")   # 30 frames @ 10fps = 3 s
        video_b = str(tmp_path / "short.mp4")  # 10 frames @ 10fps = 1 s
        make_video(video_a, num_frames=30, fps=10.0, color=(128, 128, 128))
        make_video(video_b, num_frames=10, fps=10.0, color=(128, 128, 128))
        output_dir = str(tmp_path / "output")

        report = compare_videos(
            video_a_path=video_a,
            video_b_path=video_b,
            output_dir=output_dir,
            sample_fps=2,
            ssim_threshold=0.90,
        )

        # After video_b ends, remaining frames from video_a are compared against
        # black frames, so they must appear as differences.
        assert len(report) > 0

    def test_blur_option_does_not_crash(self, tmp_path):
        """Passing --blur should not raise any exception."""
        video_path = str(tmp_path / "v.mp4")
        make_video(video_path, num_frames=10, fps=10.0, color=(80, 80, 80))
        output_dir = str(tmp_path / "output")

        compare_videos(
            video_a_path=video_path,
            video_b_path=video_path,
            output_dir=output_dir,
            sample_fps=2,
            apply_blur=True,
        )

    def test_invalid_video_raises(self, tmp_path):
        """Passing a non-existent video path must raise FileNotFoundError."""
        output_dir = str(tmp_path / "output")
        with pytest.raises(FileNotFoundError):
            compare_videos(
                video_a_path="/nonexistent/video_a.mp4",
                video_b_path="/nonexistent/video_b.mp4",
                output_dir=output_dir,
            )

    def test_timestamp_format_in_report(self, tmp_path):
        """Timestamps in the report must follow HH:MM:SS.mmm format."""
        import re
        video_a = str(tmp_path / "a.mp4")
        video_b = str(tmp_path / "b.mp4")
        make_video(video_a, num_frames=20, fps=10.0, color=(0, 0, 0))
        make_video(video_b, num_frames=20, fps=10.0, color=(255, 255, 255))
        output_dir = str(tmp_path / "output")

        report = compare_videos(
            video_a_path=video_a,
            video_b_path=video_b,
            output_dir=output_dir,
            sample_fps=2,
            ssim_threshold=0.90,
        )

        pattern = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}$")
        for entry in report:
            assert pattern.match(entry["timestamp"]), f"Bad timestamp: {entry['timestamp']}"
