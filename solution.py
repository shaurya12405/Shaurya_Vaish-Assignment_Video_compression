"""
video_compression.py
Sentio Mind · Project 2 · Smart Behavioral Video Compression

Copy this file to solution.py and fill in every TODO block.
Do not rename any function.
Run: python solution.py
Requires ffmpeg installed on your system: sudo apt install ffmpeg
"""

import cv2
import json
import base64
import subprocess
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_IN               = Path("video_sample_1.mov")
VIDEO_OUT              = Path("compressed_output.mp4")
REPORT_HTML_OUT        = Path("compression_report.html")
SEGMENTS_JSON_OUT      = Path("segments_kept.json")

PHASH_THRESHOLD        = 0.95   # similarity above this = near-duplicate, discard
MOTION_KEEP_THRESH     = 0.15   # keep frame if motion exceeds this (no face needed)
MOTION_DISCARD_THRESH  = 0.05   # definitely discard below this
CONTEXT_EVERY_SEC      = 3      # force-keep one frame every this many seconds
OUTPUT_FPS             = 12     # frame rate of the output video
OUTPUT_CRF             = 28     # ffmpeg quality: lower = better quality + larger file
# ---------------------------------------------------------------------------
# PERCEPTUAL HASH
# ---------------------------------------------------------------------------

def compute_phash(frame: np.ndarray) -> str:
    """
    Compute a perceptual hash of the frame.
    Steps: resize to 32×32 grayscale → DCT → threshold at mean → flatten to bit string.
    Return a string of '0' and '1' characters, length 64.

    You can use the imagehash library (imagehash.phash) or implement manually.
    TODO: implement
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))

    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:8, :8]

    mean_val = np.mean(dct_low)
    hash_bits = (dct_low > mean_val).flatten()

    return ''.join(['1' if b else '0' for b in hash_bits])


def phash_similarity(h1: str, h2: str) -> float:
    """
    Compare two hash strings. Return 1.0 if identical, 0.0 if completely different.
    Formula: 1.0 - (hamming_distance / length)
    TODO: implement
    """
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0

    hamming = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    return 1.0 - (hamming / len(h1))


# ---------------------------------------------------------------------------
# MOTION SCORE
# ---------------------------------------------------------------------------

def compute_motion_score(prev_gray, curr_gray: np.ndarray) -> float:
    """
    Dense optical flow between two grayscale frames. Return mean magnitude, ~0.0-1.0.
    If prev_gray is None, return 0.0.
    TODO: cv2.calcOpticalFlowFarneback
    Params: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2
    """
    if prev_gray is None:
        return 0.0

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


# ---------------------------------------------------------------------------
# FACE PRESENCE CHECK
# ---------------------------------------------------------------------------

def has_face(frame: np.ndarray, cascade) -> bool:
    """
    True if at least one face detected. Use the Haar cascade passed in.
    Equalise histogram on grayscale first for better CCTV detection.
    TODO: cascade.detectMultiScale — scaleFactor=1.1, minNeighbors=3, minSize=(20,20)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20)
    )

    return len(faces) > 0


# ---------------------------------------------------------------------------
# FRAME KEEP DECISION
# ---------------------------------------------------------------------------

def should_keep_frame(frame: np.ndarray,
                      prev_frame,
                      prev_kept_hash: str,
                      last_kept_time_sec: float,
                      current_time_sec: float,
                      cascade) -> tuple:
    """
    Apply the 5-step decision algorithm from README in order.
    Return: (keep: bool, reason: str, motion_score: float, face_found: bool)

    Reason strings (use exactly these):
      'face_detected', 'motion_above_threshold', 'context_frame',
      'face_and_motion', 'discarded_duplicate', 'discarded_static'

    TODO: implement
    """
    curr_hash = compute_phash(frame)

    # Step 1: pHash duplicate removal
    if prev_kept_hash:
        similarity = phash_similarity(curr_hash, prev_kept_hash)
        if similarity > PHASH_THRESHOLD:
            return False, "discarded_duplicate", 0.0, False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = None if prev_frame is None else cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    motion_score = compute_motion_score(prev_gray, gray)
    face_found = has_face(frame, cascade)

    # Step 3: Face override
    if face_found and motion_score >= MOTION_KEEP_THRESH:
        return True, "face_and_motion", motion_score, True

    if face_found:
        return True, "face_detected", motion_score, True

    # Step 2: Motion-based decision
    if motion_score >= MOTION_KEEP_THRESH:
        return True, "motion_above_threshold", motion_score, False

    if motion_score < MOTION_DISCARD_THRESH:
        # Step 4: Context frame
        if current_time_sec - last_kept_time_sec >= CONTEXT_EVERY_SEC:
            return True, "context_frame", motion_score, False
        return False, "discarded_static", motion_score, False

    return True, "motion_above_threshold", motion_score, False


# ---------------------------------------------------------------------------
# VIDEO WRITING
# ---------------------------------------------------------------------------

def write_frames_to_video(kept_frames: list, output_path: Path,
                          fps: float, frame_size: tuple):
    """
    Write kept_frames to a temporary file, then re-encode with ffmpeg to H.264 MP4.

    Steps:
      1. Write to temp_raw.avi using cv2.VideoWriter (mp4v codec)
      2. Call ffmpeg: ffmpeg -y -i temp_raw.avi -vcodec libx264 -crf CRF -preset fast out.mp4
      3. Delete temp_raw.avi

    TODO: implement
    """
    temp_path = "temp_raw.avi"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, frame_size)

    for frame in kept_frames:
        out.write(frame)

    out.release()

    command = [
        "ffmpeg", "-y",
        "-i", temp_path,
        "-vcodec", "libx264",
        "-crf", str(OUTPUT_CRF),
        "-preset", "fast",
        str(output_path)
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    Path(temp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_compression_report(segments: list, stats: dict, output_path: Path):
    """
    Write a self-contained HTML file showing:
      - Original vs compressed size (MB and % reduction)
      - Original vs compressed duration (seconds)
      - Processing time
      - Storyboard grid: one thumbnail per segment
      - Frames kept vs discarded count

    No CDN. Inline CSS only. Must work offline.
    TODO: implement
    """
    html = f"""
    <html>
    <head>
    <style>
    body {{ font-family: Arial; margin: 20px; }}
    .grid {{ display: flex; flex-wrap: wrap; }}
    .card {{ margin: 10px; }}
    img {{ border-radius: 8px; }}
    </style>
    </head>
    <body>

    <h1>Compression Report</h1>

    <p><b>Original Size:</b> {stats['original_size_mb']} MB</p>
    <p><b>Compressed Size:</b> {stats['compressed_size_mb']} MB</p>
    <p><b>Reduction:</b> {stats['reduction_pct']}%</p>

    <p><b>Original Duration:</b> {stats['original_duration_sec']} sec</p>
    <p><b>Compressed Duration:</b> {stats['compressed_duration_sec']} sec</p>

    <p><b>Processing Time:</b> {stats['processing_time_sec']} sec</p>

    <p><b>Frames Kept:</b> {stats['frames_kept']}</p>
    <p><b>Frames Discarded:</b> {stats['frames_discarded_reasons']['total_discarded']}</p>

    <h2>Storyboard</h2>
    <div class="grid">
    """

    for seg in segments:
        html += f"""
        <div class="card">
        <img src="data:image/jpeg;base64,{seg['thumbnail_b64']}" />
        <p>Segment {seg['segment_id']}</p>
        </div>
        """

    html += "</div></body></html>"

    with open(output_path, "w") as f:
        f.write(html)
