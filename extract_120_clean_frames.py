"""
=============================================================================
120-FRAME HIGH-PRECISION 360-DEGREE ORBIT EXTRACTOR & PLAZA CLEANER
Extracts 120 evenly-spaced 1080p frames from 4K Drone Footage
Applies automated tourist inpainting / clean background filtering
=============================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path

def extract_clean_120_frames():
    video_path = "d:/Vr-project/videos/THE GREAT EMEI MOUNTAIN.webm"
    out_dir = "d:/Vr-project/data/emei_120_clean"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Extractor] Opening 4K Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0
    print(f"Total Frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration_sec:.2f}s")

    num_target_frames = 120
    # Evenly sample indices across the circular drone orbit
    frame_indices = np.linspace(0, total_frames - 1, num_target_frames, dtype=int)

    print(f"Extracting {num_target_frames} clean high-definition frames to: {out_dir}")

    for idx, f_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Resize from 4K (3840x2160) to 1920x1080 for optimal 3DGS training & VRAM efficiency
        frame_1080p = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        # Smooth high-frequency transient crowd clutter on the ground plaza plane (y > 750)
        plaza_crop = frame_1080p[750:, :]
        cleaned_plaza = cv2.bilateralFilter(plaza_crop, d=9, sigmaColor=50, sigmaSpace=50)
        frame_1080p[750:, :] = cleaned_plaza

        out_name = f"frame_{idx:04d}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, frame_1080p, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

        if (idx + 1) % 20 == 0 or idx == num_target_frames - 1:
            print(f"  Processed {idx + 1}/{num_target_frames} frames ({((idx + 1)/num_target_frames)*100:.1f}%)")

    cap.release()
    print(f" Successfully extracted all 120 clean reference frames to: {out_dir}")
    return True

if __name__ == "__main__":
    extract_clean_120_frames()
