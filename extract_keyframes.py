"""
=============================================================================
FAST 4K KEYFRAME EXTRACTOR
Uses direct seeking to extract 60 sharp keyframes in seconds from 4K videos
=============================================================================
"""

import cv2
import os
from pathlib import Path

def extract_fast_keyframes(video_path: str, output_dir: str, target_count: int = 60, target_width: int = 1280):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = orig_h / orig_w
    target_height = int(target_width * aspect_ratio)

    print(f"\n[Extracting] {Path(video_path).name}")
    print(f"Native: {orig_w}x{orig_h} ({total_frames} frames) -> Output: {target_width}x{target_height}")

    # Calculate frame step indices across the entire video
    step = total_frames / (target_count + 1)
    frame_indices = [int(step * (i + 1)) for i in range(target_count)]

    saved_files = []
    for idx, f_pos in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_pos)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Downscale to 720p/1080p
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        out_file = output_path / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(out_file), resized, [cv2.IMWRITE_JPEG_QUALITY, 94])
        saved_files.append(str(out_file))

    cap.release()
    print(f" Successfully extracted {len(saved_files)} keyframes to: {output_dir}")
    return saved_files

if __name__ == "__main__":
    base_dir = "d:/Vr-project/videos"
    
    # 1. The Great Emei Mountain
    extract_fast_keyframes(
        os.path.join(base_dir, "THE GREAT EMEI MOUNTAIN.webm"),
        "d:/Vr-project/data/emei_frames",
        target_count=60
    )

    # 2. Pelling (West Sikkim)
    extract_fast_keyframes(
        os.path.join(base_dir, "Pelling.webm"),
        "d:/Vr-project/data/pelling_frames",
        target_count=60
    )

    # 3. India Monastery
    extract_fast_keyframes(
        os.path.join(base_dir, "india_monestry.webm"),
        "d:/Vr-project/data/india_monastery_frames",
        target_count=50
    )
