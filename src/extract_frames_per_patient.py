import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd

# ----------- CONFIG -----------
VIDEO_DIR = "vidfol"  # folder with patient videos
OUTPUT_DIR = "frames_by_patient"  # output frames root folder
FRAMES_PER_VIDEO = 50  # number of frames to sample per video
METADATA_CSV = "frames_metadata.csv"
# ------------------------------


def get_extensions(folder):
    exts = []
    for f in os.listdir(folder):
        _, e = os.path.splitext(f)
        e = e.lower()
        if e and e not in exts:
            exts.append(e)
    return exts


def list_videos(video_dir, extensions):
    return [
        f
        for f in os.listdir(video_dir)
        if os.path.isfile(os.path.join(video_dir, f))
        and os.path.splitext(f)[1].lower() in extensions
    ]


def frame_indices(cap, frames_per_video):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    return np.linspace(0, total - 1, frames_per_video, dtype=int).tolist()


def extract_frames(video_path, output_dir, frames_per_video):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, []

    indices = frame_indices(cap, frames_per_video)
    saved_files = []
    base = Path(video_path).stem
    patient_id = base.split("_")[0]  # assumes filename starts with patientID_...

    patient_dir = Path(output_dir) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        fname = f"{base}_f{idx:04d}.jpg"
        fpath = patient_dir / fname
        cv2.imwrite(str(fpath), frame)
        saved_files.append(str(fpath))

    cap.release()
    return len(saved_files), saved_files


def batch_extract(video_dir, output_dir, frames_per_video):
    os.makedirs(output_dir, exist_ok=True)
    exts = get_extensions(video_dir)
    videos = list_videos(video_dir, exts)

    all_records = []
    for v in videos:
        vid_path = os.path.join(video_dir, v)
        n, saved = extract_frames(vid_path, output_dir, frames_per_video)
        for f in saved:
            all_records.append({"video": v, "frame_file": f})
        print(f"{v} -> saved {n} frames")
    return pd.DataFrame(all_records)


if __name__ == "__main__":
    df = batch_extract(VIDEO_DIR, OUTPUT_DIR, FRAMES_PER_VIDEO)
    df.to_csv(METADATA_CSV, index=False)
    print(f"\nMetadata saved to {METADATA_CSV}")
