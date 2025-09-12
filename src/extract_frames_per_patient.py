import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd

videos_dir = "vidfol"                   
frames_root = "frames_by_patient"       
frames_per_video = 50                    
metadata_file = "frames_metadata.csv"   

def get_exts(folder):
    exts = []
    for f in os.listdir(folder):
        _, e = os.path.splitext(f)
        e = e.lower()
        if e and e not in exts:
            exts.append(e)
    return exts

def list_vids(folder, exts):
    return [
        f
        for f in sorted(os.listdir(folder))
        if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in exts
    ]

def pick_frames(cap, count):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    return np.linspace(0, total - 1, count, dtype=int).tolist()

def save_frames(video_path, out_root, count):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, []

    indices = pick_frames(cap, count)
    saved = []
    base = Path(video_path).stem
    patient_id = base.split("_")[0]

    patient_folder = Path(out_root) / patient_id
    patient_folder.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        fname = f"{base}_f{int(idx):04d}.jpg"
        fpath = patient_folder / fname
        cv2.imwrite(str(fpath), frame)
        saved.append(str(fpath))

    cap.release()
    return len(saved), saved

def extract_all_frames(videos_folder, out_root, count):
    os.makedirs(out_root, exist_ok=True)
    exts = get_exts(videos_folder)
    vids = list_vids(videos_folder, exts)

    records = []
    for v in vids:
        n, saved_list = save_frames(os.path.join(videos_folder, v), out_root, count)
        for f in saved_list:
            records.append({"video": v, "frame_path": f})
        # print(f"{v}: saved {n} frames")
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = extract_all_frames(videos_dir, frames_root, frames_per_video)
    df.to_csv(metadata_file, index=False)
    # print(f"{len(df)} frames")
    # print(f" : {metadata_file}")
