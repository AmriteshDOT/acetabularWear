import os
import cv2
from pathlib import Path
import numpy as np
import pandas as pd


def pick_frames(cap, n):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    idxs = np.linspace(0, total - 1, n, dtype=int)
    return list(idxs)


def save_frames(vid_path, out_root, n=50):
    cap = cv2.VideoCapture(vid_path)
    saved = []
    if not cap.isOpened():
        return saved

    idxs = pick_frames(cap, n)
    base = Path(vid_path).stem
    pid = base.split("_")[0]
    out_dir = Path(out_root) / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        name = f"{base}_f{int(i):04d}.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), frame)
        saved.append(str(path))

    cap.release()
    return saved


def extract_all(videos_folder, out_root, n=10):
    os.makedirs(out_root, exist_ok=True)
    files = [
        f
        for f in os.listdir(videos_folder)
        if os.path.isfile(os.path.join(videos_folder, f))
    ]
    recs = []
    for v in files:
        saved = save_frames(os.path.join(videos_folder, v), out_root, n)
        for s in saved:
            recs.append({"video": v, "frame": s})
    return pd.DataFrame(recs)


def main():
    videos_folder = "videos"
    out_root = "frames"
    n = 15
    df = extract_all(videos_folder, out_root, n)
    # print(f"extracted {len(df)} frames : {out_root}")
    df.to_csv(Path(out_root) / "frames_log.csv", index=False)


if __name__ == "__main__":
    main()
