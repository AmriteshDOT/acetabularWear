import os

IMAGE_DIR = "vidfol"

extensions = []
filenames = os.listdir(IMAGE_DIR)

for filename in filenames:
    name, ext = os.path.splitext(filename)
    if ext not in extensions:
        extensions.append(ext.lower())
##################
import os
import cv2
import numpy as np

# Directories
VIDEO_DIR  = "vidfol"     # folder containing your .mp4/.avi/.mov videos
OUTPUT_DIR = "output"     # where to dump the extracted frames

os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAMES_PER_VIDEO = 50

videos = [
    f for f in os.listdir(VIDEO_DIR)
    if os.path.isfile(os.path.join(VIDEO_DIR, f)) and
       os.path.splitext(f)[1].lower() in extensions
]

for vid_name in videos:
    vid_path = os.path.join(VIDEO_DIR, vid_name)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        continue # skip

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        continue #noframe

    indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        base = os.path.splitext(vid_name)[0]
        out_fname = f"{base}_f{idx:04d}.jpg"
        out_path  = os.path.join(OUTPUT_DIR, out_fname)

        cv2.imwrite(out_path, frame)
        saved += 1

    cap.release()

