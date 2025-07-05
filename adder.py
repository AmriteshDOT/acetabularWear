import os
import shutil

SRC_DIRS = ['unanno', 'output']

DEST_DIR = 'combined'
os.makedirs(DEST_DIR, exist_ok=True)

for src in SRC_DIRS:
    for fname in os.listdir(src):
        src_path = os.path.join(src, fname)
        if os.path.isfile(src_path):
            dest_path = os.path.join(DEST_DIR, fname)
            shutil.copy(src_path, dest_path)

print(f"All images from {SRC_DIRS} have been copied into {DEST_DIR}/")
############
import os
import random
from PIL import Image, ImageEnhance

IMAGE_DIR = "combined"

# Make sure the folder exists
if not os.path.isdir(IMAGE_DIR):
    raise RuntimeError(f"Folder {IMAGE_DIR!r} not found")

filenames = [f for f in os.listdir(IMAGE_DIR)
             if os.path.isfile(os.path.join(IMAGE_DIR, f))]

for fname in filenames:
    base, ext = os.path.splitext(fname)
    src_path = os.path.join(IMAGE_DIR, fname)
    try:
        img = Image.open(src_path).convert("RGB")
    except Exception as e:
        continue

    # Horizontal flip
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped.save(os.path.join(IMAGE_DIR, f"{base}_flip{ext}"))

    # Random rotation
    angle = random.uniform(-15, 15) 
    rotated = img.rotate(angle, expand=True)
    rotated.save(os.path.join(IMAGE_DIR, f"{base}_rot{int(angle)}{ext}"))

    # Brightness adjustment (×1.5)
    enhancer = ImageEnhance.Brightness(img)
    bright = enhancer.enhance(1.5)
    bright.save(os.path.join(IMAGE_DIR, f"{base}_bright{ext}"))

    # Contrast adjustment (×1.5)
    enhancer = ImageEnhance.Contrast(img)
    contrast = enhancer.enhance(1.5)
    contrast.save(os.path.join(IMAGE_DIR, f"{base}_contrast{ext}"))

